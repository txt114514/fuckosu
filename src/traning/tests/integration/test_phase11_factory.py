"""单一 V2 config 到 runtime 与坐标适配器的装配验收。"""

from __future__ import annotations

import pytest

from traning.core.app.factory import (
    assemble_runtime_pipeline,
    build_frame_coordinate_transform,
    build_untrained_runtime_for_smoke,
)
from traning.core.app.runtime import V2RuntimePipeline
from traning.core.belief import PerTrackBeliefEncoder
from traning.conf import (
    BeliefConfig,
    CoordinateConfig,
    DecisionConfig,
    OutcomeConfig,
    PerceptionConfig,
    RuntimeConfig,
    RuntimeDevice,
    V2Config,
)
from traning.core.data import OsuPoint
from traning.core.outcome import DenseOutcomeModel
from traning.core.perception import PerceptionModel
from traning.core.training import RuntimeModelBundle


def _cpu_config() -> V2Config:
    """构造尺寸较小但领域契约完整的 CPU smoke 配置。"""

    return V2Config(
        perception=PerceptionConfig(
            frame_width=32,
            frame_height=32,
            embedding_dim=2,
        ),
        belief=BeliefConfig(input_dim=4, hidden_dim=4, layers=1),
        outcome=OutcomeConfig(hidden_dims=(4,), horizons_ms=(0, 16)),
        decision=DecisionConfig(horizons_ms=(0, 16)),
        coordinates=CoordinateConfig(
            source_width=513,
            source_height=385,
            transform_identity="identity-test-v1",
            affine_matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        ),
        runtime=RuntimeConfig(
            device=RuntimeDevice.CPU,
            require_cuda=False,
            amp=False,
        ),
    )


def test_untrained_builder_is_explicit_and_coordinate_config_is_shared() -> None:
    """smoke factory 可运行，但正式坐标仍来自同一显式 config。"""

    config = _cpu_config()
    pipeline = build_untrained_runtime_for_smoke(config)
    coordinates = build_frame_coordinate_transform(config)

    assert isinstance(pipeline, V2RuntimePipeline)
    mapped = coordinates.ground_truth_to_training_target(
        OsuPoint(80.0, 101.0),
        source_frame_width=513,
        source_frame_height=385,
    )
    assert (mapped.x, mapped.y) == (80.0, 101.0)
    assert coordinates.transform_identity == "identity-test-v1"


def test_assembly_rejects_model_config_drift() -> None:
    """checkpoint 模型与启动配置不一致时不得静默装配。"""

    config = _cpu_config()
    perception = PerceptionModel(config.perception)
    belief = PerTrackBeliefEncoder(config.belief, appearance_embedding_dim=2)
    outcome = DenseOutcomeModel(config.outcome, belief.flattened_hidden_dim)
    drifted_perception = PerceptionModel(
        PerceptionConfig(frame_width=64, frame_height=32, embedding_dim=2)
    )
    transform_fingerprint = build_frame_coordinate_transform(
        config
    ).transform_fingerprint

    with pytest.raises(ValueError, match="PerceptionModel config"):
        assemble_runtime_pipeline(
            config,
            models=RuntimeModelBundle(
                drifted_perception,
                belief,
                outcome,
                "checkpoint-drifted",
                transform_fingerprint,
            ),
        )
    # 正确模型组合仍能装配，证明失败不是依赖缺失。
    assert isinstance(
        assemble_runtime_pipeline(
            config,
            models=RuntimeModelBundle(
                perception,
                belief,
                outcome,
                "checkpoint-valid",
                transform_fingerprint,
            ),
        ),
        V2RuntimePipeline,
    )


def test_assembly_rejects_checkpoint_coordinate_drift() -> None:
    """相同网络结构但由旧坐标系训练的权重 bundle 必须在装配前拒绝。"""

    config = _cpu_config()
    perception = PerceptionModel(config.perception)
    belief = PerTrackBeliefEncoder(config.belief, appearance_embedding_dim=2)
    outcome = DenseOutcomeModel(config.outcome, belief.flattened_hidden_dim)
    with pytest.raises(ValueError, match="坐标变换指纹"):
        assemble_runtime_pipeline(
            config,
            models=RuntimeModelBundle(
                perception,
                belief,
                outcome,
                "checkpoint-old-coordinate",
                "transform-0123456789abcdef",
            ),
        )


def test_missing_affine_calibration_is_not_centered_fallback() -> None:
    """默认无矩阵配置必须硬失败，不猜测 playfield 居中矩形。"""

    with pytest.raises(ValueError, match="affine_matrix"):
        build_frame_coordinate_transform(
            V2Config(
                runtime=RuntimeConfig(
                    device=RuntimeDevice.CPU,
                    require_cuda=False,
                    amp=False,
                )
            )
        )
