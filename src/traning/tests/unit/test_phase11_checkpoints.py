"""Phase 11 runtime checkpoint 坐标身份与完整性门禁。"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from traning.app.factory import build_frame_coordinate_transform
from traning.belief import PerTrackBeliefEncoder
from traning.config import (
    BeliefConfig,
    CoordinateConfig,
    DecisionConfig,
    OutcomeConfig,
    PerceptionConfig,
    RuntimeConfig,
    RuntimeDevice,
    V2Config,
)
from traning.infrastructure import IntegrityError, SchemaMismatchError
from traning.outcome import DenseOutcomeModel
from traning.perception import PerceptionModel
from traning.training import (
    RuntimeModelBundle,
    load_runtime_checkpoint,
    publish_runtime_checkpoint,
)


def _config() -> V2Config:
    """构造可快速保存与恢复的小型 CPU 模型配置。"""

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
            transform_identity="checkpoint-coordinate-v1",
            affine_matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        ),
        runtime=RuntimeConfig(
            device=RuntimeDevice.CPU,
            require_cuda=False,
            amp=False,
        ),
    )


def _models(config: V2Config) -> RuntimeModelBundle:
    """构造与配置和坐标指纹一致的三模型 bundle。"""

    perception = PerceptionModel(config.perception)
    belief = PerTrackBeliefEncoder(
        config.belief,
        appearance_embedding_dim=config.perception.embedding_dim,
    )
    outcome = DenseOutcomeModel(
        config.outcome,
        belief_embedding_dim=belief.flattened_hidden_dim,
    )
    return RuntimeModelBundle(
        perception,
        belief,
        outcome,
        "runtime-checkpoint-test",
        build_frame_coordinate_transform(config).transform_fingerprint,
    )


def test_checkpoint_roundtrip_preserves_models_and_coordinate_identity(
    tmp_path: Path,
) -> None:
    """manifest、权重和坐标身份全通过后才返回可装配模型 bundle。"""

    torch.manual_seed(11011)
    config = _config()
    coordinate_transform = build_frame_coordinate_transform(config)
    source = _models(config)
    manifest = publish_runtime_checkpoint(
        tmp_path,
        config,
        source,
        coordinate_transform,
        dataset_id="phase11-dataset",
        producer_id="phase11-test",
        created_at_ms=11.0,
    )
    loaded = load_runtime_checkpoint(
        tmp_path,
        config,
        coordinate_transform,
        expected_dataset_id="phase11-dataset",
    )

    assert loaded.artifact_id == manifest.artifact.artifact_id
    assert loaded.transform_fingerprint == coordinate_transform.transform_fingerprint
    model_pairs = (
        (source.perception_model, loaded.perception_model),
        (source.belief_encoder, loaded.belief_encoder),
        (source.outcome_model, loaded.outcome_model),
    )
    for expected_model, actual_model in model_pairs:
        assert tuple(expected_model.state_dict()) == tuple(actual_model.state_dict())
        assert all(
            torch.equal(expected, actual)
            for expected, actual in zip(
                expected_model.state_dict().values(),
                actual_model.state_dict().values(),
                strict=True,
            )
        )


def test_checkpoint_rejects_weight_corruption(tmp_path: Path) -> None:
    """权重 generation 被修改后必须在 torch 解码前由 SHA-256 拒绝。"""

    config = _config()
    coordinate_transform = build_frame_coordinate_transform(config)
    manifest = publish_runtime_checkpoint(
        tmp_path,
        config,
        _models(config),
        coordinate_transform,
        dataset_id="phase11-dataset",
        producer_id="phase11-test",
        created_at_ms=11.0,
    )
    weights_path = tmp_path / manifest.weights_filename
    weights_path.write_bytes(weights_path.read_bytes() + b"corrupt")

    with pytest.raises(IntegrityError, match="SHA-256"):
        load_runtime_checkpoint(
            tmp_path,
            config,
            coordinate_transform,
            expected_dataset_id="phase11-dataset",
        )


def test_checkpoint_rejects_old_or_changed_coordinate_fingerprint(
    tmp_path: Path,
) -> None:
    """结构相同的旧坐标权重也不得绕过 manifest 指纹门禁。"""

    config = _config()
    coordinate_transform = build_frame_coordinate_transform(config)
    publish_runtime_checkpoint(
        tmp_path,
        config,
        _models(config),
        coordinate_transform,
        dataset_id="phase11-dataset",
        producer_id="phase11-test",
        created_at_ms=11.0,
    )
    manifest_path = tmp_path / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["metadata"]["transform_fingerprint"] = "transform-0123456789abcdef"
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(SchemaMismatchError, match="坐标变换指纹"):
        load_runtime_checkpoint(
            tmp_path,
            config,
            coordinate_transform,
            expected_dataset_id="phase11-dataset",
        )


def test_checkpoint_manifest_requires_coordinate_provenance(tmp_path: Path) -> None:
    """缺少变换指纹的旧 manifest schema 不能被当作新 checkpoint 加载。"""

    config = _config()
    coordinate_transform = build_frame_coordinate_transform(config)
    publish_runtime_checkpoint(
        tmp_path,
        config,
        _models(config),
        coordinate_transform,
        dataset_id="phase11-dataset",
        producer_id="phase11-test",
        created_at_ms=11.0,
    )
    manifest_path = tmp_path / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    del payload["metadata"]["transform_fingerprint"]
    manifest_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(SchemaMismatchError, match="typed schema"):
        load_runtime_checkpoint(
            tmp_path,
            config,
            coordinate_transform,
            expected_dataset_id="phase11-dataset",
        )


def test_checkpoint_rejects_wrong_dataset_identity(tmp_path: Path) -> None:
    """权重结构相同也不能绕过训练数据集身份门禁。"""

    config = _config()
    coordinate_transform = build_frame_coordinate_transform(config)
    publish_runtime_checkpoint(
        tmp_path,
        config,
        _models(config),
        coordinate_transform,
        dataset_id="phase11-dataset",
        producer_id="phase11-test",
        created_at_ms=11.0,
    )

    with pytest.raises(SchemaMismatchError, match="dataset identity"):
        load_runtime_checkpoint(
            tmp_path,
            config,
            coordinate_transform,
            expected_dataset_id="other-dataset",
        )


def test_checkpoint_gates_model_contract_not_deployment_paths(tmp_path: Path) -> None:
    """部署目录可调整，但任一三模型结构字段变化都必须拒绝。"""

    config = _config()
    coordinate_transform = build_frame_coordinate_transform(config)
    publish_runtime_checkpoint(
        tmp_path,
        config,
        _models(config),
        coordinate_transform,
        dataset_id="phase11-dataset",
        producer_id="phase11-test",
        created_at_ms=11.0,
    )
    deployment_config = replace(
        config,
        telemetry=replace(config.telemetry, directory=tmp_path / "other-telemetry"),
    )
    loaded = load_runtime_checkpoint(
        tmp_path,
        deployment_config,
        build_frame_coordinate_transform(deployment_config),
        expected_dataset_id="phase11-dataset",
    )
    assert loaded.artifact_id == "runtime-checkpoint-test"

    changed_model_config = replace(
        config,
        perception=replace(config.perception, embedding_dim=3),
    )
    with pytest.raises(SchemaMismatchError, match="模型契约摘要"):
        load_runtime_checkpoint(
            tmp_path,
            changed_model_config,
            build_frame_coordinate_transform(changed_model_config),
            expected_dataset_id="phase11-dataset",
        )


@pytest.mark.parametrize("timestamp", (float("nan"), float("inf")))
def test_checkpoint_rejects_non_finite_timestamp_before_writing(
    tmp_path: Path,
    timestamp: float,
) -> None:
    """非有限发布时间必须在创建任何 generation 前硬失败。"""

    config = _config()
    coordinate_transform = build_frame_coordinate_transform(config)

    with pytest.raises(ValueError, match="有限"):
        publish_runtime_checkpoint(
            tmp_path,
            config,
            _models(config),
            coordinate_transform,
            dataset_id="phase11-dataset",
            producer_id="phase11-test",
            created_at_ms=timestamp,
        )
    assert tuple(tmp_path.iterdir()) == ()
