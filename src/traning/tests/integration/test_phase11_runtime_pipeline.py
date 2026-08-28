"""Phase 11 正式 runtime 链路的多帧集成验收。"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch
from torch import nn

from package import AffineOsuVideoTransform
from traning.app import RuntimeStepResult, V2RuntimePipeline
from traning.belief import PerTrackBeliefEncoder, PerTrackBeliefRuntime
from traning.config import (
    BeliefConfig,
    DecisionConfig,
    OutcomeConfig,
    PerceptionConfig,
    TrackingConfig,
)
from traning.contracts import (
    BeliefState,
    DecisionAction,
    OutcomeDistribution,
    RuntimeFrame,
    TrackLifecycle,
)
from traning.decision import OptimalStoppingPlanner
from traning.data import FrameCoordinateTransform
from traning.outcome import DenseOutcomeModel
from traning.perception import DensePerceptionOutput, PerceptionRuntime
from traning.tracking import MultiObjectTracker


_RUNTIME_PATH = Path(__file__).resolve().parents[2] / "app/runtime.py"


class _PixelControlledPerceptionModel:
    """用首像素选择空帧、单目标或双目标稠密预测。"""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, image: torch.Tensor) -> DensePerceptionOutput:
        self.calls += 1
        marker = int(round(float(image[0, 0, 0, 0].cpu().item()) * 255.0))
        height, width = 2, 4
        scalar = torch.full(
            (1, 1, height, width),
            -20.0,
            device=image.device,
            dtype=image.dtype,
        )
        type_logits = torch.full(
            (1, 4, height, width),
            -20.0,
            device=image.device,
            dtype=image.dtype,
        )
        type_logits[:, 0] = 20.0
        embedding = torch.zeros(
            (1, 2, height, width),
            device=image.device,
            dtype=image.dtype,
        )
        # 默认单位向量确保即使空帧也满足稠密输出的数值边界。
        embedding[:, 0] = 1.0
        embedding[0, :, 0, 3] = torch.tensor(
            (0.0, 1.0), device=image.device, dtype=image.dtype
        )
        if marker == 1:
            scalar[0, 0, 0, 0] = 12.0
        elif marker == 2:
            scalar[0, 0, 0, 0] = 12.0
            scalar[0, 0, 0, 3] = 10.0
        elif marker == 3:
            # 两个位置不变，只交换分数次序。
            scalar[0, 0, 0, 0] = 10.0
            scalar[0, 0, 0, 3] = 12.0
        elif marker != 0:
            raise ValueError(f"未知测试帧 marker：{marker}")
        return DensePerceptionOutput(
            center_logits=scalar,
            visibility_logits=torch.full_like(scalar, 20.0),
            type_logits=type_logits,
            xy_offsets=torch.zeros(
                (1, 2, height, width), device=image.device, dtype=image.dtype
            ),
            ring_logits=torch.full_like(scalar, 20.0),
            ring_radius=torch.ones_like(scalar),
            slider_logits=torch.full_like(scalar, -20.0),
            slider_direction=torch.zeros(
                (1, 2, height, width), device=image.device, dtype=image.dtype
            ),
            spinner_logits=torch.full_like(scalar, -20.0),
            identity_embedding=embedding,
        )


class _FailingOutcomeModel(DenseOutcomeModel):
    """在 stateful 边界之后模拟不可恢复的预测异常。"""

    def predict(
        self,
        belief: BeliefState,
        horizon_ms: float,
    ) -> OutcomeDistribution:
        """稳定抛出预测异常，以验证 runtime 的失败锁存边界。"""

        raise RuntimeError(f"测试预测失败：{belief.track_id}@{float(horizon_ms):.1f}ms")


def _frame(index: int, marker: int) -> RuntimeFrame:
    """构造具有真实 RGB 字节长度的运行时帧。"""

    width, height = 8, 4
    return RuntimeFrame(
        frame_id=f"runtime-frame-{index}",
        frame_index=index,
        timestamp_ms=index * 16.0,
        width=width,
        height=height,
        image_bytes=bytes((marker,)) * width * height * 3,
    )


def _outcome_model(
    belief_dim: int,
    *,
    future_is_better: bool,
) -> DenseOutcomeModel:
    """构造仅由 horizon 控制、可精确触发 CLICK 或 WAIT 的 dense 模型。"""

    model = DenseOutcomeModel(
        OutcomeConfig(hidden_dims=(1,), horizons_ms=(0, 16)),
        belief_embedding_dim=belief_dim,
    )
    with torch.no_grad():
        trunk = model.trunk[0]
        assert isinstance(trunk, nn.Linear)
        trunk.weight.zero_()
        trunk.bias.fill_(-2.0 if future_is_better else 2.0)
        trunk.weight[0, -2] = 4.0 if future_is_better else -4.0
        model.category_head.weight.zero_()
        model.category_head.bias.zero_()
        model.category_head.weight[4, 0] = 5.0
        model.expiry_head.weight.zero_()
        model.expiry_head.bias.fill_(-20.0)
    return model


def _pipeline(
    *,
    future_is_better: bool,
    max_missed_frames: int = 1,
    fail_predictions: bool = False,
) -> tuple[
    V2RuntimePipeline,
    _PixelControlledPerceptionModel,
    MultiObjectTracker,
    PerTrackBeliefRuntime,
]:
    """组装完整的正式层级，并返回可观测的有状态组件。"""

    perception_config = PerceptionConfig(
        frame_width=8,
        frame_height=4,
        embedding_dim=2,
        score_threshold=0.1,
        nms_radius_px=0.1,
    )
    perception_model = _PixelControlledPerceptionModel()
    perception_runtime = PerceptionRuntime(perception_model, perception_config)
    tracker = MultiObjectTracker(
        TrackingConfig(
            max_distance_px=16.0,
            max_embedding_distance=0.5,
            max_missed_frames=max_missed_frames,
        )
    )
    encoder = PerTrackBeliefEncoder(
        BeliefConfig(input_dim=4, hidden_dim=4, layers=1),
        appearance_embedding_dim=2,
    )
    belief_runtime = PerTrackBeliefRuntime(encoder)
    if fail_predictions:
        outcome_model = _FailingOutcomeModel(
            OutcomeConfig(hidden_dims=(1,), horizons_ms=(0, 16)),
            belief_embedding_dim=encoder.flattened_hidden_dim,
        )
    else:
        outcome_model = _outcome_model(
            encoder.flattened_hidden_dim,
            future_is_better=future_is_better,
        )
    planner = OptimalStoppingPlanner(
        DecisionConfig(
            horizons_ms=(0, 16),
            click_cost=0.0,
            invalid_penalty=0.0,
            miss_penalty=0.0,
            expire_penalty=0.0,
            min_confidence=0.0,
            risk_lambda=0.0,
            wait_cost=0.0,
        )
    )
    return (
        V2RuntimePipeline(
            perception_runtime,
            tracker,
            belief_runtime,
            outcome_model,
            planner,
            FrameCoordinateTransform(
                source_frame_width=8,
                source_frame_height=4,
                transform_identity="runtime-test-v1",
                transform=AffineOsuVideoTransform(
                    (
                        (7.0 / 512.0, 0.0, 0.0),
                        (0.0, 3.0 / 384.0, 0.0),
                    )
                ),
            ),
        ),
        perception_model,
        tracker,
        belief_runtime,
    )


def test_multi_object_pipeline_is_stable_when_perception_score_order_swaps() -> None:
    """分数排序交换不得改变目标身份、输出顺序或正式 CLICK 绑定。"""

    torch.manual_seed(11001)
    pipeline, _model, _tracker, _belief_runtime = _pipeline(future_is_better=False)
    first = pipeline.step(_frame(0, 2))
    second = pipeline.step(_frame(1, 3))

    assert isinstance(first, RuntimeStepResult)
    assert (
        first.coordinate_transform_fingerprint
        == pipeline.coordinate_transform.transform_fingerprint
    )
    assert tuple(item.x for item in first.candidates) == (1.0, 7.0)
    assert tuple(item.x for item in second.candidates) == (1.0, 7.0)
    assert tuple(item.track_id for item in second.tracks) == (
        "track-00000001",
        "track-00000002",
    )
    assert tuple(item.candidate.x for item in second.tracks if item.candidate) == (
        1.0,
        7.0,
    )
    assert tuple(item.track_id for item in second.active_beliefs) == (
        "track-00000001",
        "track-00000002",
    )
    assert tuple((item.track_id, item.horizon_ms) for item in second.outcomes) == (
        ("track-00000001", 0.0),
        ("track-00000001", 16.0),
        ("track-00000002", 0.0),
        ("track-00000002", 16.0),
    )
    assert second.decision.action is DecisionAction.CLICK
    assert second.decision.track_id == "track-00000001"
    assert second.decision.target_position == second.active_beliefs[0].position_mean


def test_future_value_produces_explicit_wait() -> None:
    """相同正式链路在未来收益更高时必须输出 WAIT 而非当前点击。"""

    torch.manual_seed(11002)
    pipeline, _model, _tracker, _belief_runtime = _pipeline(future_is_better=True)
    result = pipeline.step(_frame(0, 1))

    assert result.decision.action is DecisionAction.WAIT
    assert result.decision.track_id is None
    assert result.decision.horizon_ms == 16.0
    assert result.decision.execute_at_ms == 16.0
    assert result.outcomes[1].expected_score > result.outcomes[0].expected_score


def test_empty_frame_context_expires_track_and_excludes_expired_belief() -> None:
    """空候选仍推进帧；EXPIRED 仅留审计 track，不得进入预测或 Decision。"""

    torch.manual_seed(11003)
    pipeline, _model, tracker, belief_runtime = _pipeline(
        future_is_better=False,
        max_missed_frames=0,
    )
    created = pipeline.step(_frame(0, 1))
    assert len(created.active_beliefs) == 1

    expired = pipeline.step(_frame(1, 0))
    assert expired.candidates == ()
    assert len(expired.tracks) == 1
    assert expired.tracks[0].lifecycle is TrackLifecycle.EXPIRED
    assert expired.active_beliefs == ()
    assert expired.outcomes == ()
    assert expired.decision.action is DecisionAction.WAIT
    assert tracker.snapshot() == ()
    assert belief_runtime.snapshot() == ()

    # 已无轨迹的后续空帧仍携带显式上下文并正常产生 WAIT。
    empty = pipeline.step(_frame(2, 0))
    assert empty.tracks == ()
    assert empty.active_beliefs == ()
    assert empty.outcomes == ()
    assert empty.decision.action is DecisionAction.WAIT


def test_frame_validation_precedes_components_and_reset_restarts_identity() -> None:
    """重复帧在感知前拒绝；reset 后帧游标、track ID 与 belief 一起重启。"""

    torch.manual_seed(11004)
    pipeline, model, tracker, belief_runtime = _pipeline(future_is_better=False)
    first = pipeline.step(_frame(0, 1))
    calls_after_first = model.calls
    tracker_snapshot = tracker.snapshot()
    belief_snapshot = belief_runtime.snapshot()

    with pytest.raises(ValueError, match="严格递增"):
        pipeline.step(_frame(0, 1))
    assert model.calls == calls_after_first
    assert tracker.snapshot() == tracker_snapshot
    assert belief_runtime.snapshot() == belief_snapshot
    assert pipeline.requires_reset is False

    with pytest.raises(ValueError, match="标定尺寸不一致"):
        pipeline.step(
            RuntimeFrame(
                frame_id="wrong-size",
                frame_index=1,
                timestamp_ms=16.0,
                width=9,
                height=4,
                image_bytes=bytes(9 * 4 * 3),
            )
        )
    assert model.calls == calls_after_first

    pipeline.reset()
    replay = pipeline.step(_frame(0, 1))
    assert replay.tracks[0].track_id == "track-00000001"
    assert replay.decision.action is first.decision.action


def test_stateful_failure_latches_pipeline_until_reset() -> None:
    """有状态边界后的异常不得允许调用方在可能不一致的状态上继续。"""

    torch.manual_seed(11005)
    pipeline, model, tracker, belief_runtime = _pipeline(
        future_is_better=False,
        fail_predictions=True,
    )
    with pytest.raises(RuntimeError, match="测试预测失败"):
        pipeline.step(_frame(0, 1))
    assert pipeline.requires_reset is True
    assert len(tracker.snapshot()) == 1
    assert len(belief_runtime.snapshot()) == 1

    calls_after_failure = model.calls
    with pytest.raises(RuntimeError, match="必须调用 reset"):
        pipeline.step(_frame(1, 1))
    assert model.calls == calls_after_failure

    pipeline.reset()
    assert pipeline.requires_reset is False
    assert tracker.snapshot() == ()
    assert belief_runtime.snapshot() == ()


def test_runtime_source_has_no_training_only_or_shortcut_dependency() -> None:
    """静态阻止训练信息、旧动作捷径和稀疏实验实现进入正式 app runtime。"""

    source = _RUNTIME_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(_RUNTIME_PATH))
    forbidden_modules = ("osu_v2",)
    forbidden_names = {
        "Any",
        "GroundTruthObject",
        "TrainingCandidateRecord",
        "TrainingSample",
        "OutcomeTrainingSample",
        "OutcomeOracle",
        "SMET",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(
                not alias.name.startswith(forbidden_modules) for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith(forbidden_modules)
        elif isinstance(node, ast.Name):
            assert node.id not in forbidden_names
        elif isinstance(node, ast.Attribute):
            assert node.attr not in forbidden_names
    lowered = source.lower()
    assert "action_logits" not in lowered
    assert "candidate_logits" not in lowered
