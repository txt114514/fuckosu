"""验证逐轨迹 Belief 的因果性、状态隔离与严格 runtime 边界。"""

from __future__ import annotations

import ast
from dataclasses import fields
from pathlib import Path

import pytest
import torch

from traning.core.belief import (
    BeliefTensorOutput,
    PerTrackBeliefEncoder,
    PerTrackBeliefRuntime,
)
from traning.conf import BeliefConfig, TrackingConfig
from traning.state import (
    AssociationStatus,
    BeliefState,
    CandidateObservation,
    ObjectTypeDistribution,
    TrackedObservation,
    TrackLifecycle,
)
from traning.core.tracking.tracker import MultiObjectTracker


_RING = ObjectTypeDistribution(1.0, 0.0, 0.0, 0.0)


def _candidate(
    frame_index: int,
    candidate_id: str,
    *,
    x: float,
    embedding: tuple[float, ...] = (1.0, 0.0),
    timestamp_ms: float | None = None,
) -> CandidateObservation:
    timestamp = frame_index * 10.0 if timestamp_ms is None else timestamp_ms
    return CandidateObservation(
        frame_id=f"frame-{frame_index}",
        frame_index=frame_index,
        timestamp_ms=timestamp,
        candidate_id=candidate_id,
        x=x,
        y=20.0,
        confidence=0.9,
        visibility_probability=0.8,
        object_type_distribution=_RING,
        appearance_embedding=embedding,
    )


def _tracked(
    track_id: str,
    frame_index: int,
    *,
    lifecycle: TrackLifecycle,
    age: int,
    x: float = 10.0,
    embedding: tuple[float, ...] = (1.0, 0.0),
    timestamp_ms: float | None = None,
    missed_frames: int = 0,
    time_since_seen_ms: float = 0.0,
) -> TrackedObservation:
    timestamp = frame_index * 10.0 if timestamp_ms is None else timestamp_ms
    candidate = None
    if lifecycle in (TrackLifecycle.NEW, TrackLifecycle.ACTIVE):
        candidate = _candidate(
            frame_index,
            f"{track_id}-candidate-{frame_index}",
            x=x,
            embedding=embedding,
            timestamp_ms=timestamp,
        )
    if lifecycle is TrackLifecycle.NEW:
        association = AssociationStatus.CREATED
        confidence = 0.9
        cost = None
    elif lifecycle is TrackLifecycle.ACTIVE:
        association = AssociationStatus.MATCHED
        confidence = 0.8
        cost = 0.2
    else:
        association = AssociationStatus.UNMATCHED
        confidence = 0.0
        cost = None
    return TrackedObservation(
        track_id=track_id,
        frame_id=f"frame-{frame_index}",
        frame_index=frame_index,
        timestamp_ms=timestamp,
        lifecycle=lifecycle,
        association=association,
        association_confidence=confidence,
        track_age=age,
        missed_frames=missed_frames,
        time_since_seen_ms=time_since_seen_ms,
        candidate=candidate,
        association_cost=cost,
    )


def _encoder() -> PerTrackBeliefEncoder:
    return PerTrackBeliefEncoder(
        BeliefConfig(input_dim=7, hidden_dim=9, layers=2, max_time_since_seen_ms=500),
        appearance_embedding_dim=2,
    )


def _assert_belief_close(first: BeliefState, second: BeliefState) -> None:
    assert first.track_id == second.track_id
    assert first.timestamp_ms == pytest.approx(second.timestamp_ms)
    assert torch.allclose(
        torch.tensor(first.belief_embedding),
        torch.tensor(second.belief_embedding),
        atol=1e-7,
        rtol=1e-7,
    )
    assert first.position_mean.x == pytest.approx(second.position_mean.x)
    assert first.position_mean.y == pytest.approx(second.position_mean.y)
    assert first.position_uncertainty.x == pytest.approx(second.position_uncertainty.x)
    assert first.position_uncertainty.y == pytest.approx(second.position_uncertainty.y)
    assert first.visibility_probability == pytest.approx(second.visibility_probability)
    assert first.uncertainty == pytest.approx(second.uncertainty)
    assert first.object_type_distribution == second.object_type_distribution
    assert first.age == second.age
    assert first.time_since_seen_ms == pytest.approx(second.time_since_seen_ms)


def test_tensor_heads_have_valid_shapes_and_backward_reaches_every_module() -> None:
    """完整 dense baseline 的 projection、GRU 与全部 head 都必须参与训练图。"""

    torch.manual_seed(5101)
    encoder = _encoder()
    features = torch.randn(4, encoder.input_feature_dim)
    output = encoder.forward_step(features)
    assert output.hidden_state.shape == (2, 4, 9)
    assert output.position_mean.shape == (4, 2)
    assert output.position_uncertainty.shape == (4, 2)
    assert output.visibility_probability.shape == (4, 1)
    assert output.type_probabilities.shape == (4, 4)
    assert output.uncertainty.shape == (4, 1)
    assert all(
        bool(torch.isfinite(getattr(output, field.name)).all())
        for field in fields(BeliefTensorOutput)
    )
    assert bool((output.position_uncertainty >= 0.0).all())
    assert bool((output.uncertainty >= 0.0).all())
    assert bool(
        (
            (output.visibility_probability >= 0.0)
            & (output.visibility_probability <= 1.0)
        ).all()
    )
    assert torch.allclose(
        output.type_probabilities.sum(dim=1), torch.ones(4), atol=1e-6, rtol=1e-6
    )

    type_weights = torch.tensor((0.0, 1.0, 2.0, 4.0))
    loss = (
        output.hidden_state.square().mean()
        + output.position_mean.square().mean()
        + output.position_uncertainty.mean()
        + output.visibility_probability.mean()
        + (output.type_probabilities * type_weights).sum(dim=1).mean()
        + output.uncertainty.mean()
    )
    loss.backward()
    module_names = (
        "projection",
        "cells",
        "position_delta_head",
        "position_uncertainty_head",
        "visibility_head",
        "type_head",
        "uncertainty_head",
    )
    for module_name in module_names:
        parameters = tuple(getattr(encoder, module_name).parameters())
        assert parameters
        assert all(parameter.grad is not None for parameter in parameters)
        assert sum(float(parameter.grad.abs().sum()) for parameter in parameters) > 0.0


def test_forward_step_is_causal_and_segmented_equals_continuous() -> None:
    """未来 suffix 不改变 prefix，传递显式 hidden 的分段递推等于连续递推。"""

    torch.manual_seed(5102)
    encoder = _encoder()
    sequence = torch.randn(
        7,
        1,
        encoder.input_feature_dim,
        requires_grad=True,
    )

    def run(
        values: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[tuple[BeliefTensorOutput, ...], torch.Tensor]:
        """逐时刻运行 encoder，并返回全部输出及最终隐状态。"""

        outputs: list[BeliefTensorOutput] = []
        current = hidden
        for features in values:
            output = encoder.forward_step(features, current)
            outputs.append(output)
            current = output.hidden_state
        assert current is not None
        return tuple(outputs), current

    continuous, _ = run(sequence)
    changed = sequence.clone()
    changed[4:] = torch.randn_like(changed[4:]) * 20.0
    changed_outputs, _ = run(changed)
    for expected, actual in zip(continuous[:4], changed_outputs[:4], strict=True):
        for field in fields(BeliefTensorOutput):
            assert torch.allclose(
                getattr(expected, field.name),
                getattr(actual, field.name),
                atol=1e-7,
                rtol=1e-7,
            )
    prefix_gradient = torch.autograd.grad(
        continuous[3].hidden_state.sum(),
        sequence,
        retain_graph=True,
    )[0]
    assert torch.count_nonzero(prefix_gradient[4:]).item() == 0

    prefix, prefix_hidden = run(sequence[:3])
    suffix, _ = run(sequence[3:], prefix_hidden)
    for expected, actual in zip(continuous, prefix + suffix, strict=True):
        assert torch.allclose(
            expected.hidden_state, actual.hidden_state, atol=1e-7, rtol=1e-7
        )
        assert torch.allclose(
            expected.position_mean, actual.position_mean, atol=1e-7, rtol=1e-7
        )


def test_runtime_isolates_tracks_from_order_and_other_track_perturbation() -> None:
    """A/B 输入反序或只扰动 B，都不得改变 A 的 belief。"""

    torch.manual_seed(5103)
    template = _encoder()

    def run(*, reverse: bool, perturb_b: bool) -> dict[str, BeliefState]:
        """在候选重排和单轨扰动条件下运行 belief 状态机。"""

        encoder = _encoder()
        encoder.load_state_dict(template.state_dict())
        runtime = PerTrackBeliefRuntime(encoder)
        first = (
            _tracked("track-a", 0, lifecycle=TrackLifecycle.NEW, age=1, x=10.0),
            _tracked("track-b", 0, lifecycle=TrackLifecycle.NEW, age=1, x=30.0),
        )
        runtime.step(tuple(reversed(first)) if reverse else first)
        second = (
            _tracked("track-a", 1, lifecycle=TrackLifecycle.ACTIVE, age=2, x=11.0),
            _tracked(
                "track-b",
                1,
                lifecycle=TrackLifecycle.ACTIVE,
                age=2,
                x=300.0 if perturb_b else 31.0,
                embedding=(-1.0, 0.0) if perturb_b else (1.0, 0.0),
            ),
        )
        return {
            belief.track_id: belief
            for belief in runtime.step(tuple(reversed(second)) if reverse else second)
        }

    canonical = run(reverse=False, perturb_b=False)
    reversed_order = run(reverse=True, perturb_b=False)
    perturbed = run(reverse=False, perturb_b=True)
    _assert_belief_close(canonical["track-a"], reversed_order["track-a"])
    _assert_belief_close(canonical["track-b"], reversed_order["track-b"])
    _assert_belief_close(canonical["track-a"], perturbed["track-a"])
    assert (
        canonical["track-b"].belief_embedding != perturbed["track-b"].belief_embedding
    )


def test_tracker_to_belief_preserves_stable_identity() -> None:
    """Tracking 的稳定 track_id 必须原样贯穿公共 BeliefState。"""

    torch.manual_seed(5104)
    tracker = MultiObjectTracker(TrackingConfig())
    runtime = PerTrackBeliefRuntime(_encoder())
    created = tracker.update((_candidate(0, "candidate-0", x=10.0),))
    first = runtime.step(created)[0]
    matched = tracker.update((_candidate(1, "candidate-1", x=11.0),))
    second = runtime.step(matched)[0]
    assert first.track_id == created[0].track_id
    assert second.track_id == first.track_id == matched[0].track_id
    assert second.age == 2


def test_missing_expired_equal_timestamp_and_clear_replay() -> None:
    """MISSING 使用 previous；EXPIRED 当帧返回后移除；clear 后可确定性重放。"""

    torch.manual_seed(5105)
    encoder = _encoder()
    with torch.no_grad():
        encoder.position_delta_head.weight.zero_()
        encoder.position_delta_head.bias.zero_()
    runtime = PerTrackBeliefRuntime(encoder)
    new = _tracked(
        "track-a", 0, lifecycle=TrackLifecycle.NEW, age=1, x=12.0, timestamp_ms=10.0
    )
    initial = runtime.step((new,))[0]
    missing = _tracked(
        "track-a",
        1,
        lifecycle=TrackLifecycle.MISSING,
        age=2,
        timestamp_ms=10.0,
        missed_frames=1,
        time_since_seen_ms=0.0,
    )
    retained = runtime.step((missing,))[0]
    assert retained.track_id == initial.track_id
    assert retained.timestamp_ms == 10.0
    assert retained.position_mean == initial.position_mean
    assert runtime.state_for("track-a") == retained

    expired = _tracked(
        "track-a",
        10,
        lifecycle=TrackLifecycle.EXPIRED,
        age=3,
        timestamp_ms=20.0,
        missed_frames=2,
        time_since_seen_ms=10.0,
    )
    expired_belief = runtime.step((expired,))[0]
    assert expired_belief.track_id == "track-a"
    assert runtime.state_for("track-a") is None
    assert runtime.snapshot() == ()

    runtime.clear()
    replayed = runtime.step((new,))[0]
    _assert_belief_close(initial, replayed)


def test_invalid_batches_preserve_snapshot_and_runtime_clock() -> None:
    """全部校验成功前不得提交 per-track state 或全局 frame clock。"""

    torch.manual_seed(5106)
    runtime = PerTrackBeliefRuntime(_encoder())
    runtime.step(
        (
            _tracked(
                "track-a", 1, lifecycle=TrackLifecycle.NEW, age=1, timestamp_ms=100.0
            ),
        )
    )
    snapshot = runtime.snapshot()
    valid_frame_20 = _tracked(
        "track-a", 20, lifecycle=TrackLifecycle.ACTIVE, age=2, timestamp_ms=100.0
    )
    invalid_batches: tuple[tuple[object, ...], ...] = (
        (),
        (valid_frame_20, valid_frame_20),
        (
            valid_frame_20,
            _tracked(
                "track-b", 21, lifecycle=TrackLifecycle.NEW, age=1, timestamp_ms=100.0
            ),
        ),
        (
            _tracked(
                "track-a", 1, lifecycle=TrackLifecycle.ACTIVE, age=2, timestamp_ms=100.0
            ),
        ),
        (
            _tracked(
                "track-a", 20, lifecycle=TrackLifecycle.ACTIVE, age=2, timestamp_ms=99.0
            ),
        ),
        (
            _tracked(
                "track-a",
                20,
                lifecycle=TrackLifecycle.ACTIVE,
                age=2,
                timestamp_ms=100.0,
                embedding=(1.0, 0.0, 0.0),
            ),
        ),
        (object(),),
    )
    for batch in invalid_batches:
        with pytest.raises((TypeError, ValueError)):
            runtime.step(batch)  # type: ignore[arg-type]
        assert runtime.snapshot() == snapshot

    # frame_index 可跳号但一次成功 step 只推进一次，且失败的 frame=20 未提交 clock。
    advanced = runtime.step((valid_frame_20,))[0]
    assert advanced.age == 2
    assert advanced.timestamp_ms == 100.0


def test_belief_model_has_no_action_gt_legacy_or_wide_type_boundary() -> None:
    """模型结构和源码不得重新引入 action/candidate head、GT、legacy 或 Any。"""

    encoder = _encoder()
    state_keys = tuple(encoder.state_dict())
    forbidden_key_fragments = (
        "action",
        "candidate_head",
        "candidate_logits",
        "selected",
    )
    assert all(
        fragment not in key
        for key in state_keys
        for fragment in forbidden_key_fragments
    )

    belief_root = Path(__file__).parents[2] / "core/belief"
    forbidden_modules = ("osu_v2", "traning.state.data", "typing.Any")
    forbidden_contracts = {
        "DecisionAction",
        "GroundTruthObject",
        "OutcomeTrainingSample",
        "TrainingCandidateRecord",
        "TrainingSample",
    }
    for source_path in belief_root.rglob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not any(
                        alias.name == module or alias.name.startswith(f"{module}.")
                        for module in forbidden_modules
                    )
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert not any(
                    module == forbidden or module.startswith(f"{forbidden}.")
                    for forbidden in forbidden_modules
                )
                assert forbidden_contracts.isdisjoint(
                    alias.name for alias in node.names
                )
                assert all(alias.name != "Any" for alias in node.names)
