"""验证 Tracking 的稳定身份、生命周期、确定性与无真值边界。"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from traning.config import TrackingConfig
from traning.contracts import (
    AssociationStatus,
    CandidateObservation,
    ObjectTypeDistribution,
    TrackLifecycle,
    TrainingCandidateRecord,
)
from traning.tracking.association import AssociationCostSpec, TrackAssociationView
from traning.tracking.tracker import MultiObjectTracker


_RING = ObjectTypeDistribution(1.0, 0.0, 0.0, 0.0)
_SLIDER = ObjectTypeDistribution(0.0, 1.0, 0.0, 0.0)


def _candidate(
    frame_index: int,
    candidate_id: str,
    *,
    x: float,
    embedding: tuple[float, ...],
    object_type: ObjectTypeDistribution = _RING,
    timestamp_ms: float | None = None,
    frame_id: str | None = None,
) -> CandidateObservation:
    """构造只含 runtime 可观测字段的候选。"""

    return CandidateObservation(
        frame_id=frame_id or f"frame-{frame_index}",
        frame_index=frame_index,
        timestamp_ms=(frame_index * 10.0 if timestamp_ms is None else timestamp_ms),
        candidate_id=candidate_id,
        x=x,
        y=20.0,
        confidence=0.9,
        visibility_probability=0.8,
        object_type_distribution=object_type,
        appearance_embedding=embedding,
    )


def _candidate_mapping(
    observations: tuple[object, ...],
) -> dict[str, str]:
    """把带 candidate 的跟踪输出投影为 candidate→track 映射。"""

    result: dict[str, str] = {}
    for observation in observations:
        candidate = getattr(observation, "candidate")
        if candidate is not None:
            result[candidate.candidate_id] = getattr(observation, "track_id")
    return result


def test_stable_track_ids_replay_and_input_order_invariance() -> None:
    """相同观测重放或反转候选输入顺序都不得改变稳定身份。"""

    frames = (
        (
            _candidate(0, "a-0", x=10.0, embedding=(1.0, 0.0)),
            _candidate(0, "b-0", x=80.0, embedding=(0.0, 1.0)),
        ),
        (
            _candidate(1, "a-1", x=12.0, embedding=(1.0, 0.0)),
            _candidate(1, "b-1", x=78.0, embedding=(0.0, 1.0)),
        ),
        (
            _candidate(2, "a-2", x=14.0, embedding=(1.0, 0.0)),
            _candidate(2, "b-2", x=76.0, embedding=(0.0, 1.0)),
        ),
    )

    def replay(*, reverse: bool) -> tuple[tuple[object, ...], ...]:
        """按指定输入次序重放帧序列并收集每帧轨迹。"""

        tracker = MultiObjectTracker(TrackingConfig())
        return tuple(
            tracker.update(tuple(reversed(frame)) if reverse else frame)
            for frame in frames
        )

    canonical = replay(reverse=False)
    assert canonical == replay(reverse=False)
    assert canonical == replay(reverse=True)
    assert _candidate_mapping(canonical[0]) == {
        "a-0": "track-00000001",
        "b-0": "track-00000002",
    }
    assert _candidate_mapping(canonical[2]) == {
        "a-2": "track-00000001",
        "b-2": "track-00000002",
    }
    assert all(
        observation.association is AssociationStatus.MATCHED
        for observation in canonical[1]
    )


def test_crossing_targets_follow_embedding_identity() -> None:
    """目标交叉时，稳定 ID 应跟随 appearance，而不是输入 slot 或当前位置。"""

    tracker = MultiObjectTracker(
        TrackingConfig(max_distance_px=64.0, max_embedding_distance=0.5)
    )
    created = tracker.update(
        (
            _candidate(0, "left-a", x=10.0, embedding=(1.0, 0.0)),
            _candidate(0, "right-b", x=30.0, embedding=(0.0, 1.0)),
        )
    )
    original = _candidate_mapping(created)
    crossed = tracker.update(
        (
            _candidate(1, "left-b", x=10.0, embedding=(0.0, 1.0)),
            _candidate(1, "right-a", x=30.0, embedding=(1.0, 0.0)),
        )
    )
    assert _candidate_mapping(crossed) == {
        "left-b": original["right-b"],
        "right-a": original["left-a"],
    }


def test_miss_expire_and_frame_gap_use_successful_update_count() -> None:
    """missed 按成功 update 计数；frame 跳号不放大计数，时间仍按 timestamp。"""

    tracker = MultiObjectTracker(TrackingConfig(max_missed_frames=2))
    track_id = tracker.update(
        (_candidate(0, "start", x=10.0, embedding=(1.0, 0.0), timestamp_ms=5.0),)
    )[0].track_id

    first_miss = tracker.update(
        (), frame_id="frame-10", frame_index=10, timestamp_ms=105.0
    )[0]
    assert first_miss.track_id == track_id
    assert first_miss.lifecycle is TrackLifecycle.MISSING
    assert first_miss.missed_frames == 1
    assert first_miss.track_age == 2
    assert first_miss.time_since_seen_ms == 100.0

    second_miss = tracker.update(
        (), frame_id="frame-100", frame_index=100, timestamp_ms=255.0
    )[0]
    assert second_miss.lifecycle is TrackLifecycle.MISSING
    assert second_miss.missed_frames == 2
    assert second_miss.track_age == 3
    assert second_miss.time_since_seen_ms == 250.0

    expired = tracker.update(
        (), frame_id="frame-101", frame_index=101, timestamp_ms=405.0
    )[0]
    assert expired.lifecycle is TrackLifecycle.EXPIRED
    assert expired.missed_frames == 3
    assert expired.time_since_seen_ms == 400.0
    assert tracker.snapshot() == ()
    assert (
        tracker.update((), frame_id="frame-102", frame_index=102, timestamp_ms=415.0)
        == ()
    )

    replacement = tracker.update(
        (_candidate(103, "return", x=10.0, embedding=(1.0, 0.0), timestamp_ms=425.0),)
    )[0]
    assert replacement.track_id == "track-00000002"
    assert replacement.lifecycle is TrackLifecycle.NEW

    expire_immediately = MultiObjectTracker(TrackingConfig(max_missed_frames=0))
    expire_immediately.update((_candidate(0, "only", x=0.0, embedding=(1.0, 0.0)),))
    assert (
        expire_immediately.update(
            (), frame_id="frame-1", frame_index=1, timestamp_ms=10.0
        )[0].lifecycle
        is TrackLifecycle.EXPIRED
    )


def test_equal_cost_tie_break_is_stable() -> None:
    """完全相同成本必须按 track_id、candidate_id 决胜且不受输入顺序影响。"""

    def run(*, reverse: bool) -> tuple[tuple[str, str], ...]:
        """按指定候选次序执行相同成本的稳定匹配场景。"""

        tracker = MultiObjectTracker(TrackingConfig())
        initial = (
            _candidate(0, "candidate-b", x=10.0, embedding=(1.0, 0.0)),
            _candidate(0, "candidate-a", x=10.0, embedding=(1.0, 0.0)),
        )
        tracker.update(tuple(reversed(initial)) if reverse else initial)
        tied = (
            _candidate(1, "candidate-d", x=10.0, embedding=(1.0, 0.0)),
            _candidate(1, "candidate-c", x=10.0, embedding=(1.0, 0.0)),
        )
        output = tracker.update(tuple(reversed(tied)) if reverse else tied)
        return tuple(
            (observation.track_id, observation.candidate.candidate_id)
            for observation in output
            if observation.candidate is not None
        )

    expected = (
        ("track-00000001", "candidate-c"),
        ("track-00000002", "candidate-d"),
    )
    assert run(reverse=False) == expected
    assert run(reverse=True) == expected


def test_invalid_frame_updates_are_transactional() -> None:
    """重复、乱序或帧身份混杂的失败 update 不得推进任何 tracker 状态。"""

    tracker = MultiObjectTracker(TrackingConfig())
    tracker.update(
        (_candidate(1, "valid-1", x=10.0, embedding=(1.0, 0.0), timestamp_ms=100.0),)
    )
    snapshot = tracker.snapshot()
    invalid_updates = (
        lambda: tracker.update(
            (_candidate(1, "repeat", x=10.0, embedding=(1.0, 0.0), timestamp_ms=100.0),)
        ),
        lambda: tracker.update(
            (
                _candidate(
                    0, "backward", x=10.0, embedding=(1.0, 0.0), timestamp_ms=90.0
                ),
            )
        ),
        lambda: tracker.update(
            (
                _candidate(
                    2, "time-back", x=10.0, embedding=(1.0, 0.0), timestamp_ms=99.0
                ),
            )
        ),
        lambda: tracker.update(
            (
                _candidate(
                    2,
                    "mixed-a",
                    x=10.0,
                    embedding=(1.0, 0.0),
                    frame_id="a",
                    timestamp_ms=110.0,
                ),
                _candidate(
                    2,
                    "mixed-b",
                    x=11.0,
                    embedding=(1.0, 0.0),
                    frame_id="b",
                    timestamp_ms=110.0,
                ),
            )
        ),
        lambda: tracker.update(
            (
                _candidate(
                    2, "duplicate", x=10.0, embedding=(1.0, 0.0), timestamp_ms=110.0
                ),
                _candidate(
                    2, "duplicate", x=11.0, embedding=(1.0, 0.0), timestamp_ms=110.0
                ),
            )
        ),
    )
    for invalid_update in invalid_updates:
        with pytest.raises(ValueError):
            invalid_update()
        assert tracker.snapshot() == snapshot

    matched = tracker.update(
        (_candidate(2, "valid-2", x=11.0, embedding=(1.0, 0.0), timestamp_ms=110.0),)
    )[0]
    assert matched.track_id == snapshot[0].track_id
    assert matched.track_age == 2


def test_tracking_rejects_training_records_and_statically_has_no_gt_dependency() -> (
    None
):
    """正式 tracking 入口只收 runtime candidate，源码不导入 GT 或 legacy。"""

    candidate = _candidate(0, "runtime", x=10.0, embedding=(1.0, 0.0))
    training_record = TrainingCandidateRecord(
        sample_id="sample-1",
        observation=candidate,
        matched_object=None,
        is_selected=False,
        temporal_target=(),
    )
    tracker = MultiObjectTracker(TrackingConfig())
    with pytest.raises(TypeError, match="CandidateObservation"):
        tracker.update((training_record,))  # type: ignore[arg-type]
    assert tracker.snapshot() == ()

    tracking_root = Path(__file__).parents[2] / "tracking"
    forbidden_modules = ("osu_v2", "traning.contracts.data")
    forbidden_contracts = {
        "GroundTruthObject",
        "TrainingCandidateRecord",
        "TrainingSample",
        "OutcomeTrainingSample",
    }
    for source_path in tracking_root.rglob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(
                    not any(
                        alias.name == module or alias.name.startswith(f"{module}.")
                        for module in forbidden_modules
                    )
                    for alias in node.names
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


@pytest.mark.parametrize(
    ("config", "next_candidate"),
    (
        (
            TrackingConfig(max_distance_px=10.0),
            _candidate(1, "far", x=21.0, embedding=(1.0, 0.0)),
        ),
        (
            TrackingConfig(max_embedding_distance=0.25),
            _candidate(1, "different-embedding", x=10.0, embedding=(0.0, 1.0)),
        ),
        (
            TrackingConfig(
                max_distance_px=10.0,
                spatial_weight=1.0,
                embedding_weight=0.0,
                type_weight=0.0,
                max_total_cost=0.4,
            ),
            _candidate(1, "total-cost", x=15.0, embedding=(1.0, 0.0)),
        ),
        (
            TrackingConfig(
                max_distance_px=10.0,
                min_association_confidence=0.6,
                spatial_weight=1.0,
                embedding_weight=0.0,
                type_weight=0.0,
            ),
            _candidate(1, "low-confidence", x=15.0, embedding=(1.0, 0.0)),
        ),
    ),
)
def test_association_gates_split_tracks(
    config: TrackingConfig,
    next_candidate: CandidateObservation,
) -> None:
    """任一显式门限拒绝配对时，旧轨迹 missing、候选创建新轨迹。"""

    tracker = MultiObjectTracker(config)
    tracker.update((_candidate(0, "initial", x=10.0, embedding=(1.0, 0.0)),))
    output = tracker.update((next_candidate,))
    assert tuple(observation.lifecycle for observation in output) == (
        TrackLifecycle.MISSING,
        TrackLifecycle.NEW,
    )


def test_invalid_embeddings_fail_without_mutating_state() -> None:
    """embedding 维数不一致或零范数必须硬失败，不能截断或静默降级。"""

    tracker = MultiObjectTracker(TrackingConfig())
    tracker.update((_candidate(0, "initial", x=10.0, embedding=(1.0, 0.0)),))
    snapshot = tracker.snapshot()
    wrong_dimension = _candidate(
        1,
        "wrong-dimension",
        x=10.0,
        embedding=(1.0, 0.0, 0.0),
    )
    with pytest.raises(ValueError):
        tracker.update((wrong_dimension,))
    assert tracker.snapshot() == snapshot

    # 零范数已经在 CandidateObservation 边界被拒绝，无法进入 tracker 状态机。
    with pytest.raises(ValueError, match="零向量"):
        _candidate(1, "zero-norm", x=10.0, embedding=(0.0, 0.0))
    assert tracker.snapshot() == snapshot


@pytest.mark.parametrize(
    ("weights", "expected_total"),
    (
        ((1.0, 0.0, 0.0), 0.5),
        ((0.0, 1.0, 0.0), 0.5),
        ((0.0, 0.0, 1.0), 1.0),
    ),
)
def test_one_hot_config_weights_select_exact_cost_component(
    weights: tuple[float, float, float],
    expected_total: float,
) -> None:
    """三项 one-hot 配置必须分别选择空间、embedding 和类型成本。"""

    config = TrackingConfig(
        max_distance_px=10.0,
        max_embedding_distance=1.0,
        min_association_confidence=0.0,
        spatial_weight=weights[0],
        embedding_weight=weights[1],
        type_weight=weights[2],
        max_total_cost=1.0,
    )
    previous = _candidate(
        0,
        "previous",
        x=10.0,
        embedding=(1.0, 0.0),
        object_type=_RING,
    )
    current = _candidate(
        1,
        "current",
        x=15.0,
        embedding=(0.0, 1.0),
        object_type=_SLIDER,
    )
    cost = AssociationCostSpec.from_config(config).cost(
        TrackAssociationView("track-1", previous, missed_frames=0), current
    )
    assert cost is not None
    assert cost.total == pytest.approx(expected_total)
    assert cost.confidence == pytest.approx(1.0 - expected_total)
