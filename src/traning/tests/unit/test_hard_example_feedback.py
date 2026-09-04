"""验证 hard-example 帧级反馈的聚合、身份绑定和 split 隔离。"""

from __future__ import annotations

from pathlib import Path

import pytest
from package import AffineOsuVideoTransform

from traning.state import DataSplit
from traning.core.data import FrameCoordinateTransform, SegmentTrainingDataset
from traning.core.evaluation import (
    EvaluationTag,
    PrimaryError,
    SequenceEvaluationEvent,
    SequenceScore,
    build_sequence_evaluation_events,
)
from traning.lib.infrastructure import (
    IntegrityError,
    SchemaMismatchError,
    atomic_write_json,
    read_json_object,
)
from traning.lib.data import SegmentAnnotation, SegmentRecord
from traning.lib.data.annotation import (
    DifficultyAnnotation,
    SourceAnnotation,
)
from traning.core.training import (
    EvaluationSplitEvent,
    HardExampleDestination,
    HardExampleExclusionReason,
    HardExampleFeedbackStore,
    ParameterVector,
    build_hard_example_plan,
)


_SEQUENCE_ID = "long_sequence/02__item_000001_long_sequence_000008"
_DATASET_ID = f"dataset-{'d' * 64}"
_CONFIG_SHA256 = "c" * 64
_AFFINE_MATRIX = ((2.0, 0.0, 100.0), (0.0, 1.5, 50.0))


def _parameters(*, score_threshold: float = 0.05) -> ParameterVector:
    """构造 registry 内的 canonical source proposal。"""

    return ParameterVector(0.001, score_threshold, 64, 0.1, 0.0, 0.0)


def _transform() -> FrameCoordinateTransform:
    """构造测试数据与 feedback store 共享的坐标身份。"""

    return FrameCoordinateTransform(
        source_frame_width=1280,
        source_frame_height=720,
        transform_identity="hard-feedback-test",
        transform=AffineOsuVideoTransform(_AFFINE_MATRIX),
    )


def _record(sequence_id: str, *, duration_ms: int = 2_000) -> SegmentRecord:
    """构造无需读取视频即可展开到 frame 105 的 segment 记录。"""

    annotation = SegmentAnnotation(
        schema_version=1,
        segment_id=sequence_id,
        dataset_dimension="long_sequence",
        category="long_sequence",
        difficulty=DifficultyAnnotation(
            approach_preempt_ms=600.0,
            circle_radius_osu_pixels=32.0,
        ),
        source=SourceAnnotation(
            folder_name="item_000001",
            osu_filename="test.osu",
            clip_start_ms=0,
            clip_end_ms=duration_ms,
        ),
        hit_objects=(),
    )
    return SegmentRecord(
        key=sequence_id,
        item_name="item_000001",
        category="long_sequence",
        dataset_dimension="long_sequence",
        directory=Path("/unused"),
        video_path=Path("/unused/video.mp4"),
        annotation_path=Path("/unused/annotation.json"),
        annotation=annotation,
    )


def _dataset(
    *,
    records: tuple[SegmentRecord, ...] | None = None,
) -> SegmentTrainingDataset:
    """构造只使用 typed frame references 的 TRAIN dataset。"""

    return SegmentTrainingDataset(
        (_record(_SEQUENCE_ID),) if records is None else records,
        split=DataSplit.TRAIN,
        sample_fps=60.0,
        frame_step=1,
        max_frames_per_segment=None,
        visibility_post_ms=100.0,
        coordinate_transform=_transform(),
    )


def _store(path: Path, dataset: SegmentTrainingDataset) -> HardExampleFeedbackStore:
    """用固定 run/data/config/transform 身份构造反馈 store。"""

    fingerprint = dataset.transform_fingerprint
    assert fingerprint is not None
    return HardExampleFeedbackStore(
        path,
        run_id="hard-feedback-run",
        dataset_id=_DATASET_ID,
        config_sha256=_CONFIG_SHA256,
        transform_fingerprint=fingerprint,
        train_dataset=dataset,
    )


def _failed_event(
    number: int,
    *,
    sample_id: str = _SEQUENCE_ID,
    frame_index: int = 105,
    primary_error: PrimaryError = PrimaryError.SPATIAL,
    tag: EvaluationTag = EvaluationTag.SPATIAL_MISS,
) -> SequenceEvaluationEvent:
    """构造可用于聚合或 split 隔离的 canonical 失败事件。"""

    return SequenceEvaluationEvent(
        event_id=f"sequence-event-{number:064x}",
        sample_id=sample_id,
        frame_index=frame_index,
        passed=False,
        primary_error=primary_error,
        error_tags=(tag,),
        target_id="target-1",
        click_index=number,
        click_x=100.0,
        click_y=80.0,
    )


def _passed_event(number: int) -> SequenceEvaluationEvent:
    """构造应被审计但不得进入权重的 TRAIN 通过事件。"""

    return SequenceEvaluationEvent(
        event_id=f"sequence-event-{number:064x}",
        sample_id=_SEQUENCE_ID,
        frame_index=36,
        passed=True,
        primary_error=PrimaryError.NONE,
        error_tags=(),
        target_id="target-36",
        click_index=number,
        click_x=411.75,
        click_y=230.40,
    )


def test_frame_105_unresolved_maps_exactly_to_decision_dataset_frame(
    tmp_path: Path,
) -> None:
    """旧 frame 105 的 unresolved 事件必须精确映射到 Decision 与索引 105。"""

    dataset = _dataset()
    event = build_sequence_evaluation_events(
        _SEQUENCE_ID,
        105,
        SequenceScore(
            clicks=(),
            resolved_targets=(),
            unresolved_target_ids=("target-105",),
        ),
    )[0]
    plan = build_hard_example_plan((EvaluationSplitEvent(event, DataSplit.TRAIN),))
    artifact = _store(tmp_path / "feedback.json", dataset).persist(
        plan,
        source_trial_index=24,
        source_parameters=_parameters(),
        evaluated=True,
        created_at_ms=1.0,
    )

    assert event.primary_error is PrimaryError.DECISION
    assert artifact.source_events[0].event_id == event.event_id
    assert artifact.frame_weights == artifact.weights_for(
        HardExampleDestination.DECISION
    )
    weight = artifact.frame_weights[0]
    assert weight.identity == (_SEQUENCE_ID, 105)
    assert weight.aggregation_key == (
        _SEQUENCE_ID,
        105,
        HardExampleDestination.DECISION,
    )
    assert weight.dataset_index == 105
    assert weight.effective_weight == pytest.approx(2.0)


def test_routes_aggregate_by_frame_and_destination_with_bonus_and_cap(
    tmp_path: Path,
) -> None:
    """同帧同领域 route 按规定公式聚合，且不同领域不会互相加权。"""

    events = (
        _failed_event(1),
        _failed_event(2),
        _failed_event(
            3,
            primary_error=PrimaryError.DECISION,
            tag=EvaluationTag.NO_ACTIVE_TARGET,
        ),
    )
    plan = build_hard_example_plan(
        tuple(EvaluationSplitEvent(event, DataSplit.TRAIN) for event in events)
    )
    artifact = _store(tmp_path / "feedback.json", _dataset()).persist(
        plan,
        source_trial_index=0,
        source_parameters=_parameters(),
        evaluated=True,
        bonus=0.75,
        max_weight=2.0,
        created_at_ms=2.0,
    )

    perception = artifact.weights_for(HardExampleDestination.PERCEPTION)[0]
    decision = artifact.weights_for(HardExampleDestination.DECISION)[0]
    assert perception.route_weight_sum == pytest.approx(2.0)
    assert perception.effective_weight == pytest.approx(2.0)
    assert decision.route_weight_sum == pytest.approx(1.0)
    assert decision.effective_weight == pytest.approx(1.75)


def test_validation_and_test_remain_excluded_from_persisted_weights(
    tmp_path: Path,
) -> None:
    """validation/test 只能进入 excluded 审计，不能污染下一 trial 权重。"""

    train_failure = _failed_event(10)
    validation_failure = _failed_event(11, sample_id="validation-sequence")
    test_failure = _failed_event(
        12,
        sample_id="test-sequence",
        primary_error=PrimaryError.DECISION,
        tag=EvaluationTag.NO_ACTIVE_TARGET,
    )
    train_passed = _passed_event(13)
    plan = build_hard_example_plan(
        (
            EvaluationSplitEvent(test_failure, DataSplit.TEST),
            EvaluationSplitEvent(train_failure, DataSplit.TRAIN),
            EvaluationSplitEvent(validation_failure, DataSplit.VALIDATION),
            EvaluationSplitEvent(train_passed, DataSplit.TRAIN),
        )
    )
    artifact = _store(tmp_path / "feedback.json", _dataset()).persist(
        plan,
        source_trial_index=3,
        source_parameters=_parameters(),
        evaluated=True,
        created_at_ms=3.0,
    )

    assert tuple(item.event_id for item in artifact.source_events) == (
        train_failure.event_id,
    )
    assert tuple(item.event_ids for item in artifact.frame_weights) == (
        (train_failure.event_id,),
    )
    excluded = {item.event_id: item for item in artifact.excluded}
    assert excluded[validation_failure.event_id].reason is (
        HardExampleExclusionReason.NON_TRAIN_SPLIT
    )
    assert excluded[test_failure.event_id].reason is (
        HardExampleExclusionReason.NON_TRAIN_SPLIT
    )
    assert excluded[train_passed.event_id].reason is HardExampleExclusionReason.PASSED


def test_unevaluated_trial_is_explicitly_persisted_as_empty_feedback(
    tmp_path: Path,
) -> None:
    """未到 evaluation 的 trial 仍有恢复点，但不得伪造 event 或权重。"""

    dataset = _dataset()
    store = _store(tmp_path / "feedback.json", dataset)
    artifact = store.persist(
        None,
        source_trial_index=7,
        source_parameters=_parameters(),
        evaluated=False,
        created_at_ms=4.0,
    )
    loaded = store.load(
        expected_source_trial_index=7,
        expected_source_parameters=_parameters(),
    )

    assert not artifact.evaluated
    assert artifact.source_events == artifact.frame_weights == artifact.excluded == ()
    assert loaded == artifact
    with pytest.raises(ValueError, match="plan 必须是 None"):
        store.persist(
            build_hard_example_plan(()),
            source_trial_index=8,
            source_parameters=_parameters(),
            evaluated=False,
        )


def test_payload_tampering_is_rejected_before_typed_load(tmp_path: Path) -> None:
    """任一 payload 字段被改写都必须触发完整 SHA 校验失败。"""

    dataset = _dataset()
    store = _store(tmp_path / "feedback.json", dataset)
    plan = build_hard_example_plan(
        (EvaluationSplitEvent(_failed_event(20), DataSplit.TRAIN),)
    )
    store.persist(
        plan,
        source_trial_index=2,
        source_parameters=_parameters(),
        evaluated=True,
        created_at_ms=5.0,
    )
    payload = read_json_object(store.path)
    frame_weights = payload["frame_weights"]
    assert isinstance(frame_weights, list) and isinstance(frame_weights[0], dict)
    frame_weights[0]["effective_weight"] = 99.0
    atomic_write_json(store.path, payload)

    with pytest.raises(IntegrityError, match="SHA-256"):
        store.load(
            expected_source_trial_index=2,
            expected_source_parameters=_parameters(),
        )


@pytest.mark.parametrize(
    ("store_overrides", "trial_index", "parameters"),
    (
        ({"run_id": "another-run"}, 5, _parameters()),
        ({"dataset_id": f"dataset-{'e' * 64}"}, 5, _parameters()),
        ({"config_sha256": "f" * 64}, 5, _parameters()),
        ({}, 6, _parameters()),
        ({}, 5, _parameters(score_threshold=0.06)),
    ),
)
def test_load_rejects_run_data_config_trial_or_parameter_identity_mismatch(
    tmp_path: Path,
    store_overrides: dict[str, str],
    trial_index: int,
    parameters: ParameterVector,
) -> None:
    """恢复必须同时匹配 run/data/config/source trial 和完整 proposal。"""

    dataset = _dataset()
    path = tmp_path / "feedback.json"
    _store(path, dataset).persist(
        None,
        source_trial_index=5,
        source_parameters=_parameters(),
        evaluated=False,
        created_at_ms=6.0,
    )
    fingerprint = dataset.transform_fingerprint
    assert fingerprint is not None
    context = {
        "run_id": "hard-feedback-run",
        "dataset_id": _DATASET_ID,
        "config_sha256": _CONFIG_SHA256,
        "transform_fingerprint": fingerprint,
    }
    context.update(store_overrides)
    mismatched = HardExampleFeedbackStore(
        path,
        train_dataset=dataset,
        **context,
    )

    with pytest.raises(SchemaMismatchError, match="不一致"):
        mismatched.load(
            expected_source_trial_index=trial_index,
            expected_source_parameters=parameters,
        )


def test_load_remaps_sequence_frame_and_rejects_stale_dataset_index(
    tmp_path: Path,
) -> None:
    """恢复时必须用当前 TRAIN dataset 重解析帧，拒绝旧索引静默指错样本。"""

    path = tmp_path / "feedback.json"
    original = _dataset()
    plan = build_hard_example_plan(
        (EvaluationSplitEvent(_failed_event(30), DataSplit.TRAIN),)
    )
    _store(path, original).persist(
        plan,
        source_trial_index=9,
        source_parameters=_parameters(),
        evaluated=True,
        created_at_ms=7.0,
    )
    shifted = _dataset(
        records=(
            _record("another-sequence"),
            _record(_SEQUENCE_ID),
        )
    )

    with pytest.raises(SchemaMismatchError, match="dataset_index"):
        _store(path, shifted).load(
            expected_source_trial_index=9,
            expected_source_parameters=_parameters(),
        )
