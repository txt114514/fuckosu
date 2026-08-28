"""Phase 9 canonical hard-example routing 验收。"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from traning.contracts import DataSplit
from traning.evaluation import (
    EvaluationTag,
    PrimaryError,
    SequenceScore,
    SequenceEvaluationEvent,
    build_sequence_evaluation_events,
)
from traning.training.hard_examples import (
    EvaluationSplitEvent,
    HardExampleConsumer,
    HardExampleDestination,
    HardExampleExclusionReason,
    HardExampleRoute,
    HardExampleWeight,
    build_hard_example_plan,
)


_SOURCE_PATH = Path(__file__).resolve().parents[2] / "training/hard_examples.py"


def _event_id(number: int) -> str:
    return f"sequence-event-{number:064x}"


def _failed_event(
    number: int,
    primary_error: PrimaryError,
    tag: EvaluationTag,
) -> SequenceEvaluationEvent:
    return SequenceEvaluationEvent(
        event_id=_event_id(number),
        sample_id="sample-1",
        frame_index=number,
        passed=False,
        primary_error=primary_error,
        error_tags=(tag,),
        target_id="target-1",
        click_index=number,
        click_x=100.0,
        click_y=80.0,
    )


def _passed_event(number: int) -> SequenceEvaluationEvent:
    return SequenceEvaluationEvent(
        event_id=_event_id(number),
        sample_id="sample-1",
        frame_index=number,
        passed=True,
        primary_error=PrimaryError.NONE,
        error_tags=(),
        target_id="target-1",
        click_index=number,
        click_x=100.0,
        click_y=80.0,
    )


def test_frame_105_unresolved_routes_only_to_decision() -> None:
    """零点击未解析目标必须保持 decision identity，绝不转为空间错误。"""

    score = SequenceScore(
        clicks=(), resolved_targets=(), unresolved_target_ids=("target-105",)
    )
    event = build_sequence_evaluation_events("sample-105", 105, score)[0]
    plan = build_hard_example_plan((EvaluationSplitEvent(event, DataSplit.TRAIN),))

    assert event.primary_error is PrimaryError.DECISION
    assert len(plan.weights) == 1
    assert plan.weights[0].event is event
    assert plan.weights[0].route.destination is HardExampleDestination.DECISION
    assert plan.weights[0].event.primary_error is not PrimaryError.SPATIAL


def test_registry_routes_three_primary_errors_and_skips_passed() -> None:
    """注册表必须路由三类主错误，并跳过已通过样本。"""

    inputs = (
        EvaluationSplitEvent(
            _failed_event(1, PrimaryError.SPATIAL, EvaluationTag.SPATIAL_MISS),
            DataSplit.TRAIN,
        ),
        EvaluationSplitEvent(
            _failed_event(2, PrimaryError.TEMPORAL, EvaluationTag.EARLY_CLICK),
            DataSplit.TRAIN,
        ),
        EvaluationSplitEvent(
            _failed_event(3, PrimaryError.DECISION, EvaluationTag.NO_ACTIVE_TARGET),
            DataSplit.TRAIN,
        ),
        EvaluationSplitEvent(_passed_event(4), DataSplit.TRAIN),
    )
    plan = build_hard_example_plan(inputs)

    assert [route.destination for route in plan.routes] == [
        HardExampleDestination.PERCEPTION,
        HardExampleDestination.OUTCOME,
        HardExampleDestination.DECISION,
    ]
    assert [weight.weight for weight in plan.weights] == [1.0, 1.0, 1.0]
    assert len(plan.excluded) == 1
    assert plan.excluded[0].reason is HardExampleExclusionReason.PASSED


def test_validation_and_test_are_explicitly_excluded_from_weights() -> None:
    """验证集与测试集事件不得进入训练重加权。"""

    event_validation = _failed_event(
        10, PrimaryError.SPATIAL, EvaluationTag.SPATIAL_MISS
    )
    event_test = _failed_event(
        11, PrimaryError.DECISION, EvaluationTag.NO_ACTIVE_TARGET
    )
    plan = build_hard_example_plan(
        (
            EvaluationSplitEvent(event_test, DataSplit.TEST),
            EvaluationSplitEvent(event_validation, DataSplit.VALIDATION),
        )
    )

    assert plan.weights == ()
    assert {item.event for item in plan.excluded} == {event_validation, event_test}
    assert all(
        item.reason is HardExampleExclusionReason.NON_TRAIN_SPLIT
        for item in plan.excluded
    )


def test_all_split_is_rejected() -> None:
    """无法确定用途的 ALL split 必须被明确拒绝。"""

    with pytest.raises(ValueError, match="DataSplit.ALL"):
        EvaluationSplitEvent(
            _failed_event(20, PrimaryError.SPATIAL, EvaluationTag.SPATIAL_MISS),
            DataSplit.ALL,
        )


def test_consumer_views_preserve_same_event_object_and_error_identity() -> None:
    """各 hard-example 消费视图必须共享同一事件及错误身份。"""

    event = _failed_event(30, PrimaryError.DECISION, EvaluationTag.NO_ACTIVE_TARGET)
    plan = build_hard_example_plan((EvaluationSplitEvent(event, DataSplit.TRAIN),))
    consumer_events = tuple(
        plan.events_for(consumer)[0] for consumer in HardExampleConsumer
    )

    assert all(item is event for item in consumer_events)
    assert all(item.primary_error is PrimaryError.DECISION for item in consumer_events)


def test_replay_is_stably_sorted_and_duplicate_event_id_is_rejected() -> None:
    """重放必须稳定排序，并拒绝重复事件 ID。"""

    first = EvaluationSplitEvent(
        _failed_event(41, PrimaryError.TEMPORAL, EvaluationTag.LATE_CLICK),
        DataSplit.TRAIN,
    )
    second = EvaluationSplitEvent(
        _failed_event(40, PrimaryError.SPATIAL, EvaluationTag.SPATIAL_MISS),
        DataSplit.TRAIN,
    )
    forward = build_hard_example_plan((first, second))
    reverse = build_hard_example_plan((second, first))
    assert forward == reverse
    assert [route.event.event_id for route in forward.routes] == sorted(
        (first.event.event_id, second.event.event_id)
    )
    with pytest.raises(ValueError, match="重复 event_id"):
        build_hard_example_plan((first, first))


@pytest.mark.parametrize("weight", (0.0, -1.0, float("nan"), float("inf")))
def test_hard_example_weight_must_be_finite_positive(weight: float) -> None:
    """hard-example 权重必须是有限正数。"""

    source = EvaluationSplitEvent(
        _failed_event(50, PrimaryError.SPATIAL, EvaluationTag.SPATIAL_MISS),
        DataSplit.TRAIN,
    )
    route = HardExampleRoute(source, HardExampleDestination.PERCEPTION)
    with pytest.raises(ValueError, match="有限正数"):
        HardExampleWeight(route, weight)


def test_hard_example_source_has_no_forbidden_dependency_or_rescoring() -> None:
    """路由层不得读取视觉旁路、oracle 或重新调用 scorer。"""

    tree = ast.parse(_SOURCE_PATH.read_text(encoding="utf-8"))
    forbidden_names = {
        "Any",
        "candidate",
        "overlay",
        "OutcomeOracle",
        "score_click_sequence",
        "score_point",
        "score_slider",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(not alias.name.startswith("osu_v2") for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert not module.startswith(("osu_v2", "traning.outcome.oracle"))
        if isinstance(node, ast.Name):
            assert node.id not in forbidden_names
        if isinstance(node, ast.Attribute):
            assert node.attr not in {"candidate", "overlay"}
