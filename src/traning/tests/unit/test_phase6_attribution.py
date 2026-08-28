"""Phase 6 typed sequence attribution 验收。"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from traning.evaluation import (
    SCORE_VERSION,
    EvaluationTag,
    PredictedClick,
    PrimaryError,
    SequenceEvaluationEvent,
    TargetObject,
    build_sequence_evaluation_events,
    score_click_sequence,
)


_OSU_V2_ROOT = Path(__file__).resolve().parents[2]
_VALID_EVENT_ID = "sequence-event-" + "0" * 64


def test_frame_105_without_clicks_is_only_decision_unresolved() -> None:
    """无点击帧不得被 overlay/candidate 证据改写为空间错误。"""

    target = TargetObject(
        "target-105",
        "circle",
        1050.0,
        1050.0,
        x=100.0,
        y=80.0,
    )
    score = score_click_sequence((target,), (), circle_radius=20.0)
    events = build_sequence_evaluation_events("sample-105", 105, score)

    assert len(events) == 1
    event = events[0]
    assert event.frame_index == 105
    assert event.passed is False
    assert event.primary_error is PrimaryError.DECISION
    assert event.error_tags == (EvaluationTag.UNRESOLVED_TARGET,)
    assert event.target_id == "target-105"
    assert event.click_index is None


def test_hit_miss_and_frequency_limited_mapping_is_exact() -> None:
    """每个 click evaluation 必须一对一保留状态与 canonical 标签。"""

    targets = (
        TargetObject("a", "circle", 1000.0, 1000.0, x=100.0, y=80.0),
        TargetObject("b", "circle", 1200.0, 1200.0, x=160.0, y=80.0),
    )
    clicks = (
        PredictedClick(1005.0, 102.0, 82.0),
        PredictedClick(1020.0, 101.0, 81.0),
        PredictedClick(1205.0, 161.0, 79.0),
        PredictedClick(1500.0, 0.0, 0.0),
    )
    result = score_click_sequence(targets, clicks, circle_radius=20.0)
    events = build_sequence_evaluation_events("sample-map", 12, result)

    assert [event.click_index for event in events] == [0, 1, 2, 3]
    assert [event.passed for event in events] == [True, False, True, False]
    assert [event.primary_error for event in events] == [
        PrimaryError.NONE,
        PrimaryError.DECISION,
        PrimaryError.NONE,
        PrimaryError.DECISION,
    ]
    assert [event.error_tags for event in events] == [
        (),
        (EvaluationTag.FREQUENCY_LIMITED,),
        (),
        (EvaluationTag.NO_ACTIVE_TARGET,),
    ]
    assert [event.target_id for event in events] == ["a", None, "b", None]


def test_event_replay_and_unresolved_order_are_stable() -> None:
    """相同领域输入必须产生完全相同的顺序和 canonical hash。"""

    targets = (
        TargetObject("z", "circle", 2000.0, 2000.0, x=0.0, y=0.0),
        TargetObject("a", "circle", 1000.0, 1000.0, x=0.0, y=0.0),
    )
    result = score_click_sequence(targets, (), circle_radius=10.0)
    first = build_sequence_evaluation_events("sample-stable", 8, result)
    second = build_sequence_evaluation_events("sample-stable", 8, result)

    assert first == second
    assert [event.target_id for event in first] == ["a", "z"]
    assert len({event.event_id for event in first}) == 2
    assert all(len(event.event_id) == len("sequence-event-") + 64 for event in first)


@pytest.mark.parametrize(
    "overrides",
    (
        {"event_id": "not-canonical"},
        {"frame_index": True},
        {"passed": True},
        {"primary_error": PrimaryError.SPATIAL},
        {"error_tags": (EvaluationTag.UNRESOLVED_TARGET, EvaluationTag.SPATIAL_MISS)},
        {"target_id": None},
        {"click_index": 0},
        {"score_version": "wrong-version"},
    ),
)
def test_unresolved_event_contract_is_strict(overrides: dict[str, object]) -> None:
    """未解析目标事件的 invariant 不允许消费者自行降级或改写。"""

    values: dict[str, object] = {
        "event_id": _VALID_EVENT_ID,
        "sample_id": "sample-contract",
        "frame_index": 105,
        "passed": False,
        "primary_error": PrimaryError.DECISION,
        "error_tags": (EvaluationTag.UNRESOLVED_TARGET,),
        "target_id": "target-105",
        "click_index": None,
        "score_version": SCORE_VERSION,
    }
    values.update(overrides)
    with pytest.raises((TypeError, ValueError)):
        SequenceEvaluationEvent(**values)  # type: ignore[arg-type]


def test_click_event_contract_rejects_contradictory_pass_and_tags() -> None:
    """通过 click 必须无错误；失败 click 必须具有错误域和标签。"""

    with pytest.raises(ValueError):
        SequenceEvaluationEvent(
            _VALID_EVENT_ID,
            "sample-contract",
            1,
            True,
            PrimaryError.SPATIAL,
            (EvaluationTag.SPATIAL_MISS,),
            "target-1",
            0,
        )
    with pytest.raises(ValueError):
        SequenceEvaluationEvent(
            _VALID_EVENT_ID,
            "sample-contract",
            1,
            False,
            PrimaryError.NONE,
            (),
            "target-1",
            0,
        )


def test_attribution_source_has_no_legacy_any_or_rescoring() -> None:
    """归因层只能投影 SequenceScore，不得读取视觉旁路或重新评分。"""

    path = _OSU_V2_ROOT / "evaluation/attribution.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    forbidden_names = {
        "Any",
        "candidate",
        "overlay",
        "score_click_sequence",
        "score_point",
        "score_slider",
        "spatial_coefficient",
        "temporal_coefficient",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(not alias.name.startswith("osu_v2") for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            assert node.module is None or not node.module.startswith("osu_v2")
        if isinstance(node, ast.Name):
            assert node.id not in forbidden_names
        if isinstance(node, ast.Attribute):
            assert node.attr not in {"candidate", "overlay"}
