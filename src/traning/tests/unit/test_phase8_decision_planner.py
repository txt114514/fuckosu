"""Phase 8 确定性最优停止规划器验收。"""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import pytest

from traning.conf import DecisionConfig
from traning.state import (
    BeliefState,
    DecisionAction,
    ObjectTypeDistribution,
    OutcomeDistribution,
    Point2D,
)
from traning.core.decision.planner import OptimalStoppingPlanner


_PLANNER_PATH = Path(__file__).resolve().parents[2] / "core/decision/planner.py"


def _config(**changes: float | tuple[int, ...]) -> DecisionConfig:
    base = DecisionConfig(
        horizons_ms=(0, 16),
        click_cost=0.0,
        invalid_penalty=0.0,
        miss_penalty=0.0,
        expire_penalty=0.0,
        risk_lambda=0.0,
        wait_cost=0.0,
        min_confidence=0.0,
    )
    return replace(base, **changes)


def _belief(track_id: str, x: float) -> BeliefState:
    return BeliefState(
        track_id=track_id,
        timestamp_ms=1000.0,
        belief_embedding=(x, 0.2),
        position_mean=Point2D(x, 20.0),
        position_uncertainty=Point2D(1.0, 1.0),
        visibility_probability=0.9,
        object_type_distribution=ObjectTypeDistribution(0.7, 0.1, 0.1, 0.1),
        age=2,
        time_since_seen_ms=0.0,
        uncertainty=0.1,
    )


def _outcome(
    track_id: str,
    horizon_ms: float,
    value: float,
    *,
    success: float = 1.0,
    variance: float = 0.0,
) -> OutcomeDistribution:
    return OutcomeDistribution(
        track_id=track_id,
        horizon_ms=horizon_ms,
        p_invalid=1.0 - success,
        p_miss=0.0,
        p_low_score=0.0,
        p_medium_score=0.0,
        p_high_score=success,
        p_expire=0.0,
        expected_score=value,
        variance=variance,
    )


def _two_track_inputs() -> tuple[
    tuple[BeliefState, ...], tuple[OutcomeDistribution, ...]
]:
    beliefs = (_belief("track-b", 20.0), _belief("track-a", 10.0))
    outcomes = (
        _outcome("track-b", 0.0, 0.5),
        _outcome("track-a", 16.0, 0.1),
        _outcome("track-a", 0.0, 0.8),
        _outcome("track-b", 16.0, 0.1),
    )
    return beliefs, outcomes


def test_clicks_current_track_with_maximum_utility_and_preserves_binding() -> None:
    """规划器必须点击当前效用最高轨迹并保留目标绑定。"""

    beliefs, outcomes = _two_track_inputs()
    result = OptimalStoppingPlanner(_config()).plan(beliefs, outcomes, 1000.0)

    assert result.action is DecisionAction.CLICK
    assert result.track_id == "track-a"
    assert result.target_position == beliefs[1].position_mean
    assert result.outcome == outcomes[2]
    assert result.execute_at_ms == 1000.0
    assert result.horizon_ms == 0.0
    assert result.expected_utility == pytest.approx(0.8)
    assert result.wait_utility == pytest.approx(0.1)


def test_waits_when_future_utility_is_clearly_better() -> None:
    """未来效用明显更高时规划器必须选择等待。"""

    belief = _belief("track-a", 10.0)
    outcomes = (
        _outcome("track-a", 0.0, 0.2),
        _outcome("track-a", 16.0, 0.9),
    )
    result = OptimalStoppingPlanner(_config(wait_cost=0.1)).plan(
        (belief,), outcomes, 1000.0
    )

    assert result.action is DecisionAction.WAIT
    assert result.track_id is None
    assert result.target_position is None
    assert result.outcome is None
    assert result.execute_at_ms == 1016.0
    assert result.horizon_ms == 16.0
    assert result.expected_utility == pytest.approx(0.8)


def test_risk_penalty_changes_selected_track() -> None:
    """风险惩罚必须能使规划器从高方差轨迹转向稳健轨迹。"""

    beliefs = (_belief("risky", 10.0), _belief("steady", 20.0))
    outcomes = (
        _outcome("risky", 0.0, 0.8, variance=1.0),
        _outcome("risky", 16.0, 0.0),
        _outcome("steady", 0.0, 0.7, variance=0.0),
        _outcome("steady", 16.0, 0.0),
    )

    no_risk = OptimalStoppingPlanner(_config(risk_lambda=0.0)).plan(
        beliefs, outcomes, 1000.0
    )
    risk_aware = OptimalStoppingPlanner(_config(risk_lambda=0.2)).plan(
        beliefs, outcomes, 1000.0
    )

    assert no_risk.track_id == "risky"
    assert risk_aware.track_id == "steady"


def test_input_slot_and_order_do_not_change_selection() -> None:
    """输入槽位与排列顺序不得改变最终决策。"""

    beliefs, outcomes = _two_track_inputs()
    planner = OptimalStoppingPlanner(_config())

    first = planner.plan(beliefs, outcomes, 1000.0)
    reordered = planner.plan(
        tuple(reversed(beliefs)), tuple(reversed(outcomes)), 1000.0
    )

    assert first == reordered


def test_stable_track_tie_and_click_now_tie_policy() -> None:
    """轨迹同效用时必须稳定决胜，并优先立即点击。"""

    beliefs = (_belief("track-z", 20.0), _belief("track-a", 10.0))
    outcomes = tuple(
        _outcome(track_id, horizon, 0.5)
        for track_id in ("track-z", "track-a")
        for horizon in (0.0, 16.0)
    )
    result = OptimalStoppingPlanner(_config()).plan(beliefs, outcomes, 1000.0)

    assert result.action is DecisionAction.CLICK
    assert result.track_id == "track-a"
    assert result.expected_utility == result.wait_utility


def test_low_current_success_forces_wait() -> None:
    """当前成功概率低于门槛时必须强制等待。"""

    belief = _belief("track-a", 10.0)
    outcomes = (
        _outcome("track-a", 0.0, 0.9, success=0.4),
        _outcome("track-a", 16.0, 0.1, success=0.4),
    )
    result = OptimalStoppingPlanner(_config(min_confidence=0.5)).plan(
        (belief,), outcomes, 1000.0
    )

    assert result.action is DecisionAction.WAIT
    assert result.confidence == pytest.approx(0.4)


def test_empty_inputs_return_explicit_costed_wait() -> None:
    """空输入必须返回携带等待成本的显式 WAIT。"""

    result = OptimalStoppingPlanner(_config(wait_cost=0.2)).plan((), (), 1000.0)

    assert result.action is DecisionAction.WAIT
    assert result.expected_utility == pytest.approx(-0.2)
    assert result.wait_utility == pytest.approx(-0.2)
    assert result.execute_at_ms == 1016.0


def test_rejects_duplicate_incomplete_unknown_extra_and_nonfinite_inputs() -> None:
    """规划器必须拒绝重复、不完整、额外轨迹及非有限输入。"""

    planner = OptimalStoppingPlanner(_config())
    belief = _belief("track-a", 10.0)
    current = _outcome("track-a", 0.0, 0.5)
    future = _outcome("track-a", 16.0, 0.6)

    with pytest.raises(ValueError, match="唯一"):
        planner.plan((belief, belief), (current, future), 1000.0)
    with pytest.raises(ValueError, match="必须提供"):
        planner.plan((belief,), (current,), 1000.0)
    with pytest.raises(ValueError, match="未知"):
        planner.plan(
            (belief,),
            (current, future, _outcome("unknown", 0.0, 0.1)),
            1000.0,
        )
    with pytest.raises(ValueError, match="只接受"):
        planner.plan(
            (belief,),
            (current, future, _outcome("track-a", 32.0, 0.7)),
            1000.0,
        )
    with pytest.raises(ValueError, match="有限"):
        planner.plan((belief,), (current, future), float("nan"))
    with pytest.raises(ValueError, match="规划时间一致"):
        planner.plan((belief,), (current, future), 1001.0)


def test_rejects_duplicate_track_horizon_outcome() -> None:
    """同一轨迹与 horizon 的重复 outcome 必须被拒绝。"""

    planner = OptimalStoppingPlanner(_config())
    belief = _belief("track-a", 10.0)
    current = _outcome("track-a", 0.0, 0.5)
    future = _outcome("track-a", 16.0, 0.6)

    with pytest.raises(ValueError, match="不得重复"):
        planner.plan((belief,), (current, current, future), 1000.0)


def test_planner_ast_has_no_forbidden_runtime_inputs_or_shortcuts() -> None:
    """规划器源码不得读取禁用训练信息或旧捷径字段。"""

    tree = ast.parse(_PLANNER_PATH.read_text(encoding="utf-8"))
    names = {
        identifier.lower()
        for node in ast.walk(tree)
        for identifier in ((node.id,) if isinstance(node, ast.Name) else ())
    }
    attributes = {
        node.attr.lower() for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    imports = {
        alias.name.lower()
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    forbidden_fragments = ("image", "groundtruth", "oracle", "candidate", "logit")

    assert "any" not in imports
    assert "argmax" not in names | attributes
    assert not any(
        fragment in identifier
        for fragment in forbidden_fragments
        for identifier in names | attributes | imports
    )
