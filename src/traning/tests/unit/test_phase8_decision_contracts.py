"""Phase 8 decision contract 与配置验收。"""

from __future__ import annotations

from dataclasses import replace

import pytest

from traning.conf import (
    DecisionConfig,
    OutcomeConfig,
    V2Config,
    load_v2_config,
    v2_config_to_dict,
)
from traning.state import (
    DecisionAction,
    DecisionResult,
    OutcomeDistribution,
    Point2D,
)


def _outcome(
    *, track_id: str = "track-1", horizon_ms: float = 0.0
) -> OutcomeDistribution:
    return OutcomeDistribution(
        track_id=track_id,
        horizon_ms=horizon_ms,
        p_invalid=0.05,
        p_miss=0.10,
        p_low_score=0.15,
        p_medium_score=0.25,
        p_high_score=0.45,
        p_expire=0.0,
        expected_score=0.7,
        variance=0.1,
    )


def _click(**overrides: object) -> DecisionResult:
    values: dict[str, object] = {
        "action": DecisionAction.CLICK,
        "track_id": "track-1",
        "execute_at_ms": 1000.0,
        "expected_utility": 0.8,
        "wait_utility": 0.5,
        "confidence": 0.9,
        "horizon_ms": 0.0,
        "target_position": Point2D(100.0, 80.0),
        "outcome": _outcome(),
    }
    values.update(overrides)
    return DecisionResult(**values)  # type: ignore[arg-type]


def _wait(**overrides: object) -> DecisionResult:
    values: dict[str, object] = {
        "action": DecisionAction.WAIT,
        "track_id": None,
        "execute_at_ms": 1000.0,
        "expected_utility": 0.6,
        "wait_utility": 0.6,
        "confidence": 0.8,
        "horizon_ms": 16.0,
        "target_position": None,
        "outcome": None,
    }
    values.update(overrides)
    return DecisionResult(**values)  # type: ignore[arg-type]


def test_decision_config_round_trip_includes_risk_and_wait_cost() -> None:
    """新增字段必须完整穿过 typed→dict→loader 边界。"""

    config = replace(
        V2Config(),
        decision=replace(V2Config().decision, risk_lambda=0.25, wait_cost=0.03),
    )
    payload = v2_config_to_dict(config)
    decision_payload = payload["decision"]
    assert isinstance(decision_payload, dict)
    assert decision_payload["risk_lambda"] == 0.25
    assert decision_payload["wait_cost"] == 0.03
    assert load_v2_config(payload) == config


def test_decision_action_must_be_enum() -> None:
    """动作字段必须使用 DecisionAction 枚举而非裸字符串。"""

    with pytest.raises(TypeError, match="DecisionAction"):
        _wait(action="wait")


def test_click_requires_immediate_horizon_and_complete_audit_fields() -> None:
    """CLICK 只表达立即执行，且保留目标与 outcome 审计证据。"""

    result = _click()
    assert result.action is DecisionAction.CLICK
    assert result.horizon_ms == 0.0
    assert result.track_id == result.outcome.track_id  # type: ignore[union-attr]
    with pytest.raises(ValueError, match="horizon_ms=0"):
        _click(horizon_ms=16.0)
    with pytest.raises(ValueError, match="outcome.*horizon_ms=0"):
        _click(outcome=_outcome(horizon_ms=16.0))
    with pytest.raises(ValueError, match="必须包含"):
        _click(target_position=None)
    with pytest.raises(ValueError, match="track_id 必须一致"):
        _click(outcome=_outcome(track_id="other-track"))
    with pytest.raises(ValueError, match="不得低于"):
        _click(expected_utility=0.4, wait_utility=0.5)


def test_wait_requires_positive_horizon_and_no_bound_target_or_outcome() -> None:
    """WAIT_ONE_STEP 必须选择正 horizon 且不得伪装成延迟 CLICK。"""

    result = _wait()
    assert result.action is DecisionAction.WAIT
    assert result.horizon_ms > 0.0
    with pytest.raises(ValueError, match="正数 horizon"):
        _wait(horizon_ms=0.0)
    with pytest.raises(ValueError, match="不得绑定"):
        _wait(track_id="track-1")
    with pytest.raises(ValueError, match="不得绑定"):
        _wait(outcome=_outcome())
    with pytest.raises(ValueError, match="必须等于"):
        _wait(expected_utility=0.5, wait_utility=0.6)


@pytest.mark.parametrize("config_type", (OutcomeConfig, DecisionConfig))
@pytest.mark.parametrize("horizons", ((16, 32), (0,), (0, 0, 16), (0, -1, 16)))
def test_horizons_require_click_now_and_positive_wait_step(
    config_type: type[OutcomeConfig] | type[DecisionConfig],
    horizons: tuple[int, ...],
) -> None:
    """horizon 集必须包含立即点击点和至少一个正等待步长。"""

    with pytest.raises((TypeError, ValueError)):
        config_type(horizons_ms=horizons)


@pytest.mark.parametrize(
    ("field_name", "value", "error"),
    (
        ("risk_lambda", -0.1, ValueError),
        ("risk_lambda", float("nan"), ValueError),
        ("risk_lambda", True, TypeError),
        ("wait_cost", -0.1, ValueError),
        ("wait_cost", float("inf"), ValueError),
        ("wait_cost", "0.1", TypeError),
    ),
)
def test_decision_cost_fields_are_strict_finite_nonnegative(
    field_name: str,
    value: object,
    error: type[Exception],
) -> None:
    """决策成本字段必须是有限且非负的严格数值。"""

    with pytest.raises(error):
        DecisionConfig(**{field_name: value})  # type: ignore[arg-type]


def test_loader_rejects_unknown_or_invalid_decision_fields() -> None:
    """配置加载器必须拒绝未知键与错误类型的决策字段。"""

    with pytest.raises(ValueError, match="未知键"):
        load_v2_config({"schema_version": 1, "decision": {"risk_lamda": 0.1}})
    with pytest.raises(TypeError):
        load_v2_config({"schema_version": 1, "decision": {"risk_lambda": "0.1"}})
