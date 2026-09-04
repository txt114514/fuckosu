"""Phase 8 CLICK utility 公式、风险敏感性与边界验收。"""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from traning.conf import DecisionConfig
from traning.state import OutcomeDistribution
from traning.core.decision.utility import ClickUtility, compute_click_utility


def _outcome(**changes: float | str) -> OutcomeDistribution:
    values: dict[str, float | str] = {
        "track_id": "track-1",
        "horizon_ms": 16.0,
        "p_invalid": 0.1,
        "p_miss": 0.2,
        "p_low_score": 0.2,
        "p_medium_score": 0.3,
        "p_high_score": 0.2,
        "p_expire": 0.4,
        "expected_score": 0.8,
        "variance": 0.25,
    }
    values.update(changes)
    return OutcomeDistribution(**values)  # type: ignore[arg-type]


def _config(**changes: float) -> DecisionConfig:
    values: dict[str, float] = {
        "risk_lambda": 0.5,
        "click_cost": 0.1,
        "invalid_penalty": 2.0,
        "miss_penalty": 3.0,
        "expire_penalty": 4.0,
    }
    values.update(changes)
    return DecisionConfig(**values)


def test_click_utility_matches_hand_computed_formula() -> None:
    """唯一 utility 公式和成功概率必须逐项匹配手算结果。"""

    outcome = _outcome()
    result = compute_click_utility(outcome, _config())
    assert result.value == pytest.approx(-1.825)
    assert result.success_probability == pytest.approx(0.7)
    assert result.track_id == outcome.track_id
    assert result.horizon_ms == outcome.horizon_ms
    assert result.outcome is outcome


def test_risk_and_each_penalty_only_reduce_value_by_its_term() -> None:
    """risk 与各 penalty 的边际影响必须等于对应概率或方差。"""

    outcome = _outcome()
    baseline = compute_click_utility(outcome, _config()).value
    changes = (
        ("risk_lambda", outcome.variance),
        ("click_cost", 1.0),
        ("invalid_penalty", outcome.p_invalid),
        ("miss_penalty", outcome.p_miss),
        ("expire_penalty", outcome.p_expire),
    )
    for field_name, expected_reduction in changes:
        updated = compute_click_utility(
            outcome, _config(**{field_name: getattr(_config(), field_name) + 1.0})
        )
        assert baseline - updated.value == pytest.approx(expected_reduction)


def test_compute_is_pure_and_result_is_frozen() -> None:
    """计算不得修改输入，结果必须保留原 Outcome 对象且不可变。"""

    outcome = _outcome()
    config = _config()
    original_outcome = replace(outcome)
    original_config = replace(config)
    first = compute_click_utility(outcome, config)
    second = compute_click_utility(outcome, config)
    assert first == second
    assert outcome == original_outcome
    assert config == original_config
    with pytest.raises((FrozenInstanceError, AttributeError)):
        first.value = 0.0  # type: ignore[misc]


def test_click_utility_rejects_identity_probability_and_finite_contradictions() -> None:
    """结果 DTO 不允许脱离绑定 Outcome 或伪造成功概率。"""

    outcome = _outcome()
    with pytest.raises(ValueError, match="track_id"):
        ClickUtility("other", outcome.horizon_ms, 0.0, 0.7, outcome)
    with pytest.raises(ValueError, match="horizon_ms"):
        ClickUtility(outcome.track_id, 99.0, 0.0, 0.7, outcome)
    with pytest.raises(ValueError, match=r"low\+medium\+high"):
        ClickUtility(outcome.track_id, outcome.horizon_ms, 0.0, 0.6, outcome)
    with pytest.raises(ValueError, match="有限"):
        ClickUtility(outcome.track_id, outcome.horizon_ms, float("inf"), 0.7, outcome)


def test_compute_rejects_wrong_typed_inputs() -> None:
    """核心入口只接受 canonical OutcomeDistribution 与 DecisionConfig。"""

    with pytest.raises(TypeError, match="OutcomeDistribution"):
        compute_click_utility(object(), _config())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="DecisionConfig"):
        compute_click_utility(_outcome(), object())  # type: ignore[arg-type]


def test_utility_source_has_no_forbidden_decision_shortcuts() -> None:
    """Utility 层不得读取图像、GT、oracle、logits、argmax 或 legacy。"""

    path = Path(__file__).resolve().parents[2] / "core/decision/utility.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    forbidden_names = {
        "Any",
        "GroundTruthObject",
        "TrainingCandidateRecord",
        "action_logits",
        "argmax",
        "candidate_logits",
        "image",
        "oracle",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            assert node.id not in forbidden_names
        elif isinstance(node, ast.Attribute):
            assert node.attr not in forbidden_names
        elif isinstance(node, ast.Import):
            assert all(not alias.name.startswith("osu_v2") for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.module is None or not node.module.startswith("osu_v2")
