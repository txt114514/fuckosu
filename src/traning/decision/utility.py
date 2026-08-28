"""仅由 Outcome 分布与显式成本计算 CLICK 未来价值。"""

from __future__ import annotations

import math
from dataclasses import dataclass

from traning.config import DecisionConfig
from traning.contracts import OutcomeDistribution


@dataclass(frozen=True, slots=True)
class ClickUtility:
    """绑定原始 OutcomeDistribution 的单轨迹单 horizon 点击价值。"""

    track_id: str
    horizon_ms: float
    value: float
    success_probability: float
    outcome: OutcomeDistribution

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, OutcomeDistribution):
            raise TypeError("outcome 必须是 OutcomeDistribution")
        if self.track_id != self.outcome.track_id:
            raise ValueError("track_id 必须与 outcome.track_id 一致")
        if self.horizon_ms != self.outcome.horizon_ms:
            raise ValueError("horizon_ms 必须与 outcome.horizon_ms 一致")
        _require_finite(self.value, "value")
        _require_finite(self.success_probability, "success_probability")
        if not 0.0 <= self.success_probability <= 1.0:
            raise ValueError("success_probability 必须位于 [0, 1]")
        expected_success = (
            self.outcome.p_low_score
            + self.outcome.p_medium_score
            + self.outcome.p_high_score
        )
        if not math.isclose(
            self.success_probability,
            expected_success,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("success_probability 必须等于 low+medium+high 概率和")


def compute_click_utility(
    outcome: OutcomeDistribution, config: DecisionConfig
) -> ClickUtility:
    """按唯一风险惩罚公式计算 CLICK utility，不读取图像或动作 logits。"""

    if not isinstance(outcome, OutcomeDistribution):
        raise TypeError("outcome 必须是 OutcomeDistribution")
    if not isinstance(config, DecisionConfig):
        raise TypeError("config 必须是 DecisionConfig")
    value = (
        outcome.expected_score
        - config.risk_lambda * outcome.variance
        - config.click_cost
        - config.invalid_penalty * outcome.p_invalid
        - config.miss_penalty * outcome.p_miss
        - config.expire_penalty * outcome.p_expire
    )
    _require_finite(value, "click utility")
    success_probability = (
        outcome.p_low_score + outcome.p_medium_score + outcome.p_high_score
    )
    return ClickUtility(
        track_id=outcome.track_id,
        horizon_ms=outcome.horizon_ms,
        value=value,
        success_probability=success_probability,
        outcome=outcome,
    )


def _require_finite(value: float, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} 必须是数值")
    if not math.isfinite(float(value)):
        raise ValueError(f"{field_name} 必须是有限数值")


__all__ = ("ClickUtility", "compute_click_utility")
