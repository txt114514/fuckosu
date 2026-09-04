"""反事实结果类别与分布契约。"""

from dataclasses import dataclass
from enum import IntEnum

from .common import (
    require_identifier,
    require_nonnegative,
    require_probability,
    require_probability_sum,
)


OUTCOME_LOW_SCORE_UPPER = 0.5
OUTCOME_MEDIUM_SCORE_UPPER = 0.8


class OutcomeCategory(IntEnum):
    """Outcome 模型固定的五类互斥标签与 tensor channel 顺序。"""

    INVALID = 0
    MISS = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4


@dataclass(frozen=True, slots=True)
class OutcomePrediction:
    """五类互斥结果及独立过期事件的预测分布。"""

    track_id: str
    horizon_ms: float
    p_invalid: float
    p_miss: float
    p_low_score: float
    p_medium_score: float
    p_high_score: float
    p_expire: float
    expected_score: float
    variance: float

    def __post_init__(self) -> None:
        require_identifier(self.track_id, "track_id")
        require_nonnegative(self.horizon_ms, "horizon_ms")
        require_probability_sum(
            (
                self.p_invalid,
                self.p_miss,
                self.p_low_score,
                self.p_medium_score,
                self.p_high_score,
            ),
            "outcome probabilities",
        )
        require_probability(self.p_expire, "p_expire")
        require_nonnegative(self.expected_score, "expected_score")
        require_nonnegative(self.variance, "variance")


# 旧名称是 identity alias；OutcomePrediction 是注册表中的规范名称。
OutcomeDistribution = OutcomePrediction


__all__ = [
    "OUTCOME_LOW_SCORE_UPPER",
    "OUTCOME_MEDIUM_SCORE_UPPER",
    "OutcomeCategory",
    "OutcomeDistribution",
    "OutcomePrediction",
]
