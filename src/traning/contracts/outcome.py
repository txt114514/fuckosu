"""已弃用兼容转发；新代码必须导入对应的 conf、core、lib 或 state 路径。"""

from traning.state.outcome import (
    OUTCOME_LOW_SCORE_UPPER,
    OUTCOME_MEDIUM_SCORE_UPPER,
    OutcomeCategory,
    OutcomeDistribution,
)

__deprecated__ = True
__all__ = [
    "OUTCOME_LOW_SCORE_UPPER",
    "OUTCOME_MEDIUM_SCORE_UPPER",
    "OutcomeCategory",
    "OutcomeDistribution",
]
