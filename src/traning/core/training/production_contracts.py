"""生产训练门禁、指标和最终产物的强类型契约。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

from traning.conf import V2Config
from traning.core.training.optimization import TrialObservation
from traning.core.training.orchestration import StageResult


def _probability(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} 必须是数值")
    if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} 必须位于 [0, 1]")


def _nonnegative(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} 必须是数值")
    if not math.isfinite(float(value)) or float(value) < 0.0:
        raise ValueError(f"{name} 必须是有限非负数")


@dataclass(frozen=True, slots=True)
class ProductionGateSpec:
    """各阶段唯一使用的生产验收阈值。"""

    perception_min_recall: float = 0.50
    tracking_max_id_switch_rate: float = 0.10
    belief_max_position_mae_px: float = 12.0
    outcome_max_nll: float = 1.35
    outcome_max_brier: float = 0.75
    outcome_max_ece: float = 0.25
    decision_min_oracle_agreement: float = 0.70
    golden_min_hit_rate: float = 0.70

    def __post_init__(self) -> None:
        for name in (
            "perception_min_recall",
            "tracking_max_id_switch_rate",
            "outcome_max_ece",
            "decision_min_oracle_agreement",
            "golden_min_hit_rate",
        ):
            _probability(name, getattr(self, name))
        for name in (
            "belief_max_position_mae_px",
            "outcome_max_nll",
            "outcome_max_brier",
        ):
            _nonnegative(name, getattr(self, name))


@dataclass(slots=True)
class ProductionTrialMetrics:
    """一个 trial 从所有真实阶段累计得到的指标快照。"""

    perception_loss: float = 0.0
    perception_recall: float = 0.0
    tracking_id_switches: int = 0
    tracking_assignments: int = 0
    belief_position_mae_px: float = 0.0
    outcome_nll: float = 0.0
    outcome_brier: float = 0.0
    outcome_ece: float = 0.0
    expected_score_mae: float = 0.0
    decision_oracle_agreement: float = 0.0
    decision_hard_agreement: float = 1.0
    decision_utility: float = 0.0
    wait_click_ratio: float = 0.0
    golden_hit_rate: float = 0.0
    training_steps: int = 0
    hard_example_count: int = 0

    @property
    def tracking_id_switch_rate(self) -> float:
        """按可连续比较的目标分配数归一化 ID switch。"""

        if self.tracking_assignments == 0:
            return 1.0
        return self.tracking_id_switches / self.tracking_assignments

    @property
    def objective(self) -> float:
        """越大越好的稳定多阶段搜索目标；门禁仍由布尔验收决定。"""

        return float(
            2.0 * self.golden_hit_rate
            + self.perception_recall
            + self.decision_oracle_agreement
            + 0.25 * self.decision_hard_agreement
            - 0.25 * self.tracking_id_switch_rate
            - 0.05 * self.belief_position_mae_px
            - 0.10 * self.outcome_nll
            - 0.05 * self.outcome_brier
        )


@dataclass(frozen=True, slots=True)
class ProductionTrainingResult:
    """全门禁通过后返回的 winning trial 与已验证 checkpoint。"""

    observation: TrialObservation
    winning_config: V2Config
    checkpoint_directory: Path
    stage_results: tuple[StageResult, ...] = ()
    metrics: ProductionTrialMetrics | None = None
    resumed: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.observation, TrialObservation):
            raise TypeError("observation 必须是 TrialObservation")
        if not self.observation.acceptance.passed:
            raise ValueError("ProductionTrainingResult 只能引用全门禁通过的 trial")
        if not isinstance(self.winning_config, V2Config):
            raise TypeError("winning_config 必须是 V2Config")
        if not isinstance(self.checkpoint_directory, Path):
            raise TypeError("checkpoint_directory 必须是 pathlib.Path")
        if any(not isinstance(item, StageResult) for item in self.stage_results):
            raise TypeError("stage_results 只能包含 StageResult")
        if self.metrics is not None and not isinstance(
            self.metrics, ProductionTrialMetrics
        ):
            raise TypeError("metrics 必须是 ProductionTrialMetrics 或 None")
        if not isinstance(self.resumed, bool):
            raise TypeError("resumed 必须是 bool")


__all__ = (
    "ProductionGateSpec",
    "ProductionTrainingResult",
    "ProductionTrialMetrics",
)
