"""确定性的 curriculum gate 与 typed ASHA 调度。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from package import CurriculumStage


CURRICULUM_ORDER: tuple[CurriculumStage, ...] = (
    CurriculumStage.BASIC,
    CurriculumStage.MULTI_OBJECT,
    CurriculumStage.COMPLEX,
    CurriculumStage.FULL,
)


@dataclass(frozen=True, slots=True)
class CurriculumGate:
    """一个可审计的 curriculum stage gate。"""

    name: str
    passed: bool

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or not self.name
            or self.name != self.name.strip()
        ):
            raise ValueError("gate name 必须是非空且无首尾空格的字符串")
        if not isinstance(self.passed, bool):
            raise TypeError("gate passed 必须是 bool")


class CurriculumAction(str, Enum):
    """curriculum 的明确调度动作。"""

    ADVANCE = "advance"
    HOLD = "hold"
    COMPLETE = "complete"


@dataclass(frozen=True, slots=True)
class CurriculumDecision:
    """当前 stage gate 的确定性决策。"""

    current_stage: CurriculumStage
    next_stage: CurriculumStage
    action: CurriculumAction


def decide_curriculum(
    current_stage: CurriculumStage,
    gates: tuple[CurriculumGate, ...],
) -> CurriculumDecision:
    """仅在至少一个 gate 且全部通过时前进。"""

    if not isinstance(current_stage, CurriculumStage):
        raise TypeError("current_stage 必须是 CurriculumStage")
    if not isinstance(gates, tuple) or any(
        not isinstance(gate, CurriculumGate) for gate in gates
    ):
        raise TypeError("gates 必须是 CurriculumGate tuple")
    names = tuple(gate.name for gate in gates)
    if len(set(names)) != len(names):
        raise ValueError("curriculum gate name 必须唯一")
    if not gates or not all(gate.passed for gate in gates):
        return CurriculumDecision(
            current_stage=current_stage,
            next_stage=current_stage,
            action=CurriculumAction.HOLD,
        )
    index = CURRICULUM_ORDER.index(current_stage)
    if index == len(CURRICULUM_ORDER) - 1:
        return CurriculumDecision(
            current_stage=current_stage,
            next_stage=current_stage,
            action=CurriculumAction.COMPLETE,
        )
    return CurriculumDecision(
        current_stage=current_stage,
        next_stage=CURRICULUM_ORDER[index + 1],
        action=CurriculumAction.ADVANCE,
    )


@dataclass(frozen=True, slots=True)
class AshaRung:
    """ASHA 的递增资源预算与晋级比例。"""

    index: int
    budget: int
    promotion_fraction: float

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or not isinstance(self.index, int):
            raise TypeError("rung index 必须是整数")
        if isinstance(self.budget, bool) or not isinstance(self.budget, int):
            raise TypeError("rung budget 必须是整数")
        if self.index < 0 or self.budget < 1:
            raise ValueError("rung index 不得为负且 budget 必须为正")
        if isinstance(self.promotion_fraction, bool) or not isinstance(
            self.promotion_fraction, (int, float)
        ):
            raise TypeError("promotion_fraction 必须是数值")
        if not math.isfinite(float(self.promotion_fraction)) or not (
            0 < self.promotion_fraction <= 1
        ):
            raise ValueError("promotion_fraction 必须位于 (0, 1]")


@dataclass(frozen=True, slots=True)
class AshaTrial:
    """到达某 rung 的 trial 观测。"""

    trial_id: str
    rung_index: int
    objective: float
    gate_passed: bool

    def __post_init__(self) -> None:
        if (
            not isinstance(self.trial_id, str)
            or not self.trial_id
            or self.trial_id != self.trial_id.strip()
        ):
            raise ValueError("trial_id 必须是非空且无首尾空格的字符串")
        if isinstance(self.rung_index, bool) or not isinstance(self.rung_index, int):
            raise TypeError("rung_index 必须是整数")
        if not isinstance(self.gate_passed, bool):
            raise TypeError("gate_passed 必须是 bool")
        if isinstance(self.objective, bool) or not isinstance(
            self.objective, (int, float)
        ):
            raise TypeError("objective 必须是数值")
        if not math.isfinite(float(self.objective)):
            raise ValueError("objective 必须有限")


class AshaAction(str, Enum):
    """ASHA 对 trial 的明确动作。"""

    PROMOTE = "promote"
    PRUNE = "prune"
    CONTINUE = "continue"


@dataclass(frozen=True, slots=True)
class AshaDecision:
    """一个 trial 在当前 rung 的确定性动作。"""

    trial_id: str
    action: AshaAction
    current_rung: int
    next_rung: int | None


@dataclass(frozen=True, slots=True)
class AshaScheduler:
    """先执行严格 gate，再按 objective 与 trial_id 稳定排名。"""

    rungs: tuple[AshaRung, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.rungs, tuple) or not self.rungs:
            raise TypeError("rungs 必须是非空 AshaRung tuple")
        if any(not isinstance(rung, AshaRung) for rung in self.rungs):
            raise TypeError("rungs 只能包含 AshaRung")
        if tuple(rung.index for rung in self.rungs) != tuple(range(len(self.rungs))):
            raise ValueError("rung index 必须从 0 连续递增")
        budgets = tuple(rung.budget for rung in self.rungs)
        if budgets != tuple(sorted(set(budgets))):
            raise ValueError("rung budget 必须严格递增")

    def decide(
        self, rung_index: int, trials: tuple[AshaTrial, ...]
    ) -> tuple[AshaDecision, ...]:
        """同 rung gate 失败必剪枝；其余按 top fraction 晋级。"""

        if isinstance(rung_index, bool) or not isinstance(rung_index, int):
            raise TypeError("rung_index 必须是整数")
        if not 0 <= rung_index < len(self.rungs):
            raise ValueError("rung_index 越界")
        if not isinstance(trials, tuple) or any(
            not isinstance(trial, AshaTrial) for trial in trials
        ):
            raise TypeError("trials 必须是 AshaTrial tuple")
        if any(trial.rung_index != rung_index for trial in trials):
            raise ValueError("trials 必须全部属于请求 rung")
        trial_ids = tuple(trial.trial_id for trial in trials)
        if len(set(trial_ids)) != len(trial_ids):
            raise ValueError("trial_id 必须唯一")

        rung = self.rungs[rung_index]
        ranked = tuple(
            sorted(
                (trial for trial in trials if trial.gate_passed),
                key=lambda trial: (-trial.objective, trial.trial_id),
            )
        )
        terminal = rung_index == len(self.rungs) - 1
        promote_count = (
            len(ranked)
            if terminal
            else max(1, math.ceil(len(ranked) * rung.promotion_fraction))
        )
        promoted_ids = {trial.trial_id for trial in ranked[:promote_count]}
        decisions = []
        for trial in sorted(trials, key=lambda item: item.trial_id):
            if not trial.gate_passed or trial.trial_id not in promoted_ids:
                action = AshaAction.PRUNE
                next_rung = None
            elif terminal:
                action = AshaAction.CONTINUE
                next_rung = None
            else:
                action = AshaAction.PROMOTE
                next_rung = rung_index + 1
            decisions.append(
                AshaDecision(
                    trial_id=trial.trial_id,
                    action=action,
                    current_rung=rung_index,
                    next_rung=next_rung,
                )
            )
        return tuple(decisions)


__all__ = (
    "CURRICULUM_ORDER",
    "AshaAction",
    "AshaDecision",
    "AshaRung",
    "AshaScheduler",
    "AshaTrial",
    "CurriculumAction",
    "CurriculumDecision",
    "CurriculumGate",
    "CurriculumStage",
    "decide_curriculum",
)
