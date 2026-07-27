"""保存训练编排可变的当前位置、预算与已完成阶段。"""

from __future__ import annotations

from dataclasses import dataclass, field

from traning.state.experiment_schema import CurriculumStage


@dataclass
class RunState:
    """进程内训练状态；持久化边界由 checkpoint/experiment 契约负责。"""

    trial_id: str | None = None
    stage: str = "data_input"
    curriculum_stage: CurriculumStage = CurriculumStage.BASIC
    rung: int = 0
    budget_steps: int = 0
    global_step: int = 0
    completed_stages: list[str] = field(default_factory=list)


__all__ = ["RunState"]
