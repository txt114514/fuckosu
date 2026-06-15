from __future__ import annotations

from dataclasses import dataclass, field

from traning.state.experiment_schema import CurriculumStage


@dataclass
class RunState:
    trial_id: str | None = None
    stage: str = "data_input"
    curriculum_stage: CurriculumStage = CurriculumStage.BASIC
    rung: int = 0
    budget_steps: int = 0
    global_step: int = 0
    completed_stages: list[str] = field(default_factory=list)


__all__ = ["RunState"]
