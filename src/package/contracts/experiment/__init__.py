from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from math import isfinite

from package.contracts.base import ContractMixin


class SearchMethod(StrEnum):
    RANDOM = "random"
    TPE = "tpe"


class TrialStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    PRUNED = "pruned"
    PROMOTED = "promoted"
    COMPLETED = "completed"
    FAILED = "failed"


class CurriculumStage(StrEnum):
    BASIC = "basic"
    MULTI_OBJECT = "multi_object"
    COMPLEX = "complex"
    FULL = "full"


@dataclass(frozen=True)
class TrialParametersRef(ContractMixin):
    architecture: dict[str, object] = field(default_factory=dict)
    training: dict[str, object] = field(default_factory=dict)
    inference: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class TrialRef(ContractMixin):
    trial_id: str
    experiment_name: str
    seed: int
    search_method: SearchMethod
    parameters: TrialParametersRef = field(default_factory=TrialParametersRef)
    status: TrialStatus = TrialStatus.PENDING
    curriculum_stage: CurriculumStage = CurriculumStage.BASIC
    rung: int = 0
    budget_steps: int = 0
    parent_trial_id: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.trial_id or not self.experiment_name:
            raise ValueError("trial identity fields must not be empty")
        if self.seed < 0 or self.rung < 0 or self.budget_steps < 0:
            raise ValueError("trial counters must be nonnegative")
        if not isinstance(self.search_method, SearchMethod):
            object.__setattr__(self, "search_method", SearchMethod(self.search_method))
        if not isinstance(self.status, TrialStatus):
            object.__setattr__(self, "status", TrialStatus(self.status))
        if not isinstance(self.curriculum_stage, CurriculumStage):
            object.__setattr__(
                self,
                "curriculum_stage",
                CurriculumStage(self.curriculum_stage),
            )
        if not isinstance(self.parameters, TrialParametersRef):
            object.__setattr__(
                self,
                "parameters",
                TrialParametersRef.from_mapping(self.parameters),
            )
        for name, value in self.metrics.items():
            if not isinstance(name, str) or not isfinite(value):
                raise ValueError("trial metrics must be string to finite float")


@dataclass(frozen=True)
class CheckpointRef(ContractMixin):
    checkpoint_id: str
    trial_id: str
    path: str
    curriculum_stage: CurriculumStage = CurriculumStage.BASIC
    rung: int = 0
    global_step: int = 0
    parent_checkpoint_id: str | None = None
    includes_optimizer: bool = True

    def __post_init__(self) -> None:
        if not self.checkpoint_id or not self.trial_id or not self.path:
            raise ValueError("checkpoint identity fields must not be empty")
        if self.rung < 0 or self.global_step < 0:
            raise ValueError("checkpoint counters must be nonnegative")
        if not isinstance(self.curriculum_stage, CurriculumStage):
            object.__setattr__(
                self,
                "curriculum_stage",
                CurriculumStage(self.curriculum_stage),
            )


@dataclass(frozen=True)
class ScoreVersionRef(ContractMixin):
    score_version: str
    dataset_version: str | None = None
    code_version: str | None = None

    def __post_init__(self) -> None:
        if not self.score_version:
            raise ValueError("score_version must not be empty")


__all__ = [
    "CheckpointRef",
    "CurriculumStage",
    "ScoreVersionRef",
    "SearchMethod",
    "TrialParametersRef",
    "TrialRef",
    "TrialStatus",
]
