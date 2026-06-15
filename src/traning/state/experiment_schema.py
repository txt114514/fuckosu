from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


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


class TrialParameters(BaseModel):
    architecture: dict[str, object] = Field(default_factory=dict)
    training: dict[str, object] = Field(default_factory=dict)
    inference: dict[str, object] = Field(default_factory=dict)


class TrialMetadata(BaseModel):
    trial_id: str
    experiment_name: str
    seed: int
    search_method: SearchMethod
    parameters: TrialParameters
    status: TrialStatus = TrialStatus.PENDING
    curriculum_stage: CurriculumStage = CurriculumStage.BASIC
    rung: int = 0
    budget_steps: int
    consumed_steps: int = 0
    parent_trial_id: str | None = None
    code_version: str | None = None
    data_version: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)

    @field_validator("rung", "budget_steps", "consumed_steps")
    @classmethod
    def _nonnegative_integer(cls, value: int) -> int:
        if value < 0:
            raise ValueError("trial counters must be nonnegative")
        return value


class EvaluationRunMetadata(BaseModel):
    evaluation_id: str
    trial_id: str
    checkpoint_id: str
    dataset_version: str
    inference_parameters: dict[str, object] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)


class ExperimentMetadata(BaseModel):
    name: str
    search_method: SearchMethod = SearchMethod.TPE
    objective_names: tuple[str, ...] = (
        "quality_score",
        "peak_vram_mb",
        "latency_ms",
    )
    base_seeds: tuple[int, ...] = (2026,)
    data_version: str | None = None
    code_version: str | None = None


__all__ = [
    "CurriculumStage",
    "EvaluationRunMetadata",
    "ExperimentMetadata",
    "SearchMethod",
    "TrialMetadata",
    "TrialParameters",
    "TrialStatus",
]
