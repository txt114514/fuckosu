from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from traning.core.optimization.attribution import AttributionSummary
from traning.core.optimization.parameter_search.curriculum import (
    CurriculumGateResult,
    evaluate_curriculum_gate,
)
from traning.core.optimization.parameter_search.hard_examples import (
    HardExampleSamplingPlan,
    build_hard_example_sampling_plan,
)
from traning.core.optimization.parameter_search.planner import OptimizationPlan
from traning.core.optimization.scoring import TrialScoreReport
from traning.state import (
    CurriculumStage,
    SearchMethod,
    TrialMetadata,
    TrialParameters,
    TrialStatus,
)


OPTIMIZATION_RECORD_VERSION = "optimization-execution-v1"


@dataclass(frozen=True)
class TrainingJobSpec:
    trial_id: str
    curriculum_stage: CurriculumStage
    rung: int
    budget_steps: int
    parameters: TrialParameters
    parent_checkpoint_path: Path | None = None
    hard_example_weights: Mapping[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "curriculum_stage": self.curriculum_stage.value,
            "rung": self.rung,
            "budget_steps": self.budget_steps,
            "parameters": self.parameters.model_dump(mode="json"),
            "parent_checkpoint_path": (
                str(self.parent_checkpoint_path)
                if self.parent_checkpoint_path is not None
                else None
            ),
            "hard_example_weights": dict(self.hard_example_weights),
        }


@dataclass(frozen=True)
class OptimizationExecution:
    version: str
    created_at_utc: str
    trial: TrialMetadata
    source_trial_id: str
    score: Mapping[str, Any]
    attribution: Mapping[str, Any]
    plan: Mapping[str, Any]
    curriculum_gate: Mapping[str, Any]
    hard_examples: Mapping[str, Any]
    job: TrainingJobSpec

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "created_at_utc": self.created_at_utc,
            "trial": self.trial.model_dump(mode="json"),
            "source_trial_id": self.source_trial_id,
            "score": dict(self.score),
            "attribution": dict(self.attribution),
            "plan": dict(self.plan),
            "curriculum_gate": dict(self.curriculum_gate),
            "hard_examples": dict(self.hard_examples),
            "job": self.job.as_dict(),
        }


@dataclass(frozen=True)
class OptimizationExecutorConfig:
    experiment_name: str = "optimization"
    seed: int = 2026
    base_budget_steps: int = 100
    budget_multiplier_per_rung: float = 3.0
    output_dir: Path = Path("runs/optimization_trials")
    code_version: str | None = None
    data_version: str | None = None

    def __post_init__(self) -> None:
        if not self.experiment_name:
            raise ValueError("experiment_name must not be empty")
        if self.base_budget_steps < 1:
            raise ValueError("base_budget_steps must be positive")
        if self.budget_multiplier_per_rung < 1:
            raise ValueError("budget_multiplier_per_rung must be >= 1")


class JsonlTrialStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, execution: OptimizationExecution) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(
                json.dumps(
                    execution.as_dict(),
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            file.write("\n")

    def load(self) -> tuple[dict[str, Any], ...]:
        if not self.path.exists():
            return ()
        records: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
        return tuple(records)


def _apply_section_updates(
    base: Mapping[str, object],
    updates: Mapping[str, Any],
) -> dict[str, object]:
    merged: dict[str, object] = dict(base)
    for key, value in updates.items():
        if key.endswith("_delta"):
            target_key = key.removesuffix("_delta")
            current = merged.get(target_key, 0)
            merged[target_key] = (
                current + value
                if isinstance(current, (int, float))
                and isinstance(value, (int, float))
                else value
            )
        elif key.endswith("_multiplier"):
            target_key = key.removesuffix("_multiplier")
            current = merged.get(target_key)
            merged[target_key] = (
                current * value
                if isinstance(current, (int, float))
                and isinstance(value, (int, float))
                else value
            )
        else:
            merged[key] = value
    return merged


def _apply_parameter_updates(
    parameters: TrialParameters,
    updates: Mapping[str, Mapping[str, Any]],
) -> TrialParameters:
    return TrialParameters(
        architecture=_apply_section_updates(
            parameters.architecture,
            updates.get("architecture", {}),
        ),
        training=_apply_section_updates(
            parameters.training,
            {
                **updates.get("training", {}),
                **({"sampling": updates["sampling"]} if "sampling" in updates else {}),
            },
        ),
        inference=_apply_section_updates(
            parameters.inference,
            updates.get("inference", {}),
        ),
    )


def _budget_steps(config: OptimizationExecutorConfig, rung: int) -> int:
    return max(
        1,
        round(config.base_budget_steps * config.budget_multiplier_per_rung**rung),
    )


def _next_trial_id(source_trial_id: str, rung: int, stage: CurriculumStage) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{source_trial_id}__r{rung:02d}__{stage.value}__{timestamp}"


def execute_optimization_plan(
    report: TrialScoreReport,
    attribution: AttributionSummary,
    plan: OptimizationPlan,
    *,
    base_parameters: TrialParameters | None = None,
    parent_checkpoint_path: Path | None = None,
    config: OptimizationExecutorConfig = OptimizationExecutorConfig(),
    store: JsonlTrialStore | None = None,
) -> OptimizationExecution:
    next_rung = 0 if plan.asha_action == "prune" else (
        1 if plan.next_stage != plan.current_stage else 0
    )
    trial_id = _next_trial_id(report.trial_id, next_rung, plan.next_stage)
    parameters = _apply_parameter_updates(
        base_parameters or report.parameters,
        plan.parameter_updates,
    )
    hard_examples = build_hard_example_sampling_plan(attribution)
    curriculum_gate = evaluate_curriculum_gate(report.samples)
    trial = TrialMetadata(
        trial_id=trial_id,
        experiment_name=config.experiment_name,
        seed=config.seed,
        search_method=plan.search_method,
        parameters=parameters,
        status=plan.next_status,
        curriculum_stage=plan.next_stage,
        rung=next_rung,
        budget_steps=_budget_steps(config, next_rung),
        parent_trial_id=report.trial_id,
        code_version=config.code_version,
        data_version=config.data_version,
        metrics={
            "quality_score": report.quality_score,
            "hit_count": float(report.hit_count),
            "miss_count": float(report.miss_count),
            "unresolved_count": float(report.unresolved_count),
            **dict(report.metrics),
        },
    )
    job = TrainingJobSpec(
        trial_id=trial_id,
        curriculum_stage=plan.next_stage,
        rung=next_rung,
        budget_steps=trial.budget_steps,
        parameters=parameters,
        parent_checkpoint_path=parent_checkpoint_path,
        hard_example_weights=hard_examples.sample_weights,
    )
    execution = OptimizationExecution(
        version=OPTIMIZATION_RECORD_VERSION,
        created_at_utc=datetime.now(UTC).isoformat(),
        trial=trial,
        source_trial_id=report.trial_id,
        score=report.as_dict(),
        attribution=attribution.as_dict(),
        plan=plan.as_dict(),
        curriculum_gate=curriculum_gate.as_dict(),
        hard_examples=hard_examples.as_dict(),
        job=job,
    )
    target_store = store or JsonlTrialStore(config.output_dir / "trials.jsonl")
    target_store.append(execution)
    return execution


__all__ = [
    "JsonlTrialStore",
    "OPTIMIZATION_RECORD_VERSION",
    "OptimizationExecution",
    "OptimizationExecutorConfig",
    "TrainingJobSpec",
    "execute_optimization_plan",
]
