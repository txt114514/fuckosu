"""执行优化试验计划，并持久化试验、检查点继承与评估结果。"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from math import isfinite
from pathlib import Path
from typing import Any

from traning.core.optimization.attribution import AttributionSummary
from traning.core.optimization.parameter_search.curriculum import (
    CurriculumGateResult,  # noqa: F401 - 保留既有模块属性兼容面。
    evaluate_curriculum_gate,
)
from traning.core.optimization.parameter_search.hard_examples import (
    HardExampleSamplingPlan,  # noqa: F401 - 保留既有模块属性兼容面。
    build_hard_example_sampling_plan,
)
from traning.core.optimization.parameter_search.planner import (
    OptimizationPlan,
    TrialHistoryEntry,
)
from traning.core.optimization.scoring import TrialScoreReport
from traning.state import (
    CurriculumStage,
    SearchMethod,  # noqa: F401 - 保留既有模块属性兼容面。
    TrialMetadata,
    TrialParameters,
    TrialStatus,  # noqa: F401 - 保留既有模块属性兼容面。
)


OPTIMIZATION_RECORD_VERSION = "optimization-execution-v1"


@dataclass(frozen=True)
class _ParameterSpec:
    default: int | float
    minimum: int | float
    maximum: int | float
    integer: bool = False

    def normalize(self, value: object) -> int | float:
        if not isinstance(value, (int, float)):
            raise ValueError("optimized parameter values must be numeric")
        numeric = float(value)
        if not isfinite(numeric):
            raise ValueError("optimized parameter values must be finite")
        bounded = min(float(self.maximum), max(float(self.minimum), numeric))
        if self.integer:
            return int(round(bounded))
        return bounded


# OptimizationPlan 中可以使用 delta/multiplier；落入 job 后必须全部变成有界绝对值。
_PARAMETER_SPECS: Mapping[tuple[str, str], _ParameterSpec] = {
    ("training", "spatial_learning_rate"): _ParameterSpec(1e-4, 1e-7, 1.0),
    ("training", "temporal_learning_rate"): _ParameterSpec(1e-4, 1e-7, 1.0),
    # 0 沿用训练 CLI 的“处理全部 patch”语义。
    ("training", "patch_limit"): _ParameterSpec(1, 0, 4096, integer=True),
    # 0 沿用训练 CLI 的“处理全部帧”语义；正整数是确定的缓存帧上限。
    ("training", "cache_max_frames"): _ParameterSpec(
        1,
        0,
        100_000_000,
        integer=True,
    ),
    ("training", "sequence_length"): _ParameterSpec(8, 1, 4096, integer=True),
    ("training", "candidate_slots"): _ParameterSpec(32, 1, 4096, integer=True),
    ("inference", "score_threshold"): _ParameterSpec(0.05, 0.0, 1.0),
    ("inference", "max_candidates"): _ParameterSpec(32, 1, 4096, integer=True),
    ("inference", "nms_radius_px"): _ParameterSpec(24.0, 0.0, 4096.0),
    ("inference", "slider_threshold"): _ParameterSpec(0.35, 0.0, 1.0),
    ("inference", "max_slider_paths"): _ParameterSpec(16, 1, 4096, integer=True),
}


@dataclass(frozen=True)
class TrainingJobSpec:
    trial_id: str
    curriculum_stage: CurriculumStage
    rung: int
    budget_steps: int
    parameters: TrialParameters
    parent_checkpoint_path: Path | None = None
    hard_example_weights: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.trial_id:
            raise ValueError("training job trial_id must not be empty")
        if self.rung < 0:
            raise ValueError("training job rung must be nonnegative")
        if self.budget_steps <= 0:
            raise ValueError("training job budget_steps must be positive")
        if self.hard_example_weights:
            raise ValueError(
                "hard_example_weights are not supported until weighted sampling "
                "is connected to the training dataset"
            )

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


class SQLiteTrialStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS optimization_trials (
              trial_id TEXT PRIMARY KEY,
              source_trial_id TEXT NOT NULL,
              created_at_utc TEXT NOT NULL,
              status TEXT NOT NULL,
              curriculum_stage TEXT NOT NULL,
              rung INTEGER NOT NULL,
              quality_score REAL NOT NULL,
              objective_score REAL,
              payload_json TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_optimization_trials_source
            ON optimization_trials(source_trial_id)
            """
        )
        return connection

    def append(self, execution: OptimizationExecution) -> None:
        payload = execution.as_dict()
        objective_score = payload.get("plan", {}).get("objective_score")
        with self._connect() as connection:
            # trial_id 是幂等键，重跑同一计划时更新记录而不是制造重复试验历史。
            connection.execute(
                """
                INSERT OR REPLACE INTO optimization_trials (
                  trial_id,
                  source_trial_id,
                  created_at_utc,
                  status,
                  curriculum_stage,
                  rung,
                  quality_score,
                  objective_score,
                  payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    execution.trial.trial_id,
                    execution.source_trial_id,
                    execution.created_at_utc,
                    execution.trial.status.value,
                    execution.trial.curriculum_stage.value,
                    execution.trial.rung,
                    float(execution.trial.metrics.get("quality_score", 0.0)),
                    float(objective_score)
                    if isinstance(objective_score, int | float)
                    else None,
                    json.dumps(payload, ensure_ascii=False, sort_keys=True),
                ),
            )

    def load(self) -> tuple[dict[str, Any], ...]:
        if not self.path.exists():
            return ()
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT payload_json
                FROM optimization_trials
                ORDER BY created_at_utc, trial_id
                """
            ).fetchall()
        return tuple(json.loads(row[0]) for row in rows)


def create_trial_store(
    *,
    backend: str,
    jsonl_path: Path,
    sqlite_path: Path,
) -> JsonlTrialStore | SQLiteTrialStore:
    if backend == "jsonl":
        return JsonlTrialStore(jsonl_path)
    if backend == "sqlite":
        return SQLiteTrialStore(sqlite_path)
    raise ValueError(f"unsupported trial store backend: {backend}")


def trial_history_from_records(
    records: Sequence[Mapping[str, Any]],
    *,
    score_version: str | None = None,
) -> tuple[TrialHistoryEntry, ...]:
    """把已评估 execution 记录还原为 ASHA 可比较的源 trial 历史。"""

    history: dict[tuple[str, CurriculumStage, int], TrialHistoryEntry] = {}
    for record in records:
        score = record.get("score")
        plan = record.get("plan")
        if not isinstance(score, Mapping) or not isinstance(plan, Mapping):
            continue
        if score_version is not None and score.get("score_version") != score_version:
            continue
        try:
            trial_id = str(score.get("trial_id") or record.get("source_trial_id") or "")
            stage = CurriculumStage(str(plan.get("current_stage") or "basic"))
            rung = int(plan.get("current_rung") or 0)
            quality_score = float(score["quality_score"])
            metrics_raw = score.get("metrics")
            metrics = (
                {
                    str(key): float(value)
                    for key, value in metrics_raw.items()
                    if isinstance(value, (int, float)) and isfinite(float(value))
                }
                if isinstance(metrics_raw, Mapping)
                else {}
            )
            entry = TrialHistoryEntry(
                trial_id=trial_id,
                rung=rung,
                curriculum_stage=stage,
                quality_score=quality_score,
                # execution 在评分完成后才写入；plan.next_status 描述下一步，
                # 不能拿来冒充源 trial 的评估完成状态。
                status=TrialStatus.COMPLETED,
                metrics=metrics,
            )
        except (KeyError, TypeError, ValueError):
            # 历史文件可能包含旧版本或半写入记录；忽略单条坏记录，当前
            # trial 仍可按固定阈值安全规划。
            continue
        history[(entry.trial_id, entry.curriculum_stage, entry.rung)] = entry
    return tuple(history.values())


def _apply_section_updates(
    section: str,
    base: Mapping[str, object],
    updates: Mapping[str, Any],
) -> dict[str, object]:
    merged: dict[str, object] = dict(base)
    for key, value in updates.items():
        if key.endswith("_delta"):
            target_key = key.removesuffix("_delta")
            spec = _PARAMETER_SPECS.get((section, target_key))
            if spec is None:
                raise ValueError(
                    f"unsupported optimized parameter: {section}.{target_key}"
                )
            current = merged.get(target_key, spec.default)
            if not isinstance(current, (int, float)) or not isinstance(
                value,
                (int, float),
            ):
                raise ValueError(f"optimized delta must be numeric: {section}.{key}")
            merged[target_key] = spec.normalize(float(current) + float(value))
        elif key.endswith("_multiplier"):
            target_key = key.removesuffix("_multiplier")
            spec = _PARAMETER_SPECS.get((section, target_key))
            if spec is None:
                raise ValueError(
                    f"unsupported optimized parameter: {section}.{target_key}"
                )
            current = merged.get(target_key, spec.default)
            if not isinstance(current, (int, float)) or not isinstance(
                value,
                (int, float),
            ):
                raise ValueError(
                    f"optimized multiplier must be numeric: {section}.{key}"
                )
            merged[target_key] = spec.normalize(float(current) * float(value))
        else:
            spec = _PARAMETER_SPECS.get((section, key))
            if spec is None:
                raise ValueError(f"unsupported optimized parameter: {section}.{key}")
            merged[key] = spec.normalize(value)
    for key, value in tuple(merged.items()):
        spec = _PARAMETER_SPECS.get((section, key))
        if spec is not None:
            merged[key] = spec.normalize(value)
    return merged


def _apply_parameter_updates(
    parameters: TrialParameters,
    updates: Mapping[str, Mapping[str, Any]],
) -> TrialParameters:
    return TrialParameters(
        architecture=_apply_section_updates(
            "architecture",
            parameters.architecture,
            updates.get("architecture", {}),
        ),
        training=_apply_section_updates(
            "training",
            parameters.training,
            updates.get("training", {}),
        ),
        inference=_apply_section_updates(
            "inference",
            parameters.inference,
            updates.get("inference", {}),
        ),
    )


def normalize_trial_parameters(parameters: TrialParameters) -> TrialParameters:
    """把 job 中已有参数规范成 runner 可直接消费的绝对值。"""

    return _apply_parameter_updates(parameters, {})


def training_job_from_dict(raw: Mapping[str, Any]) -> TrainingJobSpec:
    """解析并校验持久化 job；dry-run 和真实执行必须共用此入口。"""

    trial_id = str(raw.get("trial_id") or "")
    try:
        stage = CurriculumStage(str(raw.get("curriculum_stage") or "basic"))
        rung = int(raw.get("rung") or 0)
        budget_steps = int(raw.get("budget_steps") or 0)
        parameters = normalize_trial_parameters(
            TrialParameters.model_validate(raw.get("parameters") or {})
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid training job: {error}") from error
    parent_raw = raw.get("parent_checkpoint_path")
    parent = Path(str(parent_raw)) if parent_raw else None
    if parent is not None and not parent.is_file():
        raise ValueError(f"training job parent checkpoint is unavailable: {parent}")
    weights_raw = raw.get("hard_example_weights")
    if weights_raw:
        raise ValueError(
            "hard_example_weights are not supported until weighted sampling "
            "is connected to the training dataset"
        )
    return TrainingJobSpec(
        trial_id=trial_id,
        curriculum_stage=stage,
        rung=rung,
        budget_steps=budget_steps,
        parameters=parameters,
        parent_checkpoint_path=parent,
        hard_example_weights={},
    )


def _budget_steps(config: OptimizationExecutorConfig, rung: int) -> int:
    return max(
        1,
        round(config.base_budget_steps * config.budget_multiplier_per_rung**rung),
    )


def _next_trial_id(source_trial_id: str, rung: int, stage: CurriculumStage) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    root_id = source_trial_id.split("__r", 1)[0][:16]
    lineage = hashlib.sha256(source_trial_id.encode("utf-8")).hexdigest()[:8]
    return f"{root_id}__{lineage}__r{rung:02d}__{stage.value}__{timestamp}"


def execute_optimization_plan(
    report: TrialScoreReport,
    attribution: AttributionSummary,
    plan: OptimizationPlan,
    *,
    base_parameters: TrialParameters | None = None,
    parent_checkpoint_path: Path | None = None,
    config: OptimizationExecutorConfig = OptimizationExecutorConfig(),
    store: JsonlTrialStore | SQLiteTrialStore | None = None,
) -> OptimizationExecution:
    next_rung = plan.next_rung
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
            "objective_score": plan.objective_score,
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
        # 当前训练 Dataset 尚无统一的加权采样入口；保留归因记录，但不在 job
        # 中声称这些权重已经接入训练。
        hard_example_weights={},
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
    "SQLiteTrialStore",
    "TrainingJobSpec",
    "create_trial_store",
    "execute_optimization_plan",
    "normalize_trial_parameters",
    "trial_history_from_records",
    "training_job_from_dict",
]
