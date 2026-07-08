from __future__ import annotations

from collections import deque
from contextlib import suppress
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Callable, Mapping

from visualization.conf import DashboardSettings
from visualization.lib.models import (
    BestParameterRecord,
    CurrentTrainingMetrics,
    DatasetUsageState,
    PipelinePhase,
    PipelineStageState,
    ResourceState,
    TrainingDashboardState,
    TrainingEvent,
    TrainingStopState,
    utc_now,
)
from visualization.state import DashboardStateStore


def _now_seconds() -> float:
    return datetime.now(timezone.utc).timestamp()


_ACTIVE_STAGE_STATUSES = {
    "scanning",
    "converting",
    "checking",
    "running",
    "training",
    "evaluating",
}
_PASSED_STAGE_STATUSES = {"passed", "completed"}
_WARNING_STAGE_STATUSES = {"warning", "skipped"}
_FAILED_STAGE_STATUSES = {"failed", "interrupted"}
_GLOBAL_TERMINAL_STATUSES = {"failed", "interrupted"}


class NullReporter:
    def update_pipeline_stage(self, stage: PipelineStageState) -> None:
        return None

    def update_metrics(self, **metrics: Any) -> None:
        return None

    def report_score(
        self,
        *,
        score: float,
        trial_id: str | None = None,
        level: str | None = None,
        category_scores: dict[str, float] | None = None,
    ) -> None:
        return None

    def report_resource(self, resource: ResourceState) -> None:
        return None

    def report_dataset_usage(self, usage: DatasetUsageState) -> None:
        return None

    def emit_event(self, event: TrainingEvent) -> None:
        return None

    def request_stop(self, stop: TrainingStopState) -> None:
        return None

    def register_checkpoint(
        self,
        path: Path,
        *,
        is_best: bool = False,
        restorable: bool = True,
    ) -> None:
        return None

    def snapshot(self) -> TrainingDashboardState:
        return TrainingDashboardState(run_id="off", status="off")

    def close(self) -> None:
        return None


class DashboardReporter:
    def __init__(
        self,
        *,
        run_id: str,
        output_dir: Path,
        settings: DashboardSettings | None = None,
    ) -> None:
        self.settings = settings or DashboardSettings()
        self.store = DashboardStateStore(output_dir)
        self.started_at = _now_seconds()
        self._events: deque[TrainingEvent] = deque(
            maxlen=self.settings.recent_event_limit
        )
        self._state = TrainingDashboardState(run_id=run_id, status="running")
        self._best_score: float | None = None
        self._last_resource_monitor_error: str | None = None
        self._refresh_callbacks: list[Callable[[], None]] = []
        self._sync_current_parameter_status()
        self._write_state()

    def add_refresh_callback(self, callback: Callable[[], None]) -> None:
        self._refresh_callbacks.append(callback)

    def remove_refresh_callback(self, callback: Callable[[], None]) -> None:
        with suppress(ValueError):
            self._refresh_callbacks.remove(callback)

    def update_pipeline_stage(self, stage: PipelineStageState) -> None:
        previous = self._state.pipeline_stages.get(stage.stage_id)
        self._state.pipeline_stages[stage.stage_id] = stage
        self._state.phase = stage.name
        if not _protects_terminal_global_state(self._state) or stage.status in {
            "failed",
            "interrupted",
        }:
            self._state.status = stage.status
        self._touch()
        self._sync_current_parameter_status()
        event = _stage_lifecycle_event(previous, stage)
        if event is not None:
            self._record_event(event)
        self._write_state()

    def update_metrics(self, **metrics: Any) -> None:
        allowed = set(CurrentTrainingMetrics.__dataclass_fields__)
        current = self._state.metrics.as_dict()
        parameters_updated = False
        for key, value in metrics.items():
            if key in allowed:
                current[key] = value
            elif key == "current_parameters":
                self._state.current_parameters = _parameter_payload(value)
                parameters_updated = True
            elif hasattr(self._state, key):
                if _should_skip_terminal_global_metric(self._state, key, value):
                    continue
                setattr(self._state, key, value)
        self._state.metrics = CurrentTrainingMetrics(**current)
        self._touch()
        if parameters_updated:
            self.store.write_named(
                "current_parameters.json",
                self._state.current_parameters,
            )
        self._sync_current_parameter_status()
        self._write_state()

    def report_score(
        self,
        *,
        score: float,
        trial_id: str | None = None,
        level: str | None = None,
        category_scores: dict[str, float] | None = None,
    ) -> None:
        self._state.metrics.score = score
        self._state.current_trial_id = trial_id or self._state.current_trial_id
        self._state.current_level = level or self._state.current_level
        if category_scores is not None:
            self._state.category_scores = dict(category_scores)
        if self._state.metrics.parameter_best_score is None:
            self._state.metrics.parameter_best_score = score
        else:
            self._state.metrics.parameter_best_score = max(
                self._state.metrics.parameter_best_score,
                score,
            )
        if self._best_score is None or score > self._best_score:
            self._best_score = score
            self._state.metrics.run_global_best_score = score
            self._state.best_parameters = BestParameterRecord(
                trial_id=self._state.current_trial_id,
                parameters=dict(self._state.current_parameters),
                score=score,
                step=self._state.global_step,
                epoch=self._state.epoch,
                level=self._state.current_level,
                grade=self._state.current_grade,
                checkpoint_path=self._state.latest_checkpoint,
                scored_at=utc_now(),
                restorable=bool(self._state.latest_checkpoint),
            )
            self.store.write_named(
                "best_parameters.json",
                self._state.best_parameters.as_dict(),
            )
            self.store.write_named(
                "best_parameters.yaml",
                self._state.best_parameters.as_dict(),
            )
            self.emit_event(
                TrainingEvent.create(
                    event_type="score",
                    severity="success",
                    message_key="best_score_updated",
                    message_args={"score": f"{score:.6f}"},
                    level=level,
                    trial_id=trial_id,
                )
            )
        else:
            self.emit_event(
                TrainingEvent.create(
                    event_type="score",
                    severity="info",
                    message_key="score_updated",
                    message_args={"score": f"{score:.6f}"},
                    level=level,
                    trial_id=trial_id,
                )
            )
        self._touch()
        self._sync_current_parameter_status()
        self._write_state()

    def report_resource(self, resource: ResourceState) -> None:
        self._state.resources = resource
        self._touch()
        self.store.write_named("resource_history.jsonl.current.json", resource.as_dict())
        monitor_error = resource.gpu_monitor_error
        if monitor_error and monitor_error != self._last_resource_monitor_error:
            self._record_event(
                TrainingEvent.create(
                    event_type="resource",
                    severity="warning",
                    message_key="resource_monitor_warning",
                    message_args={"error": monitor_error},
                )
            )
        elif self._last_resource_monitor_error and not monitor_error:
            self._record_event(
                TrainingEvent.create(
                    event_type="resource",
                    severity="success",
                    message_key="resource_monitor_restored",
                    message_args={"source": resource.gpu_monitor_source or "GPU 监控"},
                )
            )
        self._last_resource_monitor_error = monitor_error
        self._write_state()

    def report_dataset_usage(self, usage: DatasetUsageState) -> None:
        self._state.dataset_usage = usage
        self.store.write_named("dataset_usage.json", usage.as_dict())
        self._touch()
        self._write_state()

    def emit_event(self, event: TrainingEvent) -> None:
        self._record_event(event)
        self._touch()
        self._sync_current_parameter_status()
        self._write_state()

    def _record_event(self, event: TrainingEvent) -> None:
        self._events.append(event)
        self._state.recent_events = list(self._events)
        with suppress(Exception):
            self.store.append_event(event)

    def request_stop(self, stop: TrainingStopState) -> None:
        self._state.stop_state = stop
        self._state.pipeline_phase = (
            PipelinePhase.FAILED.value
            if stop.reason != "USER_INTERRUPTED"
            else self._state.pipeline_phase
        )
        self._state.status = "interrupted" if stop.reason == "USER_INTERRUPTED" else "failed"
        self.store.write_named("stop_state.json", stop.as_dict())
        self.emit_event(
            TrainingEvent.create(
                event_type="stop",
                severity="error",
                message_key="dataset_exhausted"
                if stop.reason == "DATASET_EXHAUSTED"
                else "fatal_error",
                message_args={"error": stop.message},
                raw_message=stop.message,
            )
        )

    def register_checkpoint(
        self,
        path: Path,
        *,
        is_best: bool = False,
        restorable: bool = True,
    ) -> None:
        self._state.latest_checkpoint = str(path)
        if is_best:
            self._state.best_checkpoint = str(path)
            self._state.best_parameters = replace(
                self._state.best_parameters,
                checkpoint_path=str(path),
                restorable=restorable,
            )
        self._sync_current_parameter_status()
        self.emit_event(
            TrainingEvent.create(
                event_type="checkpoint",
                severity="success",
                message_key="checkpoint_saved",
                message_args={"path": str(path)},
            )
        )

    def snapshot(self) -> TrainingDashboardState:
        return self._state

    def close(self) -> None:
        if not _protects_terminal_global_state(self._state):
            self._state.status = "completed"
            self._state.pipeline_phase = PipelinePhase.COMPLETED.value
        self.emit_event(
            TrainingEvent.create(
                event_type="dashboard",
                severity="info",
                message_key="dashboard_stopped",
                message_args={"status": self._state.status},
            )
        )
        self._write_state()

    def _touch(self) -> None:
        self._state.elapsed_seconds = max(_now_seconds() - self.started_at, 0.0)
        self._state.updated_at = utc_now()

    def _write_state(self) -> None:
        with suppress(Exception):
            self.store.write_state(self._state)
        self._notify_refresh_callbacks()

    def _notify_refresh_callbacks(self) -> None:
        for callback in tuple(self._refresh_callbacks):
            with suppress(Exception):
                callback()

    def _sync_current_parameter_status(self) -> None:
        self._state.current_parameter_status = _parameter_status_payload(self._state)
        with suppress(Exception):
            self.store.write_named(
                "current_parameter_status.json",
                self._state.current_parameter_status,
            )


def _parameter_payload(value: Any) -> dict[str, object]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    return {"value": value}


def _stage_lifecycle_event(
    previous: PipelineStageState | None,
    stage: PipelineStageState,
) -> TrainingEvent | None:
    if stage.status == "pending":
        return None
    if previous is not None and previous.status == stage.status:
        return None
    if stage.status in _ACTIVE_STAGE_STATUSES:
        severity = "info"
        message_key = "stage_lifecycle_started"
    elif stage.status in _PASSED_STAGE_STATUSES:
        severity = "success"
        message_key = "stage_lifecycle_passed"
    elif stage.status in _WARNING_STAGE_STATUSES:
        severity = "warning" if stage.status == "warning" else "info"
        message_key = (
            "stage_lifecycle_warning"
            if stage.status == "warning"
            else "stage_lifecycle_skipped"
        )
    elif stage.status in _FAILED_STAGE_STATUSES:
        severity = "error"
        message_key = "stage_lifecycle_failed"
    else:
        return None
    return TrainingEvent.create(
        event_type="stage_lifecycle",
        severity=severity,
        message_key=message_key,
        message_args=_stage_lifecycle_message_args(stage),
        phase=stage.name,
        raw_message=_stage_lifecycle_raw_message(message_key, stage),
    )


def _stage_lifecycle_message_args(stage: PipelineStageState) -> dict[str, object]:
    args: dict[str, object] = {"stage": stage.name, "status": stage.status}
    if stage.message:
        args["message"] = stage.message
    elif stage.error_reason:
        args["message"] = stage.error_reason
    if stage.warning_count:
        args["warnings"] = stage.warning_count
    if stage.score is not None:
        args["score"] = f"{stage.score:.6f}"
    if stage.threshold is not None:
        args["threshold"] = f"{stage.threshold:.6f}"
    return args


def _stage_lifecycle_raw_message(
    message_key: str,
    stage: PipelineStageState,
) -> str | None:
    detail = stage.message or stage.error_reason
    if not detail and stage.score is None and stage.warning_count <= 0:
        return None
    label = {
        "stage_lifecycle_started": "阶段开始",
        "stage_lifecycle_passed": "阶段完成",
        "stage_lifecycle_warning": "阶段警告",
        "stage_lifecycle_skipped": "阶段跳过",
        "stage_lifecycle_failed": "阶段失败",
    }.get(message_key, "阶段")
    parts = [f"{label}：{stage.name}（{stage.status}）"]
    if detail:
        parts.append(str(detail))
    if stage.score is not None:
        score = f"评分 {stage.score:.6f}"
        if stage.threshold is not None:
            score += f" / 阈值 {stage.threshold:.6f}"
        parts.append(score)
    if stage.warning_count:
        parts.append(f"警告/未解析 {stage.warning_count}")
    return "；".join(parts)


def _protects_terminal_global_state(state: TrainingDashboardState) -> bool:
    return (
        state.stop_state is not None
        or state.status in _GLOBAL_TERMINAL_STATUSES
        or state.pipeline_phase == PipelinePhase.FAILED.value
    )


def _should_skip_terminal_global_metric(
    state: TrainingDashboardState,
    key: str,
    value: object,
) -> bool:
    if not _protects_terminal_global_state(state):
        return False
    if key == "status":
        return str(value) not in _GLOBAL_TERMINAL_STATUSES
    if key == "pipeline_phase":
        return str(value) != PipelinePhase.FAILED.value
    return False


def _parameter_status_payload(state: TrainingDashboardState) -> dict[str, object]:
    stages = [
        _stage_status_payload(stage)
        for stage in state.pipeline_stages.values()
    ]
    stage_counts: dict[str, int] = {}
    passed_tests: list[str] = []
    warning_tests: list[str] = []
    failed_tests: list[str] = []
    running_tests: list[str] = []
    pending_tests: list[str] = []
    test_scores: dict[str, float] = {}
    test_statuses: dict[str, str] = {}
    test_thresholds: dict[str, float] = {}
    for stage in stages:
        status = str(stage["status"])
        stage_counts[status] = stage_counts.get(status, 0) + 1
        name = str(stage["name"])
        test_statuses[name] = status
        score = stage.get("score")
        threshold = stage.get("threshold")
        if score is not None:
            test_scores[name] = float(score)
        if threshold is not None:
            test_thresholds[name] = float(threshold)
        if status in {"passed", "completed"}:
            passed_tests.append(name)
        elif status == "warning":
            warning_tests.append(name)
        elif status in {"failed", "interrupted"}:
            failed_tests.append(name)
        elif status in {"running", "checking"}:
            running_tests.append(name)
        elif status == "pending":
            pending_tests.append(name)
    for name, score in state.category_scores.items():
        test_scores.setdefault(str(name), float(score))
    stage_index = state.completed_levels + (1 if state.current_level else 0)
    stage_budget = state.target_global_steps
    trial_status = state.trial_status or _derive_trial_status(state)
    return {
        "trial_id": state.current_trial_id,
        "parameter_values": dict(state.current_parameters),
        "pipeline_phase": state.pipeline_phase,
        "level": state.current_level,
        "curriculum_stage": state.current_level or state.phase,
        "stage_index": stage_index,
        "total_stage_count": state.total_levels,
        "train_step": state.global_step,
        "stage_step": state.global_step,
        "stage_budget": stage_budget,
        "grade": state.current_grade,
        "promotion_status": state.promotion_status,
        "consecutive_passes": state.consecutive_passes,
        "required_passes": state.required_passes,
        "required_consecutive_passes": state.required_passes,
        "score": state.metrics.score,
        "latest_score": state.metrics.score,
        "parameter_best_score": state.metrics.parameter_best_score,
        "best_score": _best_score(state),
        "level_best_score": state.metrics.level_best_score,
        "run_global_best_score": state.metrics.run_global_best_score,
        "category_scores": dict(state.category_scores),
        "stage_counts": stage_counts,
        "passed_tests": passed_tests,
        "warning_tests": warning_tests,
        "failed_tests": failed_tests,
        "running_tests": running_tests,
        "pending_tests": pending_tests,
        "test_scores": test_scores,
        "test_statuses": test_statuses,
        "test_thresholds": test_thresholds,
        "stages": stages,
        "latest_checkpoint": state.latest_checkpoint,
        "checkpoint_path": state.latest_checkpoint,
        "best_checkpoint": state.best_checkpoint,
        "trial_status": trial_status,
        "prune_reason": state.prune_reason,
        "budget_used": state.global_step,
        "budget_total": stage_budget,
        "budget_used_ratio": _ratio(state.global_step, stage_budget),
        "stop_reason": state.stop_state.reason if state.stop_state is not None else None,
        "stop_message": state.stop_state.message if state.stop_state is not None else None,
        "updated_at": state.updated_at,
    }


def _stage_status_payload(stage: PipelineStageState) -> dict[str, object]:
    return {
        "stage_id": stage.stage_id,
        "name": stage.name,
        "status": stage.status,
        "processed": stage.processed,
        "total": stage.total,
        "warning_count": stage.warning_count,
        "error_reason": stage.error_reason,
        "message": stage.message,
        "output_path": stage.output_path,
        "blocks_training": stage.blocks_training,
        "started_at": stage.started_at,
        "ended_at": stage.ended_at,
        "score": stage.score,
        "threshold": stage.threshold,
    }


def _best_score(state: TrainingDashboardState) -> float | None:
    values = (
        state.metrics.parameter_best_score,
        state.metrics.run_global_best_score,
        state.metrics.inherited_best_score,
        state.best_parameters.score,
    )
    available = [float(value) for value in values if value is not None]
    return max(available) if available else None


def _ratio(value: int | None, total: int | None) -> float | None:
    if value is None or total in (None, 0):
        return None
    return min(max(float(value) / float(total), 0.0), 1.0)


def _derive_trial_status(state: TrainingDashboardState) -> str | None:
    if not state.current_trial_id:
        return None
    if state.stop_state is not None:
        return "failed"
    if state.status in {"failed", "interrupted"}:
        return "failed"
    if state.current_grade == "pruned":
        return "pruned"
    if state.current_grade == "promoted":
        return "promoted"
    if state.status == "completed":
        return "completed"
    if state.phase.lower().find("评估") >= 0 or state.status == "evaluating":
        return "evaluating"
    if state.status in {"running", "checking"}:
        return "training"
    if state.status in {"passed", "warning"}:
        return "passed"
    return state.status or None


class ManagedDashboardHandle:
    def __init__(self, reporter: DashboardReporter, renderer: object | None = None) -> None:
        self.reporter = reporter
        self.renderer = renderer

    def __enter__(self) -> ManagedDashboardHandle:
        if hasattr(self.renderer, "start"):
            with suppress(Exception):
                self.renderer.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool | None:
        self.close()
        return None

    def close(self) -> None:
        self.reporter.close()
        if hasattr(self.renderer, "stop"):
            with suppress(Exception):
                self.renderer.stop()


def choose_ui_mode(mode: str) -> str:
    if mode != "auto":
        return mode
    if sys.stdout.isatty() and sys.stdin.isatty():
        return "rich"
    return "plain"
