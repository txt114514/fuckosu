from __future__ import annotations

from collections import deque
from contextlib import suppress
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

from visualization.conf import DashboardSettings
from visualization.lib.models import (
    BestParameterRecord,
    CurrentTrainingMetrics,
    DatasetUsageState,
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
        self._write_state()

    def update_pipeline_stage(self, stage: PipelineStageState) -> None:
        self._state.pipeline_stages[stage.stage_id] = stage
        self._state.phase = stage.name
        self._state.status = stage.status
        self._touch()
        self._write_state()

    def update_metrics(self, **metrics: Any) -> None:
        allowed = set(CurrentTrainingMetrics.__dataclass_fields__)
        current = self._state.metrics.as_dict()
        for key, value in metrics.items():
            if key in allowed:
                current[key] = value
            elif hasattr(self._state, key):
                setattr(self._state, key, value)
        self._state.metrics = CurrentTrainingMetrics(**current)
        self._touch()
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
        if category_scores:
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
        self._write_state()

    def report_resource(self, resource: ResourceState) -> None:
        self._state.resources = resource
        self._touch()
        self.store.write_named("resource_history.jsonl.current.json", resource.as_dict())
        self._write_state()

    def report_dataset_usage(self, usage: DatasetUsageState) -> None:
        self._state.dataset_usage = usage
        self.store.write_named("dataset_usage.json", usage.as_dict())
        self._touch()
        self._write_state()

    def emit_event(self, event: TrainingEvent) -> None:
        self._events.append(event)
        self._state.recent_events = list(self._events)
        with suppress(Exception):
            self.store.append_event(event)
        self._touch()
        self._write_state()

    def request_stop(self, stop: TrainingStopState) -> None:
        self._state.stop_state = stop
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
        self._state.status = "completed" if self._state.stop_state is None else self._state.status
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
        if hasattr(self.renderer, "stop"):
            with suppress(Exception):
                self.renderer.stop()
        self.reporter.close()


def choose_ui_mode(mode: str) -> str:
    if mode != "auto":
        return mode
    if sys.stdout.isatty() and sys.stdin.isatty():
        return "rich"
    return "plain"
