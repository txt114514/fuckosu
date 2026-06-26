from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from visualization.lib.models import (
    DatasetUsageState,
    PipelineStageState,
    ResourceState,
    TrainingDashboardState,
    TrainingEvent,
    TrainingStopState,
)


class TrainingReporter(Protocol):
    def update_pipeline_stage(self, stage: PipelineStageState) -> None: ...
    def update_metrics(self, **metrics: Any) -> None: ...
    def report_score(
        self,
        *,
        score: float,
        trial_id: str | None = None,
        level: str | None = None,
        category_scores: dict[str, float] | None = None,
    ) -> None: ...
    def report_resource(self, resource: ResourceState) -> None: ...
    def report_dataset_usage(self, usage: DatasetUsageState) -> None: ...
    def emit_event(self, event: TrainingEvent) -> None: ...
    def request_stop(self, stop: TrainingStopState) -> None: ...
    def register_checkpoint(
        self,
        path: Path,
        *,
        is_best: bool = False,
        restorable: bool = True,
    ) -> None: ...
    def snapshot(self) -> TrainingDashboardState: ...
    def close(self) -> None: ...


class DashboardHandle(Protocol):
    reporter: TrainingReporter

    def __enter__(self) -> DashboardHandle: ...
    def __exit__(self, exc_type, exc, traceback) -> bool | None: ...
    def close(self) -> None: ...
