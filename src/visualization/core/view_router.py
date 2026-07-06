from __future__ import annotations

from typing import Literal

from rich.console import Group

from visualization.lib.models import PipelinePhase, TrainingDashboardState
from visualization.core.startup_view import STARTUP_PAGES, render_startup_view
from visualization.core.training_view import TRAINING_PAGES, render_training_view


DashboardViewKind = Literal["startup", "training"]

_STARTUP_PHASES = {
    PipelinePhase.STARTUP.value,
    PipelinePhase.DATA_PREPARATION.value,
    PipelinePhase.PRETRAIN_CHECK.value,
    PipelinePhase.PROGRESSIVE_PREPARATION.value,
}


def dashboard_view_kind(state: TrainingDashboardState) -> DashboardViewKind:
    if state.pipeline_phase in _STARTUP_PHASES:
        return "startup"
    return "training"


def dashboard_pages(state: TrainingDashboardState) -> tuple[str, ...]:
    if dashboard_view_kind(state) == "startup":
        return STARTUP_PAGES
    return TRAINING_PAGES


def render_dashboard_view(
    state: TrainingDashboardState,
    *,
    compact: bool = False,
    page: str = "overview",
) -> Group:
    if dashboard_view_kind(state) == "startup":
        return render_startup_view(state, compact=compact, page=page)
    return render_training_view(state, compact=compact, page=page)
