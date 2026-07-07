from __future__ import annotations

from typing import Literal

from rich.console import Console, Group
from rich.panel import Panel

from visualization.lib.models import PipelinePhase, TrainingDashboardState
from visualization.core.startup_view import (
    STARTUP_PAGES,
    dashboard_panels as startup_dashboard_panels,
    render_startup_view,
)
from visualization.core.training_view import (
    TRAINING_PAGES,
    dashboard_panels as training_dashboard_panels,
    render_training_view,
)


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


def dashboard_panel_items(state: TrainingDashboardState) -> tuple[Panel, ...]:
    if dashboard_view_kind(state) == "startup":
        return startup_dashboard_panels(state)
    return training_dashboard_panels(state)


def render_dashboard_page(
    state: TrainingDashboardState,
    *,
    page_index: int,
    terminal_height: int,
    terminal_width: int,
) -> tuple[Group, int]:
    panels = dashboard_panel_items(state)
    pages = _paginate_panels(
        panels,
        terminal_height=max(terminal_height, 6),
        terminal_width=max(terminal_width, 40),
    )
    page_count = max(len(pages), 1)
    selected_index = min(max(page_index, 0), page_count - 1)
    selected = pages[selected_index] if pages else panels
    footer = Panel(
        (
            f"页面：{selected_index + 1}/{page_count}；"
            "上方向键上一页，下方向键下一页，Tab/空格下一页，b上一页，Ctrl+C中断"
        ),
        title="翻页",
    )
    return Group(*selected, footer), page_count


def render_dashboard_view(
    state: TrainingDashboardState,
    *,
    compact: bool = False,
    page: str = "overview",
) -> Group:
    if dashboard_view_kind(state) == "startup":
        return render_startup_view(state, compact=compact, page=page)
    return render_training_view(state, compact=compact, page=page)


def _paginate_panels(
    panels: tuple[Panel, ...],
    *,
    terminal_height: int,
    terminal_width: int,
) -> tuple[tuple[Panel, ...], ...]:
    available_height = max(terminal_height - 4, 1)
    pages: list[tuple[Panel, ...]] = []
    current: list[Panel] = []
    current_height = 0
    for panel in panels:
        height = _renderable_height(panel, terminal_width)
        if current and current_height + height > available_height:
            pages.append(tuple(current))
            current = []
            current_height = 0
        current.append(panel)
        current_height += height
    if current:
        pages.append(tuple(current))
    return tuple(pages)


def _renderable_height(panel: Panel, terminal_width: int) -> int:
    console = Console(
        record=True,
        width=max(terminal_width, 40),
        color_system=None,
        force_terminal=False,
    )
    try:
        console.print(panel)
        return max(len(console.export_text().splitlines()), 1)
    except Exception:
        return 4
