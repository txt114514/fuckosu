from __future__ import annotations

from rich.console import Group
from rich.panel import Panel

from visualization.conf.messages import (
    display_pipeline_phase,
    display_status,
    display_text,
)
from visualization.core.panels.events_panel import render_events_panel
from visualization.core.panels.pipeline_panel import render_pipeline_panel
from visualization.core.panels.resources_panel import render_resources_panel
from visualization.lib.models import TrainingDashboardState


STARTUP_PAGES: tuple[str, ...] = ("overview", "resources", "events")


def render_startup_view(
    state: TrainingDashboardState,
    *,
    compact: bool = False,
    page: str = "overview",
) -> Group:
    header = render_startup_header(state, compact=compact, page=page)
    if compact:
        selected = page if page in STARTUP_PAGES else "overview"
        panels = {
            "overview": (
                render_pipeline_panel(state),
                render_resources_panel(state, compact=True),
            ),
            "resources": (render_resources_panel(state),),
            "events": (render_events_panel(state, limit=8),),
        }[selected]
        return Group(header, *panels)
    return Group(
        header,
        render_pipeline_panel(state),
        render_resources_panel(state),
        render_events_panel(state),
    )


def dashboard_panels(state: TrainingDashboardState) -> tuple[Panel, ...]:
    return (
        render_startup_header(state),
        render_pipeline_panel(state),
        render_resources_panel(state),
        render_events_panel(state),
    )


def render_startup_header(
    state: TrainingDashboardState,
    *,
    compact: bool = False,
    page: str = "overview",
) -> Panel:
    return Panel(
        f"流程阶段：{display_pipeline_phase(state.pipeline_phase)}\n"
        f"当前流程：{display_text(state.phase)}\n"
        f"当前状态：{display_status(state.status)}\n"
        "目标：确认系统是否具备进入正式训练的条件"
        + (_compact_hint(page, STARTUP_PAGES) if compact else ""),
        title="启动检查与渐进训练准备",
    )


def _compact_hint(page: str, pages: tuple[str, ...]) -> str:
    current = pages.index(page) + 1 if page in pages else 1
    return (
        f"\n紧凑页：{current}/{len(pages)}；"
        "Tab/空格下一页，b上一页，f完整视图，Ctrl+C中断"
    )
