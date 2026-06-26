from __future__ import annotations

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel

from visualization.conf import DashboardSettings
from visualization.core.panels.best_parameters_panel import render_best_parameters_panel
from visualization.core.panels.current_learning_panel import render_current_learning_panel
from visualization.core.panels.events_panel import render_events_panel
from visualization.core.panels.overall_progress_panel import render_overall_progress_panel
from visualization.core.panels.pipeline_panel import render_pipeline_panel
from visualization.core.panels.resources_panel import render_resources_panel
from visualization.lib.reporter import DashboardReporter


class RichDashboardRenderer:
    def __init__(self, reporter: DashboardReporter, *, settings: DashboardSettings) -> None:
        self.reporter = reporter
        self.settings = settings
        self.console = Console()
        self.live: Live | None = None

    def start(self) -> None:
        self.live = Live(
            self._render(),
            console=self.console,
            refresh_per_second=self.settings.refresh_per_second,
            transient=False,
        )
        self.live.start()

    def stop(self) -> None:
        if self.live is not None:
            self.live.update(self._render())
            self.live.stop()
            self.live = None

    def _render(self):
        state = self.reporter.snapshot()
        header = Panel(
            f"当前流程：{state.phase}\n"
            f"当前状态：{state.status}\n"
            f"当前 Level：{state.current_level or '未开始'}\n"
            f"当前 Trial：{state.current_trial_id or '未分配'}",
            title="统一训练控制台",
        )
        panels = (
            render_pipeline_panel(state),
            render_best_parameters_panel(state),
            render_current_learning_panel(state),
            render_overall_progress_panel(state),
            render_resources_panel(state),
            render_events_panel(state),
        )
        return Group(header, *panels)
