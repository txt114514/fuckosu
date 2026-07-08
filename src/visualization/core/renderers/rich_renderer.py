from __future__ import annotations

import threading

from rich.console import Console
from rich.live import Live

from visualization.conf import DashboardSettings
from visualization.core.view_router import render_dashboard_page
from visualization.lib.reporter import DashboardReporter


class RichDashboardRenderer:
    def __init__(
        self, reporter: DashboardReporter, *, settings: DashboardSettings
    ) -> None:
        self.reporter = reporter
        self.settings = settings
        self.console = Console()
        self.live: Live | None = None
        self._refresh_callback = self.refresh
        self._page_index = 0
        self._page_count = 1
        self._refresh_lock = threading.Lock()

    def start(self) -> None:
        self.live = Live(
            self._render(),
            console=self.console,
            refresh_per_second=self.settings.refresh_per_second,
            transient=False,
        )
        self.live.start()
        self.reporter.add_refresh_callback(self._refresh_callback)
        self.refresh()

    def stop(self) -> None:
        self.reporter.remove_refresh_callback(self._refresh_callback)
        if self.live is not None:
            self.refresh()
            self.live.stop()
            self.live = None

    def refresh(self) -> None:
        with self._refresh_lock:
            if self.live is not None:
                self.live.update(self._render())

    def _render(self):
        state = self.reporter.snapshot()
        renderable, page_count = render_dashboard_page(
            state,
            page_index=self._page_index,
            terminal_height=self.console.size.height,
            terminal_width=self.console.size.width,
            state_path=str(self.reporter.store.state_path),
        )
        self._page_count = page_count
        if self._page_index >= page_count:
            self._page_index = max(page_count - 1, 0)
            renderable, self._page_count = render_dashboard_page(
                state,
                page_index=self._page_index,
                terminal_height=self.console.size.height,
                terminal_width=self.console.size.width,
                state_path=str(self.reporter.store.state_path),
            )
        return renderable
