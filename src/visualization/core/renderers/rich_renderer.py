"""使用 Rich Live 渲染可分页终端仪表盘。"""

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
        # 先启动 Live 再注册回调，避免首个状态事件刷新尚未初始化的渲染对象。
        self.reporter.add_refresh_callback(self._refresh_callback)
        self.refresh()

    def stop(self) -> None:
        # 先解除事件流订阅，确保 Live 停止后不会再收到跨线程刷新。
        self.reporter.remove_refresh_callback(self._refresh_callback)
        if self.live is not None:
            self.refresh()
            self.live.stop()
            self.live = None

    def refresh(self) -> None:
        # 报告器更新可能来自训练和资源采样线程，Rich Live 更新必须串行化。
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
