from __future__ import annotations

from contextlib import suppress
import select
import sys
import termios
import threading
import time
import tty

from rich.console import Console
from rich.live import Live

from visualization.conf import DashboardSettings
from visualization.core.view_router import dashboard_pages, render_dashboard_view
from visualization.lib.reporter import DashboardReporter


class RichDashboardRenderer:
    def __init__(self, reporter: DashboardReporter, *, settings: DashboardSettings) -> None:
        self.reporter = reporter
        self.settings = settings
        self.console = Console()
        self.live: Live | None = None
        self._refresh_callback = self.refresh
        self._page = "overview"
        self._force_full = False
        self._last_page_switch = time.monotonic()
        self._stop_keyboard = threading.Event()
        self._keyboard_thread: threading.Thread | None = None
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
        self._start_keyboard_listener()
        self.refresh()

    def stop(self) -> None:
        self._stop_keyboard_listener()
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
        pages = dashboard_pages(state)
        if self._page not in pages:
            self._page = pages[0]
        compact = (
            not self._force_full
            and self.console.size.height <= self.settings.compact_terminal_height
        )
        if compact:
            self._advance_page_if_due(pages)
        return render_dashboard_view(state, compact=compact, page=self._page)

    def _start_keyboard_listener(self) -> None:
        if self._keyboard_thread is not None:
            return
        if not sys.stdin.isatty():
            return
        self._keyboard_thread = threading.Thread(
            target=self._keyboard_loop,
            name="rich-dashboard-keyboard",
            daemon=True,
        )
        self._keyboard_thread.start()

    def _stop_keyboard_listener(self) -> None:
        self._stop_keyboard.set()
        if self._keyboard_thread is not None:
            self._keyboard_thread.join(timeout=0.5)
            self._keyboard_thread = None

    def _keyboard_loop(self) -> None:
        fd = sys.stdin.fileno()
        with suppress(Exception):
            old_settings = termios.tcgetattr(fd)
        if "old_settings" not in locals():
            return
        try:
            tty.setcbreak(fd)
            while not self._stop_keyboard.is_set():
                ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                if not ready:
                    continue
                char = sys.stdin.read(1)
                if self._handle_key(char):
                    self.refresh()
        finally:
            with suppress(Exception):
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def _handle_key(self, char: str) -> bool:
        pages = dashboard_pages(self.reporter.snapshot())
        if char in {"\t", " "}:
            self._page = _relative_page(self._page, pages, 1)
            self._last_page_switch = time.monotonic()
            return True
        if char in {"b", "B"}:
            self._page = _relative_page(self._page, pages, -1)
            self._last_page_switch = time.monotonic()
            return True
        if char in {"f", "F"}:
            self._force_full = not self._force_full
            return True
        if char.isdigit():
            index = int(char) - 1
            if 0 <= index < len(pages):
                self._page = pages[index]
                self._last_page_switch = time.monotonic()
                return True
        return False

    def _advance_page_if_due(self, pages: tuple[str, ...]) -> None:
        if len(pages) <= 1:
            return
        now = time.monotonic()
        if now - self._last_page_switch < self.settings.auto_page_seconds:
            return
        self._page = _relative_page(self._page, pages, 1)
        self._last_page_switch = now


def _relative_page(current: str, pages: tuple[str, ...], offset: int) -> str:
    if not pages:
        return current
    index = pages.index(current) if current in pages else 0
    return pages[(index + offset) % len(pages)]
