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
from visualization.core.view_router import render_dashboard_page
from visualization.lib.reporter import DashboardReporter


class RichDashboardRenderer:
    def __init__(self, reporter: DashboardReporter, *, settings: DashboardSettings) -> None:
        self.reporter = reporter
        self.settings = settings
        self.console = Console()
        self.live: Live | None = None
        self._refresh_callback = self.refresh
        self._page_index = 0
        self._page_count = 1
        self._stop_keyboard = threading.Event()
        self._keyboard_thread: threading.Thread | None = None
        self._refresh_lock = threading.Lock()

    def start(self) -> None:
        self._stop_keyboard.clear()
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
                if char == "\x1b":
                    char += self._read_escape_tail()
                if self._handle_key(char):
                    self.refresh()
        finally:
            with suppress(Exception):
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def _handle_key(self, char: str) -> bool:
        if char in {"\t", " ", "\x1b[B"}:
            return self._set_page_index(self._page_index + 1)
        if char in {"b", "B", "\x1b[A"}:
            return self._set_page_index(self._page_index - 1)
        if char.isdigit():
            index = int(char) - 1
            return self._set_page_index(index)
        return False

    def _read_escape_tail(self) -> str:
        tail: list[str] = []
        deadline = time.monotonic() + 0.05
        while len(tail) < 8:
            timeout = max(deadline - time.monotonic(), 0.0)
            if timeout <= 0:
                break
            ready, _, _ = select.select([sys.stdin], [], [], timeout)
            if not ready:
                break
            item = sys.stdin.read(1)
            tail.append(item)
            if tail[0] == "[" and item.isalpha():
                break
        return "".join(tail)

    def _set_page_index(self, index: int) -> bool:
        page_count = max(self._page_count, 1)
        selected = min(max(index, 0), page_count - 1)
        if selected == self._page_index:
            return False
        self._page_index = selected
        return True
