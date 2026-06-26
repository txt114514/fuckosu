from __future__ import annotations

from typing import Protocol


class DashboardRenderer(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...
