"""定义仪表盘渲染器最小生命周期协议。"""

from __future__ import annotations

from typing import Protocol


class DashboardRenderer(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...
