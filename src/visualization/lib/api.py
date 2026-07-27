"""提供低耦合的仪表盘报告器工厂，并支持完全关闭可视化。"""

from __future__ import annotations

from pathlib import Path
from importlib import import_module

from visualization.conf import DashboardSettings
from visualization.lib.reporter import NullReporter


def create_dashboard_reporter(
    *,
    run_id: str,
    output_dir: Path,
    progress_ui: str = "auto",
    progress_language: str = "zh-CN",
):
    if progress_ui == "off":
        return _NullDashboardHandle()
    settings = DashboardSettings(mode=progress_ui, language=progress_language)
    # 延迟导入 core，保持训练代码只依赖轻量 lib 边界，并避免加载可选 GUI/Rich 实现。
    build_dashboard_handle = import_module(
        "visualization.core.controller"
    ).build_dashboard_handle
    return build_dashboard_handle(
        run_id=run_id,
        output_dir=output_dir,
        settings=settings,
    )


class _NullDashboardHandle:
    reporter = NullReporter()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return None

    def close(self) -> None:
        return None
