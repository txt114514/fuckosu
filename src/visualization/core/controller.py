from __future__ import annotations

from pathlib import Path

from visualization.conf import DashboardSettings
from visualization.lib.models import TrainingEvent
from visualization.lib.reporter import (
    DashboardReporter,
    ManagedDashboardHandle,
    choose_ui_mode,
)


def build_dashboard_handle(
    *,
    run_id: str,
    output_dir: Path,
    settings: DashboardSettings,
) -> ManagedDashboardHandle:
    reporter = DashboardReporter(
        run_id=run_id,
        output_dir=output_dir,
        settings=settings,
    )
    mode = choose_ui_mode(settings.mode)
    renderer = None
    if mode == "rich":
        try:
            from visualization.core.renderers.rich_renderer import RichDashboardRenderer

            renderer = RichDashboardRenderer(reporter, settings=settings)
        except Exception as error:
            reporter.emit_event(
                TrainingEvent.create(
                    event_type="ui",
                    severity="warning",
                    message_key="fatal_error",
                    message_args={"error": f"Rich 初始化失败，切换 plain：{error}"},
                )
            )
            from visualization.core.renderers.plain_renderer import PlainDashboardRenderer

            renderer = PlainDashboardRenderer(reporter, settings=settings)
    elif mode == "plain":
        from visualization.core.renderers.plain_renderer import PlainDashboardRenderer

        renderer = PlainDashboardRenderer(reporter, settings=settings)
    return ManagedDashboardHandle(reporter=reporter, renderer=renderer)
