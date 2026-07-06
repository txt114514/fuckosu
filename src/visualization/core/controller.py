from __future__ import annotations

import os
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
            _launch_multi_terminal_panels(
                reporter=reporter,
                run_id=run_id,
                output_dir=output_dir,
            )
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


def _launch_multi_terminal_panels(
    *,
    reporter: DashboardReporter,
    run_id: str,
    output_dir: Path,
) -> None:
    if os.environ.get("OSU_AI_TMUX_UI_ATTACHED") == "1":
        return
    try:
        from visualization.core.multi_terminal import launch_panel_terminals

        result = launch_panel_terminals(
            run_id=run_id,
            dashboard_dir=output_dir,
            cwd=Path.cwd(),
        )
    except Exception as error:
        result = None
        reporter.emit_event(
            TrainingEvent.create(
                event_type="ui.multi_terminal",
                severity="warning",
                message_key="fatal_error",
                message_args={"error": f"多终端面板启动失败：{error}"},
            )
        )
    if result is None:
        return
    reporter.emit_event(
        TrainingEvent.create(
            event_type="ui.multi_terminal",
            severity="success" if result.status == "launched" else "warning",
            message_key="fatal_error"
            if result.status != "launched"
            else "dashboard_started",
            message_args={"error": result.message, "run_id": result.session_name or run_id},
            raw_message=result.message,
        )
    )
