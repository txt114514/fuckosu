from rich.panel import Panel
from rich.table import Table

from visualization.conf.messages import SEVERITY_NAMES, render_message
from visualization.lib.models import TrainingDashboardState


def render_events_panel(
    state: TrainingDashboardState,
    *,
    limit: int = 12,
) -> Panel:
    table = Table.grid(expand=True)
    table.add_column("时间")
    table.add_column("级别")
    table.add_column("事件")
    for event in state.recent_events[-limit:]:
        table.add_row(
            event.timestamp[-8:],
            SEVERITY_NAMES.get(event.severity, event.severity),
            event.raw_message or render_message(event.message_key, event.message_args),
        )
    if not state.recent_events:
        table.add_row("--:--:--", "信息", "暂无事件")
    return Panel(table, title="最近事件与警告")
