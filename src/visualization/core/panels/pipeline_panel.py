from rich.panel import Panel
from rich.table import Table

from visualization.conf.messages import STATUS_NAMES
from visualization.lib.models import TrainingDashboardState


def render_pipeline_panel(state: TrainingDashboardState) -> Panel:
    table = Table.grid(expand=True)
    table.add_column("阶段")
    table.add_column("状态")
    table.add_column("进度")
    table.add_column("警告")
    for stage in state.pipeline_stages.values():
        status = STATUS_NAMES.get(stage.status, stage.status)
        total = stage.total if stage.total is not None else "?"
        table.add_row(
            stage.name,
            status,
            f"{stage.processed}/{total}",
            str(stage.warning_count),
        )
    if not state.pipeline_stages:
        table.add_row("检测流程", "等待中", "0/?", "0")
    return Panel(table, title="检测流程")
