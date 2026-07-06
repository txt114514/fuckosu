from rich.panel import Panel
from rich.table import Table

from visualization.lib.models import TrainingDashboardState


def render_resources_panel(
    state: TrainingDashboardState,
    *,
    compact: bool = False,
) -> Panel:
    resource = state.resources
    table = Table.grid(expand=True)
    table.add_column("资源")
    table.add_column("值")
    table.add_row("GPU", resource.gpu_name or "不可用")
    table.add_row("当前 allocated", _gb(resource.gpu_allocated_gb))
    table.add_row("当前 reserved", _gb(resource.gpu_reserved_gb))
    if not compact:
        table.add_row("峰值 allocated", _gb(resource.gpu_peak_allocated_gb))
    table.add_row("峰值 reserved", _gb(resource.gpu_peak_reserved_gb))
    if not compact:
        table.add_row("总显存", _gb(resource.gpu_total_gb))
        table.add_row("GPU 利用率", _pct(resource.gpu_utilization))
        table.add_row("GPU 温度", _num(resource.gpu_temperature_c, " C"))
    table.add_row("CPU", _pct(resource.cpu_percent))
    table.add_row("进程内存", _gb(resource.process_memory_gb))
    table.add_row("磁盘剩余", _gb(resource.disk_free_gb))
    return Panel(table, title="本机资源")


def _gb(value: float | None) -> str:
    return "不可用" if value is None else f"{value:.3f} GB"


def _pct(value: float | None) -> str:
    return "不可用" if value is None else f"{value:.1f}%"


def _num(value: float | None, suffix: str) -> str:
    return "不可用" if value is None else f"{value:.1f}{suffix}"
