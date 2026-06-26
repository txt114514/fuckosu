from rich.panel import Panel
from rich.table import Table

from visualization.lib.models import TrainingDashboardState


def render_current_learning_panel(state: TrainingDashboardState) -> Panel:
    table = Table.grid(expand=True)
    table.add_column("指标")
    table.add_column("值")
    table.add_row("当前 loss", _fmt(state.metrics.loss))
    table.add_row("滑动平均 loss", _fmt(state.metrics.moving_average_loss))
    table.add_row("当前评分", _fmt(state.metrics.score))
    table.add_row("当前等级", state.current_grade or "未评级")
    table.add_row("历史最高等级", state.best_grade or "未评级")
    table.add_row("晋升状态", state.promotion_status or "未评级")
    table.add_row(
        "连续通过",
        f"{state.consecutive_passes}/{state.required_passes or '?'}",
    )
    for name, value in sorted(state.category_scores.items()):
        table.add_row(name, f"{value:.6f}")
    return Panel(table, title="当前参数学习及评分")


def _fmt(value: float | None) -> str:
    return "无" if value is None else f"{value:.6f}"
