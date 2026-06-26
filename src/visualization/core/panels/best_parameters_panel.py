from rich.panel import Panel
from rich.table import Table

from visualization.lib.models import TrainingDashboardState


def render_best_parameters_panel(state: TrainingDashboardState) -> Panel:
    best = state.best_parameters
    table = Table.grid(expand=True)
    table.add_column("指标")
    table.add_column("值")
    table.add_row("当前参数当前评分", _fmt(state.metrics.score))
    table.add_row("当前参数历史最高", _fmt(state.metrics.parameter_best_score))
    table.add_row("当前 Level 最佳", _fmt(state.metrics.level_best_score))
    table.add_row("本次全局最高", _fmt(state.metrics.run_global_best_score))
    table.add_row("继承历史最高", _fmt(state.metrics.inherited_best_score))
    table.add_row("最佳 Trial", best.trial_id or "无")
    table.add_row("最佳 Step", str(best.step or 0))
    table.add_row("最佳等级", best.grade or "未评级")
    table.add_row("最佳检查点", best.checkpoint_path or "无")
    return Panel(table, title="最佳参数记录")


def _fmt(value: float | None) -> str:
    return "无" if value is None else f"{value:.6f}"
