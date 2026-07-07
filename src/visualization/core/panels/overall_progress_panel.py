from rich.panel import Panel
from rich.table import Table

from visualization.lib.models import TrainingDashboardState


def render_overall_progress_panel(state: TrainingDashboardState) -> Panel:
    usage = state.dataset_usage
    table = Table.grid(expand=True)
    table.add_column("项目")
    table.add_column("值")
    table.add_row("总体等级", f"{state.completed_levels}/{state.total_levels}")
    table.add_row("训练轮次", f"{state.epoch}/{state.total_epochs or '?'}")
    table.add_row("全局步数", f"{state.global_step}/{state.target_global_steps or '?'}")
    table.add_row("空间步数", f"{state.spatial_step}/{state.spatial_target or '?'}")
    table.add_row("时序步数", f"{state.temporal_step}/{state.temporal_target or '?'}")
    table.add_row("唯一片段", f"{usage.unique_segments}/{usage.total_segments}")
    table.add_row("唯一帧", f"{usage.unique_frames}/{usage.total_frames}")
    table.add_row("缓存帧", str(usage.cached_frames))
    table.add_row("候选", str(usage.generated_candidates))
    table.add_row("速度", f"{state.steps_per_second or 0:.3f} 步/秒")
    return Panel(table, title="总体训练进度")
