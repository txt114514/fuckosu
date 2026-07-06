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
    counts = _stage_counts(state)
    table.add_row("已通过测试", str(counts.get("passed", 0) + counts.get("completed", 0)))
    table.add_row(
        "运行中测试",
        str(counts.get("running", 0) + counts.get("checking", 0)),
    )
    table.add_row("警告测试", str(counts.get("warning", 0)))
    table.add_row(
        "失败测试",
        str(counts.get("failed", 0) + counts.get("interrupted", 0)),
    )
    latest_passed = _status_names(state, "passed_tests")
    latest_failed = _status_names(state, "failed_tests")
    latest_warning = _status_names(state, "warning_tests")
    if latest_passed:
        table.add_row("最近通过", _compact_names(latest_passed))
    if latest_warning:
        table.add_row("最近警告", _compact_names(latest_warning))
    if latest_failed:
        table.add_row("失败项目", _compact_names(latest_failed))
    for name, value in sorted(state.category_scores.items()):
        table.add_row(name, f"{value:.6f}")
    return Panel(table, title="当前参数学习及评分")


def _fmt(value: float | None) -> str:
    return "无" if value is None else f"{value:.6f}"


def _stage_counts(state: TrainingDashboardState) -> dict[str, int]:
    value = state.current_parameter_status.get("stage_counts")
    if isinstance(value, dict):
        counts: dict[str, int] = {}
        for key, item in value.items():
            try:
                counts[str(key)] = int(item)
            except (TypeError, ValueError):
                continue
        return counts
    counts: dict[str, int] = {}
    for stage in state.pipeline_stages.values():
        counts[stage.status] = counts.get(stage.status, 0) + 1
    return counts


def _status_names(state: TrainingDashboardState, key: str) -> list[str]:
    value = state.current_parameter_status.get(key)
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item]


def _compact_names(names: list[str], *, limit: int = 3) -> str:
    selected = names[-limit:]
    text = "、".join(selected)
    remaining = len(names) - len(selected)
    if remaining > 0:
        return f"{text} 等 {len(names)} 项"
    return text
