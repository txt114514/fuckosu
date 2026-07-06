from collections.abc import Mapping, Sequence

from rich.panel import Panel
from rich.table import Table

from visualization.lib.models import TrainingDashboardState


_MAX_PARAMETER_ROWS = 10
_PRIORITY_PARAMETER_KEYS = (
    "parameter_group_id",
    "device",
    "split",
    "training.spatial_max_steps",
    "training.temporal_max_steps",
    "training.spatial_learning_rate",
    "training.temporal_learning_rate",
    "candidate_cache.cache_max_frames",
    "candidate_cache.max_candidates",
    "temporal.sequence_length",
    "temporal.candidate_slots",
    "evaluation.min_click_interval_ms",
)


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
    flat_parameters = _flatten_parameters(state.current_parameters)
    table.add_row("当前参数数量", str(len(flat_parameters)))
    for key, value in _selected_parameter_rows(flat_parameters):
        table.add_row(f"参数 {key}", _fmt_parameter_value(value))
    return Panel(table, title="最佳参数记录")


def _fmt(value: float | None) -> str:
    return "无" if value is None else f"{value:.6f}"


def _selected_parameter_rows(
    flat_parameters: Mapping[str, object],
) -> list[tuple[str, object]]:
    selected: list[tuple[str, object]] = []
    seen: set[str] = set()
    for key in _PRIORITY_PARAMETER_KEYS:
        if key in flat_parameters:
            selected.append((key, flat_parameters[key]))
            seen.add(key)
    for key in sorted(flat_parameters):
        if len(selected) >= _MAX_PARAMETER_ROWS:
            break
        if key not in seen:
            selected.append((key, flat_parameters[key]))
            seen.add(key)
    return selected[:_MAX_PARAMETER_ROWS]


def _flatten_parameters(
    parameters: Mapping[str, object],
    *,
    prefix: str = "",
) -> dict[str, object]:
    flat: dict[str, object] = {}
    for key, value in parameters.items():
        text_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flat.update(_flatten_parameters(value, prefix=text_key))
        else:
            flat[text_key] = value
    return flat


def _fmt_parameter_value(value: object) -> str:
    if value is None:
        return "无"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (str, int, bool)):
        return str(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return ", ".join(str(item) for item in value[:5])
    return str(value)
