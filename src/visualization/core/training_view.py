from __future__ import annotations

from collections.abc import Mapping, Sequence

from rich.console import Group
from rich.panel import Panel
from rich.table import Table

from visualization.conf.messages import PIPELINE_PHASE_NAMES, STATUS_NAMES
from visualization.core.panels.best_parameters_panel import render_best_parameters_panel
from visualization.core.panels.current_learning_panel import render_current_learning_panel
from visualization.core.panels.events_panel import render_events_panel
from visualization.core.panels.overall_progress_panel import render_overall_progress_panel
from visualization.core.panels.resources_panel import render_resources_panel
from visualization.lib.models import TrainingDashboardState


_MAX_PARAMETER_ROWS = 14
TRAINING_PAGES: tuple[str, ...] = (
    "overview",
    "parameters",
    "tests",
    "scores",
    "resources",
    "events",
)


def render_training_view(
    state: TrainingDashboardState,
    *,
    compact: bool = False,
    page: str = "overview",
) -> Group:
    if compact:
        selected = page if page in TRAINING_PAGES else "overview"
        return Group(
            render_current_trial_panel(
                state,
                compact=True,
                page=selected,
                pages=TRAINING_PAGES,
            ),
            *_compact_training_panels(state, selected),
        )
    return Group(
        render_current_trial_panel(state),
        render_parameters_panel(state),
        render_tests_panel(state),
        render_best_parameters_panel(state),
        render_current_learning_panel(state),
        render_overall_progress_panel(state),
        render_resources_panel(state),
        render_events_panel(state),
    )


def render_current_trial_panel(
    state: TrainingDashboardState,
    *,
    compact: bool = False,
    page: str = "overview",
    pages: tuple[str, ...] = TRAINING_PAGES,
) -> Panel:
    runtime = state.current_parameter_status
    table = Table.grid(expand=True)
    table.add_column("字段")
    table.add_column("值")
    table.add_row("试验", state.current_trial_id or "无活动试验")
    table.add_row("流程阶段", _phase_name(state.pipeline_phase))
    table.add_row("当前阶段", str(runtime.get("curriculum_stage") or state.phase))
    table.add_row("状态", _status_name(runtime.get("trial_status")))
    table.add_row(
        "步数",
        _progress(runtime.get("stage_step"), runtime.get("stage_budget")),
    )
    table.add_row("当前得分", _fmt_float(runtime.get("latest_score")))
    table.add_row("最高得分", _fmt_float(runtime.get("best_score")))
    if not compact:
        table.add_row(
            "连续通过",
            _progress(
                runtime.get("consecutive_passes"),
                runtime.get("required_consecutive_passes"),
            ),
        )
        table.add_row("检查点", str(runtime.get("checkpoint_path") or "无"))
        if runtime.get("prune_reason"):
            table.add_row("淘汰原因", str(runtime["prune_reason"]))
    else:
        table.add_row("页面", _page_hint(page, pages))
    return Panel(table, title="当前试验")


def render_parameters_panel(
    state: TrainingDashboardState,
    *,
    max_rows: int = _MAX_PARAMETER_ROWS,
) -> Panel:
    table = Table.grid(expand=True)
    table.add_column("参数")
    table.add_column("值")
    if not state.current_trial_id:
        table.add_row("无活动试验", "无")
    else:
        flat = _flatten_mapping(state.current_parameters)
        if not flat:
            table.add_row("参数", "无")
        for key, value in list(flat.items())[:max_rows]:
            table.add_row(key, _fmt_value(value))
    return Panel(table, title="参数")


def render_tests_panel(
    state: TrainingDashboardState,
    *,
    max_rows: int | None = None,
) -> Panel:
    runtime = state.current_parameter_status
    statuses = _mapping(runtime.get("test_statuses"))
    scores = _mapping(runtime.get("test_scores"))
    thresholds = _mapping(runtime.get("test_thresholds"))
    table = Table.grid(expand=True)
    table.add_column("测试")
    table.add_column("状态")
    table.add_column("得分")
    table.add_column("阈值")
    names = tuple(dict.fromkeys((*statuses.keys(), *scores.keys(), *thresholds.keys())))
    if not names:
        table.add_row("测试状态", "等待中", "无", "无")
    rows = names if max_rows is None else names[:max_rows]
    for name in rows:
        status = str(statuses.get(name, "pending"))
        table.add_row(
            name,
            STATUS_NAMES.get(status, status),
            _fmt_float(scores.get(name)),
            _fmt_float(thresholds.get(name)),
        )
    if max_rows is not None and len(names) > max_rows:
        table.add_row(f"其余 {len(names) - max_rows} 项", "切到完整视图", "无", "无")
    return Panel(table, title="测试")


def _compact_training_panels(
    state: TrainingDashboardState,
    page: str,
) -> tuple[Panel, ...]:
    if page == "parameters":
        return (render_parameters_panel(state, max_rows=12),)
    if page == "tests":
        return (render_tests_panel(state, max_rows=14),)
    if page == "scores":
        return (render_score_summary_panel(state),)
    if page == "resources":
        return (
            render_overall_progress_panel(state),
            render_resources_panel(state, compact=True),
        )
    if page == "events":
        return (render_events_panel(state, limit=10),)
    return (
        render_score_summary_panel(state),
        render_tests_panel(state, max_rows=8),
    )


def render_score_summary_panel(state: TrainingDashboardState) -> Panel:
    runtime = state.current_parameter_status
    table = Table.grid(expand=True)
    table.add_column("指标")
    table.add_column("值")
    table.add_row("当前 loss", _fmt_float(state.metrics.loss))
    table.add_row("当前评分", _fmt_float(state.metrics.score))
    table.add_row("参数最高", _fmt_float(state.metrics.parameter_best_score))
    table.add_row("本轮最高", _fmt_float(state.metrics.run_global_best_score))
    table.add_row("当前等级", state.current_grade or "未评级")
    table.add_row("晋升状态", state.promotion_status or "未评级")
    table.add_row(
        "连续通过",
        _progress(
            runtime.get("consecutive_passes"),
            runtime.get("required_consecutive_passes"),
        ),
    )
    if state.best_parameters.trial_id:
        table.add_row("最佳 Trial", state.best_parameters.trial_id)
    return Panel(table, title="评分摘要")


def _page_hint(page: str, pages: tuple[str, ...]) -> str:
    labels = {
        "overview": "概览",
        "parameters": "参数",
        "tests": "测试",
        "scores": "评分",
        "resources": "资源",
        "events": "事件",
    }
    index = pages.index(page) + 1 if page in pages else 1
    return (
        f"{index}/{len(pages)} {labels.get(page, page)}；"
        "1-6切页，Tab/空格下一页，b上一页，f完整视图，Ctrl+C中断"
    )


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _flatten_mapping(
    value: Mapping[str, object],
    *,
    prefix: str = "",
) -> dict[str, object]:
    flat: dict[str, object] = {}
    for key in sorted(value):
        item = value[key]
        text_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, Mapping):
            flat.update(_flatten_mapping(item, prefix=text_key))
        else:
            flat[text_key] = item
    return flat


def _progress(value: object, total: object) -> str:
    if value is None and total is None:
        return "N/A"
    if total in (None, 0):
        return str(value if value is not None else "N/A")
    return f"{value if value is not None else 0} / {total}"


def _fmt_float(value: object) -> str:
    if value is None:
        return "无"
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def _fmt_value(value: object) -> str:
    if value is None:
        return "无"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (str, int, bool)):
        return str(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return ", ".join(str(item) for item in value[:5])
    return str(value)


def _phase_name(value: object) -> str:
    text = str(value) if value is not None else ""
    return PIPELINE_PHASE_NAMES.get(text, text or "无")


def _status_name(value: object) -> str:
    text = str(value) if value is not None else ""
    return STATUS_NAMES.get(text, text or "无")
