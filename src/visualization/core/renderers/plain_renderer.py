from __future__ import annotations

from visualization.conf import DashboardSettings
from visualization.conf.messages import (
    display_grade,
    display_status,
    display_text,
    display_view_kind,
)
from visualization.core.view_router import dashboard_view_kind
from visualization.lib.reporter import DashboardReporter


class PlainDashboardRenderer:
    def __init__(
        self,
        reporter: DashboardReporter,
        *,
        settings: DashboardSettings,
    ) -> None:
        self.reporter = reporter
        self.settings = settings

    def start(self) -> None:
        state = self.reporter.snapshot()
        print(
            f"[训练控制台:{display_view_kind(dashboard_view_kind(state))}] "
            f"运行：{state.run_id}，状态：{display_status(state.status)}",
            flush=True,
        )

    def stop(self) -> None:
        state = self.reporter.snapshot()
        metrics = state.metrics
        resources = state.resources
        counts = _stage_counts(state.current_parameter_status)
        print(
            f"[训练控制台:{display_view_kind(dashboard_view_kind(state))}] "
            f"阶段：{display_text(state.phase)}；评分：{_fmt(metrics.score)}；"
            f"最高：{_fmt(metrics.run_global_best_score)}；"
            f"等级：{display_grade(state.current_grade)}；"
            f"晋升：{display_text(state.promotion_status or '未评级')}；"
            f"通过：{counts.get('passed', 0) + counts.get('completed', 0)}；"
            f"警告：{counts.get('warning', 0)}；"
            f"失败：{counts.get('failed', 0) + counts.get('interrupted', 0)}；"
            f"显存峰值：{_fmt(resources.gpu_peak_reserved_gb)}",
            flush=True,
        )


def _stage_counts(status: dict[str, object]) -> dict[str, int]:
    value = status.get("stage_counts")
    if not isinstance(value, dict):
        return {}
    counts: dict[str, int] = {}
    for key, item in value.items():
        try:
            counts[str(key)] = int(item)
        except (TypeError, ValueError):
            continue
    return counts


def _fmt(value: object) -> str:
    return "无" if value is None else str(value)
