from __future__ import annotations

from visualization.conf import DashboardSettings
from visualization.lib.reporter import DashboardReporter


class PlainDashboardRenderer:
    def __init__(self, reporter: DashboardReporter, *, settings: DashboardSettings) -> None:
        self.reporter = reporter
        self.settings = settings

    def start(self) -> None:
        state = self.reporter.snapshot()
        print(
            f"[训练控制台] 运行：{state.run_id}，状态：{state.status}",
            flush=True,
        )

    def stop(self) -> None:
        state = self.reporter.snapshot()
        metrics = state.metrics
        resources = state.resources
        print(
            "[训练控制台] "
            f"阶段：{state.phase}；评分：{metrics.score}; "
            f"最高：{metrics.run_global_best_score}; "
            f"等级：{state.current_grade or '未评级'}；"
            f"晋升：{state.promotion_status or '未评级'}；"
            f"显存峰值：{resources.gpu_peak_reserved_gb}",
            flush=True,
        )
