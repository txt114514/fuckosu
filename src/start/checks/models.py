"""适配共享检查报告与训练数据输入报告的组合结果。"""

from __future__ import annotations

from typing import Any

from package.checks import CheckStatus, StartupCheckReport, StartupCheckResult


class TrainingStartupCheckReport:
    """把可阻断的统一检查报告与只读数据输入统计绑定返回。"""

    def __init__(self, report: StartupCheckReport, data_input: Any):
        self.report = report
        self.data_input = data_input

    @property
    def ok(self) -> bool:
        return self.report.ok

    def raise_for_errors(self) -> None:
        self.report.raise_for_errors()

    def as_dict(self) -> dict[str, Any]:
        return {
            "report": self.report.as_dict(),
            "data_input": _data_input_report_dict(self.data_input),
        }


def _data_input_report_dict(report: Any) -> dict[str, Any]:
    return {
        "split": report.split,
        "segment_count": report.segment_count,
        "frame_count_estimate": report.frame_count_estimate,
        "item_counts": report.item_counts,
        "category_counts": report.category_counts,
        "dimension_counts": report.dimension_counts,
        "issue_count": report.issue_count,
        "issues": report.issues,
        "ok": report.ok,
    }


__all__ = [
    "CheckStatus",
    "StartupCheckReport",
    "StartupCheckResult",
    "TrainingStartupCheckReport",
]
