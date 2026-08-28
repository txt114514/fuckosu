"""声明启动层对共享检查结果与 canonical 数据质量报告的组合类型。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from package.checks import CheckStatus, StartupCheckReport, StartupCheckResult
from traning.contracts import DataQualityReport


@dataclass(frozen=True, slots=True)
class TrainingStartupCheckReport:
    """把 V2 启动检查与同一次 canonical 数据质量检查绑定。"""

    report: StartupCheckReport
    data_quality: DataQualityReport | None

    @property
    def ok(self) -> bool:
        """只使用共享检查报告的阻断结论。"""

        return self.report.ok

    @property
    def results(self) -> tuple[StartupCheckResult, ...]:
        """暴露只读检查项，便于 UI 在不识别内部 DTO 时展示。"""

        return self.report.results

    @property
    def warnings(self) -> tuple[StartupCheckResult, ...]:
        """返回可见但不阻断的检查项。"""

        return self.report.warnings

    def raise_for_errors(self) -> None:
        """保留共享报告的统一失败语义。"""

        self.report.raise_for_errors()

    def as_dict(self) -> dict[str, Any]:
        """序列化报告；缺质量报告不会伪装成通过。"""

        return {
            "report": self.report.as_dict(),
            "data_quality": _data_quality_report_dict(self.data_quality),
        }


def _data_quality_report_dict(
    report: DataQualityReport | None,
) -> dict[str, Any] | None:
    """把 canonical 质量报告转换为稳定的 JSON 数据。"""

    if report is None:
        return None
    return {
        "ok": report.ok,
        "issues": tuple(
            {
                "code": issue.code,
                "severity": issue.severity.value,
                "blocks_training": issue.blocks_training,
                "sample_id": issue.sample_id,
                "message": issue.message,
                "details": dict(issue.details),
            }
            for issue in report.issues
        ),
    }


__all__ = (
    "CheckStatus",
    "StartupCheckReport",
    "StartupCheckResult",
    "TrainingStartupCheckReport",
)
