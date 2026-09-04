"""只消费 canonical 质量门结果的数据 pipeline。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from traning.state import DataQualityReport, TrainingSample
from traning.core.data.quality import (
    DataQualityContext,
    DataQualityGate,
    DatasetSummary,
)


@dataclass(frozen=True, slots=True)
class DataPipelineResult:
    """通过质量门后发布给下游的 typed 数据结果。"""

    samples: tuple[TrainingSample, ...]
    summary: DatasetSummary
    quality_report: DataQualityReport

    def __post_init__(self) -> None:
        if not isinstance(self.samples, tuple) or any(
            not isinstance(sample, TrainingSample) for sample in self.samples
        ):
            raise TypeError("samples 必须是 TrainingSample 元组")
        if not isinstance(self.summary, DatasetSummary):
            raise TypeError("summary 必须是 DatasetSummary")
        if not isinstance(self.quality_report, DataQualityReport):
            raise TypeError("quality_report 必须是 DataQualityReport")
        if not self.quality_report.ok:
            raise ValueError("DataPipelineResult 只能承载通过质量门的数据")


class QualityGateBlockedError(RuntimeError):
    """数据质量门阻断训练时的明确失败。"""

    def __init__(self, report: DataQualityReport) -> None:
        if not isinstance(report, DataQualityReport):
            raise TypeError("report 必须是 DataQualityReport")
        if report.ok:
            raise ValueError("不能用通过质量门的 report 构造阻断异常")
        self.report = report
        blocking_count = sum(issue.blocks_training for issue in report.issues)
        super().__init__(f"数据质量门阻断训练：{blocking_count} 个阻断问题")


def require_quality(report: DataQualityReport) -> None:
    """仅按 canonical ``report.ok`` 决定是否阻断。"""

    if not isinstance(report, DataQualityReport):
        raise TypeError("report 必须是 DataQualityReport")
    if not report.ok:
        raise QualityGateBlockedError(report)


@dataclass(frozen=True, slots=True)
class DataPipeline:
    """确定性整理样本并执行唯一质量门的编排器。"""

    quality_gate: DataQualityGate = DataQualityGate()

    def __post_init__(self) -> None:
        if not isinstance(self.quality_gate, DataQualityGate):
            raise TypeError("quality_gate 必须是 DataQualityGate")

    def run(self, samples: Sequence[TrainingSample]) -> DataPipelineResult:
        """运行 pipeline；阻断时抛出携带完整 report 的异常。"""

        context = DataQualityContext.from_samples(samples)
        report = self.quality_gate.evaluate(context)
        require_quality(report)
        return DataPipelineResult(
            samples=context.samples,
            summary=context.summary,
            quality_report=report,
        )


def run_data_pipeline(
    samples: Sequence[TrainingSample],
    *,
    quality_gate: DataQualityGate | None = None,
) -> DataPipelineResult:
    """函数式入口；不提供跳过质量门的路径。"""

    pipeline = DataPipeline(
        quality_gate=quality_gate if quality_gate is not None else DataQualityGate()
    )
    return pipeline.run(samples)


__all__ = (
    "DataPipeline",
    "DataPipelineResult",
    "QualityGateBlockedError",
    "require_quality",
    "run_data_pipeline",
)
