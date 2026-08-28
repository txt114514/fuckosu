"""以 canonical 数据质量门为入口的训练阶段编排。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from traning.contracts import DataQualityReport
from traning.data.pipeline import QualityGateBlockedError, require_quality
from traning.training.optimization import TrialAcceptance


class TrainingStage(str, Enum):
    """训练流水线的固定阶段。"""

    PERCEPTION = "perception"
    TRACKING = "tracking"
    BELIEF = "belief"
    OUTCOME = "outcome"
    DECISION = "decision"
    EVALUATION = "evaluation"


STAGE_REGISTRY: tuple[TrainingStage, ...] = (
    TrainingStage.PERCEPTION,
    TrainingStage.TRACKING,
    TrainingStage.BELIEF,
    TrainingStage.OUTCOME,
    TrainingStage.DECISION,
    TrainingStage.EVALUATION,
)


class ExecutionStatus(str, Enum):
    """阶段与整体编排共享的终态。"""

    PASSED = "passed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class StageResult:
    """单个训练阶段的明确结果。"""

    stage: TrainingStage
    status: ExecutionStatus
    message: str = ""
    acceptance: TrialAcceptance | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.stage, TrainingStage):
            raise TypeError("stage 必须是 TrainingStage")
        if not isinstance(self.status, ExecutionStatus):
            raise TypeError("status 必须是 ExecutionStatus")
        if not isinstance(self.message, str) or self.message != self.message.strip():
            raise ValueError("message 必须是无首尾空格的字符串")
        if self.acceptance is not None and not isinstance(
            self.acceptance, TrialAcceptance
        ):
            raise TypeError("acceptance 必须是 TrialAcceptance 或 None")
        # 只有 evaluation 成功结束后才有资格发布 canonical 全门禁结果。
        expects_acceptance = (
            self.stage is TrainingStage.EVALUATION
            and self.status is ExecutionStatus.PASSED
        )
        if expects_acceptance != (self.acceptance is not None):
            raise ValueError("只有 PASSED evaluation stage 必须携带 TrialAcceptance")


class StageRunner(Protocol):
    """各领域训练实现必须满足的最小运行协议。"""

    def run(self, stage: TrainingStage) -> StageResult:
        """运行一个阶段并返回对应 typed 结果。"""


@dataclass(frozen=True, slots=True)
class OrchestrationResult:
    """质量门、阶段执行与最终 acceptance 的整体审计结果。"""

    status: ExecutionStatus
    quality_passed: bool
    stage_results: tuple[StageResult, ...]
    acceptance: TrialAcceptance | None

    def __post_init__(self) -> None:
        if not isinstance(self.status, ExecutionStatus):
            raise TypeError("status 必须是 ExecutionStatus")
        if not isinstance(self.quality_passed, bool):
            raise TypeError("quality_passed 必须是 bool")
        if self.acceptance is not None and not isinstance(
            self.acceptance, TrialAcceptance
        ):
            raise TypeError("acceptance 必须是 TrialAcceptance 或 None")
        if not isinstance(self.stage_results, tuple) or any(
            not isinstance(result, StageResult) for result in self.stage_results
        ):
            raise TypeError("stage_results 必须是 StageResult tuple")
        evaluation_acceptances = tuple(
            result.acceptance
            for result in self.stage_results
            if result.acceptance is not None
        )
        if len(evaluation_acceptances) > 1:
            raise ValueError("编排结果不得包含多个 TrialAcceptance")
        if evaluation_acceptances != (
            () if self.acceptance is None else (self.acceptance,)
        ):
            raise ValueError("最终 acceptance 必须与 evaluation stage 使用同一对象")
        fully_passed = (
            self.quality_passed
            and tuple(result.stage for result in self.stage_results) == STAGE_REGISTRY
            and all(
                result.status is ExecutionStatus.PASSED for result in self.stage_results
            )
            and self.acceptance is not None
            and self.acceptance.passed
        )
        if (self.status is ExecutionStatus.PASSED) != fully_passed:
            raise ValueError("PASSED 只允许表示完整阶段及最终 acceptance 全部通过")


@dataclass(frozen=True, slots=True)
class TrainingOrchestrator:
    """按固定 registry 顺序执行训练并在首个失败处停止。"""

    runner: StageRunner

    def run(
        self,
        quality_report: DataQualityReport,
    ) -> OrchestrationResult:
        """先消费唯一质量门，再从 evaluation 取得 canonical acceptance。"""

        if not isinstance(quality_report, DataQualityReport):
            raise TypeError("quality_report 必须是 DataQualityReport")
        try:
            require_quality(quality_report)
        except QualityGateBlockedError:
            return OrchestrationResult(
                status=ExecutionStatus.FAILED,
                quality_passed=False,
                stage_results=(),
                acceptance=None,
            )

        results: list[StageResult] = []
        for stage in STAGE_REGISTRY:
            result = self.runner.run(stage)
            if not isinstance(result, StageResult):
                raise TypeError("runner 必须返回 StageResult")
            if result.stage is not stage:
                raise ValueError("runner 返回的 stage 与请求阶段不一致")
            results.append(result)
            if result.status is ExecutionStatus.FAILED:
                return OrchestrationResult(
                    status=ExecutionStatus.FAILED,
                    quality_passed=True,
                    stage_results=tuple(results),
                    acceptance=None,
                )

        acceptance = results[-1].acceptance
        if acceptance is None:  # pragma: no cover - StageResult 已保证
            raise RuntimeError("evaluation stage 缺少 TrialAcceptance")
        if not acceptance.data:
            raise ValueError("quality report 已通过，但 TrialAcceptance.data 为 False")
        final_status = (
            ExecutionStatus.PASSED if acceptance.passed else ExecutionStatus.FAILED
        )
        return OrchestrationResult(
            status=final_status,
            quality_passed=True,
            stage_results=tuple(results),
            acceptance=acceptance,
        )


__all__ = (
    "STAGE_REGISTRY",
    "ExecutionStatus",
    "OrchestrationResult",
    "StageResult",
    "StageRunner",
    "TrainingOrchestrator",
    "TrainingStage",
)
