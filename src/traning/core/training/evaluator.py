"""把真实阶段 runner 接入持续参数搜索的具体 TrialEvaluator。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from traning.state import DataQualityReport
from traning.core.data.pipeline import require_quality

from .optimization import ParameterVector, TrialAcceptance, TrialObservation
from .orchestration import (
    ExecutionStatus,
    OrchestrationResult,
    StageResult,
    StageRunner,
    TrainingOrchestrator,
    TrainingStage,
)


TrialRunnerFactory = Callable[[ParameterVector, int], StageRunner]
"""为每个 proposal 构造全新阶段 runner 的工厂类型。"""

TrialObjectiveFunction = Callable[[ParameterVector, int, OrchestrationResult], float]
"""从完整编排结果提取优化目标的纯函数类型。"""


@dataclass(slots=True)
class _LazyTrialRunner:
    """把昂贵 runner 构造延迟到数据质量门实际通过之后。"""

    factory: TrialRunnerFactory
    parameters: ParameterVector
    trial_index: int
    _runner: StageRunner | None = field(default=None, init=False)

    def run(self, stage: TrainingStage) -> StageResult:
        """首次阶段调用时构造一次 runner，之后保持同一 trial 状态。"""

        if self._runner is None:
            runner = self.factory(self.parameters, self.trial_index)
            if not callable(getattr(runner, "run", None)):
                raise TypeError("runner_factory 必须返回实现 run 的 StageRunner")
            self._runner = runner
        return self._runner.run(stage)


@dataclass(frozen=True, slots=True)
class OrchestratedTrialEvaluator:
    """以质量门和真实阶段 runner 求值每个搜索 proposal。

    阶段返回 ``FAILED`` 会变成未通过的 ``TrialObservation``，交还搜索控制器
    选择下一组参数；只有异常才表示执行边界损坏并终止进程。
    """

    quality_report: DataQualityReport
    runner_factory: TrialRunnerFactory
    objective_function: TrialObjectiveFunction

    def __post_init__(self) -> None:
        if not isinstance(self.quality_report, DataQualityReport):
            raise TypeError("quality_report 必须是 DataQualityReport")
        if not callable(self.runner_factory):
            raise TypeError("runner_factory 必须可调用")
        if not callable(self.objective_function):
            raise TypeError("objective_function 必须可调用")

    def evaluate(
        self,
        parameters: ParameterVector,
        trial_index: int,
    ) -> TrialObservation:
        """完整执行一个 proposal，并把失败门禁保留为可继续搜索的观测。"""

        if not isinstance(parameters, ParameterVector):
            raise TypeError("parameters 必须是 ParameterVector")
        if isinstance(trial_index, bool) or not isinstance(trial_index, int):
            raise TypeError("trial_index 必须是整数")
        if trial_index < 0:
            raise ValueError("trial_index 不得为负数")
        # 数据质量对整次搜索固定，无法靠换超参数修复；必须在 runner 构造前阻断，
        # 否则 max_trials=None 会对同一份坏数据无限提出无意义 proposal。
        require_quality(self.quality_report)
        lazy_runner = _LazyTrialRunner(
            self.runner_factory,
            parameters,
            trial_index,
        )
        result = TrainingOrchestrator(lazy_runner).run(self.quality_report)
        acceptance = acceptance_from_orchestration(result)
        objective = self.objective_function(parameters, trial_index, result)
        return TrialObservation(
            trial_index=trial_index,
            parameters=parameters,
            objective=objective,
            acceptance=acceptance,
        )


def acceptance_from_orchestration(
    result: OrchestrationResult,
) -> TrialAcceptance:
    """把阶段失败映成 gate，而不把普通未通过误当成程序异常。"""

    if result.acceptance is not None:
        return result.acceptance
    stage_status = {
        stage_result.stage: stage_result.status is ExecutionStatus.PASSED
        for stage_result in result.stage_results
    }
    return TrialAcceptance(
        data=result.quality_passed,
        perception=stage_status.get(TrainingStage.PERCEPTION, False),
        tracking=stage_status.get(TrainingStage.TRACKING, False),
        belief=stage_status.get(TrainingStage.BELIEF, False),
        outcome=stage_status.get(TrainingStage.OUTCOME, False),
        decision=stage_status.get(TrainingStage.DECISION, False),
        golden=False,
    )


__all__ = (
    "OrchestratedTrialEvaluator",
    "TrialObjectiveFunction",
    "TrialRunnerFactory",
    "acceptance_from_orchestration",
)
