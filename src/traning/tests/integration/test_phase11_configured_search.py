"""正式 V2 配置与持续参数搜索的端到端验收。"""

from __future__ import annotations

from pathlib import Path

import pytest

from traning.app import initial_parameter_vector, run_configured_search
from traning.config import OptimizationConfig, V2Config
from traning.contracts import (
    DataQualityIssue,
    DataQualityReport,
    DataQualitySeverity,
)
from traning.data import QualityGateBlockedError
from traning.telemetry import StateStore, TelemetryReporter
from traning.training import (
    ParameterVector,
    OrchestratedTrialEvaluator,
    SearchExhaustedError,
    STAGE_REGISTRY,
    ExecutionStatus,
    StageResult,
    TrainingStage,
    TrialAcceptance,
    TrialObservation,
)


class _PassesOnThirdTrial:
    """记录实际执行次数，并仅让第三组参数全门禁通过。"""

    def __init__(self) -> None:
        self.parameters: list[ParameterVector] = []

    def evaluate(
        self,
        parameters: ParameterVector,
        trial_index: int,
    ) -> TrialObservation:
        """返回与 proposal 身份一致的确定性 trial 观测。"""

        self.parameters.append(parameters)
        passed = trial_index == 2
        return TrialObservation(
            trial_index=trial_index,
            parameters=parameters,
            objective=float(trial_index),
            acceptance=TrialAcceptance(
                passed,
                passed,
                passed,
                passed,
                passed,
                passed,
                passed,
            ),
        )


class _ConcreteStageRunner:
    """用 typed StageResult 表示普通训练失败，而不是抛异常结束搜索。"""

    def __init__(self, trial_index: int) -> None:
        self.trial_index = trial_index
        self.calls: list[TrainingStage] = []

    def run(self, stage: TrainingStage) -> StageResult:
        """第一轮早停、第二轮 gate 未过、第三轮完整通过。"""

        self.calls.append(stage)
        if self.trial_index == 0 and stage is TrainingStage.PERCEPTION:
            return StageResult(stage, ExecutionStatus.FAILED, "perception gate failed")
        if stage is not TrainingStage.EVALUATION:
            return StageResult(stage, ExecutionStatus.PASSED)
        passed = self.trial_index == 2
        acceptance = TrialAcceptance(
            True,
            True,
            True,
            True,
            True,
            True,
            passed,
        )
        return StageResult(
            stage,
            ExecutionStatus.PASSED,
            acceptance=acceptance,
        )


def test_default_config_executes_third_trial_instead_of_stopping_at_two() -> None:
    """直接固定用户旧运行的提前停止回归。"""

    evaluator = _PassesOnThirdTrial()
    result = run_configured_search(V2Config(), evaluator)

    assert result.trial_index == 2
    assert result.acceptance.passed
    assert len(evaluator.parameters) == 3
    assert len(set(evaluator.parameters)) == 3
    assert all(item.score_threshold >= 0.0 for item in evaluator.parameters)


def test_concrete_orchestrated_evaluator_continues_after_stage_failures() -> None:
    """真实阶段 FAILED 与最终 gate 未过都应继续提案，直到第三轮全通过。"""

    runners: list[_ConcreteStageRunner] = []

    def build_runner(
        _parameters: ParameterVector,
        trial_index: int,
    ) -> _ConcreteStageRunner:
        """记录每个 proposal 获得的独立有状态 runner。"""

        runner = _ConcreteStageRunner(trial_index)
        runners.append(runner)
        return runner

    evaluator = OrchestratedTrialEvaluator(
        quality_report=DataQualityReport(issues=()),
        runner_factory=build_runner,
        objective_function=lambda _parameters, trial_index, _result: float(trial_index),
    )
    result = run_configured_search(V2Config(), evaluator)

    assert result.trial_index == 2
    assert runners[0].calls == [TrainingStage.PERCEPTION]
    assert tuple(runners[1].calls) == STAGE_REGISTRY
    assert tuple(runners[2].calls) == STAGE_REGISTRY


def test_blocking_data_quality_stops_search_before_constructing_runner(
    tmp_path: Path,
) -> None:
    """固定坏数据不是参数失败；无预算搜索也必须首轮立即阻断。"""

    runner_factory_calls = 0

    def build_runner(
        _parameters: ParameterVector,
        _trial_index: int,
    ) -> _ConcreteStageRunner:
        """若质量门正确前置，本工厂永远不应被调用。"""

        nonlocal runner_factory_calls
        runner_factory_calls += 1
        return _ConcreteStageRunner(0)

    evaluator = OrchestratedTrialEvaluator(
        quality_report=DataQualityReport(
            issues=(
                DataQualityIssue(
                    code="missing-source",
                    severity=DataQualitySeverity.ERROR,
                    blocks_training=True,
                    sample_id="broken-sample",
                    message="缺少训练源文件",
                ),
            )
        ),
        runner_factory=build_runner,
        objective_function=lambda _parameters, _trial_index, _result: 0.0,
    )

    store = StateStore(tmp_path / "blocked-telemetry")
    reporter = TelemetryReporter("blocked-search", store)
    with pytest.raises(QualityGateBlockedError):
        run_configured_search(V2Config(), evaluator, reporter=reporter)
    assert runner_factory_calls == 0
    assert tuple(event.event_type for event in store.history().events) == (
        "search.failed",
    )


def test_explicit_trial_budget_is_reported_as_exhausted_not_success() -> None:
    """用户显式预算为 2 时必须产生 typed EXHAUSTED。"""

    evaluator = _PassesOnThirdTrial()
    config = V2Config(optimization=OptimizationConfig(max_trials=2))

    with pytest.raises(SearchExhaustedError) as captured:
        run_configured_search(config, evaluator)
    assert captured.value.decision.trial_count == 2
    assert len(evaluator.parameters) == 2


def test_initial_vector_is_collectively_derived_from_domain_config() -> None:
    """初始 proposal 不再由零散 job 字典逐字段改写。"""

    config = V2Config()
    vector = initial_parameter_vector(config)

    assert vector.learning_rate == config.training.learning_rate
    assert vector.score_threshold == config.perception.score_threshold
    assert vector.max_candidates == config.perception.max_candidates
    assert vector.risk_lambda == config.decision.risk_lambda
    assert vector.wait_cost == config.decision.wait_cost
    assert vector.min_confidence == config.decision.min_confidence


def test_configured_search_publishes_every_trial_and_explicit_terminal(
    tmp_path: Path,
) -> None:
    """Dashboard 看到真实 trial 历史和 PASSED，而非停留在运行中。"""

    evaluator = _PassesOnThirdTrial()
    store = StateStore(tmp_path / "telemetry")
    reporter = TelemetryReporter("search-run", store)

    run_configured_search(V2Config(), evaluator, reporter=reporter)

    event_types = tuple(item.event_type for item in store.history().events)
    assert event_types == (
        "search.trial.completed",
        "search.trial.completed",
        "search.trial.completed",
        "search.passed",
    )
