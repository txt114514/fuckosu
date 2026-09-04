"""Phase 9 确定性参数搜索、接受门禁与终止状态验收。"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from traning.core.training.optimization import (
    PARAMETER_REGISTRY,
    DeterministicSearchController,
    ParameterRegistry,
    ParameterSpec,
    ParameterVector,
    SearchExhaustedError,
    SearchStatus,
    TrialAcceptance,
    TrialObservation,
    run_search,
)


def _initial() -> ParameterVector:
    return ParameterVector(
        learning_rate=0.001,
        score_threshold=0.2,
        max_candidates=32,
        risk_lambda=0.1,
        wait_cost=0.0,
        min_confidence=0.1,
    )


def _acceptance(*, passed: bool) -> TrialAcceptance:
    return TrialAcceptance(
        data=True,
        perception=True,
        tracking=True,
        belief=True,
        outcome=True,
        decision=True,
        golden=passed,
    )


def _observation(
    trial_index: int,
    parameters: ParameterVector,
    *,
    objective: float = 1.0,
    passed: bool = False,
) -> TrialObservation:
    return TrialObservation(
        trial_index=trial_index,
        parameters=parameters,
        objective=objective,
        acceptance=_acceptance(passed=passed),
    )


def test_parameter_registry_clamps_legacy_negative_threshold() -> None:
    """注册表必须把旧负阈值钳制到合法搜索边界。"""

    threshold = next(
        spec for spec in PARAMETER_REGISTRY.specs if spec.name == "score_threshold"
    )

    assert threshold.quantize(-0.01) == 0.0
    assert threshold.quantize(1.01) == 1.0


def test_controller_continues_after_high_objective_when_a_gate_fails() -> None:
    """任一门禁失败时，即使目标分很高也必须继续搜索。"""

    controller = DeterministicSearchController(seed=17, max_trials=None)
    initial = _initial()
    history: tuple[TrialObservation, ...] = ()

    for trial_index in range(3):
        decision = controller.decide(initial, history)
        assert decision.status is SearchStatus.RUNNING
        assert decision.proposal is not None
        history = (
            *history,
            _observation(
                trial_index,
                decision.proposal,
                objective=1_000_000.0,
                passed=False,
            ),
        )

    next_decision = controller.decide(initial, history)

    assert next_decision.status is SearchStatus.RUNNING
    assert next_decision.proposal not in tuple(item.parameters for item in history)


def test_controller_passes_only_when_every_acceptance_gate_passes() -> None:
    """只有全部接受门禁通过时控制器才能进入 PASSED。"""

    initial = _initial()
    controller = DeterministicSearchController(seed=3, max_trials=1)
    first = controller.decide(initial, ())
    assert first.proposal is not None

    result = controller.decide(
        initial,
        (_observation(0, first.proposal, objective=-5.0, passed=True),),
    )

    assert result.status is SearchStatus.PASSED
    assert result.best_observation is not None
    assert result.best_observation.objective == -5.0


def test_controller_reports_budget_exhaustion_explicitly() -> None:
    """显式 trial 预算耗尽时必须返回 EXHAUSTED 状态。"""

    initial = _initial()
    controller = DeterministicSearchController(seed=9, max_trials=2)
    history: tuple[TrialObservation, ...] = ()

    for trial_index in range(2):
        decision = controller.decide(initial, history)
        assert decision.proposal is not None
        history = (*history, _observation(trial_index, decision.proposal))

    exhausted = controller.decide(initial, history)

    assert exhausted.status is SearchStatus.EXHAUSTED
    assert exhausted.proposal is None
    assert exhausted.trial_count == 2


def test_same_seed_and_history_produce_same_proposal() -> None:
    """相同随机种子与历史必须生成完全相同的提案。"""

    initial = _initial()
    first_controller = DeterministicSearchController(seed=42, max_trials=None)
    first = first_controller.decide(initial, ())
    assert first.proposal is not None
    history = (_observation(0, first.proposal),)

    left = first_controller.decide(initial, history)
    right = DeterministicSearchController(seed=42, max_trials=None).decide(
        initial, history
    )

    assert left == right


def test_every_published_proposal_is_in_range_and_quantized() -> None:
    """每个公开提案都必须位于范围内并符合量化步长。"""

    initial = _initial()
    controller = DeterministicSearchController(seed=21, max_trials=None)
    history: tuple[TrialObservation, ...] = ()

    for trial_index in range(5):
        decision = controller.decide(initial, history)
        assert decision.proposal is not None
        for spec in PARAMETER_REGISTRY.specs:
            value = getattr(decision.proposal, spec.name)
            spec.validate(value)
            assert spec.quantize(value) == value
        history = (*history, _observation(trial_index, decision.proposal))


def test_finite_quantized_space_has_a_real_exhausted_terminal_state() -> None:
    """有限量化空间遍历完后必须产生真实 EXHAUSTED 终态。"""

    initial = _initial()
    values = tuple(getattr(initial, spec.name) for spec in PARAMETER_REGISTRY.specs)
    one_point_registry = ParameterRegistry(
        tuple(
            ParameterSpec(
                name=spec.name,
                parameter_type=spec.parameter_type,
                minimum=float(value),
                maximum=float(value),
                step=spec.step,
            )
            for spec, value in zip(PARAMETER_REGISTRY.specs, values, strict=True)
        )
    )
    controller = DeterministicSearchController(
        seed=0, max_trials=None, registry=one_point_registry
    )
    first = controller.decide(initial, ())
    assert first.proposal == initial

    exhausted = controller.decide(initial, (_observation(0, initial),))

    assert one_point_registry.space_size == 1
    assert exhausted.status is SearchStatus.EXHAUSTED


class _ThirdTrialPasses:
    def __init__(self) -> None:
        self.calls = 0

    def evaluate(
        self, parameters: ParameterVector, trial_index: int
    ) -> TrialObservation:
        """在第三次 trial 返回全门禁通过的观测。"""

        self.calls += 1
        return _observation(trial_index, parameters, passed=trial_index == 2)


class _NeverPasses:
    def evaluate(
        self, parameters: ParameterVector, trial_index: int
    ) -> TrialObservation:
        """始终返回未完全通过的 trial 观测。"""

        return _observation(trial_index, parameters)


def test_run_search_keeps_going_until_full_acceptance() -> None:
    """无预算搜索必须持续到出现全门禁通过的 trial。"""

    evaluator = _ThirdTrialPasses()

    result = run_search(evaluator, _initial(), seed=5, max_trials=None)

    assert evaluator.calls == 3
    assert result.trial_index == 2
    assert result.acceptance.passed


def test_run_search_raises_typed_error_on_exhaustion() -> None:
    """搜索预算耗尽必须抛出携带终态的 typed 错误。"""

    with pytest.raises(SearchExhaustedError) as captured:
        run_search(_NeverPasses(), _initial(), seed=5, max_trials=2)

    assert captured.value.decision.status is SearchStatus.EXHAUSTED
    assert captured.value.decision.trial_count == 2


def test_optimization_contract_has_no_any_or_legacy_dependency() -> None:
    """优化契约不得依赖宽泛 Any、legacy 或错误顶层包。"""

    source_path = Path(__file__).parents[2] / "core/training" / "optimization.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }

    assert "Any" not in source
    assert not any(
        name == "traning" or name.startswith("traning.") for name in imported_names
    )
    assert not any(
        name == "training" or name.startswith("training.") for name in imported_names
    )
