"""Phase 9 训练编排、curriculum 与 ASHA 验收。"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from traning.contracts import (
    DataQualityIssue,
    DataQualityReport,
    DataQualitySeverity,
)
from traning.training.orchestration import (
    STAGE_REGISTRY,
    ExecutionStatus,
    StageResult,
    TrainingOrchestrator,
    TrainingStage,
)
from traning.training.optimization import TrialAcceptance
from traning.training.scheduling import (
    AshaAction,
    AshaRung,
    AshaScheduler,
    AshaTrial,
    CurriculumAction,
    CurriculumGate,
    CurriculumStage,
    decide_curriculum,
)


_TRAINING_ROOT = Path(__file__).resolve().parents[2] / "training"


def _quality_report(*, blocked: bool = False) -> DataQualityReport:
    issues = (
        (
            DataQualityIssue(
                code="artificial-quality",
                severity=DataQualitySeverity.ERROR,
                blocks_training=True,
                sample_id=None,
                message="人工阻断问题",
            ),
        )
        if blocked
        else ()
    )
    return DataQualityReport(issues=issues)


@dataclass
class RecordingRunner:
    """记录阶段调用并可注入失败与最终接受结果的测试 runner。"""

    fail_at: TrainingStage | None = None
    acceptance: TrialAcceptance = field(
        default_factory=lambda: TrialAcceptance(
            True, True, True, True, True, True, True
        )
    )
    calls: list[TrainingStage] = field(default_factory=list)

    def run(self, stage: TrainingStage) -> StageResult:
        """记录阶段并返回按测试设置生成的阶段结果。"""

        self.calls.append(stage)
        status = (
            ExecutionStatus.FAILED if stage is self.fail_at else ExecutionStatus.PASSED
        )
        acceptance = (
            self.acceptance
            if stage is TrainingStage.EVALUATION and status is ExecutionStatus.PASSED
            else None
        )
        return StageResult(stage=stage, status=status, acceptance=acceptance)


def test_blocking_quality_report_causes_zero_runner_calls() -> None:
    """数据质量阻断时不得调用任何训练阶段。"""

    runner = RecordingRunner()
    result = TrainingOrchestrator(runner).run(_quality_report(blocked=True))

    assert result.status is ExecutionStatus.FAILED
    assert result.quality_passed is False
    assert result.stage_results == ()
    assert result.acceptance is None
    assert runner.calls == []


def test_stages_run_in_fixed_registry_order_and_all_pass() -> None:
    """所有阶段必须按固定注册表顺序运行并汇总通过。"""

    runner = RecordingRunner()
    result = TrainingOrchestrator(runner).run(_quality_report())

    assert tuple(runner.calls) == STAGE_REGISTRY
    assert tuple(item.stage for item in result.stage_results) == STAGE_REGISTRY
    assert all(item.status is ExecutionStatus.PASSED for item in result.stage_results)
    assert result.status is ExecutionStatus.PASSED


def test_stage_failure_stops_immediately_and_stays_failed() -> None:
    """阶段失败必须立即短路，且编排结果保持失败。"""

    runner = RecordingRunner(fail_at=TrainingStage.BELIEF)
    result = TrainingOrchestrator(runner).run(_quality_report())

    assert tuple(runner.calls) == STAGE_REGISTRY[:3]
    assert result.stage_results[-1].status is ExecutionStatus.FAILED
    assert result.status is ExecutionStatus.FAILED
    assert result.acceptance is None


def test_final_acceptance_is_required_after_all_stages() -> None:
    """阶段全部成功后仍必须满足最终 acceptance 门禁。"""

    runner = RecordingRunner(
        acceptance=TrialAcceptance(True, True, True, True, True, True, False)
    )
    result = TrainingOrchestrator(runner).run(_quality_report())

    assert tuple(runner.calls) == STAGE_REGISTRY
    assert result.status is ExecutionStatus.FAILED
    assert result.acceptance is runner.acceptance
    assert result.acceptance.passed is False


def test_quality_and_acceptance_data_gate_cannot_disagree() -> None:
    """质量报告与 acceptance 的 data 门禁不得互相矛盾。"""

    runner = RecordingRunner(
        acceptance=TrialAcceptance(False, True, True, True, True, True, True)
    )

    with pytest.raises(ValueError, match="Acceptance.data"):
        TrainingOrchestrator(runner).run(_quality_report())


@pytest.mark.parametrize(
    ("stage", "expected"),
    (
        (CurriculumStage.BASIC, CurriculumStage.MULTI_OBJECT),
        (CurriculumStage.MULTI_OBJECT, CurriculumStage.COMPLEX),
        (CurriculumStage.COMPLEX, CurriculumStage.FULL),
    ),
)
def test_curriculum_advances_only_when_all_gates_pass(
    stage: CurriculumStage, expected: CurriculumStage
) -> None:
    """课程阶段只有在全部门禁通过时才能推进。"""

    passed = (
        CurriculumGate("accuracy", True),
        CurriculumGate("robustness", True),
    )
    decision = decide_curriculum(stage, passed)

    assert decision.action is CurriculumAction.ADVANCE
    assert decision.next_stage is expected

    held = decide_curriculum(
        stage,
        (CurriculumGate("accuracy", True), CurriculumGate("robustness", False)),
    )
    assert held.action is CurriculumAction.HOLD
    assert held.next_stage is stage


def test_empty_curriculum_gates_hold_and_full_stage_completes() -> None:
    """空课程门禁必须保持，FULL 通过后必须明确完成。"""

    assert decide_curriculum(CurriculumStage.BASIC, ()).action is CurriculumAction.HOLD
    completed = decide_curriculum(
        CurriculumStage.FULL, (CurriculumGate("final", True),)
    )
    assert completed.action is CurriculumAction.COMPLETE
    assert completed.next_stage is CurriculumStage.FULL


def _asha() -> AshaScheduler:
    return AshaScheduler(
        rungs=(
            AshaRung(index=0, budget=10, promotion_fraction=0.5),
            AshaRung(index=1, budget=30, promotion_fraction=0.5),
        )
    )


def test_asha_strict_gate_overrides_high_objective_and_top_fraction_promotes() -> None:
    """ASHA 严格门禁优先于高分，并只晋升通过者中的头部比例。"""

    trials = (
        AshaTrial("failed-best", 0, 100.0, False),
        AshaTrial("second", 0, 0.8, True),
        AshaTrial("first", 0, 0.9, True),
        AshaTrial("third", 0, 0.7, True),
    )
    decisions = _asha().decide(0, trials)
    actions = {decision.trial_id: decision.action for decision in decisions}

    assert actions == {
        "failed-best": AshaAction.PRUNE,
        "first": AshaAction.PROMOTE,
        "second": AshaAction.PROMOTE,
        "third": AshaAction.PRUNE,
    }
    assert all(
        decision.next_rung == 1
        for decision in decisions
        if decision.action is AshaAction.PROMOTE
    )


def test_asha_ties_are_stable_and_terminal_rung_continues() -> None:
    """ASHA 同分决策必须稳定，末级通过者应继续完成流程。"""

    tied = (
        AshaTrial("trial-z", 0, 0.5, True),
        AshaTrial("trial-a", 0, 0.5, True),
    )
    forward = _asha().decide(0, tied)
    reversed_order = _asha().decide(0, tuple(reversed(tied)))

    assert forward == reversed_order
    assert {item.trial_id: item.action for item in forward} == {
        "trial-a": AshaAction.PROMOTE,
        "trial-z": AshaAction.PRUNE,
    }

    terminal = _asha().decide(1, (AshaTrial("winner", 1, 0.9, True),))
    assert terminal[0].action is AshaAction.CONTINUE
    assert terminal[0].next_rung is None


def test_asha_rejects_non_increasing_budgets_and_mixed_rungs() -> None:
    """ASHA 必须拒绝非递增预算和混合 rung 输入。"""

    with pytest.raises(ValueError, match="严格递增"):
        AshaScheduler(
            rungs=(
                AshaRung(0, 10, 0.5),
                AshaRung(1, 10, 0.5),
            )
        )
    with pytest.raises(ValueError, match="请求 rung"):
        _asha().decide(0, (AshaTrial("wrong", 1, 0.9, True),))


def test_phase9_modules_do_not_depend_on_sqlite_legacy_ui_or_any() -> None:
    """Phase 9 模块不得依赖 SQLite、legacy UI 或宽泛 Any。"""

    identifiers: set[str] = set()
    type_any_names: set[str] = set()
    for path in (
        _TRAINING_ROOT / "orchestration.py",
        _TRAINING_ROOT / "scheduling.py",
    ):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        identifiers.update(
            alias.name.lower()
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        )
        identifiers.update(
            node.id.lower() for node in ast.walk(tree) if isinstance(node, ast.Name)
        )
        type_any_names.update(
            node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
        )

    assert "Any" not in type_any_names
    assert not any(
        fragment in identifier
        for fragment in ("sqlite", "legacy", "traning")
        for identifier in identifiers
    )
    assert not any(
        identifier == "ui" or identifier.startswith("ui.") for identifier in identifiers
    )
