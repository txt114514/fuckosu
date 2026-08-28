"""Phase 10 Rich/Qt 只读 telemetry projection 验收。"""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from traning.evaluation.attribution import (
    EvaluationTag,
    PrimaryError,
    SequenceEvaluationEvent,
)
from traning.telemetry.reporter import (
    DASHBOARD_SCHEMA_VERSION,
    DashboardMetrics,
    DashboardResources,
    DashboardSnapshot,
)
from traning.visualization import (
    DashboardMetric,
    DashboardMetricRow,
    QtDashboardRenderer,
    QtEvaluationColumn,
    QtMetricColumn,
    RichDashboardRenderer,
)


_SOURCE_PATH = Path(__file__).resolve().parents[2] / "visualization/renderers.py"


def _failed_evaluation() -> SequenceEvaluationEvent:
    """复现 frame 105 的 canonical decision/unresolved 语义。"""

    return SequenceEvaluationEvent(
        event_id=f"sequence-event-{'1' * 64}",
        sample_id="long-sequence-000008",
        frame_index=105,
        passed=False,
        primary_error=PrimaryError.DECISION,
        error_tags=(EvaluationTag.UNRESOLVED_TARGET,),
        target_id="target-105",
        click_index=None,
    )


def _snapshot(
    evaluation: SequenceEvaluationEvent | None = None,
) -> DashboardSnapshot:
    """构造覆盖 Phase 10 所有必需指标的不可变快照。"""

    return DashboardSnapshot(
        schema_version=DASHBOARD_SCHEMA_VERSION,
        run_id="run-phase-10",
        timestamp_ms=160.0,
        metrics=DashboardMetrics(
            step=12,
            loss=0.125,
            score=0.875,
            perception_recall=0.95,
            tracking_id_switches=2,
            outcome_nll=0.21,
            outcome_brier=0.14,
            outcome_ece=0.03,
            expected_score_error=0.04,
            decision_utility=0.72,
            wait_click_ratio=0.25,
        ),
        resources=DashboardResources(
            gpu_utilization=0.84,
            vram_used_mb=4096.0,
            vram_total_mb=8192.0,
            throughput=321.5,
        ),
        evaluation=evaluation,
    )


def _rich_rows(snapshot: DashboardSnapshot) -> tuple[DashboardMetricRow, ...]:
    """按 Rich 模型中的固定分区顺序展开指标行。"""

    model = RichDashboardRenderer.render(snapshot)
    return tuple(row for section in model.sections for row in section.rows)


def test_rich_and_qt_show_the_complete_metric_registry_in_stable_order() -> None:
    """Rich 与 Qt 必须按稳定顺序展示完整指标注册表。"""

    snapshot = _snapshot()
    rich = RichDashboardRenderer.render(snapshot)
    qt = QtDashboardRenderer.render(snapshot)
    rich_rows = tuple(row for section in rich.sections for row in section.rows)

    expected_metrics = tuple(DashboardMetric)
    assert tuple(row.metric for row in rich_rows) == expected_metrics
    assert tuple(row.metric for row in qt.metrics.rows) == expected_metrics
    assert tuple(row.rank for row in rich_rows) == tuple(range(len(expected_metrics)))
    assert qt.metrics.columns == tuple(QtMetricColumn)
    assert qt.evaluations.columns == tuple(QtEvaluationColumn)

    values = {row.metric: row.value for row in qt.metrics.rows}
    assert values == {
        DashboardMetric.STEP: 12,
        DashboardMetric.LOSS: 0.125,
        DashboardMetric.SCORE: 0.875,
        DashboardMetric.PERCEPTION_RECALL: 0.95,
        DashboardMetric.TRACKING_ID_SWITCHES: 2,
        DashboardMetric.OUTCOME_NLL: 0.21,
        DashboardMetric.OUTCOME_BRIER: 0.14,
        DashboardMetric.OUTCOME_ECE: 0.03,
        DashboardMetric.EXPECTED_SCORE_ERROR: 0.04,
        DashboardMetric.DECISION_UTILITY: 0.72,
        DashboardMetric.WAIT_CLICK_RATIO: 0.25,
        DashboardMetric.THROUGHPUT: 321.5,
        DashboardMetric.GPU_UTILIZATION: 84.0,
        DashboardMetric.VRAM_USED_MB: 4096.0,
        DashboardMetric.VRAM_TOTAL_MB: 8192.0,
    }
    assert rich.schema_version == DASHBOARD_SCHEMA_VERSION
    assert qt.run_id == snapshot.run_id


def test_evaluation_projection_preserves_identity_pass_and_primary_error() -> None:
    """评估投影必须保留对象身份、通过状态和主错误。"""

    evaluation = _failed_evaluation()
    snapshot = _snapshot(evaluation)
    rich = RichDashboardRenderer.render(snapshot)
    qt = QtDashboardRenderer.render(snapshot)

    rich_row = rich.evaluations[0]
    qt_row = qt.evaluations.rows[0]
    assert rich_row.event is evaluation
    assert qt_row.event is evaluation
    assert rich_row.passed is False
    assert qt_row.passed is False
    assert rich_row.primary_error is PrimaryError.DECISION
    assert qt_row.primary_error is PrimaryError.DECISION
    assert rich_row.error_tags == (EvaluationTag.UNRESOLVED_TARGET,)
    assert rich_row.frame_index == 105


def test_passed_evaluation_is_not_reinterpreted_as_an_error() -> None:
    """已通过的评估不得被可视化层重新解释为错误。"""

    evaluation = SequenceEvaluationEvent(
        event_id=f"sequence-event-{'2' * 64}",
        sample_id="passed-sample",
        frame_index=36,
        passed=True,
        primary_error=PrimaryError.NONE,
        error_tags=(),
        target_id="target-36",
        click_index=0,
        click_x=100.0,
        click_y=80.0,
    )
    row = RichDashboardRenderer.render(_snapshot(evaluation)).evaluations[0]

    assert row.event is evaluation
    assert row.passed is True
    assert row.primary_error is PrimaryError.NONE
    assert row.error_tags == ()


def test_renderers_are_deterministic_and_models_are_frozen() -> None:
    """渲染输出必须确定，且视图模型必须保持不可变。"""

    snapshot = _snapshot(_failed_evaluation())
    rich_first = RichDashboardRenderer.render(snapshot)
    rich_second = RichDashboardRenderer.render(snapshot)
    qt_first = QtDashboardRenderer.render(snapshot)
    qt_second = QtDashboardRenderer.render(snapshot)

    assert rich_first == rich_second
    assert qt_first == qt_second
    assert _rich_rows(snapshot) == _rich_rows(snapshot)
    with pytest.raises(FrozenInstanceError):
        rich_first.run_id = "mutated"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        qt_first.metrics.rows = ()  # type: ignore[misc]
    assert snapshot.run_id == "run-phase-10"
    assert snapshot.evaluation is not None


def test_missing_optional_telemetry_keeps_all_slots_without_invention() -> None:
    """缺少可选遥测时必须保留槽位且不能虚构数据。"""

    snapshot = DashboardSnapshot(
        schema_version=DASHBOARD_SCHEMA_VERSION,
        run_id="empty-run",
        timestamp_ms=0.0,
    )
    rich = RichDashboardRenderer.render(snapshot)
    qt = QtDashboardRenderer.render(snapshot)
    rich_rows = tuple(row for section in rich.sections for row in section.rows)

    assert tuple(row.metric for row in rich_rows) == tuple(DashboardMetric)
    assert all(row.value is None for row in rich_rows)
    assert all(row.display_value == "—" for row in qt.metrics.rows)
    assert rich.evaluations == ()
    assert qt.evaluations.rows == ()


def test_renderer_rejects_mapping_instead_of_accepting_mutable_live_state() -> None:
    """渲染器必须拒绝可变映射，只接受冻结快照。"""

    with pytest.raises(TypeError, match="DashboardSnapshot"):
        RichDashboardRenderer.render({"run_id": "legacy"})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="DashboardSnapshot"):
        QtDashboardRenderer.render({"run_id": "legacy"})  # type: ignore[arg-type]


def test_renderer_source_has_no_gui_store_io_or_semantic_side_channel() -> None:
    """Renderer 只能读 reporter snapshot，不得接触 store、scorer 或质量门禁。"""

    tree = ast.parse(_SOURCE_PATH.read_text(encoding="utf-8"))
    forbidden_modules = (
        "PySide6",
        "rich",
        "osu_v2",
        "traning.data.quality",
        "traning.evaluation.scoring",
        "traning.outcome.oracle",
        "traning.telemetry.store",
    )
    forbidden_calls = {
        "SequenceEvaluationEvent",
        "PrimaryError",
        "open",
        "write",
        "write_bytes",
        "write_text",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(
                not alias.name.startswith(forbidden_modules) for alias in node.names
            )
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert not module.startswith(forbidden_modules)
        if isinstance(node, ast.Name):
            assert node.id != "Any"
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                assert node.func.id not in forbidden_calls
            elif isinstance(node.func, ast.Attribute):
                assert node.func.attr not in forbidden_calls
