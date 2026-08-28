"""Phase 10 reporter 与不可变 dashboard projection 验收。"""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, fields, replace
from pathlib import Path

import pytest

from traning.contracts.telemetry import TelemetryEvent
from traning.evaluation.attribution import (
    EvaluationTag,
    PrimaryError,
    SequenceEvaluationEvent,
)
from traning.telemetry.events import (
    TELEMETRY_SCHEMA_VERSION,
    EvaluationEvent,
    MetricsEvent,
    ResourceEvent,
)
from traning.telemetry.reporter import (
    DASHBOARD_SCHEMA_VERSION,
    DashboardMetrics,
    DashboardResources,
    DashboardSnapshot,
    TelemetryReporter,
)
from traning.telemetry.store import StateStore


_REPORTER_PATH = Path(__file__).resolve().parents[2] / "telemetry/reporter.py"


def _canonical_evaluation() -> SequenceEvaluationEvent:
    """构造 frame 105 同型的零点击未解析 decision 事件。"""

    return SequenceEvaluationEvent(
        event_id=f"sequence-event-{105:064x}",
        sample_id="long_sequence_000008",
        frame_index=105,
        passed=False,
        primary_error=PrimaryError.DECISION,
        error_tags=(EvaluationTag.UNRESOLVED_TARGET,),
        target_id="target-105",
        click_index=None,
    )


def _metrics(run_id: str = "run-1", timestamp_ms: float = 10.0) -> MetricsEvent:
    return MetricsEvent(
        schema_version=TELEMETRY_SCHEMA_VERSION,
        timestamp_ms=timestamp_ms,
        run_id=run_id,
        step=7,
        loss=0.25,
        perception_recall=0.9,
        tracking_id_switches=2,
        outcome_nll=0.31,
        outcome_brier=0.18,
        outcome_ece=0.04,
        expected_score_error=0.07,
        decision_utility=-0.02,
        wait_click_ratio=0.5,
        score=0.82,
    )


def _resources(run_id: str = "run-1") -> ResourceEvent:
    return ResourceEvent(
        schema_version=TELEMETRY_SCHEMA_VERSION,
        timestamp_ms=20.0,
        run_id=run_id,
        step=7,
        gpu_utilization=0.75,
        vram_used_mb=4096.0,
        vram_total_mb=8192.0,
        throughput=123.5,
    )


def test_reporter_publishes_all_required_metrics_and_resources(tmp_path: Path) -> None:
    """reporter 必须完整投影指标和资源事件的全部字段。"""

    reporter = TelemetryReporter("run-1", StateStore(tmp_path))
    reporter.publish(_metrics())
    reporter.publish(_resources())

    snapshot = reporter.snapshot()
    assert snapshot.schema_version == DASHBOARD_SCHEMA_VERSION
    assert snapshot.run_id == "run-1"
    assert snapshot.timestamp_ms == 20.0
    assert snapshot.metrics == DashboardMetrics(
        step=7,
        loss=0.25,
        perception_recall=0.9,
        tracking_id_switches=2,
        outcome_nll=0.31,
        outcome_brier=0.18,
        outcome_ece=0.04,
        expected_score_error=0.07,
        decision_utility=-0.02,
        wait_click_ratio=0.5,
        score=0.82,
    )
    assert snapshot.resources == DashboardResources(
        gpu_utilization=0.75,
        vram_used_mb=4096.0,
        vram_total_mb=8192.0,
        throughput=123.5,
    )


def test_canonical_evaluation_object_and_semantics_are_preserved(
    tmp_path: Path,
) -> None:
    """评估事件进入 dashboard 后必须保持对象身份与判定语义。"""

    evaluation = _canonical_evaluation()
    reporter = TelemetryReporter("run-1", StateStore(tmp_path))
    reporter.publish(
        EvaluationEvent(TELEMETRY_SCHEMA_VERSION, 30.0, "run-1", evaluation)
    )

    snapshot = reporter.snapshot()
    assert snapshot.evaluation is evaluation
    assert snapshot.evaluation.passed is False
    assert snapshot.evaluation.primary_error is PrimaryError.DECISION
    assert snapshot.evaluation.error_tags == (EvaluationTag.UNRESOLVED_TARGET,)


def test_reporter_has_no_side_cache_and_projects_store_latest_state(
    tmp_path: Path,
) -> None:
    """reporter 不得维护旁路缓存，并只投影 store 的最新状态。"""

    store = StateStore(tmp_path)
    reporter = TelemetryReporter("run-1", store)
    assert reporter.snapshot() == DashboardSnapshot(1, "run-1", 0.0)

    # 直接发布到唯一 store 后，原 reporter 必须立即看到新快照。
    store.publish(_metrics())
    assert reporter.snapshot().metrics is not None
    assert reporter.snapshot().metrics.step == 7


def test_generic_quality_event_is_not_reinterpreted_by_dashboard(
    tmp_path: Path,
) -> None:
    """dashboard 不得擅自重解释通用质量事件。"""

    reporter = TelemetryReporter("run-1", StateStore(tmp_path))
    evaluation = _canonical_evaluation()
    reporter.publish(
        EvaluationEvent(TELEMETRY_SCHEMA_VERSION, 30.0, "run-1", evaluation)
    )
    reporter.publish(
        TelemetryEvent(
            schema_version=TELEMETRY_SCHEMA_VERSION,
            event_type="quality",
            timestamp_ms=40.0,
            run_id="run-1",
            metrics=(("quality_ok", 0.0),),
            payload=(("blocks_training", True),),
        )
    )

    snapshot = reporter.snapshot()
    assert snapshot.timestamp_ms == 40.0
    assert snapshot.metrics is None
    assert snapshot.evaluation is evaluation
    assert not hasattr(snapshot, "quality_ok")


def test_reporter_rejects_cross_run_events_before_publication(tmp_path: Path) -> None:
    """不同 run 的事件必须在持久化前被 reporter 拒绝。"""

    store = StateStore(tmp_path)
    reporter = TelemetryReporter("run-1", store)
    with pytest.raises(ValueError, match="run_id"):
        reporter.publish(_metrics(run_id="run-2"))
    assert store.history().metrics == ()


def test_dashboard_contracts_are_frozen_and_cover_required_fields() -> None:
    """dashboard 契约必须不可变且覆盖规定字段。"""

    assert {item.name for item in fields(DashboardMetrics)} == {
        "step",
        "loss",
        "perception_recall",
        "tracking_id_switches",
        "outcome_nll",
        "outcome_brier",
        "outcome_ece",
        "expected_score_error",
        "decision_utility",
        "wait_click_ratio",
        "score",
    }
    assert {item.name for item in fields(DashboardResources)} == {
        "gpu_utilization",
        "vram_used_mb",
        "vram_total_mb",
        "throughput",
    }
    snapshot = DashboardSnapshot(1, "run-1", 0.0)
    with pytest.raises(FrozenInstanceError):
        snapshot.timestamp_ms = 1.0  # type: ignore[misc]
    with pytest.raises(ValueError, match="gpu_utilization"):
        DashboardResources(50.0, 1.0, 2.0, 3.0)
    valid_metrics = DashboardMetrics(
        step=0,
        loss=0.0,
        perception_recall=0.0,
        tracking_id_switches=0,
        outcome_nll=0.0,
        outcome_brier=0.0,
        outcome_ece=0.0,
        expected_score_error=0.0,
        decision_utility=0.0,
        wait_click_ratio=0.0,
        score=0.0,
    )
    with pytest.raises(ValueError, match="outcome_ece"):
        replace(valid_metrics, outcome_ece=1.01)


def test_reporter_source_never_recomputes_quality_or_evaluation() -> None:
    """Reporter 不得读 passed/primary_error/blocks_training 发明新语义。"""

    tree = ast.parse(_REPORTER_PATH.read_text(encoding="utf-8"))
    forbidden_names = {"Any", "DataQualityIssue", "DataQualityReport"}
    forbidden_attributes = {"blocks_training", "passed", "primary_error", "ok"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            assert node.id not in forbidden_names
        if isinstance(node, ast.Attribute):
            assert node.attr not in forbidden_attributes
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith(("osu_v2", "visualization"))
