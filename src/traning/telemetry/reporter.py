"""把 typed telemetry store 投影为不可变 dashboard 快照。"""

from __future__ import annotations

from dataclasses import dataclass

from traning.contracts.common import (
    require_finite,
    require_identifier,
    require_nonnegative,
    require_probability,
)
from traning.contracts.telemetry import TelemetryEvent
from traning.evaluation.attribution import SequenceEvaluationEvent
from traning.telemetry.events import (
    EvaluationEvent,
    MetricsEvent,
    PublishableTelemetryEvent,
    ResourceEvent,
    TELEMETRY_SCHEMA_VERSION,
)
from traning.telemetry.store import StateStore, StoreSnapshot


DASHBOARD_SCHEMA_VERSION = 1
"""Rich 与 Qt 共同消费的 dashboard schema 版本。"""


@dataclass(frozen=True, slots=True)
class DashboardMetrics:
    """训练与推理各层的完整、不可变指标投影。"""

    step: int
    loss: float
    perception_recall: float
    tracking_id_switches: int
    outcome_nll: float
    outcome_brier: float
    outcome_ece: float
    expected_score_error: float
    decision_utility: float
    wait_click_ratio: float
    score: float

    def __post_init__(self) -> None:
        if isinstance(self.step, bool) or not isinstance(self.step, int):
            raise TypeError("step 必须是整数")
        if self.step < 0:
            raise ValueError("step 不得为负数")
        if isinstance(self.tracking_id_switches, bool) or not isinstance(
            self.tracking_id_switches, int
        ):
            raise TypeError("tracking_id_switches 必须是整数")
        if self.tracking_id_switches < 0:
            raise ValueError("tracking_id_switches 不得为负数")
        require_nonnegative(self.loss, "loss")
        require_probability(self.perception_recall, "perception_recall")
        require_probability(self.outcome_ece, "outcome_ece")
        for field_name, value in (
            ("outcome_nll", self.outcome_nll),
            ("outcome_brier", self.outcome_brier),
            ("expected_score_error", self.expected_score_error),
            ("wait_click_ratio", self.wait_click_ratio),
        ):
            require_nonnegative(value, field_name)
        require_finite(self.decision_utility, "decision_utility")
        require_finite(self.score, "score")


@dataclass(frozen=True, slots=True)
class DashboardResources:
    """GPU、显存与吞吐资源的不可变投影。"""

    gpu_utilization: float
    vram_used_mb: float
    vram_total_mb: float
    throughput: float

    def __post_init__(self) -> None:
        # 与 ResourceEvent 保持同一单位：0..1 ratio，Renderer 再格式化为百分比。
        require_probability(self.gpu_utilization, "gpu_utilization")
        for field_name, value in (
            ("vram_used_mb", self.vram_used_mb),
            ("vram_total_mb", self.vram_total_mb),
            ("throughput", self.throughput),
        ):
            require_nonnegative(value, field_name)
        if self.vram_used_mb > self.vram_total_mb:
            raise ValueError("vram_used_mb 不得大于 vram_total_mb")


@dataclass(frozen=True, slots=True)
class DashboardSnapshot:
    """Renderer 唯一允许消费的 versioned dashboard 快照。"""

    schema_version: int
    run_id: str
    timestamp_ms: float
    metrics: DashboardMetrics | None = None
    resources: DashboardResources | None = None
    evaluation: SequenceEvaluationEvent | None = None

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or not isinstance(
            self.schema_version, int
        ):
            raise TypeError("schema_version 必须是整数")
        if self.schema_version != DASHBOARD_SCHEMA_VERSION:
            raise ValueError(
                f"dashboard schema_version 必须是 {DASHBOARD_SCHEMA_VERSION}"
            )
        require_identifier(self.run_id, "run_id")
        require_nonnegative(self.timestamp_ms, "timestamp_ms")
        if self.metrics is not None and not isinstance(self.metrics, DashboardMetrics):
            raise TypeError("metrics 必须是 DashboardMetrics 或 None")
        if self.resources is not None and not isinstance(
            self.resources, DashboardResources
        ):
            raise TypeError("resources 必须是 DashboardResources 或 None")
        if self.evaluation is not None and not isinstance(
            self.evaluation, SequenceEvaluationEvent
        ):
            raise TypeError("evaluation 必须是 SequenceEvaluationEvent 或 None")


def project_dashboard(run_id: str, snapshot: StoreSnapshot) -> DashboardSnapshot:
    """逐字段投影 store 的最新事件，不推导领域判断或复制评估事件。"""

    require_identifier(run_id, "run_id")
    if not isinstance(snapshot, StoreSnapshot):
        raise TypeError("snapshot 必须是 StoreSnapshot")
    if snapshot.schema_version != TELEMETRY_SCHEMA_VERSION:
        raise ValueError("不支持的 telemetry snapshot schema_version")

    # store 是 live state 的唯一权威；这里仅消费其不可变 latest-channel 快照。
    latest_events = tuple(
        event
        for event in (
            snapshot.metrics,
            snapshot.resources,
            snapshot.evaluation,
            snapshot.event,
        )
        if event is not None
    )
    if any(event.run_id != run_id for event in latest_events):
        raise ValueError("StoreSnapshot 含有其他 run_id 的事件")
    timestamp_ms = max(
        (event.timestamp_ms for event in latest_events),
        default=0.0,
    )

    metrics = _project_metrics(snapshot.metrics)
    resources = _project_resources(snapshot.resources)
    # 直接暴露信封内的 canonical 对象，不重构 primary_error/pass。
    evaluation = snapshot.evaluation.event if snapshot.evaluation is not None else None
    return DashboardSnapshot(
        schema_version=DASHBOARD_SCHEMA_VERSION,
        run_id=run_id,
        timestamp_ms=timestamp_ms,
        metrics=metrics,
        resources=resources,
        evaluation=evaluation,
    )


def _project_metrics(event: MetricsEvent | None) -> DashboardMetrics | None:
    """把完整 MetricsEvent 原值复制到展示契约。"""

    if event is None:
        return None
    return DashboardMetrics(
        step=event.step,
        loss=event.loss,
        perception_recall=event.perception_recall,
        tracking_id_switches=event.tracking_id_switches,
        outcome_nll=event.outcome_nll,
        outcome_brier=event.outcome_brier,
        outcome_ece=event.outcome_ece,
        expected_score_error=event.expected_score_error,
        decision_utility=event.decision_utility,
        wait_click_ratio=event.wait_click_ratio,
        score=event.score,
    )


def _project_resources(event: ResourceEvent | None) -> DashboardResources | None:
    """把完整 ResourceEvent 原值复制到展示契约。"""

    if event is None:
        return None
    return DashboardResources(
        gpu_utilization=event.gpu_utilization,
        vram_used_mb=event.vram_used_mb,
        vram_total_mb=event.vram_total_mb,
        throughput=event.throughput,
    )


@dataclass(frozen=True, slots=True)
class TelemetryReporter:
    """训练/推理线程与唯一 StateStore 之间的极薄 typed 发布边界。"""

    run_id: str
    store: StateStore

    def __post_init__(self) -> None:
        require_identifier(self.run_id, "run_id")
        if not isinstance(self.store, StateStore):
            raise TypeError("store 必须是 StateStore")

    def publish(self, event: PublishableTelemetryEvent) -> None:
        """校验 run identity 后把 typed event 原对象交给 store。"""

        if not isinstance(
            event,
            (MetricsEvent, ResourceEvent, EvaluationEvent, TelemetryEvent),
        ):
            raise TypeError("event 必须是 typed telemetry event")
        if event.run_id != self.run_id:
            raise ValueError("event.run_id 与 reporter.run_id 不一致")
        self.store.publish(event)

    def snapshot(self) -> DashboardSnapshot:
        """从 store 的原子快照即时生成 renderer 输入，不维护旁路缓存。"""

        return project_dashboard(self.run_id, self.store.snapshot())


__all__ = (
    "DASHBOARD_SCHEMA_VERSION",
    "DashboardMetrics",
    "DashboardResources",
    "DashboardSnapshot",
    "TelemetryReporter",
    "project_dashboard",
)
