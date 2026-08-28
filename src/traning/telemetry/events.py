"""定义 V2 遥测四通道使用的强类型不可变事件。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

from traning.config.versions import TELEMETRY_SCHEMA_VERSION
from traning.contracts.common import (
    require_finite,
    require_identifier,
    require_nonnegative,
    require_probability,
)
from traning.contracts.telemetry import TelemetryEvent
from traning.evaluation.attribution import SequenceEvaluationEvent


class TelemetryChannel(str, Enum):
    """正式持久化通道；枚举值就是稳定 JSON ``record_type``。"""

    METRICS = "metrics"
    RESOURCES = "resources"
    EVALUATION = "evaluation"
    EVENTS = "events"


@dataclass(frozen=True, slots=True)
class ChannelSpec:
    """把通道与固定文件名集中注册，避免各消费者自行拼接路径。"""

    channel: TelemetryChannel
    filename: str


CHANNEL_SPECS: tuple[ChannelSpec, ...] = (
    ChannelSpec(TelemetryChannel.METRICS, "metrics.jsonl"),
    ChannelSpec(TelemetryChannel.RESOURCES, "resources.jsonl"),
    ChannelSpec(TelemetryChannel.EVALUATION, "evaluation.jsonl"),
    ChannelSpec(TelemetryChannel.EVENTS, "events.jsonl"),
)
"""四个且仅四个正式 JSONL 通道的唯一规格表。"""


def _validate_header(schema_version: int, timestamp_ms: float, run_id: str) -> None:
    """校验所有事件共享的 schema、时间戳和运行标识。"""

    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise TypeError("schema_version 必须是整数")
    if schema_version != TELEMETRY_SCHEMA_VERSION:
        raise ValueError(f"schema_version 必须是 {TELEMETRY_SCHEMA_VERSION}")
    require_nonnegative(timestamp_ms, "timestamp_ms")
    require_identifier(run_id, "run_id")


def _validate_step(step: int) -> None:
    """拒绝布尔值、非整数和负训练步。"""

    if isinstance(step, bool) or not isinstance(step, int):
        raise TypeError("step 必须是整数")
    if step < 0:
        raise ValueError("step 不得为负数")


def _validate_count(value: int, field_name: str) -> None:
    """校验不会被 bool 冒充的非负计数。"""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} 必须是整数")
    if value < 0:
        raise ValueError(f"{field_name} 不得为负数")


@dataclass(frozen=True, slots=True)
class MetricsEvent:
    """一次训练/评估步的完整模型质量与决策指标快照。"""

    schema_version: int
    timestamp_ms: float
    run_id: str
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
    score: float = 0.0

    def __post_init__(self) -> None:
        _validate_header(self.schema_version, self.timestamp_ms, self.run_id)
        _validate_step(self.step)
        require_nonnegative(self.loss, "loss")
        require_probability(self.perception_recall, "perception_recall")
        _validate_count(self.tracking_id_switches, "tracking_id_switches")
        require_nonnegative(self.outcome_nll, "outcome_nll")
        require_nonnegative(self.outcome_brier, "outcome_brier")
        require_probability(self.outcome_ece, "outcome_ece")
        require_nonnegative(self.expected_score_error, "expected_score_error")
        require_finite(self.decision_utility, "decision_utility")
        require_nonnegative(self.wait_click_ratio, "wait_click_ratio")
        require_finite(self.score, "score")


@dataclass(frozen=True, slots=True)
class ResourceEvent:
    """一次资源采样，覆盖 GPU、显存和端到端吞吐量。"""

    schema_version: int
    timestamp_ms: float
    run_id: str
    step: int
    gpu_utilization: float
    vram_used_mb: float
    vram_total_mb: float
    throughput: float

    def __post_init__(self) -> None:
        _validate_header(self.schema_version, self.timestamp_ms, self.run_id)
        _validate_step(self.step)
        require_probability(self.gpu_utilization, "gpu_utilization")
        require_nonnegative(self.vram_used_mb, "vram_used_mb")
        require_nonnegative(self.vram_total_mb, "vram_total_mb")
        require_nonnegative(self.throughput, "throughput")
        if self.vram_used_mb > self.vram_total_mb:
            raise ValueError("vram_used_mb 不得超过 vram_total_mb")


@dataclass(frozen=True, slots=True)
class EvaluationEvent:
    """canonical 评分事件的无损遥测信封。

    信封只补充运行和时间信息，不复制、不重判 ``passed``、主要错误域或
    错误标签。因而进程内 snapshot 可保持原 ``SequenceEvaluationEvent`` 的
    对象身份，磁盘恢复则保持值语义完全一致。
    """

    schema_version: int
    timestamp_ms: float
    run_id: str
    event: SequenceEvaluationEvent

    def __post_init__(self) -> None:
        _validate_header(self.schema_version, self.timestamp_ms, self.run_id)
        if not isinstance(self.event, SequenceEvaluationEvent):
            raise TypeError("event 必须是 SequenceEvaluationEvent")


PublishableTelemetryEvent: TypeAlias = (
    MetricsEvent | ResourceEvent | EvaluationEvent | TelemetryEvent
)
"""``StateStore.publish`` 接受的封闭事件联合。"""


def event_channel(event: PublishableTelemetryEvent) -> TelemetryChannel:
    """返回事件的唯一持久化通道，拒绝联合外的运行时对象。"""

    if isinstance(event, MetricsEvent):
        return TelemetryChannel.METRICS
    if isinstance(event, ResourceEvent):
        return TelemetryChannel.RESOURCES
    if isinstance(event, EvaluationEvent):
        return TelemetryChannel.EVALUATION
    if isinstance(event, TelemetryEvent):
        return TelemetryChannel.EVENTS
    raise TypeError("event 必须是正式的 typed telemetry event")


__all__ = (
    "CHANNEL_SPECS",
    "TELEMETRY_SCHEMA_VERSION",
    "ChannelSpec",
    "EvaluationEvent",
    "MetricsEvent",
    "PublishableTelemetryEvent",
    "ResourceEvent",
    "TelemetryChannel",
    "event_channel",
)
