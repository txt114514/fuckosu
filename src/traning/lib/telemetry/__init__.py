"""V2 强类型遥测事件、状态存储和 dashboard 投影公开入口。"""

from traning.lib.telemetry.events import (
    CHANNEL_SPECS,
    TELEMETRY_SCHEMA_VERSION,
    ChannelSpec,
    EvaluationEvent,
    MetricsEvent,
    PublishableTelemetryEvent,
    ResourceEvent,
    TelemetryChannel,
    event_channel,
)
from traning.lib.telemetry.reporter import (
    DASHBOARD_SCHEMA_VERSION,
    DashboardMetrics,
    DashboardResources,
    DashboardSnapshot,
    TelemetryReporter,
    project_dashboard,
)
from traning.lib.telemetry.store import StateStore, StoreSnapshot, TelemetryHistory

__all__ = (
    "CHANNEL_SPECS",
    "DASHBOARD_SCHEMA_VERSION",
    "TELEMETRY_SCHEMA_VERSION",
    "ChannelSpec",
    "DashboardMetrics",
    "DashboardResources",
    "DashboardSnapshot",
    "EvaluationEvent",
    "MetricsEvent",
    "PublishableTelemetryEvent",
    "ResourceEvent",
    "StateStore",
    "StoreSnapshot",
    "TelemetryChannel",
    "TelemetryHistory",
    "TelemetryReporter",
    "event_channel",
    "project_dashboard",
)
