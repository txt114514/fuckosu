"""遥测事件的稳定跨线程快照契约。"""

from dataclasses import dataclass

from .common import JSONValue, require_finite, require_identifier, require_nonnegative


@dataclass(frozen=True, slots=True)
class MemoryReport:
    """PyTorch allocator 的峰值与当前显存快照。"""

    cuda_available: bool
    max_allocated_gib: float | None
    max_reserved_gib: float | None
    current_allocated_gib: float | None
    current_reserved_gib: float | None


@dataclass(frozen=True, slots=True)
class TelemetryEvent:
    """带 schema 版本的不可变事件快照。"""

    schema_version: int
    event_type: str
    timestamp_ms: float
    run_id: str
    metrics: tuple[tuple[str, float], ...] = ()
    payload: tuple[tuple[str, JSONValue], ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or not isinstance(
            self.schema_version, int
        ):
            raise TypeError("schema_version 必须是整数")
        if self.schema_version < 1:
            raise ValueError("schema_version 必须至少为 1")
        require_identifier(self.event_type, "event_type")
        require_identifier(self.run_id, "run_id")
        require_nonnegative(self.timestamp_ms, "timestamp_ms")
        metric_keys = tuple(key for key, _ in self.metrics)
        if any(not key for key in metric_keys) or len(metric_keys) != len(
            set(metric_keys)
        ):
            raise ValueError("metrics 的键必须非空且唯一")
        for key, value in self.metrics:
            require_finite(value, f"metrics[{key}]")
        payload_keys = tuple(key for key, _ in self.payload)
        if any(not key for key in payload_keys) or len(payload_keys) != len(
            set(payload_keys)
        ):
            raise ValueError("payload 的键必须非空且唯一")


__all__ = ["MemoryReport", "TelemetryEvent"]
