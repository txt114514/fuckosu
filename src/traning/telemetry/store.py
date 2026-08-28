"""提供线程安全、可恢复且与 UI 解耦的遥测状态存储。"""

from __future__ import annotations

import json
import math
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from traning.contracts.common import JSONObject, JSONValue
from traning.contracts.telemetry import TelemetryEvent
from traning.evaluation.attribution import (
    EvaluationCoordinateSpace,
    EvaluationTag,
    PrimaryError,
    SequenceEvaluationEvent,
)
from traning.infrastructure.errors import (
    AtomicWriteError,
    IntegrityError,
    SchemaMismatchError,
)
from traning.infrastructure.persistence import atomic_write_jsonl

from .events import (
    CHANNEL_SPECS,
    TELEMETRY_SCHEMA_VERSION,
    EvaluationEvent,
    MetricsEvent,
    PublishableTelemetryEvent,
    ResourceEvent,
    TelemetryChannel,
    event_channel,
)


_METRICS_KEYS = frozenset(
    {
        "schema_version",
        "record_type",
        "timestamp_ms",
        "run_id",
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
)
_RESOURCE_KEYS = frozenset(
    {
        "schema_version",
        "record_type",
        "timestamp_ms",
        "run_id",
        "step",
        "gpu_utilization",
        "vram_used_mb",
        "vram_total_mb",
        "throughput",
    }
)
_EVALUATION_KEYS = frozenset(
    {
        "schema_version",
        "record_type",
        "timestamp_ms",
        "run_id",
        "event_id",
        "sample_id",
        "frame_index",
        "passed",
        "primary_error",
        "error_tags",
        "target_id",
        "click_index",
        "score_version",
        "coordinate_space",
        "coordinate_transform_fingerprint",
        "source_frame_width",
        "source_frame_height",
        "click_x",
        "click_y",
        "spatial_error_osu",
        "temporal_error_ms",
    }
)
_EVENT_KEYS = frozenset(
    {
        "schema_version",
        "record_type",
        "event_type",
        "timestamp_ms",
        "run_id",
        "metrics",
        "payload",
    }
)


@dataclass(frozen=True, slots=True)
class StoreSnapshot:
    """四通道最新值的不可变跨线程视图。"""

    schema_version: int
    metrics: MetricsEvent | None
    resources: ResourceEvent | None
    evaluation: EvaluationEvent | None
    event: TelemetryEvent | None
    metrics_count: int
    resources_count: int
    evaluation_count: int
    event_count: int


@dataclass(frozen=True, slots=True)
class TelemetryHistory:
    """一次锁内复制得到的四通道完整历史。"""

    schema_version: int
    metrics: tuple[MetricsEvent, ...]
    resources: tuple[ResourceEvent, ...]
    evaluations: tuple[EvaluationEvent, ...]
    events: tuple[TelemetryEvent, ...]


class StateStore:
    """把 typed events 追加到固定 JSONL，并提供只读 snapshot/history。

    单个进程内的发布和读取由同一把可重入锁串行化。记录先完整编码到内存，
    再以 ``O_APPEND`` 写入固定通道并 ``fsync``；只有持久化成功后才会更新内存
    历史，因此 UI 永远不会看到尚未耐久化的状态。
    """

    def __init__(
        self,
        directory: Path,
        *,
        schema_version: int = TELEMETRY_SCHEMA_VERSION,
    ) -> None:
        if not isinstance(directory, Path):
            raise TypeError("directory 必须是 pathlib.Path")
        if isinstance(schema_version, bool) or not isinstance(schema_version, int):
            raise TypeError("schema_version 必须是整数")
        if schema_version != TELEMETRY_SCHEMA_VERSION:
            raise ValueError(f"schema_version 必须是 {TELEMETRY_SCHEMA_VERSION}")
        self._directory = directory
        self._schema_version = schema_version
        self._lock = threading.RLock()
        self._paths = {
            spec.channel: directory / spec.filename for spec in CHANNEL_SPECS
        }
        self._metrics: tuple[MetricsEvent, ...] = ()
        self._resources: tuple[ResourceEvent, ...] = ()
        self._evaluations: tuple[EvaluationEvent, ...] = ()
        self._events: tuple[TelemetryEvent, ...] = ()
        self._run_id: str | None = None
        self._initialize_or_recover()

    @property
    def directory(self) -> Path:
        """返回调用方明确传入的目录，不做隐式 fallback。"""

        return self._directory

    def publish(self, event: PublishableTelemetryEvent) -> None:
        """耐久化一个 typed event，并在成功后原子推进内存状态。"""

        channel = event_channel(event)
        schema_version = event.schema_version
        if schema_version != self._schema_version:
            raise ValueError("event.schema_version 与 StateStore 不一致")
        if isinstance(event, TelemetryEvent):
            checked_event: PublishableTelemetryEvent = _copy_contract_event(event)
        else:
            checked_event = event
        record = _encode_event(checked_event)
        with self._lock:
            if self._run_id is not None and checked_event.run_id != self._run_id:
                raise ValueError("event.run_id 与 StateStore 已绑定 run_id 不一致")
            self._append_record(channel, record)
            self._remember(checked_event)
            if self._run_id is None:
                self._run_id = checked_event.run_id

    def snapshot(self) -> StoreSnapshot:
        """复制四通道最新状态；不暴露可变 live state。"""

        with self._lock:
            latest_event = (
                _copy_contract_event(self._events[-1]) if self._events else None
            )
            return StoreSnapshot(
                schema_version=self._schema_version,
                metrics=self._metrics[-1] if self._metrics else None,
                resources=self._resources[-1] if self._resources else None,
                evaluation=self._evaluations[-1] if self._evaluations else None,
                event=latest_event,
                metrics_count=len(self._metrics),
                resources_count=len(self._resources),
                evaluation_count=len(self._evaluations),
                event_count=len(self._events),
            )

    def history(self) -> TelemetryHistory:
        """复制完整历史；canonical evaluation 对象无需复制且保持身份。"""

        with self._lock:
            return TelemetryHistory(
                schema_version=self._schema_version,
                metrics=tuple(self._metrics),
                resources=tuple(self._resources),
                evaluations=tuple(self._evaluations),
                events=tuple(_copy_contract_event(event) for event in self._events),
            )

    def _initialize_or_recover(self) -> None:
        """创建四个新通道，或严格恢复已有的完整通道集合。"""

        try:
            self._directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise AtomicWriteError(
                f"无法创建 telemetry 目录：{self._directory}"
            ) from exc
        if not self._directory.is_dir():
            raise SchemaMismatchError(f"telemetry 路径不是目录：{self._directory}")

        existing = tuple(path.exists() for path in self._paths.values())
        if any(existing) and not all(existing):
            raise IntegrityError("telemetry 四通道文件集合不完整，拒绝静默补建")
        if not any(existing):
            for path in self._paths.values():
                atomic_write_jsonl(path, ())
            return

        for path in self._paths.values():
            if path.is_symlink() or not path.is_file():
                raise IntegrityError(f"telemetry 通道必须是本地普通文件：{path}")
        records = {
            channel: _read_channel(path, channel, self._schema_version)
            for channel, path in self._paths.items()
        }
        metrics = records[TelemetryChannel.METRICS]
        resources = records[TelemetryChannel.RESOURCES]
        evaluations = records[TelemetryChannel.EVALUATION]
        events = records[TelemetryChannel.EVENTS]
        if any(not isinstance(item, MetricsEvent) for item in metrics):
            raise IntegrityError("metrics.jsonl 含有错误事件类型")
        if any(not isinstance(item, ResourceEvent) for item in resources):
            raise IntegrityError("resources.jsonl 含有错误事件类型")
        if any(not isinstance(item, EvaluationEvent) for item in evaluations):
            raise IntegrityError("evaluation.jsonl 含有错误事件类型")
        if any(not isinstance(item, TelemetryEvent) for item in events):
            raise IntegrityError("events.jsonl 含有错误事件类型")
        run_ids = {
            item.run_id
            for channel_records in records.values()
            for item in channel_records
        }
        if len(run_ids) > 1:
            raise IntegrityError("telemetry 四通道含有多个 run_id，拒绝混合恢复")
        self._metrics = cast(tuple[MetricsEvent, ...], metrics)
        self._resources = cast(tuple[ResourceEvent, ...], resources)
        self._evaluations = cast(tuple[EvaluationEvent, ...], evaluations)
        self._events = cast(tuple[TelemetryEvent, ...], events)
        self._run_id = next(iter(run_ids), None)

    def _append_record(
        self,
        channel: TelemetryChannel,
        record: JSONObject,
    ) -> None:
        """以单个完整 UTF-8 JSON 行追加，并同步文件内容。"""

        path = self._paths[channel]
        if path.is_symlink() or not path.is_file():
            raise IntegrityError(f"telemetry 通道不是本地普通文件：{path}")
        try:
            encoded = (
                json.dumps(
                    record,
                    allow_nan=False,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
                + b"\n"
            )
        except (TypeError, ValueError, UnicodeEncodeError) as exc:
            raise SchemaMismatchError("telemetry event 无法编码为严格 JSON") from exc

        descriptor: int | None = None
        try:
            flags = os.O_WRONLY | os.O_APPEND | getattr(os, "O_CLOEXEC", 0)
            descriptor = os.open(path, flags)
            view = memoryview(encoded)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("telemetry append 未取得写入进展")
                view = view[written:]
            os.fsync(descriptor)
        except OSError as exc:
            raise AtomicWriteError(f"telemetry append 失败：{path}") from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)

    def _remember(self, event: PublishableTelemetryEvent) -> None:
        """按封闭联合类型推进唯一通道的 immutable tuple。"""

        if isinstance(event, MetricsEvent):
            self._metrics = (*self._metrics, event)
        elif isinstance(event, ResourceEvent):
            self._resources = (*self._resources, event)
        elif isinstance(event, EvaluationEvent):
            self._evaluations = (*self._evaluations, event)
        elif isinstance(event, TelemetryEvent):
            self._events = (*self._events, event)
        else:  # pragma: no cover - publish 的 event_channel 已封闭联合
            raise TypeError("无法记录未知 telemetry event")


def _encode_event(event: PublishableTelemetryEvent) -> JSONObject:
    """将 typed event 无损投影到其版本化 JSON boundary。"""

    if isinstance(event, MetricsEvent):
        return {
            "schema_version": event.schema_version,
            "record_type": TelemetryChannel.METRICS.value,
            "timestamp_ms": event.timestamp_ms,
            "run_id": event.run_id,
            "step": event.step,
            "loss": event.loss,
            "perception_recall": event.perception_recall,
            "tracking_id_switches": event.tracking_id_switches,
            "outcome_nll": event.outcome_nll,
            "outcome_brier": event.outcome_brier,
            "outcome_ece": event.outcome_ece,
            "expected_score_error": event.expected_score_error,
            "decision_utility": event.decision_utility,
            "wait_click_ratio": event.wait_click_ratio,
            "score": event.score,
        }
    if isinstance(event, ResourceEvent):
        return {
            "schema_version": event.schema_version,
            "record_type": TelemetryChannel.RESOURCES.value,
            "timestamp_ms": event.timestamp_ms,
            "run_id": event.run_id,
            "step": event.step,
            "gpu_utilization": event.gpu_utilization,
            "vram_used_mb": event.vram_used_mb,
            "vram_total_mb": event.vram_total_mb,
            "throughput": event.throughput,
        }
    if isinstance(event, EvaluationEvent):
        canonical = event.event
        return {
            "schema_version": event.schema_version,
            "record_type": TelemetryChannel.EVALUATION.value,
            "timestamp_ms": event.timestamp_ms,
            "run_id": event.run_id,
            "event_id": canonical.event_id,
            "sample_id": canonical.sample_id,
            "frame_index": canonical.frame_index,
            "passed": canonical.passed,
            "primary_error": canonical.primary_error.value,
            "error_tags": [tag.value for tag in canonical.error_tags],
            "target_id": canonical.target_id,
            "click_index": canonical.click_index,
            "score_version": canonical.score_version,
            "coordinate_space": canonical.coordinate_space.value,
            "coordinate_transform_fingerprint": (
                canonical.coordinate_transform_fingerprint
            ),
            "source_frame_width": canonical.source_frame_width,
            "source_frame_height": canonical.source_frame_height,
            "click_x": canonical.click_x,
            "click_y": canonical.click_y,
            "spatial_error_osu": canonical.spatial_error_osu,
            "temporal_error_ms": canonical.temporal_error_ms,
        }
    if isinstance(event, TelemetryEvent):
        return {
            "schema_version": event.schema_version,
            "record_type": TelemetryChannel.EVENTS.value,
            "event_type": event.event_type,
            "timestamp_ms": event.timestamp_ms,
            "run_id": event.run_id,
            "metrics": {key: value for key, value in event.metrics},
            "payload": {key: value for key, value in event.payload},
        }
    raise TypeError("无法编码未知 telemetry event")


def _read_channel(
    path: Path,
    channel: TelemetryChannel,
    schema_version: int,
) -> tuple[PublishableTelemetryEvent, ...]:
    """逐行严格解码一个通道，任何损坏都显式终止恢复。"""

    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise IntegrityError(f"无法读取 telemetry 通道：{path}") from exc
    if payload and not payload.endswith(b"\n"):
        raise IntegrityError(f"telemetry JSONL 尾行不完整：{path}")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise IntegrityError(f"telemetry JSONL 不是 UTF-8：{path}") from exc

    decoded: list[PublishableTelemetryEvent] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise IntegrityError(f"telemetry JSONL 含空行：{path}:{line_number}")
        record = _decode_json_object(line, path, line_number)
        decoded.append(_decode_record(record, channel, schema_version))
    return tuple(decoded)


def _decode_json_object(line: str, path: Path, line_number: int) -> dict[str, object]:
    """拒绝重复键、NaN/Infinity 和非 object JSON 根节点。"""

    try:
        decoded: object = json.loads(
            line,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except IntegrityError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise IntegrityError(f"telemetry JSON 损坏：{path}:{line_number}") from exc
    if not isinstance(decoded, dict):
        raise SchemaMismatchError(
            f"telemetry JSON 根节点必须是 object：{path}:{line_number}"
        )
    return cast(dict[str, object], decoded)


def _decode_record(
    record: dict[str, object],
    expected_channel: TelemetryChannel,
    schema_version: int,
) -> PublishableTelemetryEvent:
    """按目标通道解码，禁止 record_type 串台。"""

    _require_exact_value(record, "record_type", expected_channel.value)
    _require_exact_value(record, "schema_version", schema_version)
    try:
        if expected_channel is TelemetryChannel.METRICS:
            _require_keys(record, _METRICS_KEYS)
            return MetricsEvent(
                schema_version=_integer(record, "schema_version"),
                timestamp_ms=_number(record, "timestamp_ms"),
                run_id=_string(record, "run_id"),
                step=_integer(record, "step"),
                loss=_number(record, "loss"),
                perception_recall=_number(record, "perception_recall"),
                tracking_id_switches=_integer(record, "tracking_id_switches"),
                outcome_nll=_number(record, "outcome_nll"),
                outcome_brier=_number(record, "outcome_brier"),
                outcome_ece=_number(record, "outcome_ece"),
                expected_score_error=_number(record, "expected_score_error"),
                decision_utility=_number(record, "decision_utility"),
                wait_click_ratio=_number(record, "wait_click_ratio"),
                score=_number(record, "score"),
            )
        if expected_channel is TelemetryChannel.RESOURCES:
            _require_keys(record, _RESOURCE_KEYS)
            return ResourceEvent(
                schema_version=_integer(record, "schema_version"),
                timestamp_ms=_number(record, "timestamp_ms"),
                run_id=_string(record, "run_id"),
                step=_integer(record, "step"),
                gpu_utilization=_number(record, "gpu_utilization"),
                vram_used_mb=_number(record, "vram_used_mb"),
                vram_total_mb=_number(record, "vram_total_mb"),
                throughput=_number(record, "throughput"),
            )
        if expected_channel is TelemetryChannel.EVALUATION:
            _require_keys(record, _EVALUATION_KEYS)
            canonical = SequenceEvaluationEvent(
                event_id=_string(record, "event_id"),
                sample_id=_string(record, "sample_id"),
                frame_index=_integer(record, "frame_index"),
                passed=_boolean(record, "passed"),
                primary_error=PrimaryError(_string(record, "primary_error")),
                error_tags=tuple(
                    EvaluationTag(value) for value in _string_list(record, "error_tags")
                ),
                target_id=_optional_string(record, "target_id"),
                click_index=_optional_integer(record, "click_index"),
                score_version=_string(record, "score_version"),
                coordinate_space=EvaluationCoordinateSpace(
                    _string(record, "coordinate_space")
                ),
                coordinate_transform_fingerprint=_optional_string(
                    record, "coordinate_transform_fingerprint"
                ),
                source_frame_width=_optional_integer(record, "source_frame_width"),
                source_frame_height=_optional_integer(record, "source_frame_height"),
                click_x=_optional_number(record, "click_x"),
                click_y=_optional_number(record, "click_y"),
                spatial_error_osu=_optional_number(record, "spatial_error_osu"),
                temporal_error_ms=_optional_number(record, "temporal_error_ms"),
            )
            return EvaluationEvent(
                schema_version=_integer(record, "schema_version"),
                timestamp_ms=_number(record, "timestamp_ms"),
                run_id=_string(record, "run_id"),
                event=canonical,
            )
        _require_keys(record, _EVENT_KEYS)
        metric_object = _object(record, "metrics")
        payload_object = _object(record, "payload")
        return TelemetryEvent(
            schema_version=_integer(record, "schema_version"),
            event_type=_string(record, "event_type"),
            timestamp_ms=_number(record, "timestamp_ms"),
            run_id=_string(record, "run_id"),
            metrics=tuple(
                (key, _standalone_number(value, f"metrics[{key}]"))
                for key, value in sorted(metric_object.items())
            ),
            payload=tuple(
                (key, _copy_json_value(value, f"payload[{key}]"))
                for key, value in sorted(payload_object.items())
            ),
        )
    except (TypeError, ValueError) as exc:
        raise SchemaMismatchError(
            f"{expected_channel.value}.jsonl 记录不满足 typed schema"
        ) from exc


def _copy_contract_event(event: TelemetryEvent) -> TelemetryEvent:
    """深复制 JSON payload，避免可变容器成为跨线程 live state。"""

    copied_payload = tuple(
        (key, _copy_json_value(value, f"payload[{key}]"))
        for key, value in sorted(event.payload)
    )
    return TelemetryEvent(
        schema_version=event.schema_version,
        event_type=event.event_type,
        timestamp_ms=event.timestamp_ms,
        run_id=event.run_id,
        metrics=tuple(sorted(event.metrics)),
        payload=copied_payload,
    )


def _copy_json_value(value: object, context: str) -> JSONValue:
    """校验并深复制 JSON 值，同时拒绝非有限数和非字符串键。"""

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SchemaMismatchError(f"{context} 含有非有限浮点数")
        return value
    if isinstance(value, list):
        return [
            _copy_json_value(item, f"{context}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, dict):
        copied: dict[str, JSONValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise SchemaMismatchError(f"{context} 含有非字符串 key")
            copied[key] = _copy_json_value(item, f"{context}.{key}")
        return copied
    raise SchemaMismatchError(f"{context} 含有非 JSON 类型：{type(value).__name__}")


def _require_keys(record: dict[str, object], expected: frozenset[str]) -> None:
    """要求字段集合完全相等，拒绝缺字段和未知字段。"""

    actual = frozenset(record)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise SchemaMismatchError(
            f"telemetry 字段不匹配：missing={missing}, unknown={unknown}"
        )


def _require_exact_value(record: dict[str, object], key: str, expected: object) -> None:
    if key not in record or record[key] != expected:
        raise SchemaMismatchError(f"telemetry {key} 必须是 {expected!r}")


def _string(record: dict[str, object], key: str) -> str:
    value = record[key]
    if not isinstance(value, str):
        raise SchemaMismatchError(f"{key} 必须是字符串")
    return value


def _integer(record: dict[str, object], key: str) -> int:
    value = record[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise SchemaMismatchError(f"{key} 必须是整数")
    return value


def _number(record: dict[str, object], key: str) -> float:
    return _standalone_number(record[key], key)


def _standalone_number(value: object, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaMismatchError(f"{key} 必须是数值")
    converted = float(value)
    if not math.isfinite(converted):
        raise SchemaMismatchError(f"{key} 必须是有限数值")
    return converted


def _boolean(record: dict[str, object], key: str) -> bool:
    value = record[key]
    if not isinstance(value, bool):
        raise SchemaMismatchError(f"{key} 必须是 bool")
    return value


def _optional_string(record: dict[str, object], key: str) -> str | None:
    value = record[key]
    if value is None or isinstance(value, str):
        return value
    raise SchemaMismatchError(f"{key} 必须是字符串或 null")


def _optional_integer(record: dict[str, object], key: str) -> int | None:
    value = record[key]
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise SchemaMismatchError(f"{key} 必须是整数或 null")
    return value


def _optional_number(record: dict[str, object], key: str) -> float | None:
    """读取可空有限数值，并继续拒绝 bool 冒充数字。"""

    value = record[key]
    if value is None:
        return None
    return _standalone_number(value, key)


def _string_list(record: dict[str, object], key: str) -> tuple[str, ...]:
    value = record[key]
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise SchemaMismatchError(f"{key} 必须是字符串数组")
    return tuple(cast(list[str], value))


def _object(record: dict[str, object], key: str) -> dict[str, object]:
    value = record[key]
    if not isinstance(value, dict) or any(not isinstance(item, str) for item in value):
        raise SchemaMismatchError(f"{key} 必须是字符串键 object")
    return cast(dict[str, object], value)


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """object_pairs_hook：在 JSON decoder 边界拒绝重复键。"""

    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise IntegrityError(f"telemetry JSON 含重复键：{key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise IntegrityError(f"telemetry JSON 含非标准数值常量：{value}")


__all__ = (
    "StateStore",
    "StoreSnapshot",
    "TelemetryHistory",
)
