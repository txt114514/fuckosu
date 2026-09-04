"""把 canonical sequence score 投影为跨阶段共享的 typed 归因事件。"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from enum import Enum

from .scoring import SCORE_VERSION
from traning.state.common import require_transform_fingerprint

from .sequence import FrameSequenceScore, SequenceScore


class PrimaryError(str, Enum):
    """事件的唯一主要错误域。"""

    NONE = "none"
    SPATIAL = "spatial"
    TEMPORAL = "temporal"
    DECISION = "decision"


class EvaluationTag(str, Enum):
    """sequence scorer 的 canonical 标签及未解析目标标签。"""

    BETTER_SCORE_AFTER_RESOLUTION = "better_score_after_resolution"
    DUPLICATE_AFTER_HIT = "duplicate_after_hit"
    EARLY_CLICK = "early_click"
    FREQUENCY_LIMITED = "frequency_limited"
    HEAD_SPATIAL_MISS = "head_spatial_miss"
    LATE_CLICK = "late_click"
    NO_ACTIVE_TARGET = "no_active_target"
    SLIDER_PATH_MISS = "slider_path_miss"
    SPATIAL_MISS = "spatial_miss"
    UNRESOLVED_TARGET = "unresolved_target"


class EvaluationCoordinateSpace(str, Enum):
    """事件中 click_x/click_y 所属的显式坐标空间。"""

    CANONICAL_OSU = "canonical_osu"
    CALIBRATED_FRAME = "calibrated_frame"


_EVENT_ID_PATTERN = re.compile(r"sequence-event-[0-9a-f]{64}")


def _require_identifier(name: str, value: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} 必须是字符串")
    if not value or value != value.strip():
        raise ValueError(f"{name} 不得为空且不得有首尾空格")


@dataclass(frozen=True, slots=True)
class SequenceEvaluationEvent:
    """Phase 9/10 直接消费的单一 typed 归因事件。"""

    event_id: str
    sample_id: str
    frame_index: int
    passed: bool
    primary_error: PrimaryError
    error_tags: tuple[EvaluationTag, ...]
    target_id: str | None
    click_index: int | None
    score_version: str = SCORE_VERSION
    coordinate_space: EvaluationCoordinateSpace = (
        EvaluationCoordinateSpace.CANONICAL_OSU
    )
    coordinate_transform_fingerprint: str | None = None
    source_frame_width: int | None = None
    source_frame_height: int | None = None
    click_x: float | None = None
    click_y: float | None = None
    spatial_error_osu: float | None = None
    temporal_error_ms: float | None = None

    def __post_init__(self) -> None:
        _require_identifier("event_id", self.event_id)
        if _EVENT_ID_PATTERN.fullmatch(self.event_id) is None:
            raise ValueError("event_id 必须使用 canonical sequence-event SHA-256 格式")
        _require_identifier("sample_id", self.sample_id)
        if isinstance(self.frame_index, bool) or not isinstance(self.frame_index, int):
            raise TypeError("frame_index 必须是整数")
        if self.frame_index < 0:
            raise ValueError("frame_index 不得为负数")
        if not isinstance(self.passed, bool):
            raise TypeError("passed 必须是 bool")
        if not isinstance(self.primary_error, PrimaryError):
            raise TypeError("primary_error 必须是 PrimaryError")
        if not isinstance(self.error_tags, tuple) or any(
            not isinstance(tag, EvaluationTag) for tag in self.error_tags
        ):
            raise TypeError("error_tags 必须是 EvaluationTag 元组")
        if len(self.error_tags) != len(set(self.error_tags)):
            raise ValueError("error_tags 不得重复")
        if self.target_id is not None:
            _require_identifier("target_id", self.target_id)
        if self.click_index is not None:
            if isinstance(self.click_index, bool) or not isinstance(
                self.click_index, int
            ):
                raise TypeError("click_index 必须是整数或 None")
            if self.click_index < 0:
                raise ValueError("click_index 不得为负数")
        if self.score_version != SCORE_VERSION:
            raise ValueError(f"score_version 必须是 {SCORE_VERSION}")
        if not isinstance(self.coordinate_space, EvaluationCoordinateSpace):
            raise TypeError("coordinate_space 必须是 EvaluationCoordinateSpace")
        provenance = (
            self.coordinate_transform_fingerprint,
            self.source_frame_width,
            self.source_frame_height,
        )
        if self.coordinate_space is EvaluationCoordinateSpace.CALIBRATED_FRAME:
            if any(value is None for value in provenance):
                raise ValueError("calibrated_frame 事件必须携带完整坐标变换来源")
            require_transform_fingerprint(self.coordinate_transform_fingerprint)
            for field_name, value in (
                ("source_frame_width", self.source_frame_width),
                ("source_frame_height", self.source_frame_height),
            ):
                if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                    raise ValueError(f"{field_name} 必须是正整数")
        elif any(value is not None for value in provenance):
            raise ValueError("canonical_osu 事件不得伪装携带 frame 标定来源")

        if (self.click_x is None) != (self.click_y is None):
            raise ValueError("click_x 与 click_y 必须同时存在或同时为空")
        for field_name, value in (
            ("click_x", self.click_x),
            ("click_y", self.click_y),
            ("spatial_error_osu", self.spatial_error_osu),
            ("temporal_error_ms", self.temporal_error_ms),
        ):
            if value is not None:
                if isinstance(value, bool) or not isinstance(value, int | float):
                    raise TypeError(f"{field_name} 必须是数值或 None")
                if not math.isfinite(float(value)):
                    raise ValueError(f"{field_name} 必须是有限数值")
        if self.spatial_error_osu is not None and self.spatial_error_osu < 0.0:
            raise ValueError("spatial_error_osu 不得为负数")
        if (
            self.coordinate_space is EvaluationCoordinateSpace.CALIBRATED_FRAME
            and self.click_x is not None
            and (
                not 0.0 <= self.click_x < self.source_frame_width
                or not 0.0 <= self.click_y < self.source_frame_height
            )
        ):
            raise ValueError("calibrated_frame 点击超出声明的原帧像素边界")

        unresolved = (EvaluationTag.UNRESOLVED_TARGET,)
        if self.click_index is None:
            if (
                self.passed
                or self.primary_error is not PrimaryError.DECISION
                or self.error_tags != unresolved
                or self.target_id is None
                or self.click_x is not None
                or self.spatial_error_osu is not None
                or self.temporal_error_ms is not None
            ):
                raise ValueError("未解析目标事件必须是唯一的 decision/unresolved 失败")
            return
        if EvaluationTag.UNRESOLVED_TARGET in self.error_tags:
            raise ValueError("click 事件不得携带 UNRESOLVED_TARGET")
        if self.click_x is None:
            raise ValueError("click 事件必须保留其输入坐标")
        if self.passed:
            if (
                self.primary_error is not PrimaryError.NONE
                or self.error_tags
                or self.target_id is None
            ):
                raise ValueError("通过事件必须无错误且引用目标")
        elif self.primary_error is PrimaryError.NONE or not self.error_tags:
            raise ValueError("失败 click 事件必须携带主要错误域和至少一个标签")


def _canonical_event_id(parts: tuple[str, ...]) -> str:
    """对 UTF-8 字节做 length-prefix 编码，避免字段连接歧义。"""

    payload = bytearray()
    for part in parts:
        encoded = part.encode("utf-8")
        payload.extend(len(encoded).to_bytes(8, byteorder="big", signed=False))
        payload.extend(encoded)
    return f"sequence-event-{hashlib.sha256(payload).hexdigest()}"


def build_sequence_evaluation_events(
    sample_id: str,
    frame_index: int,
    result: SequenceScore | FrameSequenceScore,
) -> tuple[SequenceEvaluationEvent, ...]:
    """确定性投影 click 评分，并为每个未解析目标追加唯一事件。"""

    _require_identifier("sample_id", sample_id)
    if isinstance(frame_index, bool) or not isinstance(frame_index, int):
        raise TypeError("frame_index 必须是整数")
    if frame_index < 0:
        raise ValueError("frame_index 不得为负数")
    if not isinstance(result, SequenceScore | FrameSequenceScore):
        raise TypeError("result 必须是 SequenceScore 或 FrameSequenceScore")

    if isinstance(result, FrameSequenceScore):
        canonical_result = result.result
        coordinate_space = EvaluationCoordinateSpace.CALIBRATED_FRAME
        transform_fingerprint = result.transform_fingerprint
        source_frame_width = result.source_frame_width
        source_frame_height = result.source_frame_height
        unresolved_frame_indices = dict(result.unresolved_target_frame_indices)
    else:
        canonical_result = result
        coordinate_space = EvaluationCoordinateSpace.CANONICAL_OSU
        transform_fingerprint = None
        source_frame_width = None
        source_frame_height = None
        unresolved_frame_indices = {}

    events: list[SequenceEvaluationEvent] = []
    for click in sorted(canonical_result.clicks, key=lambda item: item.click_index):
        primary_error = PrimaryError(click.primary_error)
        error_tags = tuple(EvaluationTag(tag) for tag in click.error_tags)
        passed = click.status == "hit"
        target_id = click.target_id
        if isinstance(result, FrameSequenceScore):
            source_click = result.frame_clicks[click.click_index]
            click_x = source_click.position.x
            click_y = source_click.position.y
            event_frame_index = (
                frame_index
                if source_click.frame_index is None
                else source_click.frame_index
            )
        else:
            click_x = click.click.x
            click_y = click.click.y
            event_frame_index = frame_index
        event_id = _canonical_event_id(
            (
                SCORE_VERSION,
                sample_id,
                str(event_frame_index),
                "click",
                str(click.click_index),
                click.status,
                target_id or "",
                coordinate_space.value,
                transform_fingerprint or "",
            )
        )
        events.append(
            SequenceEvaluationEvent(
                event_id=event_id,
                sample_id=sample_id,
                frame_index=event_frame_index,
                passed=passed,
                primary_error=primary_error,
                error_tags=error_tags,
                target_id=target_id,
                click_index=click.click_index,
                coordinate_space=coordinate_space,
                coordinate_transform_fingerprint=transform_fingerprint,
                source_frame_width=source_frame_width,
                source_frame_height=source_frame_height,
                click_x=click_x,
                click_y=click_y,
                spatial_error_osu=click.spatial_error,
                temporal_error_ms=click.temporal_error_ms,
            )
        )

    unresolved_ids = tuple(sorted(canonical_result.unresolved_target_ids))
    if len(unresolved_ids) != len(set(unresolved_ids)):
        raise ValueError("unresolved_target_ids 不得重复")
    for target_id in unresolved_ids:
        _require_identifier("unresolved target_id", target_id)
        event_frame_index = unresolved_frame_indices.get(target_id, frame_index)
        event_id = _canonical_event_id(
            (
                SCORE_VERSION,
                sample_id,
                str(event_frame_index),
                "unresolved",
                target_id,
                coordinate_space.value,
                transform_fingerprint or "",
            )
        )
        events.append(
            SequenceEvaluationEvent(
                event_id=event_id,
                sample_id=sample_id,
                frame_index=event_frame_index,
                passed=False,
                primary_error=PrimaryError.DECISION,
                error_tags=(EvaluationTag.UNRESOLVED_TARGET,),
                target_id=target_id,
                click_index=None,
                coordinate_space=coordinate_space,
                coordinate_transform_fingerprint=transform_fingerprint,
                source_frame_width=source_frame_width,
                source_frame_height=source_frame_height,
            )
        )
    event_ids = tuple(event.event_id for event in events)
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("canonical event_id 发生重复")
    return tuple(events)


__all__ = (
    "EvaluationTag",
    "EvaluationCoordinateSpace",
    "PrimaryError",
    "SequenceEvaluationEvent",
    "build_sequence_evaluation_events",
)
