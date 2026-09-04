"""按 canonical 单帧事件发布 production PNG gallery 与完整 manifest。

旧 exporter 先用整段序列的 ``all(frame.passed)`` 决定目录，导致一个已通过
的单帧也会被其他帧拖进 ``failed``。本模块只消费 scorer 已发布的
``SequenceEvaluationEvent``：序列名仅用于样本身份，passed/failed 与错误域
都从当前帧事件导出，绝不重新评分或根据覆盖层猜测。
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path, PurePosixPath
import re

from traning.state import RuntimeFrame
from traning.core.data import FrameCoordinateTransform
from traning.core.evaluation import (
    FrameSequenceScore,
    PrimaryError,
    SequenceEvaluationEvent,
    TargetObject,
)
from traning.lib.infrastructure import atomic_write_json, sha256_file
from traning.lib.visualization import build_gallery_frame_overlay, render_gallery_png


PRODUCTION_GALLERY_SCHEMA_VERSION = 1
"""production gallery manifest 的唯一活动 schema。"""

PRODUCTION_GALLERY_MANIFEST_FILENAME = "manifest.json"
"""全部 PNG 成功发布后最后提交的 manifest 文件名。"""

_SAFE_COMPONENT_PATTERN = re.compile(r"[^0-9A-Za-z._-]+")


@dataclass(frozen=True, slots=True)
class ProductionGalleryRecord:
    """一张逐帧 PNG 的 canonical 分类、来源和完整性记录。"""

    sequence_id: str
    frame_index: int
    passed: bool
    primary_errors: tuple[PrimaryError, ...]
    event_ids: tuple[str, ...]
    relative_png_path: str
    png_sha256: str
    source_frame_width: int
    source_frame_height: int
    transform_fingerprint: str

    def __post_init__(self) -> None:
        """拒绝组级分类、路径穿越和跨坐标系记录。"""

        if not isinstance(self.sequence_id, str) or not self.sequence_id.strip():
            raise ValueError("sequence_id 必须是非空字符串")
        if self.sequence_id != self.sequence_id.strip():
            raise ValueError("sequence_id 不得有首尾空格")
        if (
            isinstance(self.frame_index, bool)
            or not isinstance(self.frame_index, int)
            or self.frame_index < 0
        ):
            raise ValueError("frame_index 必须是非负整数")
        if not isinstance(self.passed, bool):
            raise TypeError("passed 必须是 bool")
        if not isinstance(self.primary_errors, tuple) or any(
            not isinstance(item, PrimaryError) for item in self.primary_errors
        ):
            raise TypeError("primary_errors 必须是 PrimaryError 元组")
        if self.primary_errors != tuple(
            sorted(set(self.primary_errors), key=lambda item: item.value)
        ):
            raise ValueError("primary_errors 必须去重并稳定排序")
        if self.passed != (self.primary_errors == ()):
            raise ValueError("passed 帧必须无错误，failed 帧必须有 canonical 错误")
        if PrimaryError.NONE in self.primary_errors:
            raise ValueError("primary_errors 不得把 NONE 当成失败模块")
        if (
            not isinstance(self.event_ids, tuple)
            or not self.event_ids
            or any(not isinstance(item, str) or not item for item in self.event_ids)
        ):
            raise ValueError("event_ids 必须是非空字符串元组")
        if self.event_ids != tuple(sorted(set(self.event_ids))):
            raise ValueError("event_ids 必须去重并稳定排序")
        path = PurePosixPath(self.relative_png_path)
        if (
            not self.relative_png_path
            or path.is_absolute()
            or ".." in path.parts
            or path.suffix.lower() != ".png"
        ):
            raise ValueError("relative_png_path 必须是安全的相对 PNG 路径")
        if (
            not isinstance(self.png_sha256, str)
            or len(self.png_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.png_sha256)
        ):
            raise ValueError("png_sha256 必须是小写 SHA-256")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in (self.source_frame_width, self.source_frame_height)
        ):
            raise ValueError("source frame 尺寸必须是正整数")
        if not isinstance(
            self.transform_fingerprint, str
        ) or not self.transform_fingerprint.startswith("transform-"):
            raise ValueError("transform_fingerprint 必须是有效坐标指纹")


def render_production_sequence_gallery(
    directory: Path,
    *,
    sequence_id: str,
    frames: tuple[RuntimeFrame, ...],
    targets: tuple[TargetObject, ...],
    score: FrameSequenceScore,
    events: tuple[SequenceEvaluationEvent, ...],
    coordinate_transform: FrameCoordinateTransform,
) -> tuple[ProductionGalleryRecord, ...]:
    """按事件真实帧渲染 PNG；其他帧失败不会污染当前帧目录。"""

    if not isinstance(directory, Path):
        raise TypeError("directory 必须是 pathlib.Path")
    if not isinstance(sequence_id, str) or not sequence_id.strip():
        raise ValueError("sequence_id 必须是非空字符串")
    if sequence_id != sequence_id.strip():
        raise ValueError("sequence_id 不得有首尾空格")
    if not isinstance(frames, tuple) or any(
        not isinstance(frame, RuntimeFrame) for frame in frames
    ):
        raise TypeError("frames 必须是 RuntimeFrame 元组")
    if not isinstance(targets, tuple) or any(
        not isinstance(target, TargetObject) for target in targets
    ):
        raise TypeError("targets 必须是 TargetObject 元组")
    if not isinstance(score, FrameSequenceScore):
        raise TypeError("score 必须是 FrameSequenceScore")
    if not isinstance(events, tuple) or any(
        not isinstance(event, SequenceEvaluationEvent) for event in events
    ):
        raise TypeError("events 必须是 SequenceEvaluationEvent 元组")
    if not isinstance(coordinate_transform, FrameCoordinateTransform):
        raise TypeError("coordinate_transform 必须是 FrameCoordinateTransform")
    if any(event.sample_id != sequence_id for event in events):
        raise ValueError("gallery events 必须属于当前 sequence_id")

    frames_by_index = {frame.frame_index: frame for frame in frames}
    if len(frames_by_index) != len(frames):
        raise ValueError("frames 不得重复 frame_index")
    events_by_frame: dict[int, list[SequenceEvaluationEvent]] = {}
    for event in events:
        events_by_frame.setdefault(event.frame_index, []).append(event)
    missing_frames = tuple(sorted(set(events_by_frame).difference(frames_by_index)))
    if missing_frames:
        raise ValueError(f"gallery event 缺少来源 RuntimeFrame: {missing_frames}")

    records: list[ProductionGalleryRecord] = []
    safe_sequence = _safe_sequence_component(sequence_id)
    for frame_index in sorted(events_by_frame):
        frame_events = tuple(
            sorted(events_by_frame[frame_index], key=lambda event: event.event_id)
        )
        passed = all(event.passed for event in frame_events)
        primary_errors = tuple(
            sorted(
                {event.primary_error for event in frame_events if not event.passed},
                key=lambda item: item.value,
            )
        )
        if passed != (primary_errors == ()):
            raise ValueError("canonical event 的 passed 与 primary_error 不一致")
        module_name = (
            "none"
            if passed
            else primary_errors[0].value
            if len(primary_errors) == 1
            else "mixed"
        )
        status_name = "passed" if passed else "failed"
        relative_path = PurePosixPath(
            status_name,
            module_name,
            safe_sequence,
            f"frame_{frame_index:06d}.png",
        )
        overlay = build_gallery_frame_overlay(
            targets,
            score,
            events,
            coordinate_transform,
            frame_index=frame_index,
        )
        output_path = directory.joinpath(*relative_path.parts)
        render_gallery_png(frames_by_index[frame_index], overlay, output_path)
        records.append(
            ProductionGalleryRecord(
                sequence_id=sequence_id,
                frame_index=frame_index,
                passed=passed,
                primary_errors=primary_errors,
                event_ids=tuple(event.event_id for event in frame_events),
                relative_png_path=relative_path.as_posix(),
                png_sha256=sha256_file(output_path),
                source_frame_width=score.source_frame_width,
                source_frame_height=score.source_frame_height,
                transform_fingerprint=score.transform_fingerprint,
            )
        )
    return tuple(records)


def publish_production_gallery_manifest(
    directory: Path,
    *,
    run_id: str,
    dataset_id: str,
    trial_index: int,
    transform_fingerprint: str,
    records: tuple[ProductionGalleryRecord, ...],
) -> Path:
    """校验全部 PNG 后最后原子提交 production gallery manifest。"""

    if not isinstance(directory, Path):
        raise TypeError("directory 必须是 pathlib.Path")
    for name, value in (("run_id", run_id), ("dataset_id", dataset_id)):
        if not isinstance(value, str) or not value or value != value.strip():
            raise ValueError(f"{name} 必须非空且无首尾空格")
    if (
        isinstance(trial_index, bool)
        or not isinstance(trial_index, int)
        or trial_index < 0
    ):
        raise ValueError("trial_index 必须是非负整数")
    if not isinstance(
        transform_fingerprint, str
    ) or not transform_fingerprint.startswith("transform-"):
        raise ValueError("transform_fingerprint 必须是有效坐标指纹")
    if not isinstance(records, tuple) or any(
        not isinstance(record, ProductionGalleryRecord) for record in records
    ):
        raise TypeError("records 必须是 ProductionGalleryRecord 元组")
    keys = tuple((record.sequence_id, record.frame_index) for record in records)
    if keys != tuple(sorted(set(keys))):
        raise ValueError("gallery records 必须按 sequence/frame 去重并稳定排序")
    if any(record.transform_fingerprint != transform_fingerprint for record in records):
        raise ValueError("gallery records 与 manifest 坐标指纹不一致")
    for record in records:
        path = directory.joinpath(*PurePosixPath(record.relative_png_path).parts)
        if not path.is_file() or sha256_file(path) != record.png_sha256:
            raise ValueError(
                f"gallery PNG 缺失或摘要不匹配: {record.relative_png_path}"
            )

    manifest_path = directory / PRODUCTION_GALLERY_MANIFEST_FILENAME
    atomic_write_json(
        manifest_path,
        {
            "schema_version": PRODUCTION_GALLERY_SCHEMA_VERSION,
            "artifact_type": "traning_production_gallery",
            "run_id": run_id,
            "dataset_id": dataset_id,
            "trial_index": trial_index,
            "transform_fingerprint": transform_fingerprint,
            "record_count": len(records),
            "records": [
                {
                    "sequence_id": record.sequence_id,
                    "frame_index": record.frame_index,
                    "passed": record.passed,
                    "primary_errors": [item.value for item in record.primary_errors],
                    "event_ids": list(record.event_ids),
                    "relative_png_path": record.relative_png_path,
                    "png_sha256": record.png_sha256,
                    "source_frame_width": record.source_frame_width,
                    "source_frame_height": record.source_frame_height,
                    "transform_fingerprint": record.transform_fingerprint,
                }
                for record in records
            ],
        },
    )
    return manifest_path


def _safe_sequence_component(sequence_id: str) -> str:
    """生成可读且带摘要的单一路径组件，阻止斜杠与 ``..`` 穿越。"""

    normalized = _SAFE_COMPONENT_PATTERN.sub("_", sequence_id).strip("._-")
    prefix = normalized[:80] or "sequence"
    digest = hashlib.sha256(sequence_id.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}__{digest}"


__all__ = (
    "PRODUCTION_GALLERY_MANIFEST_FILENAME",
    "PRODUCTION_GALLERY_SCHEMA_VERSION",
    "ProductionGalleryRecord",
    "publish_production_gallery_manifest",
    "render_production_sequence_gallery",
)
