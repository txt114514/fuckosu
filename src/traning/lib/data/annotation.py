"""segment 标注的数据模型、文件校验与按时间可见物件筛选。"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator


class HitObjectAnnotation(BaseModel):
    """单个 osu! 物件在 segment 相对时间轴上的标注。"""

    model_config = ConfigDict(extra="allow")

    type: str
    start_ms: int
    end_ms: int
    x: float | None = None
    y: float | None = None
    path: tuple[tuple[float, float], ...] = ()
    repeats: int = 1
    curve_type: str = "L"
    pixel_length: float | None = None
    source_index: int | None = None

    @field_validator("end_ms")
    @classmethod
    def _valid_end(cls, value: int, info: ValidationInfo) -> int:
        start = info.data.get("start_ms")
        if start is not None and value < start:
            raise ValueError("end_ms must not be earlier than start_ms")
        return value


class DifficultyAnnotation(BaseModel):
    """生成监督目标所需的难度派生参数。"""

    model_config = ConfigDict(extra="allow")

    approach_preempt_ms: float
    circle_radius_osu_pixels: float


class SourceAnnotation(BaseModel):
    """segment 对应谱面与原始裁剪区间的来源信息。"""

    model_config = ConfigDict(extra="allow")

    folder_name: str
    osu_filename: str
    clip_start_ms: int
    clip_end_ms: int

    @field_validator("clip_end_ms")
    @classmethod
    def _valid_clip_end(cls, value: int, info: ValidationInfo) -> int:
        start = info.data.get("clip_start_ms")
        if start is not None and value <= start:
            raise ValueError("clip_end_ms must be later than clip_start_ms")
        return value


class SegmentAnnotation(BaseModel):
    """一个视频 segment 的版本化完整训练标注。"""

    model_config = ConfigDict(extra="allow")

    schema_version: int
    segment_id: str
    dataset_dimension: str
    category: str
    difficulty: DifficultyAnnotation
    source: SourceAnnotation
    hit_objects: tuple[HitObjectAnnotation, ...] = Field(default_factory=tuple)

    @property
    def duration_ms(self) -> int:
        """返回 segment 在源时间轴上的持续毫秒数。"""

        return self.source.clip_end_ms - self.source.clip_start_ms


def load_annotation(path: Path) -> SegmentAnnotation:
    """从 JSON 文件读取并严格校验一个 segment 标注。"""

    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"failed to read annotation: {path}") from error
    return SegmentAnnotation.model_validate(raw)


def visible_hit_objects(
    annotation: SegmentAnnotation,
    timestamp_ms: float,
    *,
    visibility_post_ms: float,
) -> tuple[HitObjectAnnotation, ...]:
    """返回当前帧应可见的物件，时间均为 segment 内相对毫秒。"""

    preempt = annotation.difficulty.approach_preempt_ms
    # 起点前的 approach 时间要纳入；结束后短暂保留，避免采样边界闪断监督。
    return tuple(
        hit_object
        for hit_object in annotation.hit_objects
        if hit_object.start_ms - preempt
        <= timestamp_ms
        <= max(hit_object.start_ms, hit_object.end_ms) + visibility_post_ms
    )


__all__ = [
    "DifficultyAnnotation",
    "HitObjectAnnotation",
    "SegmentAnnotation",
    "SourceAnnotation",
    "load_annotation",
    "visible_hit_objects",
]
