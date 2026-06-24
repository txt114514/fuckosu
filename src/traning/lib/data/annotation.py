from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class HitObjectAnnotation(BaseModel):
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
    def _valid_end(cls, value: int, info: Any) -> int:
        start = info.data.get("start_ms")
        if start is not None and value < start:
            raise ValueError("end_ms must not be earlier than start_ms")
        return value


class DifficultyAnnotation(BaseModel):
    model_config = ConfigDict(extra="allow")

    approach_preempt_ms: float
    circle_radius_osu_pixels: float


class SourceAnnotation(BaseModel):
    model_config = ConfigDict(extra="allow")

    folder_name: str
    osu_filename: str
    clip_start_ms: int
    clip_end_ms: int

    @field_validator("clip_end_ms")
    @classmethod
    def _valid_clip_end(cls, value: int, info: Any) -> int:
        start = info.data.get("clip_start_ms")
        if start is not None and value <= start:
            raise ValueError("clip_end_ms must be later than clip_start_ms")
        return value


class SegmentAnnotation(BaseModel):
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
        return self.source.clip_end_ms - self.source.clip_start_ms


def load_annotation(path: Path) -> SegmentAnnotation:
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
    preempt = annotation.difficulty.approach_preempt_ms
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
