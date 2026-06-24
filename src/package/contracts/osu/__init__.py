from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from typing import Any

from package.contracts.base import ContractMixin
from package.contracts.geometry import CoordinateSpace, Point2D


class OsuObjectType(StrEnum):
    CIRCLE = "circle"
    SLIDER = "slider"
    SPINNER = "spinner"


@dataclass(frozen=True)
class OsuTimingPoint(ContractMixin):
    time_ms: float
    beat_length: float
    meter: int = 4
    uninherited: bool = True

    def __post_init__(self) -> None:
        if not isfinite(self.time_ms) or not isfinite(self.beat_length):
            raise ValueError("timing point values must be finite")
        if self.meter <= 0:
            raise ValueError("meter must be positive")


@dataclass(frozen=True)
class OsuDifficulty(ContractMixin):
    circle_size: float | None = None
    approach_rate: float | None = None
    overall_difficulty: float | None = None
    hp_drain_rate: float | None = None

    def __post_init__(self) -> None:
        for value in (
            self.circle_size,
            self.approach_rate,
            self.overall_difficulty,
            self.hp_drain_rate,
        ):
            if value is not None and not isfinite(value):
                raise ValueError("difficulty values must be finite")


@dataclass(frozen=True)
class OsuHitObject(ContractMixin):
    object_id: str
    object_type: OsuObjectType
    start_ms: float
    end_ms: float
    position: Point2D | None = None
    path: tuple[Point2D, ...] = ()
    repeats: int = 1
    curve_type: str | None = None
    pixel_length: float | None = None
    source_index: int | None = None

    def __post_init__(self) -> None:
        if not self.object_id:
            raise ValueError("object_id must not be empty")
        if not isinstance(self.object_type, OsuObjectType):
            object.__setattr__(self, "object_type", OsuObjectType(self.object_type))
        if not isfinite(self.start_ms) or not isfinite(self.end_ms):
            raise ValueError("object times must be finite")
        if self.end_ms < self.start_ms:
            raise ValueError("end_ms must not be earlier than start_ms")
        if self.position is not None and not isinstance(self.position, Point2D):
            object.__setattr__(self, "position", Point2D.from_mapping(self.position))
        path = tuple(
            point if isinstance(point, Point2D) else Point2D.from_mapping(point)
            for point in self.path
        )
        object.__setattr__(self, "path", path)
        if self.object_type == OsuObjectType.CIRCLE and self.position is None:
            raise ValueError("circle objects require position")
        if self.object_type == OsuObjectType.SLIDER and len(path) < 2:
            raise ValueError("slider objects require at least two path points")
        if self.object_type == OsuObjectType.SPINNER and self.end_ms == self.start_ms:
            raise ValueError("spinner objects require duration")
        if self.repeats < 1:
            raise ValueError("repeats must be positive")
        if self.pixel_length is not None and self.pixel_length <= 0:
            raise ValueError("pixel_length must be positive when provided")
        if self.source_index is not None and self.source_index < 0:
            raise ValueError("source_index must be nonnegative")

    @classmethod
    def circle(
        cls,
        object_id: str,
        *,
        start_ms: float,
        x: float,
        y: float,
        source_index: int | None = None,
    ) -> OsuHitObject:
        return cls(
            object_id=object_id,
            object_type=OsuObjectType.CIRCLE,
            start_ms=start_ms,
            end_ms=start_ms,
            position=Point2D(x, y, CoordinateSpace.OSU),
            source_index=source_index,
        )

    @classmethod
    def slider(
        cls,
        object_id: str,
        *,
        start_ms: float,
        end_ms: float,
        path: tuple[tuple[float, float], ...],
        repeats: int = 1,
        curve_type: str | None = None,
        pixel_length: float | None = None,
        source_index: int | None = None,
    ) -> OsuHitObject:
        points = tuple(Point2D(x, y, CoordinateSpace.OSU) for x, y in path)
        return cls(
            object_id=object_id,
            object_type=OsuObjectType.SLIDER,
            start_ms=start_ms,
            end_ms=end_ms,
            position=points[0],
            path=points,
            repeats=repeats,
            curve_type=curve_type,
            pixel_length=pixel_length,
            source_index=source_index,
        )

    @classmethod
    def spinner(
        cls,
        object_id: str,
        *,
        start_ms: float,
        end_ms: float,
        source_index: int | None = None,
    ) -> OsuHitObject:
        return cls(
            object_id=object_id,
            object_type=OsuObjectType.SPINNER,
            start_ms=start_ms,
            end_ms=end_ms,
            source_index=source_index,
        )

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> OsuHitObject:
        return cls(
            object_id=data["object_id"],
            object_type=data["object_type"],
            start_ms=data["start_ms"],
            end_ms=data["end_ms"],
            position=data.get("position"),
            path=tuple(data.get("path", ())),
            repeats=data.get("repeats", 1),
            curve_type=data.get("curve_type"),
            pixel_length=data.get("pixel_length"),
            source_index=data.get("source_index"),
        )


__all__ = [
    "OsuDifficulty",
    "OsuHitObject",
    "OsuObjectType",
    "OsuTimingPoint",
]
