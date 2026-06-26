from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


OSU_PLAYFIELD_WIDTH = 512.0
OSU_PLAYFIELD_HEIGHT = 384.0
COORDINATE_TRANSFORM_VERSION = "osu-playfield-rect-v1"


@dataclass(frozen=True)
class PlayfieldRect:
    """Video-pixel rectangle containing the osu!standard playfield."""

    left: float
    top: float
    width: float
    height: float

    def __post_init__(self) -> None:
        values = (self.left, self.top, self.width, self.height)
        if any(not isinstance(value, int | float) for value in values):
            raise ValueError("playfield rect values must be numeric")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("playfield rect dimensions must be positive")

    def as_dict(self) -> dict[str, float]:
        return {
            "left": float(self.left),
            "top": float(self.top),
            "width": float(self.width),
            "height": float(self.height),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> PlayfieldRect:
        missing = {"left", "top", "width", "height"} - set(value)
        if missing:
            raise ValueError(f"playfield rect is missing: {', '.join(sorted(missing))}")
        return cls(
            left=float(value["left"]),
            top=float(value["top"]),
            width=float(value["width"]),
            height=float(value["height"]),
        )


@dataclass(frozen=True)
class CoordinateTransformSpec:
    """Stable metadata for osu/video coordinate conversion."""

    version: str
    rect: PlayfieldRect
    source: str = "explicit"

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "source": self.source,
            "video_pixel_format": {
                "left": "x coordinate in original frame pixels before crop/resize",
                "top": "y coordinate in original frame pixels before crop/resize",
                "width": "playfield width in original frame pixels",
                "height": "playfield height in original frame pixels",
            },
            "osu_playfield_size": {
                "width": OSU_PLAYFIELD_WIDTH,
                "height": OSU_PLAYFIELD_HEIGHT,
            },
            "rect": self.rect.as_dict(),
        }


@dataclass(frozen=True)
class OsuVideoTransform:
    """Map osu!standard playfield coordinates to video pixels."""

    playfield_left: float
    playfield_top: float
    playfield_width: float
    playfield_height: float

    def __post_init__(self) -> None:
        if self.playfield_width <= 0 or self.playfield_height <= 0:
            raise ValueError("playfield dimensions must be positive")

    @classmethod
    def fit_centered(
        cls,
        video_width: int,
        video_height: int,
    ) -> OsuVideoTransform:
        if video_width <= 0 or video_height <= 0:
            raise ValueError("video dimensions must be positive")
        scale = min(
            video_width / OSU_PLAYFIELD_WIDTH,
            video_height / OSU_PLAYFIELD_HEIGHT,
        )
        width = OSU_PLAYFIELD_WIDTH * scale
        height = OSU_PLAYFIELD_HEIGHT * scale
        return cls(
            playfield_left=(video_width - width) / 2.0,
            playfield_top=(video_height - height) / 2.0,
            playfield_width=width,
            playfield_height=height,
        )

    @classmethod
    def from_rect(cls, rect: PlayfieldRect | Mapping[str, Any]) -> OsuVideoTransform:
        selected = rect if isinstance(rect, PlayfieldRect) else PlayfieldRect.from_mapping(rect)
        return cls(
            playfield_left=selected.left,
            playfield_top=selected.top,
            playfield_width=selected.width,
            playfield_height=selected.height,
        )

    @property
    def rect(self) -> PlayfieldRect:
        return PlayfieldRect(
            left=self.playfield_left,
            top=self.playfield_top,
            width=self.playfield_width,
            height=self.playfield_height,
        )

    def spec(self, *, source: str = "explicit") -> CoordinateTransformSpec:
        return CoordinateTransformSpec(
            version=COORDINATE_TRANSFORM_VERSION,
            rect=self.rect,
            source=source,
        )

    @property
    def scale_x(self) -> float:
        return self.playfield_width / OSU_PLAYFIELD_WIDTH

    @property
    def scale_y(self) -> float:
        return self.playfield_height / OSU_PLAYFIELD_HEIGHT

    def osu_to_video(self, x: float, y: float) -> tuple[float, float]:
        return (
            self.playfield_left + x * self.scale_x,
            self.playfield_top + y * self.scale_y,
        )

    def video_to_osu(self, x: float, y: float) -> tuple[float, float]:
        return (
            (x - self.playfield_left) / self.scale_x,
            (y - self.playfield_top) / self.scale_y,
        )

    def osu_radius_to_video(self, radius: float) -> float:
        if abs(self.scale_x - self.scale_y) > 1e-9:
            raise ValueError("radius conversion requires uniform playfield scaling")
        return radius * self.scale_x


__all__ = [
    "COORDINATE_TRANSFORM_VERSION",
    "CoordinateTransformSpec",
    "OSU_PLAYFIELD_HEIGHT",
    "OSU_PLAYFIELD_WIDTH",
    "OsuVideoTransform",
    "PlayfieldRect",
]
