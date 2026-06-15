from __future__ import annotations

from dataclasses import dataclass


OSU_PLAYFIELD_WIDTH = 512.0
OSU_PLAYFIELD_HEIGHT = 384.0


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
    "OSU_PLAYFIELD_HEIGHT",
    "OSU_PLAYFIELD_WIDTH",
    "OsuVideoTransform",
]
