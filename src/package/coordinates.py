from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping


OSU_PLAYFIELD_WIDTH = 512.0
OSU_PLAYFIELD_HEIGHT = 384.0
COORDINATE_TRANSFORM_VERSION = "osu-playfield-rect-v1"
COORDINATE_CHAIN_VERSION = "osu-playfield-chain-v2"
CoordinateSpace = Literal[
    "beatmap",
    "source_video",
    "cropped_video",
    "model_input",
    "model_output",
    "screen",
]


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
class ImageSize:
    width: float
    height: float

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("image dimensions must be positive")

    def as_dict(self) -> dict[str, float]:
        return {"width": float(self.width), "height": float(self.height)}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ImageSize:
        missing = {"width", "height"} - set(value)
        if missing:
            raise ValueError(f"image size is missing: {', '.join(sorted(missing))}")
        return cls(width=float(value["width"]), height=float(value["height"]))


@dataclass(frozen=True)
class CoordinateTransformSpec:
    """Stable metadata for osu/video coordinate conversion."""

    version: str
    rect: PlayfieldRect
    source: str = "explicit"
    transform_status: str = "configured"
    source_size: ImageSize | None = None
    crop_rect: PlayfieldRect | None = None
    resized_size: ImageSize | None = None
    playfield_source_rect: PlayfieldRect | None = None
    chain: Mapping[str, Any] | None = None
    matrix: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "version": self.version,
            "source": self.source,
            "transform_status": self.transform_status,
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
        if self.source_size is not None:
            payload["source_size"] = self.source_size.as_dict()
        if self.crop_rect is not None:
            payload["crop_rect"] = self.crop_rect.as_dict()
        if self.resized_size is not None:
            payload["resized_size"] = self.resized_size.as_dict()
        if self.playfield_source_rect is not None:
            payload["playfield_source_rect"] = self.playfield_source_rect.as_dict()
        if self.chain is not None:
            payload["chain"] = dict(self.chain)
        if self.matrix is not None:
            payload["matrix"] = [list(row) for row in self.matrix]
        return payload


@dataclass(frozen=True)
class CoordinateTransformChain:
    """Traceable osu -> source -> crop -> training-frame transform."""

    source_size: ImageSize
    crop_rect: PlayfieldRect
    resized_size: ImageSize
    playfield_source_rect: PlayfieldRect | None
    source: str
    status: str = "configured"
    version: str = COORDINATE_CHAIN_VERSION

    @property
    def resolved(self) -> bool:
        return self.status != "unresolved" and self.playfield_source_rect is not None

    @property
    def scale_x(self) -> float:
        return self.resized_size.width / self.crop_rect.width

    @property
    def scale_y(self) -> float:
        return self.resized_size.height / self.crop_rect.height

    @property
    def playfield_frame_rect(self) -> PlayfieldRect:
        if self.playfield_source_rect is None:
            raise ValueError("coordinate transform is unresolved")
        cropped = self.source_to_crop_rect(self.playfield_source_rect)
        return PlayfieldRect(
            left=cropped.left * self.scale_x,
            top=cropped.top * self.scale_y,
            width=cropped.width * self.scale_x,
            height=cropped.height * self.scale_y,
        )

    def source_to_crop(self, x: float, y: float) -> tuple[float, float]:
        return x - self.crop_rect.left, y - self.crop_rect.top

    def crop_to_source(self, x: float, y: float) -> tuple[float, float]:
        return x + self.crop_rect.left, y + self.crop_rect.top

    def crop_to_training_frame(self, x: float, y: float) -> tuple[float, float]:
        return x * self.scale_x, y * self.scale_y

    def training_frame_to_crop(self, x: float, y: float) -> tuple[float, float]:
        return x / self.scale_x, y / self.scale_y

    def source_to_crop_rect(self, rect: PlayfieldRect) -> PlayfieldRect:
        return PlayfieldRect(
            left=rect.left - self.crop_rect.left,
            top=rect.top - self.crop_rect.top,
            width=rect.width,
            height=rect.height,
        )

    def osu_to_source(self, x: float, y: float) -> tuple[float, float]:
        if self.playfield_source_rect is None:
            raise ValueError("coordinate transform is unresolved")
        return OsuVideoTransform.from_rect(self.playfield_source_rect).osu_to_video(x, y)

    def source_to_osu(self, x: float, y: float) -> tuple[float, float]:
        if self.playfield_source_rect is None:
            raise ValueError("coordinate transform is unresolved")
        return OsuVideoTransform.from_rect(self.playfield_source_rect).video_to_osu(x, y)

    def osu_to_training_frame(self, x: float, y: float) -> tuple[float, float]:
        source_x, source_y = self.osu_to_source(x, y)
        crop_x, crop_y = self.source_to_crop(source_x, source_y)
        return self.crop_to_training_frame(crop_x, crop_y)

    def training_frame_to_osu(self, x: float, y: float) -> tuple[float, float]:
        crop_x, crop_y = self.training_frame_to_crop(x, y)
        source_x, source_y = self.crop_to_source(crop_x, crop_y)
        return self.source_to_osu(source_x, source_y)

    def model_output_to_osu(self, x: float, y: float) -> tuple[float, float]:
        return self.training_frame_to_osu(x, y)

    def osu_to_model_input(self, x: float, y: float) -> tuple[float, float]:
        return self.osu_to_training_frame(x, y)

    def model_input_to_osu(self, x: float, y: float) -> tuple[float, float]:
        return self.training_frame_to_osu(x, y)

    def to_frame_transform(self) -> OsuVideoTransform:
        return OsuVideoTransform.from_rect(self.playfield_frame_rect)

    def as_dict(self, *, include_rects: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "version": self.version,
            "source": self.source,
            "status": self.status,
            "steps": (
                "beatmap_to_source_video",
                "source_to_crop",
                "crop_to_model_input",
            ),
            "reverse_steps": (
                "model_input_to_crop",
                "crop_to_source",
                "source_video_to_beatmap",
            ),
            "spaces": {
                "authority": "beatmap",
                "source_video": self.source_size.as_dict(),
                "crop": self.crop_rect.as_dict(),
                "model_input": self.resized_size.as_dict(),
            },
            "scale": {"x": self.scale_x, "y": self.scale_y},
        }
        if include_rects:
            payload.update(
                {
                    "source_size": self.source_size.as_dict(),
                    "crop_rect": self.crop_rect.as_dict(),
                    "resized_size": self.resized_size.as_dict(),
                    "playfield_source_rect": (
                        None
                        if self.playfield_source_rect is None
                        else self.playfield_source_rect.as_dict()
                    ),
                }
            )
        return payload

    def spec(self) -> CoordinateTransformSpec:
        rect = (
            self.playfield_frame_rect
            if self.playfield_source_rect is not None
            else PlayfieldRect(0.0, 0.0, self.resized_size.width, self.resized_size.height)
        )
        return CoordinateTransformSpec(
            version=COORDINATE_TRANSFORM_VERSION,
            rect=rect,
            source=self.source,
            transform_status=self.status,
            source_size=self.source_size,
            crop_rect=self.crop_rect,
            resized_size=self.resized_size,
            playfield_source_rect=self.playfield_source_rect,
            chain=self.as_dict(include_rects=False),
        )


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


@dataclass(frozen=True)
class ScreenTransform:
    """Map authoritative osu! playfield coordinates to desktop screen pixels."""

    playfield_rect: PlayfieldRect

    @classmethod
    def from_rect(cls, rect: PlayfieldRect | Mapping[str, Any]) -> ScreenTransform:
        selected = rect if isinstance(rect, PlayfieldRect) else PlayfieldRect.from_mapping(rect)
        return cls(playfield_rect=selected)

    @property
    def scale_x(self) -> float:
        return self.playfield_rect.width / OSU_PLAYFIELD_WIDTH

    @property
    def scale_y(self) -> float:
        return self.playfield_rect.height / OSU_PLAYFIELD_HEIGHT

    def osu_to_screen(self, x: float, y: float) -> tuple[float, float]:
        return (
            self.playfield_rect.left + x * self.scale_x,
            self.playfield_rect.top + y * self.scale_y,
        )

    def screen_to_osu(self, x: float, y: float) -> tuple[float, float]:
        return (
            (x - self.playfield_rect.left) / self.scale_x,
            (y - self.playfield_rect.top) / self.scale_y,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_space": "beatmap",
            "target_space": "screen",
            "playfield_rect": self.playfield_rect.as_dict(),
            "scale": {"x": self.scale_x, "y": self.scale_y},
        }


@dataclass(frozen=True)
class AffineOsuVideoTransform:
    """Map osu! playfield coordinates with a fitted 2x3 affine matrix."""

    matrix: tuple[tuple[float, float, float], tuple[float, float, float]]

    def __post_init__(self) -> None:
        if len(self.matrix) != 2 or any(len(row) != 3 for row in self.matrix):
            raise ValueError("affine coordinate transform requires a 2x3 matrix")
        determinant = self.matrix[0][0] * self.matrix[1][1] - self.matrix[0][1] * self.matrix[1][0]
        if abs(determinant) <= 1e-9:
            raise ValueError("affine coordinate transform must be invertible")

    @classmethod
    def from_rows(
        cls,
        rows: Any,
    ) -> AffineOsuVideoTransform:
        if not isinstance(rows, (list, tuple)) or len(rows) != 2:
            raise ValueError("affine matrix must contain two rows")
        return cls(
            matrix=(
                (float(rows[0][0]), float(rows[0][1]), float(rows[0][2])),
                (float(rows[1][0]), float(rows[1][1]), float(rows[1][2])),
            )
        )

    @property
    def rect(self) -> PlayfieldRect:
        corners = (
            self.osu_to_video(0.0, 0.0),
            self.osu_to_video(OSU_PLAYFIELD_WIDTH, 0.0),
            self.osu_to_video(0.0, OSU_PLAYFIELD_HEIGHT),
            self.osu_to_video(OSU_PLAYFIELD_WIDTH, OSU_PLAYFIELD_HEIGHT),
        )
        xs = [point[0] for point in corners]
        ys = [point[1] for point in corners]
        return PlayfieldRect(
            left=min(xs),
            top=min(ys),
            width=max(xs) - min(xs),
            height=max(ys) - min(ys),
        )

    def spec(self, *, source: str = "affine_matrix", status: str = "calibrated") -> CoordinateTransformSpec:
        return CoordinateTransformSpec(
            version=COORDINATE_TRANSFORM_VERSION,
            rect=self.rect,
            source=source,
            transform_status=status,
            matrix=self.matrix,
        )

    def osu_to_video(self, x: float, y: float) -> tuple[float, float]:
        return (
            self.matrix[0][0] * x + self.matrix[0][1] * y + self.matrix[0][2],
            self.matrix[1][0] * x + self.matrix[1][1] * y + self.matrix[1][2],
        )

    def video_to_osu(self, x: float, y: float) -> tuple[float, float]:
        a, b, c = self.matrix[0]
        d, e, f = self.matrix[1]
        determinant = a * e - b * d
        tx = x - c
        ty = y - f
        return (
            (e * tx - b * ty) / determinant,
            (-d * tx + a * ty) / determinant,
        )

    def osu_radius_to_video(self, radius: float) -> float:
        scale_x = (self.matrix[0][0] ** 2 + self.matrix[1][0] ** 2) ** 0.5
        scale_y = (self.matrix[0][1] ** 2 + self.matrix[1][1] ** 2) ** 0.5
        return radius * (scale_x + scale_y) / 2.0


__all__ = [
    "AffineOsuVideoTransform",
    "COORDINATE_CHAIN_VERSION",
    "COORDINATE_TRANSFORM_VERSION",
    "CoordinateSpace",
    "CoordinateTransformChain",
    "CoordinateTransformSpec",
    "ImageSize",
    "OSU_PLAYFIELD_HEIGHT",
    "OSU_PLAYFIELD_WIDTH",
    "OsuVideoTransform",
    "PlayfieldRect",
    "ScreenTransform",
]
