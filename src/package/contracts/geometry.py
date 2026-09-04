"""定义带显式坐标空间的点、尺寸和矩形几何契约。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite

from package.contracts.base import ContractMixin


class CoordinateSpace(StrEnum):
    OSU = "osu"
    VIDEO = "video"
    FEATURE = "feature"
    PATCH = "patch"


@dataclass(frozen=True)
class Point2D(ContractMixin):
    x: float
    y: float
    space: CoordinateSpace = CoordinateSpace.OSU

    def __post_init__(self) -> None:
        if not isfinite(self.x) or not isfinite(self.y):
            raise ValueError("point coordinates must be finite")
        if not isinstance(self.space, CoordinateSpace):
            object.__setattr__(self, "space", CoordinateSpace(self.space))

    def as_tuple(self) -> tuple[float, float]:
        return self.x, self.y


@dataclass(frozen=True)
class Size2D(ContractMixin):
    width: float
    height: float
    space: CoordinateSpace = CoordinateSpace.VIDEO

    def __post_init__(self) -> None:
        if not isfinite(self.width) or not isfinite(self.height):
            raise ValueError("size dimensions must be finite")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("size dimensions must be positive")
        if not isinstance(self.space, CoordinateSpace):
            object.__setattr__(self, "space", CoordinateSpace(self.space))


@dataclass(frozen=True)
class Box2D(ContractMixin):
    left: float
    top: float
    width: float
    height: float
    space: CoordinateSpace = CoordinateSpace.VIDEO

    def __post_init__(self) -> None:
        values = (self.left, self.top, self.width, self.height)
        if any(not isfinite(value) for value in values):
            raise ValueError("rect values must be finite")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("rect dimensions must be positive")
        if not isinstance(self.space, CoordinateSpace):
            object.__setattr__(self, "space", CoordinateSpace(self.space))

    @property
    def right(self) -> float:
        return self.left + self.width

    @property
    def bottom(self) -> float:
        return self.top + self.height

    def contains(self, point: Point2D) -> bool:
        if point.space != self.space:
            raise ValueError("point and rect use different coordinate spaces")
        return self.left <= point.x <= self.right and self.top <= point.y <= self.bottom


# 历史名称仅为 identity alias；跨模块矩形契约以 Box2D 为规范名。
Rect2D = Box2D


@dataclass(frozen=True)
class Circle2D(ContractMixin):
    """带显式坐标空间中心的正半径圆。"""

    center: Point2D
    radius: float

    def __post_init__(self) -> None:
        if not isinstance(self.center, Point2D):
            raise TypeError("center must be Point2D")
        if not isfinite(self.radius) or self.radius <= 0:
            raise ValueError("radius must be finite and positive")


@dataclass(frozen=True)
class ResizeMeta(ContractMixin):
    """一次完整帧 resize 的可逆轴向映射元数据。

    映射方程为 ``target = source * scale + offset``；坐标空间标签在变换时随源、目标
    尺寸显式转换，避免像素点与 osu! 点被静默混用。
    """

    source_size: Size2D
    target_size: Size2D
    scale_x: float
    scale_y: float
    offset_x: float = 0.0
    offset_y: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.source_size, Size2D) or not isinstance(
            self.target_size,
            Size2D,
        ):
            raise TypeError("source_size and target_size must be Size2D")
        values = (self.scale_x, self.scale_y, self.offset_x, self.offset_y)
        if any(not isfinite(value) for value in values):
            raise ValueError("resize scale and offset values must be finite")
        if self.scale_x <= 0 or self.scale_y <= 0:
            raise ValueError("resize scales must be positive")

    def to_target(self, point: Point2D) -> Point2D:
        """按统一方程把源点映射到目标帧。"""

        if point.space != self.source_size.space:
            raise ValueError("point and source size use different coordinate spaces")
        return Point2D(
            x=point.x * self.scale_x + self.offset_x,
            y=point.y * self.scale_y + self.offset_y,
            space=self.target_size.space,
        )

    def to_source(self, point: Point2D) -> Point2D:
        """按统一逆方程把目标点映射回源帧。"""

        if point.space != self.target_size.space:
            raise ValueError("point and target size use different coordinate spaces")
        return Point2D(
            x=(point.x - self.offset_x) / self.scale_x,
            y=(point.y - self.offset_y) / self.scale_y,
            space=self.source_size.space,
        )


__all__ = [
    "Box2D",
    "Circle2D",
    "CoordinateSpace",
    "Point2D",
    "Rect2D",
    "ResizeMeta",
    "Size2D",
]
