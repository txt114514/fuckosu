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
class Rect2D(ContractMixin):
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
        return (
            self.left <= point.x <= self.right
            and self.top <= point.y <= self.bottom
        )


__all__ = ["CoordinateSpace", "Point2D", "Rect2D", "Size2D"]
