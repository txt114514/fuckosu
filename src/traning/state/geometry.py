"""跨模块几何契约的训练 state 公开入口。

实现权威位于 :mod:`package.contracts.geometry`；这里直接复用相同类型对象，避免训练包
再次声明同义点、尺寸、矩形、圆或 resize 元数据。
"""

from package.contracts.geometry import (
    Box2D,
    Circle2D,
    CoordinateSpace,
    Point2D,
    Rect2D,
    ResizeMeta,
    Size2D,
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
