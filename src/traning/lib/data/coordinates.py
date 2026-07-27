"""patch 局部像素、完整帧像素与步长特征网格之间的坐标换算。"""

from __future__ import annotations

from traning.lib.data.patch_stream import PatchMeta


def local_to_global(meta: PatchMeta, x: float, y: float) -> tuple[float, float]:
    """把 patch 局部图像像素平移到完整帧像素。"""

    return meta.x0 + x, meta.y0 + y


def global_to_local(meta: PatchMeta, x: float, y: float) -> tuple[float, float]:
    """把完整帧像素平移到 patch 局部图像像素。"""

    return x - meta.x0, y - meta.y0


def global_to_patch_indices(
    metas: tuple[PatchMeta, ...],
    x: float,
    y: float,
) -> tuple[int, ...]:
    """返回有效图像区包含该完整帧点的所有 patch 索引。"""

    # x1/y1 是半开边界；重叠区中的点可以同时属于多个 patch。
    return tuple(
        meta.index
        for meta in metas
        if meta.x0 <= x < meta.x1 and meta.y0 <= y < meta.y1
    )


def image_to_feature_grid(
    x: float,
    y: float,
    *,
    stride: int,
) -> tuple[float, float]:
    """按 stride 把图像像素映射到连续特征网格坐标。"""

    if stride <= 0:
        raise ValueError("stride must be positive")
    return x / stride, y / stride


def feature_grid_to_image(
    gx: float,
    gy: float,
    *,
    stride: int,
) -> tuple[float, float]:
    """按 stride 把连续特征网格坐标还原为图像像素。"""

    if stride <= 0:
        raise ValueError("stride must be positive")
    return gx * stride, gy * stride


__all__ = [
    "feature_grid_to_image",
    "global_to_local",
    "global_to_patch_indices",
    "image_to_feature_grid",
    "local_to_global",
]
