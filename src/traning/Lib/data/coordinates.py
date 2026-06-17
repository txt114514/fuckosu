from __future__ import annotations

from traning.Lib.data.patch_stream import PatchMeta


def local_to_global(meta: PatchMeta, x: float, y: float) -> tuple[float, float]:
    """Convert patch-local image coordinates to full-frame image coordinates."""

    return meta.x0 + x, meta.y0 + y


def global_to_local(meta: PatchMeta, x: float, y: float) -> tuple[float, float]:
    """Convert full-frame image coordinates to patch-local image coordinates."""

    return x - meta.x0, y - meta.y0


def global_to_patch_indices(
    metas: tuple[PatchMeta, ...],
    x: float,
    y: float,
) -> tuple[int, ...]:
    """Return patch indices whose valid image area contains a full-frame point."""

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
    """Map image-pixel coordinates to a stride-based feature grid."""

    if stride <= 0:
        raise ValueError("stride must be positive")
    return x / stride, y / stride


def feature_grid_to_image(
    gx: float,
    gy: float,
    *,
    stride: int,
) -> tuple[float, float]:
    """Map stride-based feature-grid coordinates back to image pixels."""

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
