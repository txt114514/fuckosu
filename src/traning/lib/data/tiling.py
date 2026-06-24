from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from torch import Tensor


@dataclass(frozen=True)
class PatchWindow:
    left: int
    top: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height


def _axis_starts(size: int, patch_size: int, overlap: int) -> tuple[int, ...]:
    if size <= 0 or patch_size <= 0:
        raise ValueError("image and patch dimensions must be positive")
    if not 0 <= overlap < patch_size:
        raise ValueError("overlap must be in [0, patch_size)")
    if size <= patch_size:
        return (0,)

    step = patch_size - overlap
    starts = list(range(0, size - patch_size + 1, step))
    final_start = size - patch_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return tuple(starts)


def build_patch_windows(
    image_width: int,
    image_height: int,
    *,
    patch_width: int,
    patch_height: int,
    overlap_x: int,
    overlap_y: int,
) -> tuple[PatchWindow, ...]:
    x_starts = _axis_starts(image_width, patch_width, overlap_x)
    y_starts = _axis_starts(image_height, patch_height, overlap_y)
    return tuple(
        PatchWindow(
            left=left,
            top=top,
            width=min(patch_width, image_width),
            height=min(patch_height, image_height),
        )
        for top in y_starts
        for left in x_starts
    )


def iter_patches(
    image: Tensor,
    windows: tuple[PatchWindow, ...],
) -> Iterator[tuple[PatchWindow, Tensor]]:
    if image.ndim != 3:
        raise ValueError("image tensor must use CHW layout")
    for window in windows:
        yield (
            window,
            image[
                :,
                window.top : window.bottom,
                window.left : window.right,
            ],
        )


__all__ = ["PatchWindow", "build_patch_windows", "iter_patches"]
