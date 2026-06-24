from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch
import torch.nn.functional as F

from traning.lib.data.tiling import build_patch_windows


@dataclass(frozen=True, slots=True)
class PatchMeta:
    """Full-frame coordinates for one CHW patch.

    Coordinates are image-pixel coordinates in the original frame. ``x1`` and
    ``y1`` are exclusive bounds for the valid, unpadded image area.
    """

    index: int
    x0: int
    y0: int
    x1: int
    y1: int
    frame_width: int
    frame_height: int
    valid_width: int
    valid_height: int
    patch_width: int | None = None
    patch_height: int | None = None

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def padded_width(self) -> int:
        return self.patch_width or self.valid_width

    @property
    def padded_height(self) -> int:
        return self.patch_height or self.valid_height


class PatchStream:
    """Generate padded CHW patches on CPU without invoking model code."""

    def __init__(
        self,
        *,
        patch_width: int = 512,
        patch_height: int = 512,
        overlap_x: int = 128,
        overlap_y: int = 128,
        pin_memory: bool = False,
        padding_value: float = 0.0,
    ) -> None:
        if patch_width <= 0 or patch_height <= 0:
            raise ValueError("patch dimensions must be positive")
        if not 0 <= overlap_x < patch_width:
            raise ValueError("overlap_x must be in [0, patch_width)")
        if not 0 <= overlap_y < patch_height:
            raise ValueError("overlap_y must be in [0, patch_height)")
        self.patch_width = patch_width
        self.patch_height = patch_height
        self.overlap_x = overlap_x
        self.overlap_y = overlap_y
        self.pin_memory = pin_memory
        self.padding_value = padding_value

    def metas(self, *, frame_width: int, frame_height: int) -> tuple[PatchMeta, ...]:
        """Return deterministic patch metadata covering the full frame."""

        windows = build_patch_windows(
            frame_width,
            frame_height,
            patch_width=self.patch_width,
            patch_height=self.patch_height,
            overlap_x=self.overlap_x,
            overlap_y=self.overlap_y,
        )
        metas = tuple(
            PatchMeta(
                index=index,
                x0=window.left,
                y0=window.top,
                x1=window.right,
                y1=window.bottom,
                frame_width=frame_width,
                frame_height=frame_height,
                valid_width=window.width,
                valid_height=window.height,
                patch_width=self.patch_width,
                patch_height=self.patch_height,
            )
            for index, window in enumerate(windows)
        )
        self._validate_coverage(
            metas, frame_width=frame_width, frame_height=frame_height
        )
        return metas

    def count(self, frame: torch.Tensor) -> int:
        """Return the number of patches that ``iter_patches`` would emit."""

        _, height, width = self._shape(frame)
        return len(self.metas(frame_width=width, frame_height=height))

    def iter_patches(
        self,
        frame: torch.Tensor,
    ) -> Iterator[tuple[torch.Tensor, PatchMeta]]:
        """Yield ``(patch, meta)`` pairs from a CHW image tensor.

        The yielded patch always has shape ``C x patch_height x patch_width``.
        Pixels outside ``meta.valid_width``/``meta.valid_height`` are padding.
        """

        channels, height, width = self._shape(frame)
        del channels
        for meta in self.metas(frame_width=width, frame_height=height):
            patch = frame[:, meta.y0 : meta.y1, meta.x0 : meta.x1]
            pad_right = self.patch_width - patch.shape[-1]
            pad_bottom = self.patch_height - patch.shape[-2]
            if pad_right < 0 or pad_bottom < 0:
                raise RuntimeError("patch window exceeded configured patch dimensions")
            if pad_right or pad_bottom:
                patch = F.pad(
                    patch,
                    (0, pad_right, 0, pad_bottom),
                    value=self.padding_value,
                )
            if self.pin_memory and not patch.is_cuda:
                try:
                    patch = patch.pin_memory()
                except RuntimeError:
                    pass
            yield patch, meta

    def to_device(
        self, patch: torch.Tensor, device: torch.device | str
    ) -> torch.Tensor:
        """Move a patch to a device using non-blocking transfer when possible."""

        return patch.to(device, non_blocking=self.pin_memory)

    @staticmethod
    def _shape(frame: torch.Tensor) -> tuple[int, int, int]:
        if frame.ndim != 3:
            raise ValueError("PatchStream expects CHW image tensors")
        channels, height, width = frame.shape
        if channels <= 0 or height <= 0 or width <= 0:
            raise ValueError("frame dimensions must be positive")
        return channels, height, width

    @staticmethod
    def _validate_coverage(
        metas: tuple[PatchMeta, ...],
        *,
        frame_width: int,
        frame_height: int,
    ) -> None:
        if not metas:
            raise ValueError("at least one patch is required")
        seen = {(meta.x0, meta.y0) for meta in metas}
        if len(seen) != len(metas):
            raise ValueError("duplicate patch coordinates are not allowed")
        coverage = torch.zeros((frame_height, frame_width), dtype=torch.bool)
        for meta in metas:
            coverage[meta.y0 : meta.y1, meta.x0 : meta.x1] = True
        if not bool(coverage.all()):
            raise ValueError("patch windows do not cover the full frame")


__all__ = ["PatchMeta", "PatchStream"]
