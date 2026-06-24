from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import torch
import torch.nn.functional as F

from traning.lib.data import PatchMeta


@dataclass
class FeatureCanvas:
    """CPU accumulation canvas for detached patch features."""

    channels: int
    frame_width: int
    frame_height: int
    stride: int = 8
    dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        if min(self.channels, self.frame_width, self.frame_height, self.stride) <= 0:
            raise ValueError("canvas dimensions must be positive")
        height = ceil(self.frame_height / self.stride)
        width = ceil(self.frame_width / self.stride)
        self._values = torch.zeros((self.channels, height, width), dtype=self.dtype)
        self._weights = torch.zeros((1, height, width), dtype=self.dtype)

    def write_patch(
        self,
        features: torch.Tensor,
        meta: PatchMeta,
        *,
        weight: torch.Tensor | None = None,
    ) -> None:
        """Accumulate one detached CHW or 1CHW patch feature tensor on CPU."""

        if features.ndim == 4:
            if features.shape[0] != 1:
                raise ValueError("FeatureCanvas only supports batch size 1 writes")
            features = features[0]
        if features.ndim != 3 or features.shape[0] != self.channels:
            raise ValueError("features must use CHW layout with matching channels")
        x0 = meta.x0 // self.stride
        y0 = meta.y0 // self.stride
        x1 = ceil(meta.x1 / self.stride)
        y1 = ceil(meta.y1 / self.stride)
        target_size = (y1 - y0, x1 - x0)
        patch = features.detach().to("cpu", dtype=self.dtype)
        if patch.shape[-2:] != target_size:
            patch = F.interpolate(
                patch.unsqueeze(0),
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )[0]
        if weight is None:
            patch_weight = torch.ones((1, *target_size), dtype=self.dtype)
        else:
            patch_weight = weight.detach().to("cpu", dtype=self.dtype)
            if patch_weight.ndim == 2:
                patch_weight = patch_weight.unsqueeze(0)
            if patch_weight.shape[-2:] != target_size:
                patch_weight = F.interpolate(
                    patch_weight.unsqueeze(0),
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )[0]
        self._values[:, y0:y1, x0:x1] += patch * patch_weight
        self._weights[:, y0:y1, x0:x1] += patch_weight

    def to_tensor(self) -> torch.Tensor:
        """Return the weighted average canvas as a detached CPU tensor."""

        return self._values / self._weights.clamp_min(1e-6)

    @property
    def weights(self) -> torch.Tensor:
        return self._weights


__all__ = ["FeatureCanvas"]
