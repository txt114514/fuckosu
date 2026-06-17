from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from traning.Lib.data import PatchMeta
from traning.Lib.models.local_encoder import LocalFeatures


@dataclass(frozen=True)
class FusedPatchFeatures:
    dense: torch.Tensor
    patch_meta: PatchMeta
    global_context: torch.Tensor


def _base_grid(
    meta: PatchMeta,
    *,
    height: int,
    width: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if width <= 0 or height <= 0:
        raise ValueError("feature map dimensions must be positive")
    x_extent = max(float(meta.padded_width), 1.0)
    y_extent = max(float(meta.padded_height), 1.0)
    x = meta.x0 + (torch.arange(width, device=device, dtype=dtype) + 0.5) * (
        x_extent / width
    )
    y = meta.y0 + (torch.arange(height, device=device, dtype=dtype) + 0.5) * (
        y_extent / height
    )
    if meta.frame_width <= 1:
        gx = torch.zeros_like(x)
    else:
        gx = x / (meta.frame_width - 1) * 2.0 - 1.0
    if meta.frame_height <= 1:
        gy = torch.zeros_like(y)
    else:
        gy = y / (meta.frame_height - 1) * 2.0 - 1.0
    yy, xx = torch.meshgrid(gy, gx, indexing="ij")
    grid = torch.stack((xx, yy), dim=-1)
    return grid.unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous()


def sample_global_feature(
    global_feature: torch.Tensor,
    patch_meta: PatchMeta,
    local_feature_shape: tuple[int, int],
) -> torch.Tensor:
    """Sample full-frame global features at one patch feature-grid alignment."""

    if global_feature.ndim != 4:
        raise ValueError("global_feature must use BCHW layout")
    height, width = local_feature_shape
    grid = _base_grid(
        patch_meta,
        height=height,
        width=width,
        batch_size=global_feature.shape[0],
        device=global_feature.device,
        dtype=global_feature.dtype,
    )
    return F.grid_sample(
        global_feature,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )


class GatedSparseFusion(nn.Module):
    """Fuse local patch features with sparse low-resolution global context."""

    def __init__(
        self,
        *,
        local_channels: int,
        global_channels: int,
        hidden_dim: int = 96,
        heads: int = 4,
        sampling_points: int = 4,
        layers: int = 2,
        enabled: bool = True,
    ) -> None:
        super().__init__()
        if (
            min(local_channels, global_channels, hidden_dim, heads, sampling_points)
            <= 0
        ):
            raise ValueError("fusion dimensions must be positive")
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads")
        self.heads = heads
        self.sampling_points = sampling_points
        self.enabled = enabled
        self.global_project = nn.Conv2d(global_channels, hidden_dim, kernel_size=1)
        self.context_project = nn.Conv2d(global_channels, local_channels, kernel_size=1)
        self.gate_project = nn.Conv2d(global_channels, local_channels, kernel_size=1)
        self.offset_predictor = nn.Conv2d(
            local_channels,
            heads * sampling_points * 2,
            kernel_size=1,
        )
        self.weight_predictor = nn.Conv2d(
            local_channels,
            heads * sampling_points,
            kernel_size=1,
        )
        self.sparse_project = nn.Conv2d(hidden_dim, local_channels, kernel_size=1)
        refinement_layers: list[nn.Module] = []
        for _ in range(max(layers - 1, 0)):
            refinement_layers.extend(
                [
                    nn.Conv2d(local_channels, local_channels, kernel_size=3, padding=1),
                    nn.GroupNorm(8 if local_channels % 8 == 0 else 1, local_channels),
                    nn.SiLU(inplace=True),
                ]
            )
        self.refinement = nn.Sequential(*refinement_layers)

    def forward(
        self,
        *,
        local_features: LocalFeatures,
        global_features: torch.Tensor,
        patch_meta: PatchMeta,
    ) -> FusedPatchFeatures:
        local = local_features.dense
        if local.ndim != 4:
            raise ValueError("local dense features must use BCHW layout")
        if not self.enabled:
            context = torch.zeros_like(local)
            return FusedPatchFeatures(
                dense=local,
                patch_meta=patch_meta,
                global_context=context,
            )
        context = sample_global_feature(
            global_features,
            patch_meta,
            (local.shape[-2], local.shape[-1]),
        )
        gate = torch.sigmoid(self.gate_project(context))
        fused = local * (1.0 + gate) + self.context_project(context)
        fused = fused + self.sparse_project(
            self._sparse_context(
                local=local,
                global_features=global_features,
                patch_meta=patch_meta,
            )
        )
        if len(self.refinement) > 0:
            fused = fused + self.refinement(fused)
        return FusedPatchFeatures(
            dense=fused,
            patch_meta=patch_meta,
            global_context=context,
        )

    def _sparse_context(
        self,
        *,
        local: torch.Tensor,
        global_features: torch.Tensor,
        patch_meta: PatchMeta,
    ) -> torch.Tensor:
        batch, _, height, width = local.shape
        global_hidden = self.global_project(global_features)
        base = _base_grid(
            patch_meta,
            height=height,
            width=width,
            batch_size=batch,
            device=local.device,
            dtype=local.dtype,
        )
        offsets = self.offset_predictor(local)
        offsets = offsets.view(
            batch,
            self.heads,
            self.sampling_points,
            2,
            height,
            width,
        )
        offsets = offsets.permute(0, 1, 2, 4, 5, 3).tanh() * 0.2
        weights = self.weight_predictor(local).view(
            batch,
            self.heads,
            self.sampling_points,
            height,
            width,
        )
        weights = torch.softmax(weights, dim=2)
        accumulated = torch.zeros(
            (batch, global_hidden.shape[1], height, width),
            device=local.device,
            dtype=local.dtype,
        )
        for head in range(self.heads):
            head_sum = torch.zeros_like(accumulated)
            for point in range(self.sampling_points):
                grid = torch.clamp(base + offsets[:, head, point], -1.0, 1.0)
                sampled = F.grid_sample(
                    global_hidden,
                    grid,
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=True,
                )
                head_sum = head_sum + sampled * weights[:, head, point].unsqueeze(1)
            accumulated = accumulated + head_sum
        return accumulated / self.heads


__all__ = ["FusedPatchFeatures", "GatedSparseFusion", "sample_global_feature"]
