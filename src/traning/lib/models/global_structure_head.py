from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class GlobalStructurePrediction:
    objectness: torch.Tensor
    center_heatmap: torch.Tensor
    ring_likelihood: torch.Tensor
    slider_likelihood: torch.Tensor
    spinner_likelihood: torch.Tensor
    coarse_radius: torch.Tensor
    context_tokens: torch.Tensor


class GlobalStructureHead(nn.Module):
    """Predict coarse full-frame object structure from global features."""

    def __init__(
        self,
        in_channels: int,
        *,
        hidden_channels: int | None = None,
        context_dim: int | None = None,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        hidden = hidden_channels or in_channels
        token_dim = context_dim or hidden
        self.trunk = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8 if hidden % 8 == 0 else 1, hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8 if hidden % 8 == 0 else 1, hidden),
            nn.SiLU(inplace=True),
        )
        self.objectness = nn.Conv2d(hidden, 1, kernel_size=1)
        self.center_heatmap = nn.Conv2d(hidden, 1, kernel_size=1)
        self.ring_likelihood = nn.Conv2d(hidden, 1, kernel_size=1)
        self.slider_likelihood = nn.Conv2d(hidden, 1, kernel_size=1)
        self.spinner_likelihood = nn.Conv2d(hidden, 1, kernel_size=1)
        self.coarse_radius = nn.Conv2d(hidden, 1, kernel_size=1)
        self.context_projection = nn.Conv2d(hidden, token_dim, kernel_size=1)

    def forward(self, features: torch.Tensor) -> GlobalStructurePrediction:
        if features.ndim != 4:
            raise ValueError("GlobalStructureHead expects BCHW features")
        hidden = self.trunk(features)
        context = self.context_projection(hidden)
        return GlobalStructurePrediction(
            objectness=self.objectness(hidden),
            center_heatmap=self.center_heatmap(hidden),
            ring_likelihood=self.ring_likelihood(hidden),
            slider_likelihood=self.slider_likelihood(hidden),
            spinner_likelihood=self.spinner_likelihood(hidden),
            coarse_radius=F.softplus(self.coarse_radius(hidden)),
            context_tokens=context.flatten(2).transpose(1, 2).contiguous(),
        )


__all__ = ["GlobalStructureHead", "GlobalStructurePrediction"]
