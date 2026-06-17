from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from traning.Lib.models.outputs import SpatialPrediction


OBJECT_TYPE_NAMES: tuple[str, ...] = (
    "background",
    "hit_circle",
    "approach_circle",
    "slider_head",
    "slider_body",
    "slider_tail",
    "slider_repeat",
    "spinner",
)


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class SpatialPredictionHead(nn.Module):
    """Multi-task dense heads for one fused high-resolution patch feature map."""

    def __init__(
        self,
        in_channels: int,
        *,
        hidden_channels: int | None = None,
        embedding_dim: int = 96,
        object_type_count: int = len(OBJECT_TYPE_NAMES),
    ) -> None:
        super().__init__()
        if in_channels <= 0 or embedding_dim <= 0 or object_type_count <= 0:
            raise ValueError("head dimensions must be positive")
        hidden = hidden_channels or in_channels
        self.trunk = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(hidden), hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(hidden), hidden),
            nn.SiLU(inplace=True),
        )
        head_specs = {
            "center_heatmap": 1,
            "visible_heatmap": 1,
            "xy_offset": 2,
            "object_type_logits": object_type_count,
            "ring_mask": 1,
            "ring_radius": 1,
            "slider_mask": 1,
            "slider_direction": 2,
            "spinner_mask": 1,
            "candidate_embedding": embedding_dim,
        }
        self.heads = nn.ModuleDict(
            {
                name: nn.Conv2d(hidden, channels, kernel_size=1)
                for name, channels in head_specs.items()
            }
        )

    def forward(self, features: torch.Tensor) -> SpatialPrediction:
        if features.ndim != 4:
            raise ValueError("SpatialPredictionHead expects BCHW features")
        hidden = self.trunk(features)
        direction = self.heads["slider_direction"](hidden)
        embedding = self.heads["candidate_embedding"](hidden)
        return SpatialPrediction(
            center_heatmap=self.heads["center_heatmap"](hidden),
            visible_heatmap=self.heads["visible_heatmap"](hidden),
            xy_offset=self.heads["xy_offset"](hidden),
            object_type_logits=self.heads["object_type_logits"](hidden),
            ring_mask=self.heads["ring_mask"](hidden),
            ring_radius=F.softplus(self.heads["ring_radius"](hidden)),
            slider_mask=self.heads["slider_mask"](hidden),
            slider_direction=F.normalize(direction, dim=1, eps=1e-6),
            spinner_mask=self.heads["spinner_mask"](hidden),
            candidate_embedding=F.normalize(embedding, dim=1, eps=1e-6),
        )


__all__ = ["OBJECT_TYPE_NAMES", "SpatialPredictionHead"]
