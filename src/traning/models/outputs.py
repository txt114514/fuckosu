from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SpatialPrediction:
    """Dense spatial predictions on a patch feature grid."""

    center_heatmap: torch.Tensor
    visible_heatmap: torch.Tensor
    xy_offset: torch.Tensor
    object_type_logits: torch.Tensor
    ring_mask: torch.Tensor
    ring_radius: torch.Tensor
    slider_mask: torch.Tensor
    slider_direction: torch.Tensor
    spinner_mask: torch.Tensor
    candidate_embedding: torch.Tensor


@dataclass(frozen=True)
class ActionPrediction:
    """Causal action prediction for one frame step."""

    action_logits: torch.Tensor
    selected_candidate_logits: torch.Tensor
    x: torch.Tensor
    y: torch.Tensor
    time_offset_ms: torch.Tensor
    next_hidden_state: torch.Tensor


__all__ = ["ActionPrediction", "SpatialPrediction"]
