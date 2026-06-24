from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from traning.lib.models.outputs import SpatialPrediction


@dataclass(frozen=True)
class LossWeights:
    center: float = 1.0
    visible: float = 0.5
    offset: float = 1.0
    object_type: float = 1.0
    ring: float = 0.5
    radius: float = 0.25
    slider: float = 0.75
    slider_direction: float = 0.25
    spinner: float = 0.5
    global_local: float = 0.25
    cross_patch: float = 0.25
    temporal: float = 0.1


@dataclass(frozen=True)
class SpatialLossTargets:
    center_heatmap: torch.Tensor
    visible_heatmap: torch.Tensor
    xy_offset: torch.Tensor
    object_type: torch.Tensor
    ring_mask: torch.Tensor
    ring_radius: torch.Tensor
    slider_mask: torch.Tensor
    slider_direction: torch.Tensor
    spinner_mask: torch.Tensor


def _masked_smooth_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if mask.ndim != prediction.ndim:
        while mask.ndim < prediction.ndim:
            mask = mask.unsqueeze(1)
    mask = mask.to(device=prediction.device, dtype=prediction.dtype)
    denominator = mask.sum() * prediction.shape[1]
    if float(denominator.detach().cpu()) <= 0.0:
        return prediction.sum() * 0.0
    loss = F.smooth_l1_loss(prediction, target, reduction="none") * mask
    return loss.sum() / denominator.clamp_min(1.0)


def compute_spatial_loss(
    prediction: SpatialPrediction,
    target: SpatialLossTargets,
    *,
    weights: LossWeights = LossWeights(),
) -> dict[str, torch.Tensor]:
    """Compute first-version dense multi-task spatial losses."""

    losses = {
        "center": F.binary_cross_entropy_with_logits(
            prediction.center_heatmap,
            target.center_heatmap,
        ),
        "visible": F.binary_cross_entropy_with_logits(
            prediction.visible_heatmap,
            target.visible_heatmap,
        ),
        "offset": _masked_smooth_l1(
            prediction.xy_offset,
            target.xy_offset,
            (target.center_heatmap > 0.25).to(target.center_heatmap.dtype),
        ),
        "object_type": F.cross_entropy(
            prediction.object_type_logits,
            target.object_type.long(),
        ),
        "ring": F.binary_cross_entropy_with_logits(
            prediction.ring_mask,
            target.ring_mask,
        ),
        "radius": _masked_smooth_l1(
            prediction.ring_radius,
            target.ring_radius,
            target.ring_mask,
        ),
        "slider": F.binary_cross_entropy_with_logits(
            prediction.slider_mask,
            target.slider_mask,
        ),
        "slider_direction": _masked_smooth_l1(
            prediction.slider_direction,
            target.slider_direction,
            target.slider_mask,
        ),
        "spinner": F.binary_cross_entropy_with_logits(
            prediction.spinner_mask,
            target.spinner_mask,
        ),
    }
    total = (
        weights.center * losses["center"]
        + weights.visible * losses["visible"]
        + weights.offset * losses["offset"]
        + weights.object_type * losses["object_type"]
        + weights.ring * losses["ring"]
        + weights.radius * losses["radius"]
        + weights.slider * losses["slider"]
        + weights.slider_direction * losses["slider_direction"]
        + weights.spinner * losses["spinner"]
    )
    losses["total"] = total
    return losses


def cosine_embedding_consistency_loss(
    embeddings: torch.Tensor,
    object_ids: torch.Tensor,
    *,
    margin: float = 0.4,
) -> torch.Tensor:
    """Pull embeddings for the same object together and push others apart."""

    if embeddings.ndim != 2 or object_ids.ndim != 1:
        raise ValueError("embeddings must be NF and object_ids must be N")
    if embeddings.shape[0] != object_ids.shape[0]:
        raise ValueError("embedding and object id counts must match")
    if embeddings.shape[0] < 2:
        return embeddings.new_zeros(())
    normalized = F.normalize(embeddings, dim=1, eps=1e-6)
    similarity = normalized @ normalized.t()
    same = object_ids[:, None] == object_ids[None, :]
    eye = torch.eye(same.shape[0], dtype=torch.bool, device=same.device)
    positive = (1.0 - similarity)[same & ~eye]
    negative = torch.clamp(similarity[~same] - margin, min=0.0)
    pieces = []
    if positive.numel():
        pieces.append(positive.mean())
    if negative.numel():
        pieces.append(negative.mean())
    if not pieces:
        return embeddings.new_zeros(())
    return torch.stack(pieces).mean()


def global_local_consistency_loss(
    local_logits: torch.Tensor,
    sampled_global_logits: torch.Tensor,
) -> torch.Tensor:
    """Encourage local dense predictions to agree with sampled global context."""

    return F.binary_cross_entropy_with_logits(
        local_logits,
        torch.sigmoid(sampled_global_logits.detach()),
    )


def temporal_consistency_loss(
    current: torch.Tensor,
    previous: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Penalize abrupt dense prediction changes between neighboring frames."""

    loss = F.smooth_l1_loss(current, previous.detach(), reduction="none")
    if mask is not None:
        loss = loss * mask
        denominator = mask.sum().clamp_min(1.0)
        return loss.sum() / denominator
    return loss.mean()


__all__ = [
    "LossWeights",
    "SpatialLossTargets",
    "compute_spatial_loss",
    "cosine_embedding_consistency_loss",
    "global_local_consistency_loss",
    "temporal_consistency_loss",
]
