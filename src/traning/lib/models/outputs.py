"""空间密集预测与逐帧动作预测的张量输出契约。"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SpatialPrediction:
    """patch 特征网格上的 BCHW 稠密空间预测。"""

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
    """单个因果帧步骤的 BF 动作、候选、坐标与时间预测。"""

    action_logits: torch.Tensor
    selected_candidate_logits: torch.Tensor
    x: torch.Tensor
    y: torch.Tensor
    time_offset_ms: torch.Tensor
    next_hidden_state: torch.Tensor


__all__ = ["ActionPrediction", "SpatialPrediction"]
