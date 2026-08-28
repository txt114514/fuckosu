"""Perception 多头监督损失与实例身份约束。"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

import torch
from torch import Tensor
from torch.nn import functional as F

from traning.perception.models import DensePerceptionOutput

from .targets import PerceptionTargets


@dataclass(frozen=True, slots=True)
class PerceptionLossWeights:
    """各监督头的非负权重。"""

    center: float = 1.0
    visibility: float = 1.0
    object_type: float = 1.0
    xy: float = 1.0
    ring: float = 1.0
    ring_radius: float = 1.0
    slider: float = 1.0
    slider_direction: float = 1.0
    spinner: float = 1.0
    identity: float = 1.0
    identity_margin: float = 0.25

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{field.name} 必须是数值")
            if not math.isfinite(float(value)) or value < 0.0:
                raise ValueError(f"{field.name} 必须是有限非负数")
        if self.identity_margin > 1.0:
            raise ValueError("identity_margin 不得大于 1")


@dataclass(frozen=True, slots=True)
class PerceptionLoss:
    """保留每个监督分量及其加权总和，便于训练遥测。"""

    center: Tensor
    visibility: Tensor
    object_type: Tensor
    xy: Tensor
    ring: Tensor
    ring_radius: Tensor
    slider: Tensor
    slider_direction: Tensor
    spinner: Tensor
    identity: Tensor
    total: Tensor


def compute_perception_loss(
    prediction: DensePerceptionOutput,
    targets: PerceptionTargets,
    weights: PerceptionLossWeights,
) -> PerceptionLoss:
    """计算完整 dense 监督；梯度保持贯穿所有预测 head 和上游融合网络。"""

    if not isinstance(targets, PerceptionTargets):
        raise TypeError("targets 必须是 PerceptionTargets")
    if not isinstance(weights, PerceptionLossWeights):
        raise TypeError("weights 必须是 PerceptionLossWeights")
    _validate_prediction(prediction, targets)

    # instance_ids 是显式的实例监督边界；背景与 ignore 区域不参与几何和身份损失。
    object_mask = targets.instance_ids.ge(0).unsqueeze(1)
    ring_mask = object_mask & targets.type_indices.eq(0).unsqueeze(1)
    slider_mask = object_mask & targets.type_indices.eq(1).unsqueeze(1)

    center = F.binary_cross_entropy_with_logits(
        prediction.center_logits, targets.center_heatmap
    )
    visibility = F.binary_cross_entropy_with_logits(
        prediction.visibility_logits, targets.visibility
    )
    object_type = _type_loss(prediction.type_logits, targets.type_indices)
    xy = _masked_smooth_l1(prediction.xy_offsets, targets.xy_offsets, object_mask)
    ring = _masked_binary_cross_entropy(
        prediction.ring_logits, targets.ring, object_mask
    )
    ring_radius = _masked_smooth_l1(
        prediction.ring_radius, targets.ring_radius, ring_mask
    )
    slider = _masked_binary_cross_entropy(
        prediction.slider_logits, targets.slider, object_mask
    )
    slider_direction = _masked_smooth_l1(
        prediction.slider_direction, targets.slider_direction, slider_mask
    )
    spinner = _masked_binary_cross_entropy(
        prediction.spinner_logits, targets.spinner, object_mask
    )
    identity = _instance_margin_loss(
        prediction.identity_embedding,
        targets.instance_ids,
        margin=weights.identity_margin,
    )

    components = (
        (weights.center, center),
        (weights.visibility, visibility),
        (weights.object_type, object_type),
        (weights.xy, xy),
        (weights.ring, ring),
        (weights.ring_radius, ring_radius),
        (weights.slider, slider),
        (weights.slider_direction, slider_direction),
        (weights.spinner, spinner),
        (weights.identity, identity),
    )
    total = sum((weight * loss for weight, loss in components), center.new_zeros(()))
    return PerceptionLoss(
        center=center,
        visibility=visibility,
        object_type=object_type,
        xy=xy,
        ring=ring,
        ring_radius=ring_radius,
        slider=slider,
        slider_direction=slider_direction,
        spinner=spinner,
        identity=identity,
        total=total,
    )


def _validate_prediction(
    prediction: DensePerceptionOutput,
    targets: PerceptionTargets,
) -> None:
    expected = targets.center_heatmap.shape
    batch, _, height, width = expected
    head_channels = {
        "center_logits": 1,
        "visibility_logits": 1,
        "type_logits": 4,
        "xy_offsets": 2,
        "ring_logits": 1,
        "ring_radius": 1,
        "slider_logits": 1,
        "slider_direction": 2,
        "spinner_logits": 1,
    }
    for name, channels in head_channels.items():
        tensor = getattr(prediction, name)
        if not isinstance(tensor, Tensor):
            raise TypeError(f"prediction.{name} 必须是 torch.Tensor")
        if tensor.shape != (batch, channels, height, width):
            raise ValueError(f"prediction.{name} 必须是 Bx{channels}xHxW")
        if not tensor.is_floating_point():
            raise TypeError(f"prediction.{name} 必须是浮点张量")
        if tensor.device != targets.center_heatmap.device:
            raise ValueError("prediction 与 targets 必须位于同一设备")

    embedding = prediction.identity_embedding
    if not isinstance(embedding, Tensor) or embedding.ndim != 4:
        raise ValueError("prediction.identity_embedding 必须是 BCHW 张量")
    if embedding.shape[0] != batch or embedding.shape[2:] != (height, width):
        raise ValueError("prediction.identity_embedding 的 B/H/W 必须与 targets 一致")
    if embedding.shape[1] < 1 or not embedding.is_floating_point():
        raise ValueError("prediction.identity_embedding 必须有正通道数且为浮点张量")
    if embedding.device != targets.center_heatmap.device:
        raise ValueError("prediction 与 targets 必须位于同一设备")


def _masked_binary_cross_entropy(
    logits: Tensor, target: Tensor, mask: Tensor
) -> Tensor:
    values = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return _masked_mean(values, mask)


def _type_loss(logits: Tensor, type_indices: Tensor) -> Tensor:
    valid = type_indices.ne(-1)
    if not bool(valid.any()):
        return logits.sum() * 0.0
    values = F.cross_entropy(
        logits,
        type_indices.to(dtype=torch.long),
        ignore_index=-1,
        reduction="none",
    )
    return values[valid].mean()


def _masked_smooth_l1(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    values = F.smooth_l1_loss(prediction, target, reduction="none")
    return _masked_mean(values, mask)


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    expanded_mask = mask.expand_as(values)
    count = expanded_mask.sum()
    if not bool(count):
        # 保留与对应 head 相连的零值计算图，空目标 batch 仍可安全 backward。
        return values.sum() * 0.0
    return values.masked_select(expanded_mask).mean()


def _instance_margin_loss(
    embedding: Tensor,
    instance_ids: Tensor,
    *,
    margin: float,
) -> Tensor:
    """跨整个时序 batch 计算 prototype pull 与跨实例 cosine margin。

    rasterizer 以 object_id 建立 batch 级稳定实例 ID，因此这里必须把相邻帧
    一起聚合；逐图计算会让 identity head 退化成单帧类别分离。
    """

    vectors = embedding.permute(0, 2, 3, 1)[instance_ids >= 0]
    labels = instance_ids[instance_ids >= 0]
    if vectors.shape[0] == 0:
        return embedding.sum() * 0.0
    vectors = F.normalize(vectors, dim=1)
    unique_ids = torch.unique(labels, sorted=True)
    prototypes: list[Tensor] = []
    pull_terms: list[Tensor] = []
    for instance_id in unique_ids:
        members = vectors[labels == instance_id]
        prototype = F.normalize(members.mean(dim=0), dim=0)
        prototypes.append(prototype)
        pull_terms.append((1.0 - members @ prototype).mean())

    pull = torch.stack(pull_terms).mean()
    if len(prototypes) < 2:
        return pull
    prototype_matrix = torch.stack(prototypes)
    similarity = prototype_matrix @ prototype_matrix.transpose(0, 1)
    off_diagonal = ~torch.eye(
        len(prototypes), dtype=torch.bool, device=similarity.device
    )
    push = F.relu(similarity[off_diagonal] - margin).mean()
    return pull + push


__all__ = ("PerceptionLoss", "PerceptionLossWeights", "compute_perception_loss")
