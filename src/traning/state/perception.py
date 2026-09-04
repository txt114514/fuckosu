"""感知层输出的结构化 Tensor 契约。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from torch import Tensor

from .observation import Candidate


@dataclass(frozen=True, slots=True)
class CandidateBatch:
    """已经由 ``PerceptionBoundary`` 验证的一帧候选集合。"""

    candidates: tuple[Candidate, ...]


@runtime_checkable
class SpatialPrediction(Protocol):
    """完整帧稠密预测的最小稳定接口。

    ``DensePerceptionOutput`` 无需继承本 Protocol 即可结构化满足契约，因此 state 不反向
    依赖模型实现，也不会复制一份输出 dataclass。
    """

    center_logits: Tensor
    visibility_logits: Tensor
    type_logits: Tensor
    xy_offsets: Tensor
    ring_logits: Tensor
    ring_radius: Tensor
    slider_logits: Tensor
    slider_direction: Tensor
    spinner_logits: Tensor
    identity_embedding: Tensor
    stride: int


__all__ = ["CandidateBatch", "SpatialPrediction"]
