"""跨感知、belief 与 outcome 训练器共享的最小契约。"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from torch import Tensor

from .data import FrameBatch, LabelBatch, TrainingBatch


@runtime_checkable
class LossBreakdown(Protocol):
    """所有领域 loss 分解都提供的可微分总损失。"""

    total: Tensor


# 显式别名用于文档和注册表，不引入第二套 batch 字段定义。
LossReport = LossBreakdown


__all__ = [
    "FrameBatch",
    "LabelBatch",
    "LossBreakdown",
    "LossReport",
    "TrainingBatch",
]
