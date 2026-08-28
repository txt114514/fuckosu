"""Perception 的 typed 训练目标和损失入口。"""

from .losses import PerceptionLoss, PerceptionLossWeights, compute_perception_loss
from .targets import (
    CoordinateTrainingTarget,
    PerceptionTargets,
    build_coordinate_training_targets,
    rasterize_perception_targets,
)

__all__ = (
    "CoordinateTrainingTarget",
    "PerceptionLoss",
    "PerceptionLossWeights",
    "PerceptionTargets",
    "build_coordinate_training_targets",
    "compute_perception_loss",
    "rasterize_perception_targets",
)
