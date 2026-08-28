"""OSU V2 的无 GT 单帧感知、解码、运行时与训练监督入口。"""

from .decode import decode_candidates
from .models import (
    OBJECT_TYPE_ORDER,
    DensePerceptionOutput,
    GatedFusion,
    GlobalEncoder,
    LocalEncoder,
    PerceptionModel,
    SpatialHead,
)
from .runtime import (
    DensePerceptionModel,
    PerceptionRuntime,
    RuntimeTensorFrame,
    decode_runtime_output,
    runtime_frame_to_tensor,
)
from .training import (
    CoordinateTrainingTarget,
    PerceptionLoss,
    PerceptionLossWeights,
    PerceptionTargets,
    build_coordinate_training_targets,
    compute_perception_loss,
    rasterize_perception_targets,
)

__all__ = (
    "OBJECT_TYPE_ORDER",
    "CoordinateTrainingTarget",
    "DensePerceptionModel",
    "DensePerceptionOutput",
    "GatedFusion",
    "GlobalEncoder",
    "LocalEncoder",
    "PerceptionLoss",
    "PerceptionLossWeights",
    "PerceptionModel",
    "PerceptionRuntime",
    "PerceptionTargets",
    "RuntimeTensorFrame",
    "SpatialHead",
    "build_coordinate_training_targets",
    "compute_perception_loss",
    "decode_candidates",
    "decode_runtime_output",
    "runtime_frame_to_tensor",
    "rasterize_perception_targets",
)
