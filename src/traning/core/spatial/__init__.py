"""Spatial candidate detection stage."""

from traning.lib.training import (
    decode_slider_paths,
    decode_spatial_candidates,
)
from traning.core.spatial.spatial_inference import (
    SPATIAL_CPU_TASKS,
    SPATIAL_GPU_TASKS,
    SpatialFrameInferenceResult,
    run_spatial_frame_inference,
    slider_path_to_dict,
    spatial_candidate_to_dict,
)
from traning.core.spatial.spatial_trainer import (
    SpatialTrainingResult,
    run_spatial_training,
)

__all__ = [
    "SPATIAL_CPU_TASKS",
    "SPATIAL_GPU_TASKS",
    "SpatialFrameInferenceResult",
    "SpatialTrainingResult",
    "decode_slider_paths",
    "decode_spatial_candidates",
    "run_spatial_frame_inference",
    "run_spatial_training",
    "slider_path_to_dict",
    "spatial_candidate_to_dict",
]
