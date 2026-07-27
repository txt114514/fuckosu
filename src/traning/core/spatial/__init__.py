"""空间模型训练、单帧推理与候选解码的公开入口。"""

from traning.lib.training import (
    decode_slider_paths,
    decode_spatial_candidates,
)
from traning.core.spatial.spatial_inference import (
    SPATIAL_CPU_TASKS,
    SPATIAL_GPU_TASKS,
    SpatialFrameInferenceResult,
    SpatialFrameInferenceRunner,
    prepare_spatial_frame_inference,
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
    "SpatialFrameInferenceRunner",
    "SpatialTrainingResult",
    "decode_slider_paths",
    "decode_spatial_candidates",
    "prepare_spatial_frame_inference",
    "run_spatial_frame_inference",
    "run_spatial_training",
    "slider_path_to_dict",
    "spatial_candidate_to_dict",
]
