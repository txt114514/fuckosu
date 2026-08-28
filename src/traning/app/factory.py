"""从单一 V2 配置装配正式 runtime 与离线坐标适配器。"""

from __future__ import annotations

import torch

from package import AffineOsuVideoTransform
from traning.app.runtime import V2RuntimePipeline
from traning.belief import PerTrackBeliefEncoder, PerTrackBeliefRuntime
from traning.config import V2Config
from traning.data import FrameCoordinateTransform
from traning.decision import OptimalStoppingPlanner
from traning.outcome import DenseOutcomeModel
from traning.perception import PerceptionModel, PerceptionRuntime
from traning.tracking import MultiObjectTracker
from traning.training.checkpoints import RuntimeModelBundle


def assemble_runtime_pipeline(
    config: V2Config,
    *,
    models: RuntimeModelBundle,
) -> V2RuntimePipeline:
    """用带 checkpoint/坐标身份的模型 bundle 装配唯一 V2 runtime 链路。"""

    if not isinstance(config, V2Config):
        raise TypeError("config 必须是 V2Config")
    if not isinstance(models, RuntimeModelBundle):
        raise TypeError("models 必须是 RuntimeModelBundle")
    perception_model = models.perception_model
    belief_encoder = models.belief_encoder
    outcome_model = models.outcome_model
    coordinate_transform = build_frame_coordinate_transform(config)
    if models.transform_fingerprint != coordinate_transform.transform_fingerprint:
        raise ValueError("checkpoint 模型与启动坐标变换指纹不一致")
    if perception_model.config != config.perception:
        raise ValueError("PerceptionModel config 与 V2Config 不一致")
    if belief_encoder.config != config.belief:
        raise ValueError("Belief encoder config 与 V2Config 不一致")
    if outcome_model.config != config.outcome:
        raise ValueError("Outcome model config 与 V2Config 不一致")
    if belief_encoder.appearance_embedding_dim != config.perception.embedding_dim:
        raise ValueError("Perception embedding_dim 与 Belief encoder 不一致")
    if outcome_model.belief_embedding_dim != belief_encoder.flattened_hidden_dim:
        raise ValueError("Belief embedding_dim 与 Outcome model 不一致")

    device = torch.device(config.runtime.device.value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("V2Config 要求 CUDA，但当前 PyTorch namespace 不可见 CUDA")
    belief_encoder.to(device)
    outcome_model.to(device)
    perception_runtime = PerceptionRuntime(
        perception_model,
        config.perception,
        device=device,
        amp=config.runtime.amp,
    )
    return V2RuntimePipeline(
        perception_runtime=perception_runtime,
        tracker=MultiObjectTracker(config.tracking),
        belief_runtime=PerTrackBeliefRuntime(belief_encoder),
        outcome_model=outcome_model,
        planner=OptimalStoppingPlanner(config.decision),
        coordinate_transform=coordinate_transform,
    )


def build_untrained_runtime_for_smoke(config: V2Config) -> V2RuntimePipeline:
    """构造随机权重 smoke runtime；名称显式禁止将其误当成部署 checkpoint。"""

    if not isinstance(config, V2Config):
        raise TypeError("config 必须是 V2Config")
    perception_model = PerceptionModel(config.perception)
    belief_encoder = PerTrackBeliefEncoder(
        config.belief,
        appearance_embedding_dim=config.perception.embedding_dim,
    )
    outcome_model = DenseOutcomeModel(
        config.outcome,
        belief_embedding_dim=belief_encoder.flattened_hidden_dim,
    )
    coordinate_transform = build_frame_coordinate_transform(config)
    models = RuntimeModelBundle(
        perception_model=perception_model,
        belief_encoder=belief_encoder,
        outcome_model=outcome_model,
        artifact_id="untrained-smoke",
        transform_fingerprint=coordinate_transform.transform_fingerprint,
    )
    return assemble_runtime_pipeline(
        config,
        models=models,
    )


def build_frame_coordinate_transform(config: V2Config) -> FrameCoordinateTransform:
    """从显式 V2 标定装配训练、评分与 gallery 共用的坐标对象。"""

    if not isinstance(config, V2Config):
        raise TypeError("config 必须是 V2Config")
    coordinate_config = config.coordinates
    if coordinate_config.affine_matrix is None:
        raise ValueError("V2Config 缺少 coordinates.affine_matrix")
    if coordinate_config.transform_identity == "unconfigured":
        raise ValueError("V2Config 缺少 coordinates.transform_identity")
    return FrameCoordinateTransform(
        source_frame_width=coordinate_config.source_width,
        source_frame_height=coordinate_config.source_height,
        transform_identity=coordinate_config.transform_identity,
        transform=AffineOsuVideoTransform(coordinate_config.affine_matrix),
    )


__all__ = (
    "assemble_runtime_pipeline",
    "build_frame_coordinate_transform",
    "build_untrained_runtime_for_smoke",
)
