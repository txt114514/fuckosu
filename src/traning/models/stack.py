from __future__ import annotations

import torch

from traning.conf import Settings
from traning.data import color_cue_channel_count
from traning.models.gated_sparse_fusion import GatedSparseFusion
from traning.models.global_encoder import LightweightGlobalEncoder
from traning.models.global_structure_head import GlobalStructureHead
from traning.models.local_encoder import SmallLocalEncoder
from traning.models.object_heads import SpatialPredictionHead


def build_model_stack(settings: Settings) -> dict[str, torch.nn.Module]:
    """Build the shared local/global/fusion/spatial model stack from settings."""

    local_cfg = settings.local_encoder
    global_cfg = settings.global_encoder
    fusion_cfg = settings.fusion
    input_channels = 3 + color_cue_channel_count(settings.input.color_cues)
    local = SmallLocalEncoder(
        in_channels=input_channels,
        stem_channels=local_cfg.stem_channels,
        feature_channels=local_cfg.feature_channels,
        output_stride=local_cfg.output_stride,
        gradient_checkpointing=settings.memory.gradient_checkpointing,
    )
    global_encoder = LightweightGlobalEncoder(
        in_channels=input_channels,
        input_height=global_cfg.input_height,
        input_width=global_cfg.input_width,
        feature_channels=global_cfg.feature_channels,
        backbone=global_cfg.backbone,
        pretrained=global_cfg.pretrained,
        frozen=global_cfg.frozen,
    )
    fusion = GatedSparseFusion(
        local_channels=local_cfg.feature_channels,
        global_channels=global_cfg.feature_channels,
        hidden_dim=fusion_cfg.hidden_dim,
        heads=fusion_cfg.heads,
        sampling_points=fusion_cfg.sampling_points,
        layers=fusion_cfg.layers,
        enabled=fusion_cfg.mode != "disabled",
    )
    return {
        "local": local,
        "global": global_encoder,
        "structure": GlobalStructureHead(global_cfg.feature_channels),
        "fusion": fusion,
        "head": SpatialPredictionHead(
            local_cfg.feature_channels,
            embedding_dim=local_cfg.embedding_dim,
        ),
    }


__all__ = ["build_model_stack"]
