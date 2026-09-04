"""V2 感知模型：完整 RGB 帧到 typed 稠密预测。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from traning.conf import PerceptionConfig


OBJECT_TYPE_ORDER: tuple[str, ...] = ("ring", "slider", "spinner", "unknown")
"""``type_logits`` 通道的固定领域顺序。"""


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


def _require_bchw(name: str, tensor: torch.Tensor) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} 必须是 torch.Tensor")
    if tensor.ndim != 4:
        raise ValueError(f"{name} 必须采用 BCHW 布局")


@dataclass(frozen=True, slots=True)
class LocalFeatureOutput:
    """局部编码器的 stride-8 BCHW 特征。"""

    dense: torch.Tensor
    stride: int = 8

    def __post_init__(self) -> None:
        _require_bchw("local.dense", self.dense)
        if self.stride != 8:
            raise ValueError("local stride 必须为 8")


@dataclass(frozen=True, slots=True)
class GlobalFeatureOutput:
    """全局编码器的 stride-16 BCHW 上下文。"""

    dense: torch.Tensor
    stride: int = 16

    def __post_init__(self) -> None:
        _require_bchw("global.dense", self.dense)
        if self.stride != 16:
            raise ValueError("global stride 必须为 16")


@dataclass(frozen=True, slots=True)
class FusedFeatureOutput:
    """与局部网格对齐的门控融合结果。"""

    dense: torch.Tensor
    global_context: torch.Tensor
    stride: int = 8

    def __post_init__(self) -> None:
        _require_bchw("fusion.dense", self.dense)
        _require_bchw("fusion.global_context", self.global_context)
        if self.dense.shape != self.global_context.shape:
            raise ValueError("融合特征与全局上下文 shape 必须一致")
        if self.stride != 8:
            raise ValueError("fusion stride 必须为 8")


@dataclass(frozen=True, slots=True)
class DensePerceptionOutput:
    """stride-8 网格上的完整稠密感知输出。

    分类相关字段保持 logits；``xy_offsets`` 为 feature-cell 单位，
    ``ring_radius`` 同样以 feature-cell 为单位。
    """

    center_logits: torch.Tensor
    visibility_logits: torch.Tensor
    type_logits: torch.Tensor
    xy_offsets: torch.Tensor
    ring_logits: torch.Tensor
    ring_radius: torch.Tensor
    slider_logits: torch.Tensor
    slider_direction: torch.Tensor
    spinner_logits: torch.Tensor
    identity_embedding: torch.Tensor
    stride: int = 8

    def __post_init__(self) -> None:
        tensors = (
            ("center_logits", self.center_logits, 1),
            ("visibility_logits", self.visibility_logits, 1),
            ("type_logits", self.type_logits, len(OBJECT_TYPE_ORDER)),
            ("xy_offsets", self.xy_offsets, 2),
            ("ring_logits", self.ring_logits, 1),
            ("ring_radius", self.ring_radius, 1),
            ("slider_logits", self.slider_logits, 1),
            ("slider_direction", self.slider_direction, 2),
            ("spinner_logits", self.spinner_logits, 1),
            ("identity_embedding", self.identity_embedding, None),
        )
        reference_shape = self.center_logits.shape[0], self.center_logits.shape[2:]
        for name, tensor, channels in tensors:
            _require_bchw(name, tensor)
            if channels is not None and tensor.shape[1] != channels:
                raise ValueError(f"{name} 通道数必须为 {channels}")
            if (tensor.shape[0], tensor.shape[2:]) != reference_shape:
                raise ValueError("所有稠密输出必须共享 batch 与空间网格")
        if self.identity_embedding.shape[1] < 1:
            raise ValueError("identity_embedding 通道不得为空")
        if self.stride != 8:
            raise ValueError("dense perception stride 必须为 8")


class DepthwiseSeparableConv(nn.Module):
    """深度卷积与逐点卷积组成的低成本卷积块。"""

    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1) -> None:
        super().__init__()
        if min(in_channels, out_channels, stride) < 1:
            raise ValueError("卷积通道和 stride 必须为正整数")
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """执行深度卷积、逐点投影、归一化与激活。"""

        return self.activation(self.norm(self.pointwise(self.depthwise(tensor))))


class _SeparableResidualBlock(nn.Module):
    """带可学习降采样捷径的深度可分离残差块。"""

    def __init__(self, in_channels: int, out_channels: int, *, stride: int) -> None:
        super().__init__()
        self.first = DepthwiseSeparableConv(in_channels, out_channels, stride=stride)
        self.second = DepthwiseSeparableConv(out_channels, out_channels)
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.GroupNorm(_group_count(out_channels), out_channels),
            )
        else:
            self.skip = nn.Identity()
        self.activation = nn.SiLU(inplace=True)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """融合主分支与尺寸匹配的捷径并返回残差特征。"""

        return self.activation(self.second(self.first(tensor)) + self.skip(tensor))


class LocalEncoder(nn.Module):
    """保存细粒度形状的 stride-8 局部特征编码器。"""

    def __init__(self, *, in_channels: int = 3, feature_channels: int = 48) -> None:
        super().__init__()
        if in_channels < 1 or feature_channels < 1:
            raise ValueError("local encoder 通道必须为正整数")
        stem_channels = 16
        middle_channels = 32
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(stem_channels), stem_channels),
            nn.SiLU(inplace=True),
        )
        self.stage2 = _SeparableResidualBlock(stem_channels, middle_channels, stride=2)
        self.stage4 = _SeparableResidualBlock(
            middle_channels, feature_channels, stride=2
        )
        self.stage8 = _SeparableResidualBlock(
            feature_channels, feature_channels, stride=2
        )

    def forward(self, frame: torch.Tensor) -> LocalFeatureOutput:
        """把完整帧编码为保留局部形状的 stride-8 特征。"""

        _require_bchw("frame", frame)
        dense = self.stage8(self.stage4(self.stage2(self.stem(frame))))
        return LocalFeatureOutput(dense=dense)


class _GlobalConvBlock(nn.Module):
    """全局支路使用的两层卷积降采样块。"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """连续执行两层卷积并完成一次二倍全局降采样。"""

        return self.block(tensor)


class GlobalEncoder(nn.Module):
    """以完整帧提供 stride-16 的低分辨率全局上下文。"""

    def __init__(
        self,
        *,
        in_channels: int = 3,
        feature_channels: int = 64,
        pretrained: bool = False,
        frozen: bool = False,
    ) -> None:
        super().__init__()
        if in_channels < 1 or feature_channels < 1:
            raise ValueError("global encoder 通道必须为正整数")
        if pretrained:
            raise ValueError("global_pretrained=True 需要明确权重来源，当前未提供")
        self.stage2 = _GlobalConvBlock(in_channels, 16)
        self.stage4 = _GlobalConvBlock(16, 32)
        self.stage8 = _GlobalConvBlock(32, feature_channels)
        self.stage16 = _GlobalConvBlock(feature_channels, feature_channels)
        self.requires_grad_(not frozen)

    def forward(self, frame: torch.Tensor) -> GlobalFeatureOutput:
        """把完整帧编码为 stride-16 的低分辨率全局上下文。"""

        _require_bchw("frame", frame)
        dense = self.stage16(self.stage8(self.stage4(self.stage2(frame))))
        return GlobalFeatureOutput(dense=dense)


def _normalized_grid(
    *,
    batch: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """构建与完整局部特征网格对齐的归一化采样中心。"""

    y_axis = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    x_axis = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    y_grid, x_grid = torch.meshgrid(y_axis, x_axis, indexing="ij")
    return torch.stack((x_grid, y_grid), dim=-1).unsqueeze(0).expand(batch, -1, -1, -1)


def _safe_channel_normalize(tensor: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    """逐像素归一化，并把精确零向量映射到确定性的第一坐标轴。

    原生 ``F.normalize`` 会保留零向量，使 tracking 的 cosine distance 没有
    定义。这里只给范数过小的位置加入固定微扰，正常向量保持原公式。
    """

    norm = torch.linalg.vector_norm(tensor, dim=1, keepdim=True)
    fallback = torch.zeros_like(tensor)
    fallback[:, :1] = eps
    safe_tensor = tensor + (norm <= eps).to(dtype=tensor.dtype) * fallback
    return F.normalize(safe_tensor, dim=1, eps=eps)


class GatedFusion(nn.Module):
    """通过可学习稀疏采样和逐通道门控融合局部与全局特征。"""

    def __init__(
        self,
        *,
        local_channels: int = 48,
        global_channels: int = 64,
        sampling_points: int = 4,
    ) -> None:
        super().__init__()
        if min(local_channels, global_channels, sampling_points) < 1:
            raise ValueError("fusion 维度必须为正整数")
        self.sampling_points = sampling_points
        self.global_project = nn.Conv2d(
            global_channels, local_channels, kernel_size=1, bias=False
        )
        self.offset_predictor = nn.Conv2d(
            local_channels, sampling_points * 2, kernel_size=1
        )
        self.weight_predictor = nn.Conv2d(
            local_channels, sampling_points, kernel_size=1
        )
        self.gate_predictor = nn.Conv2d(
            local_channels * 2, local_channels, kernel_size=1
        )
        self.refinement = nn.Sequential(
            nn.Conv2d(local_channels, local_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(local_channels), local_channels),
            nn.SiLU(inplace=True),
        )

    def forward(
        self,
        local: LocalFeatureOutput,
        global_features: GlobalFeatureOutput,
    ) -> FusedFeatureOutput:
        """在局部网格采样全局上下文并以逐通道门控完成融合。"""

        local_dense = local.dense
        projected_global = self.global_project(global_features.dense)
        batch, _, height, width = local_dense.shape
        base_grid = _normalized_grid(
            batch=batch,
            height=height,
            width=width,
            device=local_dense.device,
            dtype=local_dense.dtype,
        )
        offsets = self.offset_predictor(local_dense).view(
            batch, self.sampling_points, 2, height, width
        )
        offsets = offsets.permute(0, 1, 3, 4, 2).tanh() * 0.125
        weights = torch.softmax(self.weight_predictor(local_dense), dim=1)
        context = torch.zeros_like(local_dense)
        for point in range(self.sampling_points):
            sampled = F.grid_sample(
                projected_global,
                torch.clamp(base_grid + offsets[:, point], -1.0, 1.0),
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
            context = context + sampled * weights[:, point : point + 1]
        gate = torch.sigmoid(
            self.gate_predictor(torch.cat((local_dense, context), dim=1))
        )
        fused = local_dense * (1.0 + gate) + context * (1.0 - gate)
        fused = fused + self.refinement(fused)
        return FusedFeatureOutput(dense=fused, global_context=context)


class SpatialHead(nn.Module):
    """从融合特征产生全部 V2 稠密图；每个构建 head 都参与 forward。"""

    _HEAD_SPECS: tuple[tuple[str, int], ...] = (
        ("center_logits", 1),
        ("visibility_logits", 1),
        ("type_logits", len(OBJECT_TYPE_ORDER)),
        ("xy_offsets", 2),
        ("ring_logits", 1),
        ("ring_radius", 1),
        ("slider_logits", 1),
        ("slider_direction", 2),
        ("spinner_logits", 1),
    )

    def __init__(self, *, in_channels: int = 48, embedding_dim: int = 32) -> None:
        super().__init__()
        if in_channels < 1 or embedding_dim < 1:
            raise ValueError("spatial head 维度必须为正整数")
        self.trunk = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(in_channels), in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(in_channels), in_channels),
            nn.SiLU(inplace=True),
        )
        self.heads = nn.ModuleDict(
            {
                name: nn.Conv2d(in_channels, channels, kernel_size=1)
                for name, channels in self._HEAD_SPECS
            }
        )
        self.identity_head = nn.Conv2d(in_channels, embedding_dim, kernel_size=1)

    def forward(self, features: FusedFeatureOutput) -> DensePerceptionOutput:
        """由融合特征一次产生全部稠密感知图与身份向量。"""

        hidden = self.trunk(features.dense)
        maps = {name: head(hidden) for name, head in self.heads.items()}
        return DensePerceptionOutput(
            center_logits=maps["center_logits"],
            visibility_logits=maps["visibility_logits"],
            type_logits=maps["type_logits"],
            xy_offsets=0.5 * torch.tanh(maps["xy_offsets"]),
            ring_logits=maps["ring_logits"],
            ring_radius=F.softplus(maps["ring_radius"]),
            slider_logits=maps["slider_logits"],
            slider_direction=_safe_channel_normalize(maps["slider_direction"]),
            spinner_logits=maps["spinner_logits"],
            identity_embedding=_safe_channel_normalize(self.identity_head(hidden)),
        )


class PerceptionModel(nn.Module):
    """完整 RGB 帧的统一感知图：local → global → fusion → spatial。"""

    def __init__(self, config: PerceptionConfig) -> None:
        super().__init__()
        if not isinstance(config, PerceptionConfig):
            raise TypeError("config 必须是 PerceptionConfig")
        if config.input_channels != 3:
            raise ValueError("PerceptionModel 只接受 full-frame RGB（三通道）输入")
        if config.global_pretrained:
            raise ValueError("global_pretrained=True 需要明确权重来源，当前未提供")
        self.config = config
        self.local_encoder = LocalEncoder(in_channels=config.input_channels)
        self.global_encoder = GlobalEncoder(
            in_channels=config.input_channels,
            pretrained=config.global_pretrained,
            frozen=config.global_frozen,
        )
        self.fusion = GatedFusion()
        self.spatial_head = SpatialHead(embedding_dim=config.embedding_dim)

    def forward(self, frame: torch.Tensor) -> DensePerceptionOutput:
        """执行无 GT 的 full-frame local→global→fusion→spatial 推理。"""

        _require_bchw("frame", frame)
        if not frame.is_floating_point():
            raise TypeError("frame 必须是浮点 Tensor")
        if frame.shape[1] != self.config.input_channels:
            raise ValueError("frame 通道数与 PerceptionConfig 不一致")
        if frame.shape[-2:] != (
            self.config.frame_height,
            self.config.frame_width,
        ):
            raise ValueError("frame 空间尺寸与 PerceptionConfig 不一致")
        local = self.local_encoder(frame)
        global_features = self.global_encoder(frame)
        fused = self.fusion(local, global_features)
        return self.spatial_head(fused)


__all__ = (
    "OBJECT_TYPE_ORDER",
    "DensePerceptionOutput",
    "DepthwiseSeparableConv",
    "FusedFeatureOutput",
    "GatedFusion",
    "GlobalEncoder",
    "GlobalFeatureOutput",
    "LocalEncoder",
    "LocalFeatureOutput",
    "PerceptionModel",
    "SpatialHead",
)
