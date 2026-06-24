from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class LocalFeatures:
    """High-resolution patch features.

    ``dense`` uses BCHW layout at ``stride`` relative to the patch image.
    ``pyramid`` contains progressively coarser BCHW feature maps.
    """

    dense: torch.Tensor
    pyramid: dict[str, torch.Tensor]
    stride: int


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1) -> None:
        super().__init__()
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
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.pointwise(self.depthwise(x))))


class SeparableResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(
            in_channels,
            out_channels,
            stride=stride,
        )
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels)
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
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv2(self.conv1(x)) + self.skip(x))


class SmallLocalEncoder(nn.Module):
    """Small-channel local CNN for serial high-resolution patch training."""

    def __init__(
        self,
        *,
        in_channels: int = 3,
        stem_channels: int = 8,
        feature_channels: int = 48,
        output_stride: int = 8,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if output_stride != 8:
            raise ValueError("SmallLocalEncoder currently supports output_stride=8")
        if in_channels <= 0 or stem_channels <= 0 or feature_channels <= 0:
            raise ValueError("encoder channels must be positive")
        c2 = max(stem_channels * 2, feature_channels // 3)
        c4 = max(stem_channels * 4, feature_channels * 2 // 3)
        c8 = feature_channels
        self.gradient_checkpointing = gradient_checkpointing
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                stem_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(_group_count(stem_channels), stem_channels),
            nn.SiLU(inplace=True),
        )
        self.stage2 = SeparableResidualBlock(stem_channels, c2, stride=2)
        self.stage4 = SeparableResidualBlock(c2, c4, stride=2)
        self.stage8 = SeparableResidualBlock(c4, c8, stride=2)
        self.p2_project = nn.Conv2d(c2, feature_channels, kernel_size=1)
        self.p4_project = nn.Conv2d(c4, feature_channels, kernel_size=1)
        self.p8_project = nn.Conv2d(c8, feature_channels, kernel_size=1)

    def _maybe_checkpoint(
        self,
        module: Callable[[torch.Tensor], torch.Tensor],
        x: torch.Tensor,
    ) -> torch.Tensor:
        if self.gradient_checkpointing and self.training and x.requires_grad:
            return checkpoint(module, x, use_reentrant=False)
        return module(x)

    def forward(self, patch: torch.Tensor) -> LocalFeatures:
        if patch.ndim != 4:
            raise ValueError("SmallLocalEncoder expects BCHW patch tensors")
        p1 = self.stem(patch)
        p2 = self._maybe_checkpoint(self.stage2, p1)
        p4 = self._maybe_checkpoint(self.stage4, p2)
        p8 = self._maybe_checkpoint(self.stage8, p4)
        pyramid = {
            "stride2": self.p2_project(p2),
            "stride4": self.p4_project(p4),
            "stride8": self.p8_project(p8),
        }
        return LocalFeatures(dense=pyramid["stride8"], pyramid=pyramid, stride=8)


__all__ = [
    "DepthwiseSeparableConv",
    "LocalFeatures",
    "SeparableResidualBlock",
    "SmallLocalEncoder",
]
