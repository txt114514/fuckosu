from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class GlobalFeatures:
    """Low-resolution full-frame context features in BCHW layout."""

    dense: torch.Tensor
    pyramid: dict[str, torch.Tensor]
    tokens: torch.Tensor
    stride: int


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LightweightGlobalEncoder(nn.Module):
    """Offline low-resolution full-frame encoder for global object context."""

    def __init__(
        self,
        *,
        in_channels: int = 3,
        input_height: int = 360,
        input_width: int = 640,
        feature_channels: int = 64,
        backbone: str = "lightweight_cnn",
        pretrained: bool = False,
        frozen: bool = False,
    ) -> None:
        super().__init__()
        if input_height <= 0 or input_width <= 0 or feature_channels <= 0:
            raise ValueError("global encoder dimensions must be positive")
        if backbone != "lightweight_cnn":
            if pretrained:
                raise ValueError("pretrained weights require explicit external setup")
            raise NotImplementedError(
                f"global backbone {backbone!r} is only an adapter placeholder"
            )
        self.input_height = input_height
        self.input_width = input_width
        c2 = max(feature_channels // 4, 16)
        c4 = max(feature_channels // 2, 32)
        c8 = feature_channels
        self.stage2 = _ConvBlock(in_channels, c2, stride=2)
        self.stage4 = _ConvBlock(c2, c4, stride=2)
        self.stage8 = _ConvBlock(c4, c8, stride=2)
        self.stage16 = _ConvBlock(c8, feature_channels, stride=2)
        if frozen:
            for parameter in self.parameters():
                parameter.requires_grad_(False)

    def forward(self, frame: torch.Tensor) -> GlobalFeatures:
        if frame.ndim != 4:
            raise ValueError("LightweightGlobalEncoder expects BCHW frame tensors")
        resized = F.interpolate(
            frame,
            size=(self.input_height, self.input_width),
            mode="bilinear",
            align_corners=False,
        )
        p2 = self.stage2(resized)
        p4 = self.stage4(p2)
        p8 = self.stage8(p4)
        p16 = self.stage16(p8)
        tokens = p16.flatten(2).transpose(1, 2).contiguous()
        return GlobalFeatures(
            dense=p16,
            pyramid={
                "stride2": p2,
                "stride4": p4,
                "stride8": p8,
                "stride16": p16,
            },
            tokens=tokens,
            stride=16,
        )


__all__ = ["GlobalFeatures", "LightweightGlobalEncoder"]
