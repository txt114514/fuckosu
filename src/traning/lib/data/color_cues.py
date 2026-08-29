"""从归一化 RGB 帧提取确定性的 osu! 配色、白色字形和边缘提示通道。"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F


ColorCueMode = Literal["disabled", "osu_basic"]

OSU_BASIC_COLOR_CUE_CHANNELS = 3

OSU_OBJECT_COLOR_ANCHORS_RGB: tuple[tuple[float, float, float], ...] = (
    (1.00, 0.10, 0.24),
    (1.00, 0.38, 0.00),
    (1.00, 0.78, 0.00),
    (0.10, 0.95, 0.18),
    (0.00, 0.74, 1.00),
    (0.66, 0.24, 1.00),
)


def color_cue_channel_count(mode: ColorCueMode) -> int:
    """返回指定颜色提示模式追加的确定性通道数量。"""

    if mode == "disabled":
        return 0
    if mode == "osu_basic":
        return OSU_BASIC_COLOR_CUE_CHANNELS
    raise ValueError(f"unsupported color cue mode: {mode}")


def append_color_cues(frame: torch.Tensor, *, mode: ColorCueMode) -> torch.Tensor:
    """在归一化 CHW RGB 帧后追加确定性的 osu! 视觉提示通道。"""

    if mode == "disabled":
        return frame
    cues = extract_osu_basic_color_cues(frame)
    return torch.cat((frame, cues.to(device=frame.device, dtype=frame.dtype)), dim=0)


def extract_osu_basic_color_cues(frame: torch.Tensor) -> torch.Tensor:
    """返回 ``3xHxW`` 的配色、白色字形和物件边缘响应。"""

    if frame.ndim != 3 or frame.shape[0] != 3:
        raise ValueError("color cue extraction expects a normalized 3xHxW RGB frame")
    rgb = frame.to(dtype=torch.float32).clamp(0.0, 1.0)
    red, green, blue = rgb.unbind(dim=0)
    value = torch.maximum(torch.maximum(red, green), blue)
    minimum = torch.minimum(torch.minimum(red, green), blue)
    saturation = (value - minimum) / value.clamp_min(1e-6)

    # 三类提示互补：物件配色、低饱和高亮数字/边框，以及受前两者约束的边缘。
    palette = _palette_response(rgb, saturation=saturation, value=value)
    white_glyph = _white_glyph_response(saturation=saturation, value=value)
    object_edge = _object_edge_response(
        rgb,
        object_prior=torch.maximum(palette, white_glyph),
    )
    return torch.stack((palette, white_glyph, object_edge), dim=0).clamp(0.0, 1.0)


def _palette_response(
    rgb: torch.Tensor,
    *,
    saturation: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    # 用色度而非原始 RGB 比较锚点，可降低整体亮度变化对颜色匹配的影响。
    chroma = rgb / rgb.sum(dim=0, keepdim=True).clamp_min(1e-6)
    anchors = torch.tensor(
        OSU_OBJECT_COLOR_ANCHORS_RGB,
        device=rgb.device,
        dtype=rgb.dtype,
    )
    anchor_chroma = anchors / anchors.sum(dim=1, keepdim=True).clamp_min(1e-6)
    distance = (
        (chroma.unsqueeze(0) - anchor_chroma[:, :, None, None]).square().sum(dim=1)
    )
    nearest = distance.amin(dim=0)
    color_match = torch.exp(-nearest / 0.018)
    return color_match * saturation.clamp(0.0, 1.0) * value.clamp(0.0, 1.0)


def _white_glyph_response(
    *,
    saturation: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    bright = ((value - 0.55) / 0.35).clamp(0.0, 1.0)
    low_saturation = ((0.32 - saturation) / 0.32).clamp(0.0, 1.0)
    return bright * low_saturation


def _object_edge_response(
    rgb: torch.Tensor,
    *,
    object_prior: torch.Tensor,
) -> torch.Tensor:
    # Sobel 响应再乘物件先验，抑制背景纹理被误当成 osu! 轮廓。
    gray = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]).view(
        1, 1, rgb.shape[-2], rgb.shape[-1]
    )
    kernel_x = torch.tensor(
        ((-1.0, 0.0, 1.0), (-2.0, 0.0, 2.0), (-1.0, 0.0, 1.0)),
        device=rgb.device,
        dtype=rgb.dtype,
    ).view(1, 1, 3, 3)
    kernel_y = torch.tensor(
        ((-1.0, -2.0, -1.0), (0.0, 0.0, 0.0), (1.0, 2.0, 1.0)),
        device=rgb.device,
        dtype=rgb.dtype,
    ).view(1, 1, 3, 3)
    grad_x = F.conv2d(gray, kernel_x, padding=1)
    grad_y = F.conv2d(gray, kernel_y, padding=1)
    edge = torch.sqrt(grad_x.square() + grad_y.square())[0, 0]
    edge = edge / edge.amax().clamp_min(1e-6)
    return edge * object_prior.clamp(0.0, 1.0)


__all__ = [
    "ColorCueMode",
    "OSU_BASIC_COLOR_CUE_CHANNELS",
    "OSU_OBJECT_COLOR_ANCHORS_RGB",
    "append_color_cues",
    "color_cue_channel_count",
    "extract_osu_basic_color_cues",
]
