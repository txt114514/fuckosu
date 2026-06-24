from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SyntheticStructure:
    """Small synthetic image bundle for model and fusion smoke tests."""

    image: torch.Tensor
    mask: torch.Tensor
    center: tuple[float, float] | None = None
    radius: float | None = None
    path: tuple[tuple[float, float], ...] = ()


def _coordinate_grid(width: int, height: int) -> tuple[torch.Tensor, torch.Tensor]:
    if width <= 0 or height <= 0:
        raise ValueError("synthetic image dimensions must be positive")
    y = torch.arange(height, dtype=torch.float32).view(height, 1)
    x = torch.arange(width, dtype=torch.float32).view(1, width)
    return x, y


def _image_from_mask(mask: torch.Tensor, *, channels: int = 3) -> torch.Tensor:
    return mask.float().unsqueeze(0).repeat(channels, 1, 1)


def make_cross_patch_ring(
    *,
    width: int = 768,
    height: int = 768,
    center: tuple[float, float] = (384.0, 384.0),
    radius: float = 210.0,
    thickness: float = 8.0,
) -> SyntheticStructure:
    """Create a ring whose circumference crosses four 512px patches."""

    x, y = _coordinate_grid(width, height)
    distance = torch.sqrt((x - center[0]).square() + (y - center[1]).square())
    mask = (distance - radius).abs() <= thickness / 2.0
    return SyntheticStructure(
        image=_image_from_mask(mask),
        mask=mask,
        center=center,
        radius=radius,
    )


def make_boundary_circle(
    *,
    width: int = 768,
    height: int = 512,
    center: tuple[float, float] = (512.0, 256.0),
    radius: float = 48.0,
) -> SyntheticStructure:
    """Create a filled circle centered on a typical patch boundary."""

    x, y = _coordinate_grid(width, height)
    mask = (x - center[0]).square() + (y - center[1]).square() <= radius**2
    return SyntheticStructure(
        image=_image_from_mask(mask),
        mask=mask,
        center=center,
        radius=radius,
    )


def make_cross_patch_slider(
    *,
    width: int = 1152,
    height: int = 512,
    start: tuple[float, float] = (120.0, 256.0),
    end: tuple[float, float] = (1032.0, 256.0),
    thickness: float = 12.0,
) -> SyntheticStructure:
    """Create a long straight slider spanning multiple patch windows."""

    x, y = _coordinate_grid(width, height)
    sx, sy = start
    ex, ey = end
    vx = ex - sx
    vy = ey - sy
    length_sq = max(vx * vx + vy * vy, 1.0)
    t = ((x - sx) * vx + (y - sy) * vy) / length_sq
    t = torch.clamp(t, 0.0, 1.0)
    nearest_x = sx + t * vx
    nearest_y = sy + t * vy
    distance = torch.sqrt((x - nearest_x).square() + (y - nearest_y).square())
    mask = distance <= thickness / 2.0
    return SyntheticStructure(
        image=_image_from_mask(mask),
        mask=mask,
        path=(start, end),
    )


def make_spinner(
    *,
    width: int = 768,
    height: int = 768,
    center: tuple[float, float] = (384.0, 384.0),
    radius: float = 260.0,
) -> SyntheticStructure:
    """Create a large spinner-like disk with a bright rim."""

    x, y = _coordinate_grid(width, height)
    distance = torch.sqrt((x - center[0]).square() + (y - center[1]).square())
    disk = distance <= radius
    rim = (distance - radius).abs() <= 6.0
    image = torch.stack((rim.float(), disk.float() * 0.5, disk.float() * 0.25))
    return SyntheticStructure(image=image, mask=disk, center=center, radius=radius)


def make_noise_background(
    *,
    width: int = 512,
    height: int = 512,
    seed: int = 2026,
) -> SyntheticStructure:
    """Create deterministic noise for background robustness smoke tests."""

    generator = torch.Generator().manual_seed(seed)
    image = torch.rand((3, height, width), generator=generator)
    mask = torch.zeros((height, width), dtype=torch.bool)
    return SyntheticStructure(image=image, mask=mask)


__all__ = [
    "SyntheticStructure",
    "make_boundary_circle",
    "make_cross_patch_ring",
    "make_cross_patch_slider",
    "make_noise_background",
    "make_spinner",
]
