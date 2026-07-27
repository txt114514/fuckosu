"""将 osu! slider 曲线按像素步长采样成跨模块复用的稠密折线。"""

from __future__ import annotations

from math import ceil

from slider.curve import Curve, Position


def sample_slider_path(
    path: tuple[tuple[float, float], ...],
    *,
    curve_type: str,
    pixel_length: float | None,
    sample_step_pixels: float = 4.0,
) -> tuple[tuple[float, float], ...]:
    """按近似像素间隔采样 osu! slider，返回包含两端点的稠密折线。"""
    if len(path) < 2 or not pixel_length:
        return path
    if sample_step_pixels <= 0:
        raise ValueError("sample_step_pixels must be positive")

    curve = Curve.from_kind_and_points(
        curve_type,
        [Position(x, y) for x, y in path],
        pixel_length,
    )
    # +1 同时保留 t=0 和 t=1；ceil 保证相邻参数采样不粗于请求步长。
    sample_count = max(2, ceil(pixel_length / sample_step_pixels) + 1)
    return tuple(
        (float(position.x), float(position.y))
        for position in (
            curve(index / (sample_count - 1))
            for index in range(sample_count)
        )
    )


__all__ = ["sample_slider_path"]
