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
    """Sample an osu! slider curve as a dense polyline."""
    if len(path) < 2 or not pixel_length:
        return path
    if sample_step_pixels <= 0:
        raise ValueError("sample_step_pixels must be positive")

    curve = Curve.from_kind_and_points(
        curve_type,
        [Position(x, y) for x, y in path],
        pixel_length,
    )
    sample_count = max(2, ceil(pixel_length / sample_step_pixels) + 1)
    return tuple(
        (float(position.x), float(position.y))
        for position in (
            curve(index / (sample_count - 1))
            for index in range(sample_count)
        )
    )


__all__ = ["sample_slider_path"]
