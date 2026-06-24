from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
import math

import torch

from package.coordinates import OsuVideoTransform
from traning.lib.data import PatchMeta
from traning.lib.models import OBJECT_TYPE_NAMES
from traning.lib.training.losses import SpatialLossTargets


DEFAULT_APPROACH_PREEMPT_MS = 1000.0
DEFAULT_CIRCLE_RADIUS_OSU_PIXELS = 32.0
APPROACH_RADIUS_EXPANSION = 3.0

OBJECT_TYPE_TO_ID = {name: index for index, name in enumerate(OBJECT_TYPE_NAMES)}


def build_spatial_loss_targets(
    sample: Mapping[str, Any],
    patch_meta: PatchMeta,
    feature_size: Sequence[int],
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> SpatialLossTargets:
    """Rasterize one frame sample into dense targets for one patch feature grid."""

    feature_height, feature_width = _normalize_feature_size(feature_size)
    selected_device = torch.device(device) if device is not None else torch.device("cpu")
    target = _empty_targets(
        feature_height=feature_height,
        feature_width=feature_width,
        device=selected_device,
        dtype=dtype,
    )
    grid = _patch_grid(
        patch_meta,
        feature_height=feature_height,
        feature_width=feature_width,
        device=selected_device,
        dtype=dtype,
    )
    transform = OsuVideoTransform.fit_centered(
        patch_meta.frame_width,
        patch_meta.frame_height,
    )
    radius_osu = _finite_float(
        sample.get("circle_radius_osu_pixels"),
        DEFAULT_CIRCLE_RADIUS_OSU_PIXELS,
    )
    hit_radius = transform.osu_radius_to_video(radius_osu)
    timestamp_ms = _finite_float(sample.get("timestamp_ms"), 0.0)
    preempt_ms = _finite_float(
        sample.get("approach_preempt_ms"),
        DEFAULT_APPROACH_PREEMPT_MS,
    )

    objects = sample.get("visible_hit_objects")
    if objects is None:
        objects = sample.get("hit_objects", ())
    for item in objects or ():
        if not isinstance(item, Mapping):
            continue
        kind = _object_kind(item)
        if kind == "spinner":
            _paint_spinner(target, grid, transform=transform)
        elif kind == "slider":
            _paint_slider(
                target,
                grid,
                item,
                transform=transform,
                hit_radius=hit_radius,
            )
        elif kind == "circle":
            _paint_circle(
                target,
                grid,
                item,
                transform=transform,
                hit_radius=hit_radius,
                timestamp_ms=timestamp_ms,
                preempt_ms=preempt_ms,
            )
    return SpatialLossTargets(
        center_heatmap=target["center_heatmap"],
        visible_heatmap=target["visible_heatmap"],
        xy_offset=target["xy_offset"],
        object_type=target["object_type"],
        ring_mask=target["ring_mask"],
        ring_radius=target["ring_radius"],
        slider_mask=target["slider_mask"],
        slider_direction=target["slider_direction"],
        spinner_mask=target["spinner_mask"],
    )


def _normalize_feature_size(feature_size: Sequence[int]) -> tuple[int, int]:
    if len(feature_size) != 2:
        raise ValueError("feature_size must contain height and width")
    height, width = int(feature_size[0]), int(feature_size[1])
    if height <= 0 or width <= 0:
        raise ValueError("feature dimensions must be positive")
    return height, width


def _empty_targets(
    *,
    feature_height: int,
    feature_width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    return {
        "center_heatmap": torch.zeros(
            (1, 1, feature_height, feature_width),
            device=device,
            dtype=dtype,
        ),
        "visible_heatmap": torch.zeros(
            (1, 1, feature_height, feature_width),
            device=device,
            dtype=dtype,
        ),
        "xy_offset": torch.zeros(
            (1, 2, feature_height, feature_width),
            device=device,
            dtype=dtype,
        ),
        "object_type": torch.zeros(
            (1, feature_height, feature_width),
            device=device,
            dtype=torch.long,
        ),
        "ring_mask": torch.zeros(
            (1, 1, feature_height, feature_width),
            device=device,
            dtype=dtype,
        ),
        "ring_radius": torch.zeros(
            (1, 1, feature_height, feature_width),
            device=device,
            dtype=dtype,
        ),
        "slider_mask": torch.zeros(
            (1, 1, feature_height, feature_width),
            device=device,
            dtype=dtype,
        ),
        "slider_direction": torch.zeros(
            (1, 2, feature_height, feature_width),
            device=device,
            dtype=dtype,
        ),
        "spinner_mask": torch.zeros(
            (1, 1, feature_height, feature_width),
            device=device,
            dtype=dtype,
        ),
    }


def _patch_grid(
    patch_meta: PatchMeta,
    *,
    feature_height: int,
    feature_width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor | float]:
    padded_width = max(float(patch_meta.padded_width), 1.0)
    padded_height = max(float(patch_meta.padded_height), 1.0)
    cell_width = padded_width / feature_width
    cell_height = padded_height / feature_height
    xs = (torch.arange(feature_width, device=device, dtype=dtype) + 0.5) * cell_width
    ys = (torch.arange(feature_height, device=device, dtype=dtype) + 0.5) * cell_height
    local_y, local_x = torch.meshgrid(ys, xs, indexing="ij")
    valid_mask = (
        (local_x < float(patch_meta.valid_width))
        & (local_y < float(patch_meta.valid_height))
    )
    return {
        "local_x": local_x,
        "local_y": local_y,
        "valid_mask": valid_mask,
        "cell_width": cell_width,
        "cell_height": cell_height,
        "cell_size": (cell_width + cell_height) / 2.0,
        "patch_x0": float(patch_meta.x0),
        "patch_y0": float(patch_meta.y0),
    }


def _finite_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    return parsed


def _object_kind(item: Mapping[str, Any]) -> str:
    kind = str(item.get("type", "")).strip().lower().replace("-", "_")
    if "spinner" in kind:
        return "spinner"
    if "slider" in kind:
        return "slider"
    if "circle" in kind:
        return "circle"
    return ""


def _set_type(
    target: dict[str, torch.Tensor],
    mask: torch.Tensor,
    object_type: str,
) -> None:
    if object_type not in OBJECT_TYPE_TO_ID:
        raise ValueError(f"unknown object type: {object_type}")
    target["object_type"][0][mask] = OBJECT_TYPE_TO_ID[object_type]


def _set_heatmap_max(
    tensor: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    current = tensor[0, 0]
    current[mask] = torch.maximum(current[mask], values[mask])


def _point_to_local(
    point: tuple[float, float],
    transform: OsuVideoTransform,
    grid: Mapping[str, torch.Tensor | float],
) -> tuple[float, float]:
    video_x, video_y = transform.osu_to_video(point[0], point[1])
    return video_x - float(grid["patch_x0"]), video_y - float(grid["patch_y0"])


def _object_points(item: Mapping[str, Any]) -> tuple[tuple[float, float], ...]:
    raw_path = item.get("path") or ()
    points: list[tuple[float, float]] = []
    for raw_point in raw_path:
        if not isinstance(raw_point, Sequence) or len(raw_point) < 2:
            continue
        x = _finite_float(raw_point[0], float("nan"))
        y = _finite_float(raw_point[1], float("nan"))
        if math.isfinite(x) and math.isfinite(y):
            points.append((x, y))

    x = _finite_float(item.get("x"), float("nan"))
    y = _finite_float(item.get("y"), float("nan"))
    if math.isfinite(x) and math.isfinite(y):
        point = (x, y)
        if not points or _distance(point, points[0]) > 1e-3:
            points.insert(0, point)
    return tuple(points)


def _distance(first: tuple[float, float], second: tuple[float, float]) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def _distance_to_point(
    grid: Mapping[str, torch.Tensor | float],
    *,
    local_x: float,
    local_y: float,
) -> torch.Tensor:
    gx = grid["local_x"]
    gy = grid["local_y"]
    if not isinstance(gx, torch.Tensor) or not isinstance(gy, torch.Tensor):
        raise TypeError("grid coordinates must be tensors")
    return torch.sqrt((gx - local_x).square() + (gy - local_y).square())


def _paint_center(
    target: dict[str, torch.Tensor],
    grid: Mapping[str, torch.Tensor | float],
    *,
    local_x: float,
    local_y: float,
    radius: float,
    object_type: str,
) -> None:
    distance = _distance_to_point(grid, local_x=local_x, local_y=local_y)
    valid_mask = grid["valid_mask"]
    if not isinstance(valid_mask, torch.Tensor):
        raise TypeError("valid_mask must be a tensor")
    cell_size = float(grid["cell_size"])
    sigma = max(radius / max(cell_size, 1e-6) * 0.12, 0.75)
    distance_cells = distance / max(cell_size, 1e-6)
    heatmap = torch.exp(-0.5 * distance_cells.square() / (sigma * sigma))
    _set_heatmap_max(target["center_heatmap"], heatmap, valid_mask)

    disk_radius = max(radius, cell_size * 1.25)
    disk_mask = (distance <= disk_radius) & valid_mask
    _set_heatmap_max(target["visible_heatmap"], torch.ones_like(distance), disk_mask)
    _set_type(target, disk_mask, object_type)
    _write_offset(target, grid, local_x=local_x, local_y=local_y)


def _write_offset(
    target: dict[str, torch.Tensor],
    grid: Mapping[str, torch.Tensor | float],
    *,
    local_x: float,
    local_y: float,
) -> None:
    cell_width = float(grid["cell_width"])
    cell_height = float(grid["cell_height"])
    col = int(local_x / max(cell_width, 1e-6))
    row = int(local_y / max(cell_height, 1e-6))
    _, _, height, width = target["center_heatmap"].shape
    if not (0 <= row < height and 0 <= col < width):
        return
    target["center_heatmap"][0, 0, row, col] = torch.maximum(
        target["center_heatmap"][0, 0, row, col],
        target["center_heatmap"].new_tensor(1.0),
    )
    center_x = (col + 0.5) * cell_width
    center_y = (row + 0.5) * cell_height
    target["xy_offset"][0, 0, row, col] = (local_x - center_x) / max(
        cell_width,
        1e-6,
    )
    target["xy_offset"][0, 1, row, col] = (local_y - center_y) / max(
        cell_height,
        1e-6,
    )


def _paint_circle(
    target: dict[str, torch.Tensor],
    grid: Mapping[str, torch.Tensor | float],
    item: Mapping[str, Any],
    *,
    transform: OsuVideoTransform,
    hit_radius: float,
    timestamp_ms: float,
    preempt_ms: float,
) -> None:
    points = _object_points(item)
    if not points:
        return
    local_x, local_y = _point_to_local(points[0], transform, grid)
    distance = _distance_to_point(grid, local_x=local_x, local_y=local_y)
    valid_mask = grid["valid_mask"]
    if not isinstance(valid_mask, torch.Tensor):
        raise TypeError("valid_mask must be a tensor")

    start_ms = _finite_float(item.get("start_ms"), timestamp_ms)
    time_to_hit = max(start_ms - timestamp_ms, 0.0)
    preempt = max(preempt_ms, 1.0)
    approach_progress = max(0.0, min(time_to_hit / preempt, 1.0))
    approach_radius = hit_radius * (1.0 + APPROACH_RADIUS_EXPANSION * approach_progress)
    cell_size = float(grid["cell_size"])
    ring_band = max(cell_size * 1.5, hit_radius * 0.12)
    ring_mask = (torch.abs(distance - approach_radius) <= ring_band) & valid_mask
    _set_heatmap_max(target["ring_mask"], torch.ones_like(distance), ring_mask)
    target["ring_radius"][0, 0][ring_mask] = approach_radius / max(cell_size, 1e-6)
    _set_heatmap_max(target["visible_heatmap"], torch.ones_like(distance), ring_mask)
    _set_type(target, ring_mask, "approach_circle")

    _paint_center(
        target,
        grid,
        local_x=local_x,
        local_y=local_y,
        radius=hit_radius,
        object_type="hit_circle",
    )


def _paint_slider(
    target: dict[str, torch.Tensor],
    grid: Mapping[str, torch.Tensor | float],
    item: Mapping[str, Any],
    *,
    transform: OsuVideoTransform,
    hit_radius: float,
) -> None:
    points = _object_points(item)
    if not points:
        return
    local_points = tuple(_point_to_local(point, transform, grid) for point in points)
    if len(local_points) >= 2:
        _paint_slider_body(
            target,
            grid,
            local_points,
            tube_radius=max(hit_radius * 0.45, float(grid["cell_size"]) * 1.25),
        )
    else:
        _paint_center(
            target,
            grid,
            local_x=local_points[0][0],
            local_y=local_points[0][1],
            radius=hit_radius,
            object_type="slider_head",
        )
        return

    _paint_center(
        target,
        grid,
        local_x=local_points[0][0],
        local_y=local_points[0][1],
        radius=hit_radius,
        object_type="slider_head",
    )
    _paint_center(
        target,
        grid,
        local_x=local_points[-1][0],
        local_y=local_points[-1][1],
        radius=hit_radius,
        object_type="slider_tail",
    )
    repeats = int(max(_finite_float(item.get("repeats"), 1.0), 1.0))
    if repeats > 1:
        _paint_repeat_points(target, grid, local_points, repeats=repeats, radius=hit_radius)


def _paint_slider_body(
    target: dict[str, torch.Tensor],
    grid: Mapping[str, torch.Tensor | float],
    points: tuple[tuple[float, float], ...],
    *,
    tube_radius: float,
) -> None:
    gx = grid["local_x"]
    gy = grid["local_y"]
    valid_mask = grid["valid_mask"]
    if not (
        isinstance(gx, torch.Tensor)
        and isinstance(gy, torch.Tensor)
        and isinstance(valid_mask, torch.Tensor)
    ):
        raise TypeError("grid coordinates and valid_mask must be tensors")
    nearest_distance = torch.full_like(gx, float("inf"))
    direction_x = torch.zeros_like(gx)
    direction_y = torch.zeros_like(gx)
    for start, end in zip(points, points[1:]):
        ax, ay = start
        bx, by = end
        vx = bx - ax
        vy = by - ay
        length_squared = vx * vx + vy * vy
        if length_squared <= 1e-6:
            continue
        t = ((gx - ax) * vx + (gy - ay) * vy) / length_squared
        t = torch.clamp(t, 0.0, 1.0)
        projection_x = ax + t * vx
        projection_y = ay + t * vy
        distance = torch.sqrt((gx - projection_x).square() + (gy - projection_y).square())
        update = distance < nearest_distance
        nearest_distance = torch.where(update, distance, nearest_distance)
        dx, dy = _unoriented_direction(vx, vy)
        direction_x = torch.where(update, torch.full_like(direction_x, dx), direction_x)
        direction_y = torch.where(update, torch.full_like(direction_y, dy), direction_y)

    body_mask = (nearest_distance <= tube_radius) & valid_mask
    _set_heatmap_max(target["slider_mask"], torch.ones_like(gx), body_mask)
    _set_heatmap_max(target["visible_heatmap"], torch.ones_like(gx), body_mask)
    target["slider_direction"][0, 0][body_mask] = direction_x[body_mask]
    target["slider_direction"][0, 1][body_mask] = direction_y[body_mask]
    _set_type(target, body_mask, "slider_body")


def _unoriented_direction(vx: float, vy: float) -> tuple[float, float]:
    angle = math.atan2(vy, vx)
    return math.cos(2.0 * angle), math.sin(2.0 * angle)


def _paint_repeat_points(
    target: dict[str, torch.Tensor],
    grid: Mapping[str, torch.Tensor | float],
    points: tuple[tuple[float, float], ...],
    *,
    repeats: int,
    radius: float,
) -> None:
    repeat_points = [points[-1]]
    if repeats > 2:
        repeat_points.append(points[0])
    for local_x, local_y in repeat_points:
        distance = _distance_to_point(grid, local_x=local_x, local_y=local_y)
        valid_mask = grid["valid_mask"]
        if not isinstance(valid_mask, torch.Tensor):
            raise TypeError("valid_mask must be a tensor")
        mask = (distance <= max(radius * 0.45, float(grid["cell_size"]))) & valid_mask
        _set_type(target, mask, "slider_repeat")


def _paint_spinner(
    target: dict[str, torch.Tensor],
    grid: Mapping[str, torch.Tensor | float],
    *,
    transform: OsuVideoTransform,
) -> None:
    valid_mask = grid["valid_mask"]
    if not isinstance(valid_mask, torch.Tensor):
        raise TypeError("valid_mask must be a tensor")
    ones = torch.ones_like(target["spinner_mask"][0, 0])
    _set_heatmap_max(target["spinner_mask"], ones, valid_mask)
    _set_heatmap_max(target["visible_heatmap"], ones, valid_mask)
    _set_type(target, valid_mask, "spinner")
    center = (
        transform.playfield_left + transform.playfield_width * 0.5,
        transform.playfield_top + transform.playfield_height * 0.5,
    )
    local_x = center[0] - float(grid["patch_x0"])
    local_y = center[1] - float(grid["patch_y0"])
    _paint_center(
        target,
        grid,
        local_x=local_x,
        local_y=local_y,
        radius=min(transform.playfield_width, transform.playfield_height) * 0.2,
        object_type="spinner",
    )


__all__ = [
    "APPROACH_RADIUS_EXPANSION",
    "DEFAULT_APPROACH_PREEMPT_MS",
    "DEFAULT_CIRCLE_RADIUS_OSU_PIXELS",
    "OBJECT_TYPE_TO_ID",
    "build_spatial_loss_targets",
]
