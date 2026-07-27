"""融合重叠 patch 的空间预测，并解码点候选与 slider 折线路径。"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import ceil, floor

import torch
import torch.nn.functional as F

from traning.lib.data import PatchMeta
from traning.lib.models import OBJECT_TYPE_NAMES, SpatialPrediction


@dataclass(frozen=True)
class SpatialPredictionMaps:
    """融合后的完整帧 CPU 特征网格；x/y 候选最终以帧像素表示。"""

    center: torch.Tensor
    visible: torch.Tensor
    xy_offset: torch.Tensor
    object_type_probs: torch.Tensor
    ring: torch.Tensor
    ring_radius: torch.Tensor
    slider: torch.Tensor
    slider_direction: torch.Tensor
    spinner: torch.Tensor
    embedding: torch.Tensor
    weights: torch.Tensor
    frame_width: int
    frame_height: int
    stride: int


@dataclass(frozen=True)
class SpatialCandidate:
    """由一个局部极大网格单元解码出的完整帧像素候选。"""

    x: float
    y: float
    score: float
    object_type: str
    object_type_id: int
    center_score: float
    visible_score: float
    type_score: float
    ring_score: float
    ring_radius_px: float
    slider_score: float
    spinner_score: float
    embedding: tuple[float, ...]


@dataclass(frozen=True)
class SliderPathCandidate:
    """slider 连通分量恢复出的等弧长采样折线及歧义诊断。"""

    component_id: int
    score: float
    continuity: float
    ambiguous: bool
    ambiguity_reasons: tuple[str, ...]
    bbox: tuple[float, float, float, float]
    head: tuple[float, float]
    tail: tuple[float, float]
    polyline: tuple[tuple[float, float], ...]
    cell_count: int
    branch_points: int
    endpoint_count: int


@dataclass(frozen=True)
class SpatialDecodeDiagnostics:
    score_min: float
    score_mean: float
    score_max: float
    feature_variance: float
    center_min: float
    center_max: float
    visible_min: float
    visible_max: float
    type_score_min: float
    type_score_max: float
    raw_cell_count: int
    local_max_count: int
    threshold_candidate_count: int
    nms_candidate_count: int
    max_candidates: int
    score_threshold: float
    nms_radius_px: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "score_min": self.score_min,
            "score_mean": self.score_mean,
            "score_max": self.score_max,
            "feature_variance": self.feature_variance,
            "center_min": self.center_min,
            "center_max": self.center_max,
            "visible_min": self.visible_min,
            "visible_max": self.visible_max,
            "type_score_min": self.type_score_min,
            "type_score_max": self.type_score_max,
            "raw_cell_count": self.raw_cell_count,
            "local_max_count": self.local_max_count,
            "threshold_candidate_count": self.threshold_candidate_count,
            "nms_candidate_count": self.nms_candidate_count,
            "max_candidates": self.max_candidates,
            "score_threshold": self.score_threshold,
            "nms_radius_px": self.nms_radius_px,
        }


class SpatialPredictionCanvas:
    """在 CPU 上融合多个 patch 已 detach 的稠密空间预测。"""

    def __init__(
        self,
        *,
        frame_width: int,
        frame_height: int,
        stride: int,
        object_type_count: int = len(OBJECT_TYPE_NAMES),
        embedding_dim: int,
        dtype: torch.dtype = torch.float32,
        feather_edges: bool = True,
    ) -> None:
        if min(frame_width, frame_height, stride, object_type_count, embedding_dim) <= 0:
            raise ValueError("canvas dimensions must be positive")
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.stride = stride
        self.dtype = dtype
        self.feather_edges = feather_edges
        # canvas 覆盖完整帧；不能整除 stride 的右/下边缘仍保留一个特征单元。
        height = ceil(frame_height / stride)
        width = ceil(frame_width / stride)
        self._values = {
            "center": torch.zeros((1, height, width), dtype=dtype),
            "visible": torch.zeros((1, height, width), dtype=dtype),
            "xy_offset": torch.zeros((2, height, width), dtype=dtype),
            "object_type_probs": torch.zeros((object_type_count, height, width), dtype=dtype),
            "ring": torch.zeros((1, height, width), dtype=dtype),
            "ring_radius": torch.zeros((1, height, width), dtype=dtype),
            "slider": torch.zeros((1, height, width), dtype=dtype),
            "slider_direction": torch.zeros((2, height, width), dtype=dtype),
            "spinner": torch.zeros((1, height, width), dtype=dtype),
            "embedding": torch.zeros((embedding_dim, height, width), dtype=dtype),
        }
        self._weights = torch.zeros((1, height, width), dtype=dtype)

    def write_patch(self, prediction: SpatialPrediction, meta: PatchMeta) -> None:
        """裁掉 padding 后，按完整帧网格位置加权写入一个 patch 预测。"""

        payload = _prediction_to_payload(prediction, dtype=self.dtype)
        feature_height, feature_width = payload["center"].shape[-2:]
        region = _write_region(
            meta,
            feature_height=feature_height,
            feature_width=feature_width,
            frame_height=self.frame_height,
            frame_width=self.frame_width,
            stride=self.stride,
        )
        if region is None:
            return
        crop_y, crop_x, target_y, target_x = region
        # Hann 羽化降低重叠 patch 边缘的卷积 padding 伪影。
        weight = _patch_weight(
            crop_y.stop - crop_y.start,
            crop_x.stop - crop_x.start,
            dtype=self.dtype,
            feather_edges=self.feather_edges,
        )
        target_size = (target_y.stop - target_y.start, target_x.stop - target_x.start)
        if weight.shape[-2:] != target_size:
            weight = F.interpolate(
                weight.unsqueeze(0),
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )[0]
        for name, value in payload.items():
            patch = value[..., crop_y, crop_x]
            if patch.shape[-2:] != target_size:
                patch = F.interpolate(
                    patch.unsqueeze(0),
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )[0]
            self._values[name][..., target_y, target_x] += patch * weight
        self._weights[..., target_y, target_x] += weight

    def to_maps(self) -> SpatialPredictionMaps:
        """完成重叠区加权平均，并重新单位化向量型输出。"""

        weights = self._weights.clamp_min(1e-6)
        values = {name: tensor / weights for name, tensor in self._values.items()}
        values["slider_direction"] = F.normalize(
            values["slider_direction"],
            dim=0,
            eps=1e-6,
        )
        values["embedding"] = F.normalize(values["embedding"], dim=0, eps=1e-6)
        return SpatialPredictionMaps(
            center=values["center"],
            visible=values["visible"],
            xy_offset=values["xy_offset"],
            object_type_probs=values["object_type_probs"],
            ring=values["ring"],
            ring_radius=values["ring_radius"],
            slider=values["slider"],
            slider_direction=values["slider_direction"],
            spinner=values["spinner"],
            embedding=values["embedding"],
            weights=self._weights.clone(),
            frame_width=self.frame_width,
            frame_height=self.frame_height,
            stride=self.stride,
        )


def decode_spatial_candidates(
    maps: SpatialPredictionMaps,
    *,
    max_candidates: int = 32,
    score_threshold: float = 0.05,
    nms_radius_px: float = 32.0,
) -> tuple[SpatialCandidate, ...]:
    """从完整帧网格做局部极大值筛选、阈值过滤和像素空间 NMS。"""

    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive")
    if nms_radius_px < 0:
        raise ValueError("nms_radius_px must be nonnegative")
    non_background = maps.object_type_probs[1:]
    if non_background.shape[0] == 0:
        return ()
    type_score, type_index = non_background.max(dim=0)
    object_type_id = type_index + 1
    # 三项相乘要求候选同时像中心、确实可见且属于非背景类别。
    score_map = maps.center[0] * maps.visible[0].clamp_min(0.05) * type_score
    score_map = score_map * (maps.weights[0] > 0).to(score_map.dtype)
    local_max = score_map == F.max_pool2d(
        score_map.view(1, 1, *score_map.shape),
        kernel_size=3,
        stride=1,
        padding=1,
    )[0, 0]
    mask = local_max & (score_map >= score_threshold)
    coordinates = mask.nonzero(as_tuple=False)
    if coordinates.numel() == 0:
        return ()
    scores = score_map[coordinates[:, 0], coordinates[:, 1]]
    order = torch.argsort(scores, descending=True)
    selected: list[SpatialCandidate] = []
    for ordinal in order.tolist():
        row = int(coordinates[ordinal, 0])
        col = int(coordinates[ordinal, 1])
        offset_x = float(maps.xy_offset[0, row, col].clamp(-0.75, 0.75))
        offset_y = float(maps.xy_offset[1, row, col].clamp(-0.75, 0.75))
        # 网格整数索引代表 cell，0.5 移到中心，再加预测的 cell 单位 offset。
        x = (col + 0.5 + offset_x) * maps.stride
        y = (row + 0.5 + offset_y) * maps.stride
        x = min(max(x, 0.0), float(maps.frame_width - 1))
        y = min(max(y, 0.0), float(maps.frame_height - 1))
        if _is_suppressed(selected, x=x, y=y, radius=nms_radius_px):
            continue
        type_id = int(object_type_id[row, col])
        selected.append(
            SpatialCandidate(
                x=x,
                y=y,
                score=float(score_map[row, col]),
                object_type=OBJECT_TYPE_NAMES[type_id],
                object_type_id=type_id,
                center_score=float(maps.center[0, row, col]),
                visible_score=float(maps.visible[0, row, col]),
                type_score=float(type_score[row, col]),
                ring_score=float(maps.ring[0, row, col]),
                ring_radius_px=float(maps.ring_radius[0, row, col] * maps.stride),
                slider_score=float(maps.slider[0, row, col]),
                spinner_score=float(maps.spinner[0, row, col]),
                embedding=tuple(float(value) for value in maps.embedding[:, row, col]),
            )
        )
        if len(selected) >= max_candidates:
            break
    return tuple(selected)


def diagnose_spatial_candidate_decode(
    maps: SpatialPredictionMaps,
    *,
    selected_count: int,
    max_candidates: int,
    score_threshold: float,
    nms_radius_px: float,
) -> SpatialDecodeDiagnostics:
    non_background = maps.object_type_probs[1:]
    if non_background.shape[0] == 0:
        empty = maps.center.new_zeros(())
        type_score = empty
        score_map = maps.center[0] * 0.0
    else:
        type_score, _ = non_background.max(dim=0)
        score_map = maps.center[0] * maps.visible[0].clamp_min(0.05) * type_score
        score_map = score_map * (maps.weights[0] > 0).to(score_map.dtype)
    local_max = score_map == F.max_pool2d(
        score_map.view(1, 1, *score_map.shape),
        kernel_size=3,
        stride=1,
        padding=1,
    )[0, 0]
    valid = maps.weights[0] > 0
    threshold_mask = local_max & valid & (score_map >= score_threshold)
    return SpatialDecodeDiagnostics(
        score_min=float(score_map[valid].min().item()) if bool(valid.any()) else 0.0,
        score_mean=float(score_map[valid].mean().item()) if bool(valid.any()) else 0.0,
        score_max=float(score_map[valid].max().item()) if bool(valid.any()) else 0.0,
        feature_variance=float(maps.embedding[:, valid].var().item())
        if bool(valid.any())
        else 0.0,
        center_min=float(maps.center[0, valid].min().item()) if bool(valid.any()) else 0.0,
        center_max=float(maps.center[0, valid].max().item()) if bool(valid.any()) else 0.0,
        visible_min=float(maps.visible[0, valid].min().item()) if bool(valid.any()) else 0.0,
        visible_max=float(maps.visible[0, valid].max().item()) if bool(valid.any()) else 0.0,
        type_score_min=float(type_score[valid].min().item())
        if bool(valid.any()) and type_score.ndim
        else 0.0,
        type_score_max=float(type_score[valid].max().item())
        if bool(valid.any()) and type_score.ndim
        else 0.0,
        raw_cell_count=int(valid.sum().item()),
        local_max_count=int((local_max & valid).sum().item()),
        threshold_candidate_count=int(threshold_mask.sum().item()),
        nms_candidate_count=int(selected_count),
        max_candidates=int(max_candidates),
        score_threshold=float(score_threshold),
        nms_radius_px=float(nms_radius_px),
    )


def decode_slider_paths(
    maps: SpatialPredictionMaps,
    *,
    threshold: float = 0.5,
    min_cells: int = 4,
    max_paths: int = 16,
    sample_points: int = 32,
    continuity_threshold: float = 0.75,
) -> tuple[SliderPathCandidate, ...]:
    """从融合后的 CPU 画布恢复首版 slider 路径候选。"""

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if min_cells <= 0:
        raise ValueError("min_cells must be positive")
    if max_paths <= 0:
        raise ValueError("max_paths must be positive")
    if sample_points < 2:
        raise ValueError("sample_points must be at least 2")

    slider = maps.slider[0].detach().to("cpu", dtype=torch.float32)
    valid = maps.weights[0].detach().to("cpu") > 0
    mask = (slider >= threshold) & valid
    # 每个八邻域连通分量独立恢复一条路径，避免跨物件串线。
    components = _connected_components(mask)
    paths: list[SliderPathCandidate] = []
    for component_id, component in enumerate(components):
        if len(component) < min_cells:
            continue
        paths.append(
            _decode_slider_component(
                maps,
                component_id=component_id,
                component=component,
                sample_points=sample_points,
                continuity_threshold=continuity_threshold,
            )
        )
    paths.sort(key=lambda path: (path.score, path.cell_count), reverse=True)
    return tuple(paths[:max_paths])


def _prediction_to_payload(
    prediction: SpatialPrediction,
    *,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    # 分类分支在离开 GPU 前变为概率；连续回归和已单位化向量保持原值。
    tensors = {
        "center": torch.sigmoid(prediction.center_heatmap),
        "visible": torch.sigmoid(prediction.visible_heatmap),
        "xy_offset": prediction.xy_offset,
        "object_type_probs": torch.softmax(prediction.object_type_logits, dim=1),
        "ring": torch.sigmoid(prediction.ring_mask),
        "ring_radius": prediction.ring_radius,
        "slider": torch.sigmoid(prediction.slider_mask),
        "slider_direction": prediction.slider_direction,
        "spinner": torch.sigmoid(prediction.spinner_mask),
        "embedding": prediction.candidate_embedding,
    }
    payload: dict[str, torch.Tensor] = {}
    for name, value in tensors.items():
        if value.ndim != 4 or value.shape[0] != 1:
            raise ValueError("spatial prediction canvas only supports batch size 1")
        payload[name] = value.detach().to("cpu", dtype=dtype)[0]
    return payload


def _write_region(
    meta: PatchMeta,
    *,
    feature_height: int,
    feature_width: int,
    frame_height: int,
    frame_width: int,
    stride: int,
) -> tuple[slice, slice, slice, slice] | None:
    """同时计算 patch 有效特征裁片和完整帧 canvas 目标切片。"""

    # cell 尺寸来自含 padding 的固定 patch，但写入宽高只取未 padding 的有效区。
    cell_width = max(float(meta.padded_width) / feature_width, 1e-6)
    cell_height = max(float(meta.padded_height) / feature_height, 1e-6)
    valid_feature_width = min(feature_width, max(1, ceil(meta.valid_width / cell_width)))
    valid_feature_height = min(
        feature_height,
        max(1, ceil(meta.valid_height / cell_height)),
    )
    target_x0 = max(0, floor(meta.x0 / stride))
    target_y0 = max(0, floor(meta.y0 / stride))
    target_x1 = min(ceil(meta.x1 / stride), ceil(frame_width / stride))
    target_y1 = min(ceil(meta.y1 / stride), ceil(frame_height / stride))
    if target_x1 <= target_x0 or target_y1 <= target_y0:
        return None
    return (
        slice(0, valid_feature_height),
        slice(0, valid_feature_width),
        slice(target_y0, target_y1),
        slice(target_x0, target_x1),
    )


def _patch_weight(
    height: int,
    width: int,
    *,
    dtype: torch.dtype,
    feather_edges: bool,
) -> torch.Tensor:
    if not feather_edges:
        return torch.ones((1, height, width), dtype=dtype)
    y = _hann_axis(height, dtype=dtype)
    x = _hann_axis(width, dtype=dtype)
    # 最小权重防止单 patch 边缘被压成零，也保证除法数值稳定。
    return (y[:, None] * x[None, :]).clamp_min(0.05).unsqueeze(0)


def _hann_axis(size: int, *, dtype: torch.dtype) -> torch.Tensor:
    if size <= 1:
        return torch.ones((size,), dtype=dtype)
    return torch.hann_window(size + 2, periodic=False, dtype=dtype)[1:-1]


def _is_suppressed(
    selected: list[SpatialCandidate],
    *,
    x: float,
    y: float,
    radius: float,
) -> bool:
    if radius <= 0:
        return False
    radius_squared = radius * radius
    return any(
        (candidate.x - x) ** 2 + (candidate.y - y) ** 2 < radius_squared
        for candidate in selected
    )


def _connected_components(mask: torch.Tensor) -> tuple[tuple[tuple[int, int], ...], ...]:
    """用八邻域 BFS 提取二维布尔 mask 的确定性连通分量。"""

    if mask.ndim != 2:
        raise ValueError("connected components expect a 2D mask")
    height, width = mask.shape
    visited = torch.zeros_like(mask, dtype=torch.bool)
    components: list[tuple[tuple[int, int], ...]] = []
    for row_tensor, col_tensor in mask.nonzero(as_tuple=False):
        start = (int(row_tensor), int(col_tensor))
        if bool(visited[start]):
            continue
        visited[start] = True
        queue = deque((start,))
        component: list[tuple[int, int]] = []
        while queue:
            cell = queue.popleft()
            component.append(cell)
            for neighbor in _neighbor_cells(cell, height=height, width=width):
                if bool(visited[neighbor]) or not bool(mask[neighbor]):
                    continue
                visited[neighbor] = True
                queue.append(neighbor)
        components.append(tuple(component))
    return tuple(components)


def _decode_slider_component(
    maps: SpatialPredictionMaps,
    *,
    component_id: int,
    component: tuple[tuple[int, int], ...],
    sample_points: int,
    continuity_threshold: float,
) -> SliderPathCandidate:
    # 把连通分量视为无向图：度数用于端点/分叉诊断，BFS 路径作为中心线近似。
    component_set = set(component)
    degrees = {
        cell: _component_degree(cell, component_set)
        for cell in component
    }
    endpoints = tuple(cell for cell, degree in degrees.items() if degree <= 1)
    branch_points = sum(1 for degree in degrees.values() if degree > 2)
    continuity = max(0.0, 1.0 - branch_points / max(len(component), 1))
    start, end = _select_component_endpoints(component, endpoints)
    ordered = _shortest_component_path(component_set, start=start, end=end)
    if len(ordered) < 2:
        ordered = (start, end)
    ordered = _orient_slider_cells(ordered, maps)
    points = tuple(_cell_to_xy(cell, maps) for cell in ordered)
    polyline = _sample_polyline(points, sample_points=sample_points)
    rows = torch.tensor([row for row, _ in component], dtype=torch.long)
    cols = torch.tensor([col for _, col in component], dtype=torch.long)
    score = float(maps.slider[0, rows, cols].detach().to("cpu").mean())
    reasons = _slider_ambiguity_reasons(
        endpoint_count=len(endpoints),
        branch_points=branch_points,
        continuity=continuity,
        continuity_threshold=continuity_threshold,
        polyline=polyline,
    )
    return SliderPathCandidate(
        component_id=component_id,
        score=score,
        continuity=continuity,
        ambiguous=bool(reasons),
        ambiguity_reasons=reasons,
        bbox=_component_bbox(component, maps),
        head=polyline[0],
        tail=polyline[-1],
        polyline=polyline,
        cell_count=len(component),
        branch_points=branch_points,
        endpoint_count=len(endpoints),
    )


def _neighbor_cells(
    cell: tuple[int, int],
    *,
    height: int,
    width: int,
) -> tuple[tuple[int, int], ...]:
    row, col = cell
    neighbors: list[tuple[int, int]] = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            next_row = row + dy
            next_col = col + dx
            if 0 <= next_row < height and 0 <= next_col < width:
                neighbors.append((next_row, next_col))
    return tuple(neighbors)


def _component_neighbors(
    cell: tuple[int, int],
    component: set[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    row, col = cell
    return tuple(
        neighbor
        for neighbor in (
            (row - 1, col - 1),
            (row - 1, col),
            (row - 1, col + 1),
            (row, col - 1),
            (row, col + 1),
            (row + 1, col - 1),
            (row + 1, col),
            (row + 1, col + 1),
        )
        if neighbor in component
    )


def _component_degree(
    cell: tuple[int, int],
    component: set[tuple[int, int]],
) -> int:
    return len(_component_neighbors(cell, component))


def _select_component_endpoints(
    component: tuple[tuple[int, int], ...],
    endpoints: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int], tuple[int, int]]:
    candidates = endpoints if len(endpoints) >= 2 else component
    return _farthest_pair(candidates)


def _farthest_pair(
    cells: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int], tuple[int, int]]:
    if len(cells) == 1:
        return cells[0], cells[0]
    first = max(cells, key=lambda cell: _cell_distance_squared(cells[0], cell))
    second = max(cells, key=lambda cell: _cell_distance_squared(first, cell))
    return first, second


def _cell_distance_squared(
    first: tuple[int, int],
    second: tuple[int, int],
) -> int:
    return (first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2


def _shortest_component_path(
    component: set[tuple[int, int]],
    *,
    start: tuple[int, int],
    end: tuple[int, int],
) -> tuple[tuple[int, int], ...]:
    parents: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    queue = deque((start,))
    while queue:
        cell = queue.popleft()
        if cell == end:
            break
        for neighbor in _component_neighbors(cell, component):
            if neighbor in parents:
                continue
            parents[neighbor] = cell
            queue.append(neighbor)
    if end not in parents:
        return (start, end)
    path: list[tuple[int, int]] = []
    current: tuple[int, int] | None = end
    while current is not None:
        path.append(current)
        current = parents[current]
    path.reverse()
    return tuple(path)


def _orient_slider_cells(
    cells: tuple[tuple[int, int], ...],
    maps: SpatialPredictionMaps,
) -> tuple[tuple[int, int], ...]:
    if len(cells) < 2:
        return cells
    names = list(OBJECT_TYPE_NAMES)
    if "slider_head" not in names or "slider_tail" not in names:
        return cells
    head_index = names.index("slider_head")
    tail_index = names.index("slider_tail")
    probs = maps.object_type_probs.detach().to("cpu", dtype=torch.float32)
    start = cells[0]
    end = cells[-1]
    # 连通图本身无方向，用 head/tail 类别概率决定折线朝向。
    forward = float(probs[head_index, start[0], start[1]] + probs[tail_index, end[0], end[1]])
    reverse = float(probs[head_index, end[0], end[1]] + probs[tail_index, start[0], start[1]])
    if reverse > forward:
        return tuple(reversed(cells))
    return cells


def _cell_to_xy(
    cell: tuple[int, int],
    maps: SpatialPredictionMaps,
) -> tuple[float, float]:
    row, col = cell
    x = min(max((col + 0.5) * maps.stride, 0.0), float(maps.frame_width - 1))
    y = min(max((row + 0.5) * maps.stride, 0.0), float(maps.frame_height - 1))
    return x, y


def _sample_polyline(
    points: tuple[tuple[float, float], ...],
    *,
    sample_points: int,
) -> tuple[tuple[float, float], ...]:
    """按累计弧长等距重采样折线，使不同 cell 数的候选具有统一长度。"""

    if len(points) <= 1:
        point = points[0] if points else (0.0, 0.0)
        return tuple(point for _ in range(sample_points))
    distances = [0.0]
    for first, second in zip(points, points[1:]):
        segment = ((second[0] - first[0]) ** 2 + (second[1] - first[1]) ** 2) ** 0.5
        distances.append(distances[-1] + segment)
    total = distances[-1]
    if total <= 1e-6:
        return tuple(points[0] for _ in range(sample_points))
    sampled: list[tuple[float, float]] = []
    segment_index = 0
    for step in range(sample_points):
        target = total * step / (sample_points - 1)
        while (
            segment_index < len(distances) - 2
            and distances[segment_index + 1] < target
        ):
            segment_index += 1
        start_distance = distances[segment_index]
        end_distance = distances[segment_index + 1]
        start = points[segment_index]
        end = points[segment_index + 1]
        ratio = (
            0.0
            if end_distance <= start_distance
            else (target - start_distance) / (end_distance - start_distance)
        )
        sampled.append(
            (
                start[0] + (end[0] - start[0]) * ratio,
                start[1] + (end[1] - start[1]) * ratio,
            )
        )
    return tuple(sampled)


def _component_bbox(
    component: tuple[tuple[int, int], ...],
    maps: SpatialPredictionMaps,
) -> tuple[float, float, float, float]:
    rows = [row for row, _ in component]
    cols = [col for _, col in component]
    x0 = min(cols) * maps.stride
    y0 = min(rows) * maps.stride
    x1 = min((max(cols) + 1) * maps.stride, maps.frame_width)
    y1 = min((max(rows) + 1) * maps.stride, maps.frame_height)
    return float(x0), float(y0), float(x1), float(y1)


def _slider_ambiguity_reasons(
    *,
    endpoint_count: int,
    branch_points: int,
    continuity: float,
    continuity_threshold: float,
    polyline: tuple[tuple[float, float], ...],
) -> tuple[str, ...]:
    reasons: list[str] = []
    if endpoint_count != 2:
        reasons.append("endpoint_count")
    if branch_points:
        reasons.append("branch_points")
    if continuity < continuity_threshold:
        reasons.append("low_continuity")
    if len(set(polyline)) < 2:
        reasons.append("short_path")
    return tuple(reasons)


__all__ = [
    "SpatialCandidate",
    "SpatialDecodeDiagnostics",
    "SpatialPredictionCanvas",
    "SpatialPredictionMaps",
    "SliderPathCandidate",
    "diagnose_spatial_candidate_decode",
    "decode_spatial_candidates",
    "decode_slider_paths",
]
