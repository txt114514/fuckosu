from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor

import torch
import torch.nn.functional as F

from traning.data import PatchMeta
from traning.models import OBJECT_TYPE_NAMES, SpatialPrediction


@dataclass(frozen=True)
class SpatialPredictionMaps:
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


class SpatialPredictionCanvas:
    """CPU canvas for fusing detached dense spatial predictions across patches."""

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
    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive")
    if nms_radius_px < 0:
        raise ValueError("nms_radius_px must be nonnegative")
    non_background = maps.object_type_probs[1:]
    if non_background.shape[0] == 0:
        return ()
    type_score, type_index = non_background.max(dim=0)
    object_type_id = type_index + 1
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


def _prediction_to_payload(
    prediction: SpatialPrediction,
    *,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
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


__all__ = [
    "SpatialCandidate",
    "SpatialPredictionCanvas",
    "SpatialPredictionMaps",
    "decode_spatial_candidates",
]
