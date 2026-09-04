"""将稠密感知输出确定性解码为运行时候选观测。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as functional

from traning.conf import PerceptionConfig
from traning.state import (
    CandidateObservation,
    ObjectTypeDistribution,
    Point2D,
    RingAttributes,
    SliderAttributes,
    SpinnerAttributes,
)
from traning.state.common import require_identifier, require_nonnegative
from traning.core.perception.models import DensePerceptionOutput


_OFFSET_LIMIT = 0.5


@dataclass(frozen=True, slots=True)
class _DecodedCell:
    """排序和 NMS 期间使用的局部候选，不越过模块边界。"""

    row: int
    column: int
    score: float
    x: float
    y: float


def decode_candidates(
    output: DensePerceptionOutput,
    *,
    frame_id: str,
    frame_index: int,
    timestamp_ms: float,
    frame_width: int,
    frame_height: int,
    config: PerceptionConfig,
    batch_index: int = 0,
) -> tuple[CandidateObservation, ...]:
    """按复合分数、局部极大和确定性 NMS 解码一个 batch 元素。"""

    _validate_decode_request(output, batch_index, frame_width, frame_height)
    require_identifier(frame_id, "frame_id")
    if isinstance(frame_index, bool) or not isinstance(frame_index, int):
        raise TypeError("frame_index 必须是整数")
    if frame_index < 0:
        raise ValueError("frame_index 不得为负数")
    require_nonnegative(timestamp_ms, "timestamp_ms")
    center = torch.sigmoid(output.center_logits[batch_index, 0])
    visibility = torch.sigmoid(output.visibility_logits[batch_index, 0])
    type_probability = torch.softmax(output.type_logits[batch_index], dim=0)
    strongest_type = torch.amax(type_probability, dim=0)
    score_map = center * visibility * strongest_type
    local_maximum = (
        score_map
        == functional.max_pool2d(
            score_map[None, None], kernel_size=3, stride=1, padding=1
        )[0, 0]
    )
    selected_cells = torch.nonzero(
        local_maximum & (score_map >= config.score_threshold), as_tuple=False
    ).to(device="cpu")

    map_height, map_width = score_map.shape
    scale_x = frame_width / map_width
    scale_y = frame_height / map_height
    offsets = output.xy_offsets[batch_index]
    sortable: list[_DecodedCell] = []
    for row_tensor, column_tensor in selected_cells:
        row = int(row_tensor.item())
        column = int(column_tensor.item())
        offset_x = _finite_scalar(offsets[0, row, column], "xy_offsets.x")
        offset_y = _finite_scalar(offsets[1, row, column], "xy_offsets.y")
        x = (column + 0.5 + _clamp_offset(offset_x)) * scale_x
        y = (row + 0.5 + _clamp_offset(offset_y)) * scale_y
        x = min(max(x, 0.0), float(frame_width - 1))
        y = min(max(y, 0.0), float(frame_height - 1))
        sortable.append(
            _DecodedCell(
                row=row,
                column=column,
                score=_finite_scalar(score_map[row, column], "score"),
                x=x,
                y=y,
            )
        )

    # 行列是完全相同分数时的稳定决胜键，不依赖设备上的 nonzero 顺序。
    sortable.sort(key=lambda item: (-item.score, item.row, item.column))
    kept = _deterministic_nms(
        sortable, radius_px=config.nms_radius_px, limit=config.max_candidates
    )
    return tuple(
        _build_observation(
            output,
            cell,
            batch_index=batch_index,
            frame_id=frame_id,
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            scale_x=scale_x,
            scale_y=scale_y,
        )
        for cell in kept
    )


def _validate_decode_request(
    output: DensePerceptionOutput,
    batch_index: int,
    frame_width: int,
    frame_height: int,
) -> None:
    if isinstance(batch_index, bool) or not isinstance(batch_index, int):
        raise TypeError("batch_index 必须是整数")
    if batch_index < 0 or batch_index >= output.center_logits.shape[0]:
        raise IndexError("batch_index 超出 DensePerceptionOutput batch 范围")
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in (frame_width, frame_height)
    ):
        raise TypeError("frame_width 和 frame_height 必须是整数")
    if frame_width < 1 or frame_height < 1:
        raise ValueError("frame_width 和 frame_height 必须为正整数")
    if output.center_logits.shape[-2] < 1 or output.center_logits.shape[-1] < 1:
        raise ValueError("DensePerceptionOutput 空间网格不得为空")
    tensor_fields = (
        ("center_logits", output.center_logits),
        ("visibility_logits", output.visibility_logits),
        ("type_logits", output.type_logits),
        ("xy_offsets", output.xy_offsets),
        ("ring_logits", output.ring_logits),
        ("ring_radius", output.ring_radius),
        ("slider_logits", output.slider_logits),
        ("slider_direction", output.slider_direction),
        ("spinner_logits", output.spinner_logits),
        ("identity_embedding", output.identity_embedding),
    )
    for field_name, tensor in tensor_fields:
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"DensePerceptionOutput.{field_name} 含有非有限数值")


def _deterministic_nms(
    candidates: list[_DecodedCell], *, radius_px: float, limit: int
) -> tuple[_DecodedCell, ...]:
    radius_squared = radius_px * radius_px
    kept: list[_DecodedCell] = []
    for candidate in candidates:
        if any(
            (candidate.x - existing.x) ** 2 + (candidate.y - existing.y) ** 2
            <= radius_squared
            for existing in kept
        ):
            continue
        kept.append(candidate)
        if len(kept) == limit:
            break
    return tuple(kept)


def _build_observation(
    output: DensePerceptionOutput,
    cell: _DecodedCell,
    *,
    batch_index: int,
    frame_id: str,
    frame_index: int,
    timestamp_ms: float,
    scale_x: float,
    scale_y: float,
) -> CandidateObservation:
    row, column = cell.row, cell.column
    # 在 CPU float64 上归一化，保证跨设备概率和满足领域契约精度。
    type_values = torch.softmax(
        output.type_logits[batch_index, :, row, column].to(
            device="cpu", dtype=torch.float64
        ),
        dim=0,
    )
    distribution = ObjectTypeDistribution(
        p_ring=float(type_values[0].item()),
        p_slider=float(type_values[1].item()),
        p_spinner=float(type_values[2].item()),
        p_unknown=float(type_values[3].item()),
    )
    type_index = int(torch.argmax(type_values).item())
    ring: RingAttributes | None = None
    slider: SliderAttributes | None = None
    spinner: SpinnerAttributes | None = None
    if type_index == 0:
        radius_cells = _finite_scalar(
            output.ring_radius[batch_index, 0, row, column], "ring_radius"
        )
        if radius_cells < 0.0:
            raise ValueError("ring_radius 不得为负数")
        ring = RingAttributes(
            probability=_sigmoid_scalar(
                output.ring_logits[batch_index, 0, row, column]
            ),
            radius_px=radius_cells * (scale_x + scale_y) / 2.0,
        )
    elif type_index == 1:
        direction_x = _finite_scalar(
            output.slider_direction[batch_index, 0, row, column], "slider_direction.x"
        )
        direction_y = _finite_scalar(
            output.slider_direction[batch_index, 1, row, column], "slider_direction.y"
        )
        direction_norm = math.hypot(direction_x, direction_y)
        if direction_norm == 0.0:
            raise ValueError("slider_direction 不得为零向量")
        normalized_direction = Point2D(
            direction_x / direction_norm, direction_y / direction_norm
        )
        slider = SliderAttributes(
            probability=_sigmoid_scalar(
                output.slider_logits[batch_index, 0, row, column]
            ),
            direction=normalized_direction,
            path=(),
        )
    elif type_index == 2:
        spinner = SpinnerAttributes(
            probability=_sigmoid_scalar(
                output.spinner_logits[batch_index, 0, row, column]
            )
        )

    embedding_tensor = output.identity_embedding[batch_index, :, row, column].to(
        device="cpu", dtype=torch.float64
    )
    embedding = tuple(float(value.item()) for value in embedding_tensor)
    visibility = _sigmoid_scalar(output.visibility_logits[batch_index, 0, row, column])
    return CandidateObservation(
        frame_id=frame_id,
        frame_index=frame_index,
        timestamp_ms=timestamp_ms,
        candidate_id=f"{frame_id}:candidate:{row:04d}:{column:04d}",
        x=cell.x,
        y=cell.y,
        confidence=cell.score,
        visibility_probability=visibility,
        object_type_distribution=distribution,
        appearance_embedding=embedding,
        ring=ring,
        slider=slider,
        spinner=spinner,
    )


def _finite_scalar(value: Tensor, field_name: str) -> float:
    scalar = float(value.detach().to(device="cpu", dtype=torch.float64).item())
    if not math.isfinite(scalar):
        raise ValueError(f"{field_name} 必须是有限数值")
    return scalar


def _sigmoid_scalar(value: Tensor) -> float:
    return _finite_scalar(torch.sigmoid(value), "sigmoid probability")


def _clamp_offset(value: float) -> float:
    return min(_OFFSET_LIMIT, max(-_OFFSET_LIMIT, value))


__all__ = ("decode_candidates",)
