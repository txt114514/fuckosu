"""Perception dense heads 使用的纯训练侧目标契约。"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from traning.state import GroundTruthObject, ObjectType, Point2D, TrainingSample
from traning.core.data.coordinates import (
    FrameCoordinateTransform,
    FramePixelPoint,
    OsuPoint,
)
from traning.core.perception.models import DensePerceptionOutput


_FLOAT_TARGET_CHANNELS = {
    "center_heatmap": 1,
    "visibility": 1,
    "xy_offsets": 2,
    "ring": 1,
    "ring_radius": 1,
    "slider": 1,
    "slider_direction": 2,
    "spinner": 1,
}


@dataclass(frozen=True, slots=True)
class CoordinateTrainingTarget:
    """用共享标定投影到原帧像素的单个感知监督目标。"""

    object_id: str
    object_type: ObjectType
    position: FramePixelPoint
    ring_radius_px: float | None = None
    slider_direction: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        """确保 builder 不会通过宽松字典丢失目标身份或坐标来源。"""

        if not isinstance(self.object_id, str):
            raise TypeError("object_id 必须是字符串")
        if not self.object_id or self.object_id != self.object_id.strip():
            raise ValueError("object_id 必须非空且无首尾空格")
        if not isinstance(self.object_type, ObjectType):
            raise TypeError("object_type 必须是 ObjectType")
        if not isinstance(self.position, FramePixelPoint):
            raise TypeError("position 必须是 FramePixelPoint")
        if self.ring_radius_px is not None:
            if (
                isinstance(self.ring_radius_px, bool)
                or not isinstance(self.ring_radius_px, int | float)
                or not math.isfinite(float(self.ring_radius_px))
                or self.ring_radius_px <= 0.0
            ):
                raise ValueError("ring_radius_px 必须是有限正数")
        if self.slider_direction is not None:
            if (
                not isinstance(self.slider_direction, tuple)
                or len(self.slider_direction) != 2
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int | float)
                    or not math.isfinite(float(value))
                    for value in self.slider_direction
                )
            ):
                raise TypeError("slider_direction 必须是两个有限数值组成的元组")
            if not math.isclose(
                math.hypot(*self.slider_direction), 1.0, rel_tol=1e-6, abs_tol=1e-6
            ):
                raise ValueError("slider_direction 必须是单位向量")
        if self.object_type is ObjectType.RING:
            if self.ring_radius_px is None or self.slider_direction is not None:
                raise ValueError("RING target 必须仅携带 ring_radius_px")
        elif self.object_type is ObjectType.SLIDER:
            if self.slider_direction is None or self.ring_radius_px is not None:
                raise ValueError("SLIDER target 必须仅携带 slider_direction")
        elif self.ring_radius_px is not None or self.slider_direction is not None:
            raise ValueError("SPINNER/UNKNOWN target 不得携带 ring/slider 几何")


def build_coordinate_training_targets(
    sample: TrainingSample,
    coordinate_transform: FrameCoordinateTransform,
) -> tuple[CoordinateTrainingTarget, ...]:
    """将样本中 canonical osu! GT 统一映射成稠密感知使用的原帧坐标。"""

    if not isinstance(sample, TrainingSample):
        raise TypeError("sample 必须是 TrainingSample")
    if not isinstance(coordinate_transform, FrameCoordinateTransform):
        raise TypeError("coordinate_transform 必须是 FrameCoordinateTransform")
    if sample.transform_fingerprint != coordinate_transform.transform_fingerprint:
        raise ValueError(
            "TrainingSample 与 training target builder 的坐标变换指纹不一致"
        )

    targets: list[CoordinateTrainingTarget] = []
    ordered_objects = sorted(
        sample.ground_truth_objects,
        key=lambda item: (item.start_time_ms, item.object_id),
    )
    for ground_truth in ordered_objects:
        if not isinstance(ground_truth, GroundTruthObject):
            raise TypeError("ground_truth_objects 只能包含 GroundTruthObject")
        frame_position = coordinate_transform.ground_truth_to_training_target(
            OsuPoint(ground_truth.position.x, ground_truth.position.y),
            source_frame_width=sample.width,
            source_frame_height=sample.height,
        )
        ring_radius_px = None
        slider_direction = None
        if ground_truth.object_type is ObjectType.RING:
            if ground_truth.radius_osu is None:
                raise ValueError("RING ground truth 缺少 radius_osu")
            ring_radius_px = (
                coordinate_transform.ground_truth_radius_to_training_target(
                    ground_truth.radius_osu,
                    source_frame_width=sample.width,
                    source_frame_height=sample.height,
                )
            )
        elif ground_truth.object_type is ObjectType.SLIDER:
            slider_direction = _slider_direction_to_training_target(
                coordinate_transform,
                start=ground_truth.path[0],
                end=ground_truth.path[1],
                source_frame_width=sample.width,
                source_frame_height=sample.height,
            )
        targets.append(
            CoordinateTrainingTarget(
                object_id=ground_truth.object_id,
                object_type=ground_truth.object_type,
                position=frame_position,
                ring_radius_px=ring_radius_px,
                slider_direction=slider_direction,
            )
        )
    return tuple(targets)


def _slider_direction_to_training_target(
    coordinate_transform: FrameCoordinateTransform,
    *,
    start: Point2D,
    end: Point2D,
    source_frame_width: int,
    source_frame_height: int,
) -> tuple[float, float]:
    """用共享 affine 方程转换 slider 首段，同时允许控制点越出游玩区。

    osu! 曲线控制点可以合法位于 playfield 外，只有命中中心需要 ``OsuPoint``
    的有界契约。这里不裁剪或改写真值，而是用当前 ``FrameCoordinateTransform``
    内同一个已校准 affine 变换投影两端，再将像素差规范成单位方向。
    """

    if not isinstance(start, Point2D) or not isinstance(end, Point2D):
        raise TypeError("slider path 首段必须由 Point2D 组成")
    if (
        source_frame_width != coordinate_transform.source_frame_width
        or source_frame_height != coordinate_transform.source_frame_height
    ):
        raise ValueError("source frame size 与坐标变换标定尺寸不一致")
    return coordinate_transform.ground_truth_direction_to_training_target(
        start,
        end,
        source_frame_width=source_frame_width,
        source_frame_height=source_frame_height,
    )


def rasterize_perception_targets(
    samples: tuple[TrainingSample, ...],
    prediction: DensePerceptionOutput,
    coordinate_transform: FrameCoordinateTransform,
) -> PerceptionTargets:
    """把 canonical GT 映射到与 dense 输出完全同构的监督张量。

    这里使用 decoder 的精确逆方程编码 cell 与 offset，因而训练、推理、
    canonical scoring 和 gallery 均共享同一条坐标链。一个输出 cell 无法表达
    多个中心，遇到碰撞时硬失败，禁止静默覆盖标签。
    """

    if not isinstance(samples, tuple) or any(
        not isinstance(sample, TrainingSample) for sample in samples
    ):
        raise TypeError("samples 必须是 TrainingSample 元组")
    if not isinstance(prediction, DensePerceptionOutput):
        raise TypeError("prediction 必须是 DensePerceptionOutput")
    if not isinstance(coordinate_transform, FrameCoordinateTransform):
        raise TypeError("coordinate_transform 必须是 FrameCoordinateTransform")
    batch, _, map_height, map_width = prediction.center_logits.shape
    if len(samples) != batch:
        raise ValueError("samples 数量必须与 prediction batch 大小一致")
    if map_height < 1 or map_width < 1:
        raise ValueError("prediction 输出网格不得为空")

    reference = prediction.center_logits
    float_kwargs = {"dtype": reference.dtype, "device": reference.device}
    scalar_shape = (batch, 1, map_height, map_width)
    vector_shape = (batch, 2, map_height, map_width)
    center_heatmap = torch.zeros(scalar_shape, **float_kwargs)
    visibility = torch.zeros_like(center_heatmap)
    type_indices = torch.full(
        (batch, map_height, map_width),
        -1,
        dtype=torch.long,
        device=reference.device,
    )
    xy_offsets = torch.zeros(vector_shape, **float_kwargs)
    ring = torch.zeros_like(center_heatmap)
    ring_radius = torch.zeros_like(center_heatmap)
    slider = torch.zeros_like(center_heatmap)
    slider_direction = torch.zeros(vector_shape, **float_kwargs)
    spinner = torch.zeros_like(center_heatmap)
    instance_ids = torch.full_like(type_indices, -1)

    # 全 batch 使用同一 object_id 注册表，使相邻帧中的同一目标拥有同一身份标签。
    object_id_to_instance = {
        object_id: instance_id
        for instance_id, object_id in enumerate(
            sorted(
                {
                    target.object_id
                    for sample in samples
                    for target in sample.ground_truth_objects
                }
            )
        )
    }
    object_type_indices = {
        ObjectType.RING: 0,
        ObjectType.SLIDER: 1,
        ObjectType.SPINNER: 2,
        ObjectType.UNKNOWN: 3,
    }

    for batch_index, sample in enumerate(samples):
        targets = build_coordinate_training_targets(sample, coordinate_transform)
        occupied_cells: set[tuple[int, int]] = set()
        scale_x = sample.width / map_width
        scale_y = sample.height / map_height
        radius_scale = (scale_x + scale_y) / 2.0
        for target in targets:
            grid_x = target.position.x / scale_x
            grid_y = target.position.y / scale_y
            column = min(map_width - 1, math.floor(grid_x))
            row = min(map_height - 1, math.floor(grid_y))
            cell = (row, column)
            if cell in occupied_cells:
                raise ValueError(
                    f"sample {sample.sample_id!r} 的多个 GT 中心落入同一 dense cell {cell}"
                )
            occupied_cells.add(cell)

            center_heatmap[batch_index, 0, row, column] = 1.0
            visibility[batch_index, 0, row, column] = 1.0
            type_indices[batch_index, row, column] = object_type_indices[
                target.object_type
            ]
            xy_offsets[batch_index, 0, row, column] = grid_x - (column + 0.5)
            xy_offsets[batch_index, 1, row, column] = grid_y - (row + 0.5)
            instance_ids[batch_index, row, column] = object_id_to_instance[
                target.object_id
            ]

            if target.object_type is ObjectType.RING:
                if target.ring_radius_px is None:
                    raise ValueError("RING target 缺少 ring_radius_px")
                ring[batch_index, 0, row, column] = 1.0
                ring_radius[batch_index, 0, row, column] = (
                    target.ring_radius_px / radius_scale
                )
            elif target.object_type is ObjectType.SLIDER:
                if target.slider_direction is None:
                    raise ValueError("SLIDER target 缺少 slider_direction")
                slider[batch_index, 0, row, column] = 1.0
                slider_direction[batch_index, :, row, column] = reference.new_tensor(
                    target.slider_direction
                )
            elif target.object_type is ObjectType.SPINNER:
                spinner[batch_index, 0, row, column] = 1.0

    return PerceptionTargets(
        center_heatmap=center_heatmap,
        visibility=visibility,
        type_indices=type_indices,
        xy_offsets=xy_offsets,
        ring=ring,
        ring_radius=ring_radius,
        slider=slider,
        slider_direction=slider_direction,
        spinner=spinner,
        instance_ids=instance_ids,
    )


@dataclass(frozen=True, slots=True)
class PerceptionTargets:
    """与 dense 输出逐头对齐、且不会进入 runtime 的监督张量。"""

    center_heatmap: Tensor
    visibility: Tensor
    type_indices: Tensor
    xy_offsets: Tensor
    ring: Tensor
    ring_radius: Tensor
    slider: Tensor
    slider_direction: Tensor
    spinner: Tensor
    instance_ids: Tensor

    def __post_init__(self) -> None:
        tensors = {
            "center_heatmap": self.center_heatmap,
            "visibility": self.visibility,
            "type_indices": self.type_indices,
            "xy_offsets": self.xy_offsets,
            "ring": self.ring,
            "ring_radius": self.ring_radius,
            "slider": self.slider,
            "slider_direction": self.slider_direction,
            "spinner": self.spinner,
            "instance_ids": self.instance_ids,
        }
        for name, tensor in tensors.items():
            if not isinstance(tensor, Tensor):
                raise TypeError(f"{name} 必须是 torch.Tensor")

        base_shape = self.center_heatmap.shape
        if self.center_heatmap.ndim != 4 or base_shape[1] != 1:
            raise ValueError("center_heatmap 必须是 Bx1xHxW")
        batch, _, height, width = base_shape
        for name, channels in _FLOAT_TARGET_CHANNELS.items():
            tensor = tensors[name]
            if tensor.shape != (batch, channels, height, width):
                raise ValueError(f"{name} 必须是 Bx{channels}xHxW")
            if not tensor.is_floating_point():
                raise TypeError(f"{name} 必须是浮点张量")
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"{name} 含有非有限值")

        expected_index_shape = (batch, height, width)
        for name in ("type_indices", "instance_ids"):
            tensor = tensors[name]
            if tensor.shape != expected_index_shape:
                raise ValueError(f"{name} 必须是 BxHxW")
            if tensor.dtype not in (torch.int32, torch.int64):
                raise TypeError(f"{name} 必须是 int32 或 int64 张量")

        if bool(((self.type_indices < -1) | (self.type_indices > 3)).any()):
            raise ValueError("type_indices 只允许 -1 或 [0, 3]")
        if bool((self.instance_ids < -1).any()):
            raise ValueError("instance_ids 只允许 -1 或非负实例 ID")

        for name in ("center_heatmap", "visibility", "ring", "slider", "spinner"):
            tensor = tensors[name]
            if bool(((tensor < 0.0) | (tensor > 1.0)).any()):
                raise ValueError(f"{name} 必须位于 [0, 1]")

        devices = {tensor.device for tensor in tensors.values()}
        if len(devices) != 1:
            raise ValueError("全部 PerceptionTargets 张量必须位于同一设备")


__all__ = (
    "CoordinateTrainingTarget",
    "PerceptionTargets",
    "build_coordinate_training_targets",
    "rasterize_perception_targets",
)
