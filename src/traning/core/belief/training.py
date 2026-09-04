"""Temporal Belief 的 typed 监督批次、损失和状态提交辅助函数。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from traning.state import (
    BeliefState,
    ObjectType,
    ObjectTypeDistribution,
    Point2D,
    TrackedObservation,
)

from .encoder import BeliefTensorOutput, PerTrackBeliefEncoder


_OBJECT_TYPE_INDEX = {
    ObjectType.RING: 0,
    ObjectType.SLIDER: 1,
    ObjectType.SPINNER: 2,
    ObjectType.UNKNOWN: 3,
}


@dataclass(frozen=True, slots=True)
class BeliefTrainingRecord:
    """一条逐轨迹观测及其仅训练侧可见的状态监督。"""

    observation: TrackedObservation
    previous: BeliefState | None
    target_position: Point2D
    target_visibility: float
    target_object_type: ObjectType

    def __post_init__(self) -> None:
        if not isinstance(self.observation, TrackedObservation):
            raise TypeError("observation 必须是 TrackedObservation")
        if self.previous is not None and not isinstance(self.previous, BeliefState):
            raise TypeError("previous 必须是 BeliefState 或 None")
        if not isinstance(self.target_position, Point2D):
            raise TypeError("target_position 必须是 Point2D")
        if (
            isinstance(self.target_visibility, bool)
            or not isinstance(self.target_visibility, int | float)
            or not 0.0 <= float(self.target_visibility) <= 1.0
        ):
            raise ValueError("target_visibility 必须位于 [0, 1]")
        if not isinstance(self.target_object_type, ObjectType):
            raise TypeError("target_object_type 必须是 ObjectType")


@dataclass(frozen=True, slots=True)
class BeliefTrainingBatch:
    """保持轨迹身份的可微分 belief 训练批次。"""

    records: tuple[BeliefTrainingRecord, ...]
    observation_features: torch.Tensor
    previous_hidden: torch.Tensor
    target_positions: torch.Tensor
    target_visibility: torch.Tensor
    target_type_indices: torch.Tensor

    def __post_init__(self) -> None:
        batch = len(self.records)
        if batch < 1:
            raise ValueError("BeliefTrainingBatch 不得为空")
        if any(not isinstance(item, BeliefTrainingRecord) for item in self.records):
            raise TypeError("records 只能包含 BeliefTrainingRecord")
        if (
            self.observation_features.ndim != 2
            or self.observation_features.shape[0] != batch
        ):
            raise ValueError("observation_features 必须采用 [batch, features]")
        if self.previous_hidden.ndim != 3 or self.previous_hidden.shape[1] != batch:
            raise ValueError("previous_hidden 必须采用 [layers, batch, hidden]")
        expected_shapes = (
            (self.target_positions, (batch, 2), "target_positions"),
            (self.target_visibility, (batch, 1), "target_visibility"),
            (self.target_type_indices, (batch,), "target_type_indices"),
        )
        for tensor, shape, name in expected_shapes:
            if tensor.shape != shape:
                raise ValueError(f"{name} shape 必须为 {shape}")
        devices = {
            self.observation_features.device,
            self.previous_hidden.device,
            self.target_positions.device,
            self.target_visibility.device,
            self.target_type_indices.device,
        }
        if len(devices) != 1:
            raise ValueError("belief batch 全部张量必须位于同一设备")


@dataclass(frozen=True, slots=True)
class BeliefLoss:
    """Belief 位置、可见性、类型和不确定性监督分解。"""

    total: torch.Tensor
    position: torch.Tensor
    visibility: torch.Tensor
    object_type: torch.Tensor
    uncertainty: torch.Tensor
    position_mae: torch.Tensor


def collate_belief_records(
    encoder: PerTrackBeliefEncoder,
    records: tuple[BeliefTrainingRecord, ...],
) -> BeliefTrainingBatch:
    """通过 encoder 的唯一特征入口拼接逐轨迹训练记录。"""

    if not isinstance(encoder, PerTrackBeliefEncoder):
        raise TypeError("encoder 必须是 PerTrackBeliefEncoder")
    if not isinstance(records, tuple) or not records:
        raise ValueError("records 必须是非空 BeliefTrainingRecord 元组")
    if any(not isinstance(item, BeliefTrainingRecord) for item in records):
        raise TypeError("records 只能包含 BeliefTrainingRecord")
    parameter = next(encoder.parameters())
    features = torch.cat(
        tuple(
            encoder.observation_features(record.observation, record.previous)
            for record in records
        ),
        dim=0,
    )
    hidden_columns: list[torch.Tensor] = []
    for record in records:
        if record.previous is None:
            hidden = parameter.new_zeros(
                encoder.config.layers,
                encoder.config.hidden_dim,
            )
        else:
            hidden = parameter.new_tensor(record.previous.belief_embedding).reshape(
                encoder.config.layers,
                encoder.config.hidden_dim,
            )
        hidden_columns.append(hidden)
    previous_hidden = torch.stack(hidden_columns, dim=1)
    return BeliefTrainingBatch(
        records=records,
        observation_features=features,
        previous_hidden=previous_hidden,
        target_positions=parameter.new_tensor(
            tuple((item.target_position.x, item.target_position.y) for item in records)
        ),
        target_visibility=parameter.new_tensor(
            tuple((float(item.target_visibility),) for item in records)
        ),
        target_type_indices=torch.tensor(
            tuple(_OBJECT_TYPE_INDEX[item.target_object_type] for item in records),
            dtype=torch.long,
            device=parameter.device,
        ),
    )


def compute_belief_loss(
    output: BeliefTensorOutput,
    batch: BeliefTrainingBatch,
) -> BeliefLoss:
    """监督可部署 belief 字段，并让不确定性拟合真实位置残差。"""

    if not isinstance(output, BeliefTensorOutput):
        raise TypeError("output 必须是 BeliefTensorOutput")
    if not isinstance(batch, BeliefTrainingBatch):
        raise TypeError("batch 必须是 BeliefTrainingBatch")
    position_error = output.position_mean - batch.target_positions
    position = F.smooth_l1_loss(output.position_mean, batch.target_positions)
    visibility = F.binary_cross_entropy(
        output.visibility_probability,
        batch.target_visibility,
    )
    object_type = F.nll_loss(
        torch.log(output.type_probabilities.clamp_min(1e-8)),
        batch.target_type_indices,
    )
    absolute_error = position_error.detach().abs()
    uncertainty = F.smooth_l1_loss(
        output.position_uncertainty,
        absolute_error,
    ) + F.smooth_l1_loss(
        output.uncertainty,
        absolute_error.mean(dim=1, keepdim=True),
    )
    total = position + visibility + object_type + 0.25 * uncertainty
    return BeliefLoss(
        total=total,
        position=position,
        visibility=visibility,
        object_type=object_type,
        uncertainty=uncertainty,
        position_mae=absolute_error.mean(),
    )


def belief_states_from_output(
    output: BeliefTensorOutput,
    batch: BeliefTrainingBatch,
) -> tuple[BeliefState, ...]:
    """将 detached 模型输出提交为下一因果步使用的公共 BeliefState。"""

    if output.hidden_state.shape[1] != len(batch.records):
        raise ValueError("output batch 与 BeliefTrainingBatch 不一致")
    hidden = output.hidden_state.detach().cpu()
    positions = output.position_mean.detach().cpu()
    position_uncertainty = output.position_uncertainty.detach().cpu()
    visibility = output.visibility_probability.detach().cpu()
    type_probabilities = output.type_probabilities.detach().cpu()
    uncertainty = output.uncertainty.detach().cpu()
    states: list[BeliefState] = []
    for index, record in enumerate(batch.records):
        observation = record.observation
        distribution = type_probabilities[index].tolist()
        states.append(
            BeliefState(
                track_id=observation.track_id,
                timestamp_ms=observation.timestamp_ms,
                belief_embedding=tuple(
                    float(value) for value in hidden[:, index, :].reshape(-1).tolist()
                ),
                position_mean=Point2D(
                    float(positions[index, 0]),
                    float(positions[index, 1]),
                ),
                position_uncertainty=Point2D(
                    float(position_uncertainty[index, 0]),
                    float(position_uncertainty[index, 1]),
                ),
                visibility_probability=float(visibility[index, 0]),
                object_type_distribution=ObjectTypeDistribution(
                    p_ring=float(distribution[0]),
                    p_slider=float(distribution[1]),
                    p_spinner=float(distribution[2]),
                    p_unknown=float(distribution[3]),
                ),
                age=observation.track_age,
                time_since_seen_ms=observation.time_since_seen_ms,
                uncertainty=float(uncertainty[index, 0]),
            )
        )
    return tuple(states)


__all__ = (
    "BeliefLoss",
    "BeliefTrainingBatch",
    "BeliefTrainingRecord",
    "belief_states_from_output",
    "collate_belief_records",
    "compute_belief_loss",
)
