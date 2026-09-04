"""逐轨迹、无隐藏 side state 的 dense Temporal Belief 编码器。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from traning.conf import BeliefConfig
from traning.state import (
    BeliefState,
    ObjectTypeDistribution,
    Point2D,
    TrackLifecycle,
    TrackedObservation,
)


OBJECT_TYPE_ORDER: tuple[str, ...] = ("ring", "slider", "spinner", "unknown")
"""信念模型四分类通道的固定顺序。"""


@dataclass(frozen=True, slots=True)
class BeliefTensorOutput:
    """单步可训练张量输出。

    ``hidden_state`` 固定为 ``[layers, batch, hidden_dim]``，因此调用方不需要
    依赖模型内部状态即可进行完整因果递推。
    """

    hidden_state: torch.Tensor
    position_mean: torch.Tensor
    position_uncertainty: torch.Tensor
    visibility_probability: torch.Tensor
    type_probabilities: torch.Tensor
    uncertainty: torch.Tensor

    def __post_init__(self) -> None:
        tensors = (
            ("hidden_state", self.hidden_state),
            ("position_mean", self.position_mean),
            ("position_uncertainty", self.position_uncertainty),
            ("visibility_probability", self.visibility_probability),
            ("type_probabilities", self.type_probabilities),
            ("uncertainty", self.uncertainty),
        )
        for name, tensor in tensors:
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} 必须是 torch.Tensor")
        if self.hidden_state.ndim != 3:
            raise ValueError("hidden_state 必须采用 [layers, batch, hidden_dim]")
        batch = self.hidden_state.shape[1]
        expected_shapes = (
            ("position_mean", self.position_mean, (batch, 2)),
            ("position_uncertainty", self.position_uncertainty, (batch, 2)),
            ("visibility_probability", self.visibility_probability, (batch, 1)),
            ("type_probabilities", self.type_probabilities, (batch, 4)),
            ("uncertainty", self.uncertainty, (batch, 1)),
        )
        for name, tensor, shape in expected_shapes:
            if tensor.shape != shape:
                raise ValueError(f"{name} shape 必须为 {shape}")


@dataclass(frozen=True, slots=True)
class _ObservationFeatureSpec:
    """观测特征的分组规格，避免训练/契约路径产生两套字段顺序。"""

    appearance_embedding_dim: int

    @property
    def feature_dim(self) -> int:
        """返回按固定字段顺序拼接后的单轨迹观测维度。"""

        # xy、visibility、type(4)、appearance、association、time、missed、age、observed。
        return self.appearance_embedding_dim + 12


class PerTrackBeliefEncoder(nn.Module):
    """projection + 独立多层 GRUCell 的逐轨迹 dense baseline。"""

    def __init__(self, config: BeliefConfig, appearance_embedding_dim: int) -> None:
        super().__init__()
        if not isinstance(config, BeliefConfig):
            raise TypeError("config 必须是 BeliefConfig")
        if isinstance(appearance_embedding_dim, bool) or not isinstance(
            appearance_embedding_dim, int
        ):
            raise TypeError("appearance_embedding_dim 必须是整数")
        if appearance_embedding_dim < 1:
            raise ValueError("appearance_embedding_dim 必须大于 0")
        self.config = config
        self.appearance_embedding_dim = appearance_embedding_dim
        self.feature_spec = _ObservationFeatureSpec(appearance_embedding_dim)
        self.projection = nn.Sequential(
            nn.Linear(self.feature_spec.feature_dim, config.input_dim),
            nn.LayerNorm(config.input_dim),
            nn.SiLU(),
        )
        cells: list[nn.GRUCell] = []
        for layer_index in range(config.layers):
            input_size = config.input_dim if layer_index == 0 else config.hidden_dim
            cells.append(nn.GRUCell(input_size, config.hidden_dim))
        self.cells = nn.ModuleList(cells)
        self.position_delta_head = nn.Linear(config.hidden_dim, 2)
        self.position_uncertainty_head = nn.Linear(config.hidden_dim, 2)
        self.visibility_head = nn.Linear(config.hidden_dim, 1)
        self.type_head = nn.Linear(config.hidden_dim, len(OBJECT_TYPE_ORDER))
        self.uncertainty_head = nn.Linear(config.hidden_dim, 1)

    @property
    def flattened_hidden_dim(self) -> int:
        """契约 ``belief_embedding`` 的精确长度。"""

        return self.config.layers * self.config.hidden_dim

    @property
    def input_feature_dim(self) -> int:
        """训练侧 ``observation_features`` 的精确特征长度。"""

        return self.feature_spec.feature_dim

    def forward_step(
        self,
        observation_features: torch.Tensor,
        previous_hidden: torch.Tensor | None = None,
    ) -> BeliefTensorOutput:
        """执行一个可反向传播的批量因果步骤。"""

        self._validate_feature_tensor(observation_features)
        batch = observation_features.shape[0]
        if previous_hidden is None:
            previous_hidden = observation_features.new_zeros(
                self.config.layers,
                batch,
                self.config.hidden_dim,
            )
        else:
            self._validate_hidden_tensor(previous_hidden, batch=batch)
            if previous_hidden.device != observation_features.device:
                raise ValueError(
                    "previous_hidden 与 observation_features 必须在同一设备"
                )
            if previous_hidden.dtype != observation_features.dtype:
                raise TypeError(
                    "previous_hidden 与 observation_features dtype 必须一致"
                )

        layer_input = self.projection(observation_features)
        next_layers: list[torch.Tensor] = []
        for layer_index, cell in enumerate(self.cells):
            layer_hidden = cell(layer_input, previous_hidden[layer_index])
            next_layers.append(layer_hidden)
            layer_input = layer_hidden
        hidden_state = torch.stack(next_layers, dim=0)
        top_hidden = next_layers[-1]
        # 前两维始终是当前可用的位置先验，head 学习因果修正量。
        position_mean = observation_features[:, :2] + self.position_delta_head(
            top_hidden
        )
        return BeliefTensorOutput(
            hidden_state=hidden_state,
            position_mean=position_mean,
            position_uncertainty=F.softplus(self.position_uncertainty_head(top_hidden)),
            visibility_probability=torch.sigmoid(self.visibility_head(top_hidden)),
            type_probabilities=torch.softmax(self.type_head(top_hidden), dim=1),
            uncertainty=F.softplus(self.uncertainty_head(top_hidden)),
        )

    def step(
        self,
        observation: TrackedObservation,
        previous: BeliefState | None = None,
    ) -> BeliefState:
        """把一个 typed tracking 观测推进为公共 ``BeliefState``。"""

        self._validate_contract_step(observation, previous)
        parameter = next(self.parameters())
        features = self.observation_features(observation, previous)
        previous_hidden = self._hidden_from_belief(
            previous,
            device=parameter.device,
            dtype=parameter.dtype,
        )
        with torch.no_grad():
            output = self.forward_step(features, previous_hidden)
        hidden_values = tuple(
            float(value)
            for value in output.hidden_state[:, 0, :].reshape(-1).cpu().tolist()
        )
        position = output.position_mean[0].cpu().tolist()
        position_uncertainty = output.position_uncertainty[0].cpu().tolist()
        type_values = output.type_probabilities[0].cpu().tolist()
        return BeliefState(
            track_id=observation.track_id,
            timestamp_ms=observation.timestamp_ms,
            belief_embedding=hidden_values,
            position_mean=Point2D(float(position[0]), float(position[1])),
            position_uncertainty=Point2D(
                float(position_uncertainty[0]), float(position_uncertainty[1])
            ),
            visibility_probability=float(
                output.visibility_probability[0, 0].cpu().item()
            ),
            object_type_distribution=ObjectTypeDistribution(
                p_ring=float(type_values[0]),
                p_slider=float(type_values[1]),
                p_spinner=float(type_values[2]),
                p_unknown=float(type_values[3]),
            ),
            age=observation.track_age,
            time_since_seen_ms=observation.time_since_seen_ms,
            uncertainty=float(output.uncertainty[0, 0].cpu().item()),
        )

    def observation_features(
        self,
        observation: TrackedObservation,
        previous: BeliefState | None = None,
    ) -> torch.Tensor:
        """把 typed 观测编码为训练与 runtime 共用的单行特征张量。

        该公开入口避免训练侧复制字段顺序；设备和 dtype 始终跟随 encoder
        参数，调用方不能用另一套缩放或列注册表悄悄改变模型语义。
        """

        self._validate_contract_step(observation, previous)
        parameter = next(self.parameters())
        return self._observation_features(
            observation,
            previous,
            device=parameter.device,
            dtype=parameter.dtype,
        )

    def _validate_feature_tensor(self, features: torch.Tensor) -> None:
        if not isinstance(features, torch.Tensor):
            raise TypeError("observation_features 必须是 torch.Tensor")
        if features.ndim != 2 or features.shape[0] < 1:
            raise ValueError("observation_features 必须是非空 [batch, features]")
        if features.shape[1] != self.feature_spec.feature_dim:
            raise ValueError(
                f"observation_features 特征维必须为 {self.feature_spec.feature_dim}"
            )
        if not features.is_floating_point():
            raise TypeError("observation_features 必须是浮点 Tensor")
        if not bool(torch.isfinite(features).all().item()):
            raise ValueError("observation_features 必须全部有限")

    def _validate_hidden_tensor(self, hidden: torch.Tensor, *, batch: int) -> None:
        if not isinstance(hidden, torch.Tensor):
            raise TypeError("previous_hidden 必须是 torch.Tensor")
        expected = (self.config.layers, batch, self.config.hidden_dim)
        if hidden.shape != expected:
            raise ValueError(f"previous_hidden shape 必须为 {expected}")
        if not hidden.is_floating_point():
            raise TypeError("previous_hidden 必须是浮点 Tensor")
        if not bool(torch.isfinite(hidden).all().item()):
            raise ValueError("previous_hidden 必须全部有限")

    def _validate_contract_step(
        self,
        observation: TrackedObservation,
        previous: BeliefState | None,
    ) -> None:
        if not isinstance(observation, TrackedObservation):
            raise TypeError("observation 必须是 TrackedObservation")
        if previous is not None and not isinstance(previous, BeliefState):
            raise TypeError("previous 必须是 BeliefState 或 None")
        if (
            observation.candidate is not None
            and len(observation.candidate.appearance_embedding)
            != self.appearance_embedding_dim
        ):
            raise ValueError("candidate appearance embedding 维度与 encoder 不一致")
        if previous is None:
            if observation.lifecycle is not TrackLifecycle.NEW:
                raise ValueError("没有 previous belief 时只能接收 NEW 轨迹")
            return
        if observation.lifecycle is TrackLifecycle.NEW:
            raise ValueError("NEW 轨迹不得携带 previous belief")
        if previous.track_id != observation.track_id:
            raise ValueError("previous belief 与 observation 的 track_id 必须一致")
        if len(previous.belief_embedding) != self.flattened_hidden_dim:
            raise ValueError("previous belief_embedding 未携带全部 GRU 层 hidden")
        if observation.timestamp_ms < previous.timestamp_ms:
            raise ValueError("同一轨迹 timestamp_ms 不得回退")
        if observation.track_age != previous.age + 1:
            raise ValueError("observation.track_age 必须等于 previous.age + 1")

    def _observation_features(
        self,
        observation: TrackedObservation,
        previous: BeliefState | None,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        candidate = observation.candidate
        if candidate is not None:
            position = (candidate.x, candidate.y)
            visibility = candidate.visibility_probability
            distribution = candidate.object_type_distribution
            appearance = candidate.appearance_embedding
            observed = 1.0
        else:
            if previous is None:
                raise ValueError("无 candidate 的轨迹必须具有 previous belief")
            position = (previous.position_mean.x, previous.position_mean.y)
            visibility = 0.0
            distribution = previous.object_type_distribution
            appearance = (0.0,) * self.appearance_embedding_dim
            observed = 0.0
        type_values = (
            distribution.p_ring,
            distribution.p_slider,
            distribution.p_spinner,
            distribution.p_unknown,
        )
        time_scale = max(float(self.config.max_time_since_seen_ms), 1.0)
        values = (
            *position,
            visibility,
            *type_values,
            *appearance,
            observation.association_confidence,
            min(observation.time_since_seen_ms / time_scale, 1.0),
            float(observation.missed_frames),
            math.log1p(observation.track_age),
            observed,
        )
        return torch.tensor((values,), device=device, dtype=dtype)

    def _hidden_from_belief(
        self,
        previous: BeliefState | None,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if previous is None:
            return None
        hidden = torch.tensor(previous.belief_embedding, device=device, dtype=dtype)
        return hidden.reshape(self.config.layers, 1, self.config.hidden_dim)


__all__ = ("OBJECT_TYPE_ORDER", "BeliefTensorOutput", "PerTrackBeliefEncoder")
