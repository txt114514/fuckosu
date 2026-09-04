"""基于逐轨迹 belief 的 dense CLICK Outcome baseline。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from traning.conf import OutcomeConfig
from traning.state import BeliefState, OutcomeDistribution


OUTCOME_CATEGORY_COUNT = 5
"""固定类别顺序：invalid、miss、low、medium、high。"""

SCORE_REPRESENTATIVES: tuple[float, ...] = (0.0, 0.0, 0.25, 0.65, 0.90)
"""五个 canonical Outcome 类别的代表分数。"""


@dataclass(frozen=True, slots=True)
class OutcomeTensorOutput:
    """批量 Outcome 的 typed 张量输出。"""

    category_logits: torch.Tensor
    category_probabilities: torch.Tensor
    expiry_logits: torch.Tensor
    expiry_probability: torch.Tensor
    expected_score: torch.Tensor
    variance: torch.Tensor

    def __post_init__(self) -> None:
        values = (
            ("category_logits", self.category_logits),
            ("category_probabilities", self.category_probabilities),
            ("expiry_logits", self.expiry_logits),
            ("expiry_probability", self.expiry_probability),
            ("expected_score", self.expected_score),
            ("variance", self.variance),
        )
        for name, tensor in values:
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} 必须是 torch.Tensor")
        if self.category_logits.ndim != 2:
            raise ValueError("category_logits 必须采用 [batch, 5]")
        batch = self.category_logits.shape[0]
        if batch < 1:
            raise ValueError("OutcomeTensorOutput batch 不得为空")
        if self.category_logits.shape != (batch, OUTCOME_CATEGORY_COUNT):
            raise ValueError("category_logits 必须采用 [batch, 5]")
        if self.category_probabilities.shape != (batch, OUTCOME_CATEGORY_COUNT):
            raise ValueError("category_probabilities 必须采用 [batch, 5]")
        for name, tensor in values[2:]:
            if tensor.shape != (batch,):
                raise ValueError(f"{name} 必须采用 [batch]")
        reference_device = self.category_logits.device
        reference_dtype = self.category_logits.dtype
        for name, tensor in values:
            if not tensor.is_floating_point():
                raise TypeError(f"{name} 必须是浮点 Tensor")
            if tensor.device != reference_device:
                raise ValueError("所有 Outcome 输出必须在同一设备")
            if tensor.dtype != reference_dtype:
                raise TypeError("所有 Outcome 输出 dtype 必须一致")
            if not bool(torch.isfinite(tensor).all().item()):
                raise ValueError(f"{name} 必须全部有限")
        if bool((self.category_probabilities < 0).any().item()) or not torch.allclose(
            self.category_probabilities.sum(dim=1),
            self.category_probabilities.new_ones(batch),
            rtol=1e-6,
            atol=1e-6,
        ):
            raise ValueError("category_probabilities 必须是归一化非负分布")
        if bool(
            ((self.expiry_probability < 0) | (self.expiry_probability > 1)).any().item()
        ):
            raise ValueError("expiry_probability 必须位于 [0, 1]")
        if bool((self.expected_score < 0).any().item()):
            raise ValueError("expected_score 不得为负数")
        if bool((self.variance < 0).any().item()):
            raise ValueError("variance 不得为负数")


class DenseOutcomeModel(nn.Module):
    """用 dense MLP 预测 CLICK 在指定 horizon 的结果分布。

    该 baseline 的动作条件被明确固定为 CLICK。模型在输入特征末尾追加常量
    ``click_action=1``，接口不声称支持 WAIT；未来若增加其他动作，必须升级 typed API。
    """

    def __init__(self, config: OutcomeConfig, belief_embedding_dim: int) -> None:
        super().__init__()
        if not isinstance(config, OutcomeConfig):
            raise TypeError("config 必须是 OutcomeConfig")
        if isinstance(belief_embedding_dim, bool) or not isinstance(
            belief_embedding_dim, int
        ):
            raise TypeError("belief_embedding_dim 必须是整数")
        if belief_embedding_dim < 1:
            raise ValueError("belief_embedding_dim 必须大于 0")
        if config.category_count != OUTCOME_CATEGORY_COUNT:
            raise ValueError("DenseOutcomeModel 固定要求 category_count=5")
        self.config = config
        self.belief_embedding_dim = belief_embedding_dim
        self.horizon_scale_ms = max(float(max(config.horizons_ms)), 1.0)

        input_dim = belief_embedding_dim + 2
        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in config.hidden_dims:
            layers.extend((nn.Linear(previous_dim, hidden_dim), nn.SiLU()))
            previous_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.category_head = nn.Linear(previous_dim, OUTCOME_CATEGORY_COUNT)
        self.expiry_head = nn.Linear(previous_dim, 1)
        self.register_buffer(
            "score_representatives",
            torch.tensor(SCORE_REPRESENTATIVES, dtype=torch.float32),
            persistent=True,
        )

    def forward(
        self,
        belief_embedding: torch.Tensor,
        horizon_ms: torch.Tensor,
    ) -> OutcomeTensorOutput:
        """预测 CLICK 条件下五分类结果与独立 expiry 概率。"""

        self._validate_inputs(belief_embedding, horizon_ms)
        normalized_horizon = horizon_ms.unsqueeze(1) / self.horizon_scale_ms
        click_action = torch.ones_like(normalized_horizon)
        features = torch.cat(
            (belief_embedding, normalized_horizon, click_action), dim=1
        )
        hidden = self.trunk(features)
        category_logits = self.category_head(hidden)
        expiry_logits = self.expiry_head(hidden).squeeze(1)
        category_probabilities = torch.softmax(category_logits, dim=1)
        expiry_probability = torch.sigmoid(expiry_logits)
        representatives = self.score_representatives.to(
            device=category_probabilities.device,
            dtype=category_probabilities.dtype,
        )
        expected_score = (category_probabilities * representatives).sum(dim=1)
        variance = (
            category_probabilities
            * (representatives.unsqueeze(0) - expected_score.unsqueeze(1)).square()
        ).sum(dim=1)
        return OutcomeTensorOutput(
            category_logits=category_logits,
            category_probabilities=category_probabilities,
            expiry_logits=expiry_logits,
            expiry_probability=expiry_probability,
            expected_score=expected_score,
            variance=variance,
        )

    def predict(self, belief: BeliefState, horizon_ms: float) -> OutcomeDistribution:
        """从公共 belief 契约生成单轨迹 canonical OutcomeDistribution。"""

        if not isinstance(belief, BeliefState):
            raise TypeError("belief 必须是 BeliefState")
        if len(belief.belief_embedding) != self.belief_embedding_dim:
            raise ValueError("belief_embedding 维度与 DenseOutcomeModel 不一致")
        if isinstance(horizon_ms, bool) or not isinstance(horizon_ms, (int, float)):
            raise TypeError("horizon_ms 必须是数值")
        if not math.isfinite(float(horizon_ms)) or horizon_ms < 0:
            raise ValueError("horizon_ms 必须是有限非负数")
        parameter = next(self.parameters())
        embedding_tensor = torch.tensor(
            (belief.belief_embedding,),
            device=parameter.device,
            dtype=parameter.dtype,
        )
        horizon_tensor = torch.tensor(
            (float(horizon_ms),),
            device=parameter.device,
            dtype=parameter.dtype,
        )
        with torch.no_grad():
            output = self.forward(embedding_tensor, horizon_tensor)
        probabilities = output.category_probabilities[0].cpu().tolist()
        return OutcomeDistribution(
            track_id=belief.track_id,
            horizon_ms=float(horizon_ms),
            p_invalid=float(probabilities[0]),
            p_miss=float(probabilities[1]),
            p_low_score=float(probabilities[2]),
            p_medium_score=float(probabilities[3]),
            p_high_score=float(probabilities[4]),
            p_expire=float(output.expiry_probability[0].cpu().item()),
            expected_score=float(output.expected_score[0].cpu().item()),
            variance=float(output.variance[0].cpu().item()),
        )

    def _validate_inputs(
        self,
        belief_embedding: torch.Tensor,
        horizon_ms: torch.Tensor,
    ) -> None:
        for name, tensor in (
            ("belief_embedding", belief_embedding),
            ("horizon_ms", horizon_ms),
        ):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} 必须是 torch.Tensor")
            if not tensor.is_floating_point():
                raise TypeError(f"{name} 必须是浮点 Tensor")
            if not bool(torch.isfinite(tensor).all().item()):
                raise ValueError(f"{name} 必须全部有限")
        if belief_embedding.ndim != 2 or belief_embedding.shape[0] < 1:
            raise ValueError("belief_embedding 必须是非空 [batch, dim]")
        if belief_embedding.shape[1] != self.belief_embedding_dim:
            raise ValueError("belief_embedding 特征维与模型不一致")
        if horizon_ms.shape != (belief_embedding.shape[0],):
            raise ValueError("horizon_ms 必须采用 [batch]")
        if belief_embedding.device != horizon_ms.device:
            raise ValueError("belief_embedding 与 horizon_ms 必须在同一设备")
        if belief_embedding.dtype != horizon_ms.dtype:
            raise TypeError("belief_embedding 与 horizon_ms dtype 必须一致")
        if bool((horizon_ms < 0).any().item()):
            raise ValueError("horizon_ms 不得为负数")
        parameter = next(self.parameters())
        if belief_embedding.device != parameter.device:
            raise ValueError("输入与模型参数必须在同一设备")
        if belief_embedding.dtype != parameter.dtype:
            raise TypeError("输入与模型参数 dtype 必须一致")


__all__ = (
    "OUTCOME_CATEGORY_COUNT",
    "SCORE_REPRESENTATIVES",
    "DenseOutcomeModel",
    "OutcomeTensorOutput",
)
