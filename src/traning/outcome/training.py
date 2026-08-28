"""Dense Outcome 模型的批处理、损失与单步训练契约。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from traning.contracts import DataSplit
from traning.contracts.common import require_transform_fingerprint
from traning.evaluation.metrics import (
    expected_score_mae,
    expiry_brier_score,
    multiclass_brier_score,
    multiclass_nll,
    top_label_ece,
)
from traning.outcome.model import (
    OUTCOME_CATEGORY_COUNT,
    DenseOutcomeModel,
    OutcomeTensorOutput,
)
from traning.outcome.dataset.builder import CounterfactualOutcomeDataset


_FLOAT_DTYPES = (torch.float32, torch.float64)


@dataclass(frozen=True, slots=True)
class OutcomeBatch:
    """保留样本血缘且可直接输入 dense Outcome 模型的批次。"""

    split: DataSplit
    transform_fingerprint: str
    sample_ids: tuple[str, ...]
    source_sample_ids: tuple[str, ...]
    oracle_state_ids: tuple[str, ...]
    track_ids: tuple[str, ...]
    target_object_ids: tuple[str | None, ...]
    belief_embeddings: torch.Tensor
    horizon_ms: torch.Tensor
    category_targets: torch.Tensor
    expiry_targets: torch.Tensor
    score_targets: torch.Tensor
    valid_targets: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.split, DataSplit) or self.split is DataSplit.ALL:
            raise ValueError("OutcomeBatch 必须属于具体 DataSplit")
        require_transform_fingerprint(self.transform_fingerprint)
        batch_size = len(self.sample_ids)
        if batch_size < 1:
            raise ValueError("OutcomeBatch 不得为空")
        if len(set(self.sample_ids)) != batch_size:
            raise ValueError("OutcomeBatch sample_ids 必须唯一")
        provenance = (
            ("source_sample_ids", self.source_sample_ids),
            ("oracle_state_ids", self.oracle_state_ids),
            ("track_ids", self.track_ids),
            ("target_object_ids", self.target_object_ids),
        )
        if any(len(values) != batch_size for _name, values in provenance):
            raise ValueError("OutcomeBatch 血缘字段必须与 batch 对齐")
        for name, values in (
            ("sample_ids", self.sample_ids),
            ("source_sample_ids", self.source_sample_ids),
            ("oracle_state_ids", self.oracle_state_ids),
            ("track_ids", self.track_ids),
        ):
            if any(not isinstance(value, str) or not value.strip() for value in values):
                raise ValueError(f"{name} 必须包含非空字符串")
        if any(
            value is not None and (not isinstance(value, str) or not value.strip())
            for value in self.target_object_ids
        ):
            raise ValueError("target_object_ids 只能包含非空字符串或 None")

        tensors = (
            ("belief_embeddings", self.belief_embeddings),
            ("horizon_ms", self.horizon_ms),
            ("category_targets", self.category_targets),
            ("expiry_targets", self.expiry_targets),
            ("score_targets", self.score_targets),
            ("valid_targets", self.valid_targets),
        )
        for name, tensor in tensors:
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} 必须是 torch.Tensor")
        if (
            self.belief_embeddings.ndim != 2
            or self.belief_embeddings.shape[0] != batch_size
            or self.belief_embeddings.shape[1] < 1
        ):
            raise ValueError("belief_embeddings 必须采用非空 [batch, dim]")
        for name, tensor in tensors[1:]:
            if tensor.shape != (batch_size,):
                raise ValueError(f"{name} 必须采用 [batch]")
        if self.belief_embeddings.dtype not in _FLOAT_DTYPES:
            raise TypeError("belief_embeddings dtype 必须是 float32 或 float64")
        for name, tensor in (
            ("horizon_ms", self.horizon_ms),
            ("expiry_targets", self.expiry_targets),
            ("score_targets", self.score_targets),
        ):
            if tensor.dtype != self.belief_embeddings.dtype:
                raise TypeError(f"{name} dtype 必须与 belief_embeddings 一致")
        if self.category_targets.dtype is not torch.long:
            raise TypeError("category_targets dtype 必须是 torch.int64")
        if self.valid_targets.dtype is not torch.bool:
            raise TypeError("valid_targets dtype 必须是 torch.bool")
        reference_device = self.belief_embeddings.device
        if any(tensor.device != reference_device for _name, tensor in tensors):
            raise ValueError("OutcomeBatch tensor 必须位于同一设备")
        for name, tensor in tensors[:2] + tensors[3:5]:
            if not bool(torch.isfinite(tensor).all().item()):
                raise ValueError(f"{name} 必须全部有限")
        if bool((self.horizon_ms < 0).any().item()):
            raise ValueError("horizon_ms 不得为负数")
        if bool(
            (
                (self.category_targets < 0)
                | (self.category_targets >= OUTCOME_CATEGORY_COUNT)
            )
            .any()
            .item()
        ):
            raise ValueError("category_targets 含有越界类别")
        for name, tensor in (
            ("expiry_targets", self.expiry_targets),
            ("score_targets", self.score_targets),
        ):
            if bool(((tensor < 0) | (tensor > 1)).any().item()):
                raise ValueError(f"{name} 必须位于 [0, 1]")
        if bool(((self.expiry_targets != 0) & (self.expiry_targets != 1)).any().item()):
            raise ValueError("expiry_targets 必须只包含 0 或 1")


@dataclass(frozen=True, slots=True)
class OutcomeLossWeights:
    """主任务权重必须为正；score 只能作为非负辅助项。"""

    category: float = 1.0
    expiry: float = 1.0
    score: float = 0.1

    def __post_init__(self) -> None:
        for name, value in (
            ("category", self.category),
            ("expiry", self.expiry),
            ("score", self.score),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} loss weight 必须是数值")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} loss weight 必须有限")
        if self.category <= 0 or self.expiry <= 0:
            raise ValueError("category 与 expiry 主任务权重必须大于 0")
        if self.score < 0:
            raise ValueError("score 辅助权重不得为负数")


@dataclass(frozen=True, slots=True)
class OutcomeLoss:
    """单个 Outcome batch 的可审计损失分解。"""

    total: torch.Tensor
    category: torch.Tensor
    expiry: torch.Tensor
    score: torch.Tensor

    def __post_init__(self) -> None:
        _validate_scalar_tensors(
            (
                ("total", self.total),
                ("category", self.category),
                ("expiry", self.expiry),
                ("score", self.score),
            )
        )


@dataclass(frozen=True, slots=True)
class OutcomeEvaluationMetrics:
    """Outcome batch 的分类、校准、分数和 expiry 指标。"""

    multiclass_nll: torch.Tensor
    multiclass_brier: torch.Tensor
    calibration_error: torch.Tensor
    expected_score_mae: torch.Tensor
    expiry_brier: torch.Tensor

    def __post_init__(self) -> None:
        _validate_scalar_tensors(
            (
                ("multiclass_nll", self.multiclass_nll),
                ("multiclass_brier", self.multiclass_brier),
                ("calibration_error", self.calibration_error),
                ("expected_score_mae", self.expected_score_mae),
                ("expiry_brier", self.expiry_brier),
            )
        )


def collate_outcome_samples(
    dataset: CounterfactualOutcomeDataset,
    belief_embedding_dim: int,
    *,
    record_indices: tuple[int, ...] | None = None,
) -> OutcomeBatch:
    """从单一有指纹数据集组装 CPU float32 batch，禁止混合裸 records。"""

    if not isinstance(dataset, CounterfactualOutcomeDataset):
        raise TypeError("dataset 必须是 CounterfactualOutcomeDataset")
    if isinstance(belief_embedding_dim, bool) or not isinstance(
        belief_embedding_dim, int
    ):
        raise TypeError("belief_embedding_dim 必须是整数")
    if belief_embedding_dim < 1:
        raise ValueError("belief_embedding_dim 必须大于 0")
    if record_indices is None:
        samples = dataset.records
    else:
        if not isinstance(record_indices, tuple) or not record_indices:
            raise TypeError("record_indices 必须是非空整数元组")
        if any(
            isinstance(index, bool) or not isinstance(index, int)
            for index in record_indices
        ):
            raise TypeError("record_indices 只能包含整数")
        if len(record_indices) != len(set(record_indices)):
            raise ValueError("record_indices 不得重复")
        if any(index < 0 or index >= len(dataset.records) for index in record_indices):
            raise IndexError("record_indices 超出 dataset 范围")
        samples = tuple(dataset.records[index] for index in record_indices)
    split = dataset.split
    if any(
        len(sample.belief.belief_embedding) != belief_embedding_dim
        for sample in samples
    ):
        raise ValueError("belief_embedding 维度与 belief_embedding_dim 不一致")

    return OutcomeBatch(
        split=split,
        transform_fingerprint=dataset.transform_fingerprint,
        sample_ids=tuple(sample.sample_id for sample in samples),
        source_sample_ids=tuple(sample.source_sample_id for sample in samples),
        oracle_state_ids=tuple(sample.oracle_state_id for sample in samples),
        track_ids=tuple(sample.belief.track_id for sample in samples),
        target_object_ids=tuple(sample.target_object_id for sample in samples),
        belief_embeddings=torch.tensor(
            tuple(sample.belief.belief_embedding for sample in samples),
            dtype=torch.float32,
        ),
        horizon_ms=torch.tensor(
            tuple(sample.horizon_ms for sample in samples), dtype=torch.float32
        ),
        category_targets=torch.tensor(
            tuple(int(sample.target_category) for sample in samples), dtype=torch.long
        ),
        expiry_targets=torch.tensor(
            tuple(float(sample.expires) for sample in samples), dtype=torch.float32
        ),
        score_targets=torch.tensor(
            tuple(sample.target_score for sample in samples), dtype=torch.float32
        ),
        valid_targets=torch.tensor(
            tuple(sample.valid for sample in samples), dtype=torch.bool
        ),
    )


def compute_outcome_loss(
    output: OutcomeTensorOutput,
    batch: OutcomeBatch,
    weights: OutcomeLossWeights = OutcomeLossWeights(),
) -> OutcomeLoss:
    """以分类和 expiry 为主任务，expected score 仅作辅助回归。"""

    if not isinstance(output, OutcomeTensorOutput):
        raise TypeError("output 必须是 OutcomeTensorOutput")
    if not isinstance(batch, OutcomeBatch):
        raise TypeError("batch 必须是 OutcomeBatch")
    if not isinstance(weights, OutcomeLossWeights):
        raise TypeError("weights 必须是 OutcomeLossWeights")
    _validate_output_batch_alignment(output, batch)
    category = F.cross_entropy(output.category_logits, batch.category_targets)
    expiry = F.binary_cross_entropy_with_logits(
        output.expiry_logits, batch.expiry_targets
    )
    score = F.smooth_l1_loss(output.expected_score, batch.score_targets)
    total = (
        weights.category * category + weights.expiry * expiry + weights.score * score
    )
    return OutcomeLoss(total=total, category=category, expiry=expiry, score=score)


def evaluate_outcome_batch(
    output: OutcomeTensorOutput,
    batch: OutcomeBatch,
    *,
    calibration_bins: int = 15,
) -> OutcomeEvaluationMetrics:
    """复用 Phase 7 canonical metrics 评估一个 Outcome batch。"""

    if not isinstance(output, OutcomeTensorOutput):
        raise TypeError("output 必须是 OutcomeTensorOutput")
    if not isinstance(batch, OutcomeBatch):
        raise TypeError("batch 必须是 OutcomeBatch")
    _validate_output_batch_alignment(output, batch)
    return OutcomeEvaluationMetrics(
        multiclass_nll=multiclass_nll(
            output.category_probabilities, batch.category_targets
        ),
        multiclass_brier=multiclass_brier_score(
            output.category_probabilities, batch.category_targets
        ),
        calibration_error=top_label_ece(
            output.category_probabilities,
            batch.category_targets,
            bin_count=calibration_bins,
        ),
        expected_score_mae=expected_score_mae(
            output.expected_score, batch.score_targets
        ),
        expiry_brier=expiry_brier_score(
            output.expiry_probability, batch.expiry_targets
        ),
    )


def train_outcome_step(
    model: DenseOutcomeModel,
    batch: OutcomeBatch,
    optimizer: torch.optim.Optimizer,
    weights: OutcomeLossWeights = OutcomeLossWeights(),
) -> OutcomeLoss:
    """执行一个标准 dense Outcome 优化步骤并返回损失分解。"""

    if not isinstance(model, DenseOutcomeModel):
        raise TypeError("model 必须是 DenseOutcomeModel")
    if not isinstance(batch, OutcomeBatch):
        raise TypeError("batch 必须是 OutcomeBatch")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer 必须是 torch.optim.Optimizer")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(batch.belief_embeddings, batch.horizon_ms)
    loss = compute_outcome_loss(output, batch, weights)
    loss.total.backward()
    optimizer.step()
    return loss


def _validate_output_batch_alignment(
    output: OutcomeTensorOutput, batch: OutcomeBatch
) -> None:
    batch_size = len(batch.sample_ids)
    if output.category_logits.shape != (batch_size, OUTCOME_CATEGORY_COUNT):
        raise ValueError("output 与 batch size 不一致")
    if output.category_logits.device != batch.belief_embeddings.device:
        raise ValueError("output 与 batch 必须位于同一设备")
    if output.category_logits.dtype != batch.belief_embeddings.dtype:
        raise TypeError("output 与 batch 浮点 dtype 必须一致")


def _validate_scalar_tensors(values: tuple[tuple[str, torch.Tensor], ...]) -> None:
    reference_device: torch.device | None = None
    reference_dtype: torch.dtype | None = None
    for name, value in values:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} 必须是 torch.Tensor")
        if value.ndim != 0:
            raise ValueError(f"{name} 必须是标量 Tensor")
        if value.dtype not in _FLOAT_DTYPES:
            raise TypeError(f"{name} dtype 必须是 float32 或 float64")
        if not bool(torch.isfinite(value).item()):
            raise ValueError(f"{name} 必须有限")
        if reference_device is None:
            reference_device = value.device
            reference_dtype = value.dtype
        elif value.device != reference_device or value.dtype != reference_dtype:
            raise ValueError("所有结果标量必须具有相同 device 和 dtype")


__all__ = (
    "OutcomeBatch",
    "OutcomeEvaluationMetrics",
    "OutcomeLoss",
    "OutcomeLossWeights",
    "collate_outcome_samples",
    "compute_outcome_loss",
    "evaluate_outcome_batch",
    "train_outcome_step",
)
