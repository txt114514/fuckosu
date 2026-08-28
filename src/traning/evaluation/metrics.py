"""Outcome 分类、分数与过期预测的严格 tensor 指标。"""

from __future__ import annotations

import torch
from torch import Tensor


_FLOAT_DTYPES = (torch.float32, torch.float64)


def multiclass_nll(probabilities: Tensor, labels: Tensor) -> Tensor:
    """返回 batch mean NLL；零目标概率按 dtype 最小正数取有限对数。"""

    _validate_multiclass(probabilities, labels)
    selected = probabilities.gather(1, labels[:, None]).squeeze(1)
    return -torch.log(selected.clamp_min(torch.finfo(probabilities.dtype).tiny)).mean()


def multiclass_brier_score(probabilities: Tensor, labels: Tensor) -> Tensor:
    """返回各样本所有类别平方误差之和的 batch mean。"""

    _validate_multiclass(probabilities, labels)
    targets = torch.nn.functional.one_hot(
        labels, num_classes=probabilities.shape[1]
    ).to(dtype=probabilities.dtype)
    return torch.sum((probabilities - targets) ** 2, dim=1).mean()


def top_label_ece(
    probabilities: Tensor, labels: Tensor, *, bin_count: int = 15
) -> Tensor:
    """计算 top-label ECE。

    分箱为 ``[i/B, (i+1)/B)``，内部边界进入右侧箱；最后一箱包含 1。
    空箱权重为零。
    """

    _validate_multiclass(probabilities, labels)
    if isinstance(bin_count, bool) or not isinstance(bin_count, int):
        raise TypeError("bin_count 必须是整数")
    if bin_count < 1:
        raise ValueError("bin_count 必须至少为 1")
    confidence, prediction = torch.max(probabilities, dim=1)
    correct = (prediction == labels).to(dtype=probabilities.dtype)
    boundaries = (
        torch.arange(
            1, bin_count, device=probabilities.device, dtype=probabilities.dtype
        )
        / bin_count
    )
    bin_indices = torch.bucketize(confidence.contiguous(), boundaries, right=True)
    result = torch.zeros((), device=probabilities.device, dtype=probabilities.dtype)
    for bin_index in range(bin_count):
        mask = bin_indices == bin_index
        count = mask.sum()
        if int(count.item()) == 0:
            continue
        weight = count.to(dtype=probabilities.dtype) / probabilities.shape[0]
        result = result + weight * torch.abs(
            correct[mask].mean() - confidence[mask].mean()
        )
    return result


def expected_score_mae(predicted_scores: Tensor, target_scores: Tensor) -> Tensor:
    """返回逐样本 expected score 与归一化目标分数的平均绝对误差。"""

    _validate_binary_pair(
        predicted_scores,
        target_scores,
        prediction_name="predicted_scores",
        target_name="target_scores",
        binary_targets=False,
    )
    return torch.abs(predicted_scores - target_scores).mean()


def expiry_binary_cross_entropy(
    expiry_probabilities: Tensor, expiry_targets: Tensor
) -> Tensor:
    """返回过期概率的 batch mean binary cross entropy。"""

    _validate_binary_pair(
        expiry_probabilities,
        expiry_targets,
        prediction_name="expiry_probabilities",
        target_name="expiry_targets",
        binary_targets=True,
    )
    epsilon = torch.finfo(expiry_probabilities.dtype).eps
    probabilities = expiry_probabilities.clamp(epsilon, 1.0 - epsilon)
    return -(
        expiry_targets * torch.log(probabilities)
        + (1.0 - expiry_targets) * torch.log1p(-probabilities)
    ).mean()


def expiry_brier_score(expiry_probabilities: Tensor, expiry_targets: Tensor) -> Tensor:
    """返回过期概率的 batch mean Brier score。"""

    _validate_binary_pair(
        expiry_probabilities,
        expiry_targets,
        prediction_name="expiry_probabilities",
        target_name="expiry_targets",
        binary_targets=True,
    )
    return ((expiry_probabilities - expiry_targets) ** 2).mean()


def _validate_multiclass(probabilities: Tensor, labels: Tensor) -> None:
    _require_tensor(probabilities, "probabilities")
    _require_tensor(labels, "labels")
    if (
        probabilities.ndim != 2
        or probabilities.shape[0] < 1
        or probabilities.shape[1] < 2
    ):
        raise ValueError("probabilities 必须是非空的 [N, C] tensor，且 C>=2")
    if probabilities.dtype not in _FLOAT_DTYPES:
        raise TypeError("probabilities dtype 必须是 float32 或 float64")
    if labels.dtype is not torch.long:
        raise TypeError("labels dtype 必须是 torch.int64")
    if labels.ndim != 1 or labels.shape[0] != probabilities.shape[0]:
        raise ValueError("labels 必须是与 probabilities batch 对齐的 [N] tensor")
    if labels.device != probabilities.device:
        raise ValueError("probabilities 与 labels 必须位于同一 device")
    _require_finite(probabilities, "probabilities")
    if bool(((probabilities < 0.0) | (probabilities > 1.0)).any().item()):
        raise ValueError("probabilities 必须位于 [0, 1]")
    tolerance = 1e-6 if probabilities.dtype is torch.float32 else 1e-12
    sums = probabilities.sum(dim=1)
    if not bool(
        torch.allclose(sums, torch.ones_like(sums), rtol=tolerance, atol=tolerance)
    ):
        raise ValueError("probabilities 每行概率和必须为 1")
    if bool(((labels < 0) | (labels >= probabilities.shape[1])).any().item()):
        raise ValueError("labels 含有越界类别")


def _validate_binary_pair(
    predictions: Tensor,
    targets: Tensor,
    *,
    prediction_name: str,
    target_name: str,
    binary_targets: bool,
) -> None:
    _require_tensor(predictions, prediction_name)
    _require_tensor(targets, target_name)
    if predictions.dtype not in _FLOAT_DTYPES or targets.dtype != predictions.dtype:
        raise TypeError(
            f"{prediction_name} 与 {target_name} 必须具有相同 float32/float64 dtype"
        )
    if (
        predictions.ndim != 1
        or predictions.shape[0] < 1
        or targets.shape != predictions.shape
    ):
        raise ValueError(
            f"{prediction_name} 与 {target_name} 必须是相同 shape 的非空 [N] tensor"
        )
    if predictions.device != targets.device:
        raise ValueError(f"{prediction_name} 与 {target_name} 必须位于同一 device")
    _require_finite(predictions, prediction_name)
    _require_finite(targets, target_name)
    if bool(((predictions < 0.0) | (predictions > 1.0)).any().item()):
        raise ValueError(f"{prediction_name} 必须位于 [0, 1]")
    if bool(((targets < 0.0) | (targets > 1.0)).any().item()):
        raise ValueError(f"{target_name} 必须位于 [0, 1]")
    if binary_targets and bool(((targets != 0.0) & (targets != 1.0)).any().item()):
        raise ValueError(f"{target_name} 必须只包含 0 或 1")


def _require_tensor(value: Tensor, name: str) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} 必须是 torch.Tensor")


def _require_finite(value: Tensor, name: str) -> None:
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} 含有非有限数值")


__all__ = (
    "expected_score_mae",
    "expiry_binary_cross_entropy",
    "expiry_brier_score",
    "multiclass_brier_score",
    "multiclass_nll",
    "top_label_ece",
)
