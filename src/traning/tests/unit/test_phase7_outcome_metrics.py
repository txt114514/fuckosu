"""Phase 7 Outcome 指标与温度校准验收。"""

from __future__ import annotations

import math

import pytest
import torch

from traning.evaluation.metrics import (
    expected_score_mae,
    expiry_binary_cross_entropy,
    expiry_brier_score,
    multiclass_brier_score,
    multiclass_nll,
    top_label_ece,
)
from traning.outcome.calibration import (
    ScalarTemperatureCalibrator,
    evaluate_temperature_calibration,
    fit_temperature_calibrator,
)


def test_hand_computed_multiclass_metrics() -> None:
    """NLL、Brier 与 ECE 必须匹配可手算二分类 batch。"""

    probabilities = torch.tensor([[0.8, 0.2], [0.25, 0.75]], dtype=torch.float64)
    labels = torch.tensor([0, 1], dtype=torch.long)
    assert multiclass_nll(probabilities, labels).item() == pytest.approx(
        -(math.log(0.8) + math.log(0.75)) / 2.0
    )
    assert multiclass_brier_score(probabilities, labels).item() == pytest.approx(0.1025)
    assert top_label_ece(probabilities, labels, bin_count=2).item() == pytest.approx(
        0.225
    )


def test_top_label_ece_internal_boundary_enters_right_bin() -> None:
    """置信度恰为内部边界时必须进入右侧箱。"""

    probabilities = torch.tensor([[0.5, 0.5], [0.8, 0.2]], dtype=torch.float64)
    labels = torch.tensor([0, 1], dtype=torch.long)
    # 两项都进入 [0.5, 1]：accuracy=.5、mean confidence=.65。
    assert top_label_ece(probabilities, labels, bin_count=2).item() == pytest.approx(
        0.15
    )


def test_hand_computed_score_and_expiry_metrics() -> None:
    """expected score MAE 与 expiry BCE/Brier 使用独立 typed 边界。"""

    predictions = torch.tensor([0.2, 0.8], dtype=torch.float64)
    score_targets = torch.tensor([0.0, 1.0], dtype=torch.float64)
    assert expected_score_mae(predictions, score_targets).item() == pytest.approx(0.2)

    expiry_probabilities = torch.tensor([0.8, 0.25], dtype=torch.float64)
    expiry_targets = torch.tensor([1.0, 0.0], dtype=torch.float64)
    assert expiry_binary_cross_entropy(
        expiry_probabilities, expiry_targets
    ).item() == pytest.approx(-(math.log(0.8) + math.log(0.75)) / 2.0)
    assert expiry_brier_score(
        expiry_probabilities, expiry_targets
    ).item() == pytest.approx(0.05125)


@pytest.mark.parametrize(
    ("probabilities", "labels", "error"),
    (
        (torch.tensor([0.5, 0.5]), torch.tensor([0]), ValueError),
        (torch.tensor([[0.6, 0.6]]), torch.tensor([0]), ValueError),
        (torch.tensor([[0.5, float("nan")]]), torch.tensor([0]), ValueError),
        (torch.tensor([[0.5, 0.5]], dtype=torch.float16), torch.tensor([0]), TypeError),
        (torch.tensor([[0.5, 0.5]]), torch.tensor([0], dtype=torch.int32), TypeError),
        (torch.tensor([[0.5, 0.5]]), torch.tensor([2]), ValueError),
    ),
)
def test_multiclass_metrics_reject_invalid_boundary(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    error: type[Exception],
) -> None:
    """shape、dtype、finite、概率和和 label 越界均硬失败。"""

    with pytest.raises(error):
        multiclass_nll(probabilities, labels)


def test_metrics_reject_device_and_binary_shape_mismatch() -> None:
    """相关 tensor 不得跨 device，也不得依赖 broadcasting。"""

    probabilities = torch.tensor([[0.5, 0.5]])
    meta_labels = torch.empty(1, dtype=torch.long, device="meta")
    with pytest.raises(ValueError, match="device"):
        multiclass_brier_score(probabilities, meta_labels)
    with pytest.raises(ValueError, match="shape"):
        expected_score_mae(torch.tensor([0.5]), torch.tensor([[0.5]]))
    with pytest.raises(TypeError, match="dtype"):
        expiry_brier_score(torch.tensor([0.5]), torch.tensor([False]))
    with pytest.raises(ValueError, match="0 或 1"):
        expiry_brier_score(torch.tensor([0.5]), torch.tensor([0.5]))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        expiry_binary_cross_entropy(torch.tensor([1.1]), torch.tensor([1.0]))


def test_temperature_one_is_identity_and_requires_positive_scalar() -> None:
    """T=1 不改变 logits，非正或非有限温度不得构造。"""

    logits = torch.tensor([[1.0, -1.0], [0.2, 0.3]], dtype=torch.float64)
    calibrator = ScalarTemperatureCalibrator(1.0)
    assert torch.equal(calibrator.transform(logits), logits)
    for invalid in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            ScalarTemperatureCalibrator(invalid)


def test_deterministic_temperature_fit_does_not_worsen_overconfident_nll() -> None:
    """固定 validation 网格拟合应复现，并缓和含错误样本的过度自信 logits。"""

    logits = torch.tensor(
        [[8.0, -8.0], [8.0, -8.0], [-8.0, 8.0], [-8.0, 8.0]],
        dtype=torch.float64,
    )
    labels = torch.tensor([0, 1, 1, 0], dtype=torch.long)
    first = fit_temperature_calibrator(logits, labels)
    second = fit_temperature_calibrator(logits, labels)
    assert first == second
    assert first.temperature > 1.0
    evaluation = evaluate_temperature_calibration(first, logits, labels)
    assert evaluation.nll_after <= evaluation.nll_before


def test_calibration_rejects_invalid_logits_labels_and_search_spec() -> None:
    """校准同样硬拒绝 shape、dtype、device、finite、label 和搜索区间错误。"""

    logits = torch.tensor([[1.0, 0.0]])
    labels = torch.tensor([0])
    with pytest.raises(ValueError):
        fit_temperature_calibrator(logits[:, :1], labels)
    with pytest.raises(TypeError):
        fit_temperature_calibrator(logits.to(torch.float16), labels)
    with pytest.raises(ValueError):
        fit_temperature_calibrator(torch.tensor([[float("inf"), 0.0]]), labels)
    with pytest.raises(ValueError):
        fit_temperature_calibrator(logits, torch.tensor([2]))
    with pytest.raises(ValueError, match="identity"):
        fit_temperature_calibrator(
            logits, labels, log_temperature_min=0.1, log_temperature_max=2.0
        )
