"""Outcome logits 的确定性正标量温度校准。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from traning.evaluation.metrics import multiclass_nll


_FLOAT_DTYPES = (torch.float32, torch.float64)


@dataclass(frozen=True, slots=True)
class CalibrationEvaluation:
    """同一 validation logits 校准前后的 NLL。"""

    nll_before: float
    nll_after: float

    def __post_init__(self) -> None:
        for name, value in (
            ("nll_before", self.nll_before),
            ("nll_after", self.nll_after),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} 必须是有限非负数")


@dataclass(frozen=True, slots=True)
class ScalarTemperatureCalibrator:
    """以严格正标量 ``T`` 执行 ``logits / T``。"""

    temperature: float = 1.0

    def __post_init__(self) -> None:
        if isinstance(self.temperature, bool) or not isinstance(
            self.temperature, (int, float)
        ):
            raise TypeError("temperature 必须是数值")
        if not math.isfinite(float(self.temperature)) or self.temperature <= 0.0:
            raise ValueError("temperature 必须是有限正数")

    def transform(self, logits: Tensor) -> Tensor:
        """校准二维 multiclass logits，保持 dtype 与 device。"""

        _validate_logits(logits)
        return logits / self.temperature

    def probabilities(self, logits: Tensor) -> Tensor:
        """返回温度校准后的类别概率。"""

        return torch.softmax(self.transform(logits), dim=1)

    def evaluate(self, logits: Tensor, labels: Tensor) -> CalibrationEvaluation:
        """在同一 validation batch 上比较校准前后 NLL。"""

        _validate_logits_and_labels(logits, labels)
        before = multiclass_nll(torch.softmax(logits, dim=1), labels)
        after = multiclass_nll(self.probabilities(logits), labels)
        return CalibrationEvaluation(float(before.item()), float(after.item()))


def fit_temperature_calibrator(
    validation_logits: Tensor,
    validation_labels: Tensor,
    *,
    log_temperature_min: float = -4.0,
    log_temperature_max: float = 4.0,
    grid_steps: int = 257,
) -> ScalarTemperatureCalibrator:
    """用固定 log-temperature 网格确定性拟合，并保证 NLL 不劣于 T=1。"""

    _validate_logits_and_labels(validation_logits, validation_labels)
    for name, value in (
        ("log_temperature_min", log_temperature_min),
        ("log_temperature_max", log_temperature_max),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} 必须是数值")
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} 必须是有限数值")
    if log_temperature_min >= log_temperature_max:
        raise ValueError("log_temperature_min 必须小于 log_temperature_max")
    if not log_temperature_min <= 0.0 <= log_temperature_max:
        raise ValueError("搜索区间必须包含 identity temperature=1")
    if isinstance(grid_steps, bool) or not isinstance(grid_steps, int):
        raise TypeError("grid_steps 必须是整数")
    if grid_steps < 2:
        raise ValueError("grid_steps 必须至少为 2")

    search_dtype = torch.float64
    log_temperatures = torch.linspace(
        log_temperature_min,
        log_temperature_max,
        grid_steps,
        device=validation_logits.device,
        dtype=search_dtype,
    )
    # identity 是显式候选，确保网格没有正好落在 0 时也不会退化。
    log_temperatures = torch.cat(
        (
            log_temperatures,
            torch.zeros(1, device=validation_logits.device, dtype=search_dtype),
        )
    )
    logits = validation_logits.to(dtype=search_dtype)
    labels = validation_labels
    losses: list[float] = []
    with torch.no_grad():
        for log_temperature in log_temperatures:
            loss = torch.nn.functional.cross_entropy(
                logits / torch.exp(log_temperature), labels
            )
            losses.append(float(loss.item()))
    best_index = min(
        range(len(losses)),
        key=lambda index: (losses[index], abs(float(log_temperatures[index].item()))),
    )
    temperature = math.exp(float(log_temperatures[best_index].item()))
    return ScalarTemperatureCalibrator(temperature=temperature)


def evaluate_temperature_calibration(
    calibrator: ScalarTemperatureCalibrator, logits: Tensor, labels: Tensor
) -> CalibrationEvaluation:
    """函数式 typed 入口，委托 calibrator 评估 validation NLL。"""

    if not isinstance(calibrator, ScalarTemperatureCalibrator):
        raise TypeError("calibrator 必须是 ScalarTemperatureCalibrator")
    return calibrator.evaluate(logits, labels)


def _validate_logits_and_labels(logits: Tensor, labels: Tensor) -> None:
    _validate_logits(logits)
    if not isinstance(labels, Tensor):
        raise TypeError("labels 必须是 torch.Tensor")
    if labels.dtype is not torch.long:
        raise TypeError("labels dtype 必须是 torch.int64")
    if labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
        raise ValueError("labels 必须是与 logits batch 对齐的 [N] tensor")
    if labels.device != logits.device:
        raise ValueError("logits 与 labels 必须位于同一 device")
    if bool(((labels < 0) | (labels >= logits.shape[1])).any().item()):
        raise ValueError("labels 含有越界类别")


def _validate_logits(logits: Tensor) -> None:
    if not isinstance(logits, Tensor):
        raise TypeError("logits 必须是 torch.Tensor")
    if logits.dtype not in _FLOAT_DTYPES:
        raise TypeError("logits dtype 必须是 float32 或 float64")
    if logits.ndim != 2 or logits.shape[0] < 1 or logits.shape[1] < 2:
        raise ValueError("logits 必须是非空 [N, C] tensor，且 C>=2")
    if not bool(torch.isfinite(logits).all().item()):
        raise ValueError("logits 含有非有限数值")


__all__ = (
    "CalibrationEvaluation",
    "ScalarTemperatureCalibrator",
    "evaluate_temperature_calibration",
    "fit_temperature_calibrator",
)
