"""无真值注入的感知运行时图像与模型适配。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as functional

from traning.config import PerceptionConfig
from traning.contracts import CandidateObservation, RuntimeFrame
from traning.perception.decode import decode_candidates
from traning.perception.models import DensePerceptionOutput


@dataclass(frozen=True, slots=True)
class RuntimeTensorFrame:
    """显式 resize 后的 BCHW RGB tensor 及其原始帧身份。"""

    frame: RuntimeFrame
    image: Tensor

    def __post_init__(self) -> None:
        if self.image.ndim != 4 or tuple(self.image.shape[:2]) != (1, 3):
            raise ValueError("image 必须是 batch=1 的 BCHW RGB tensor")
        if not self.image.is_floating_point():
            raise TypeError("image 必须是浮点 tensor")


class DensePerceptionModel(Protocol):
    """运行时需要的最小模型调用契约。"""

    def __call__(self, image: Tensor) -> DensePerceptionOutput:
        """执行单批次稠密推理。"""


def runtime_frame_to_tensor(
    frame: RuntimeFrame, config: PerceptionConfig
) -> RuntimeTensorFrame:
    """严格按 raw RGB 解码，并 resize 为配置输入尺寸。"""

    if config.input_channels != 3:
        raise ValueError("raw RGB runtime 要求 perception.input_channels 等于 3")
    expected_size = frame.width * frame.height * 3
    if len(frame.image_bytes) != expected_size:
        raise ValueError(
            f"RuntimeFrame.image_bytes 长度必须为 {expected_size}，实际为 {len(frame.image_bytes)}"
        )
    # bytearray 提供可写独立存储，避免 torch.frombuffer 持有只读 bytes。
    pixels = torch.frombuffer(bytearray(frame.image_bytes), dtype=torch.uint8).reshape(
        frame.height, frame.width, 3
    )
    image = pixels.permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32).div_(255.0)
    if (frame.height, frame.width) != (config.frame_height, config.frame_width):
        image = functional.interpolate(
            image,
            size=(config.frame_height, config.frame_width),
            mode="bilinear",
            align_corners=False,
        )
    return RuntimeTensorFrame(frame=frame, image=image.contiguous())


def decode_runtime_output(
    tensor_frame: RuntimeTensorFrame,
    output: DensePerceptionOutput,
    config: PerceptionConfig,
) -> tuple[CandidateObservation, ...]:
    """将网络尺寸上的输出直接映回 RuntimeFrame 原始坐标。"""

    frame = tensor_frame.frame
    return decode_candidates(
        output,
        frame_id=frame.frame_id,
        frame_index=frame.frame_index,
        timestamp_ms=frame.timestamp_ms,
        frame_width=frame.width,
        frame_height=frame.height,
        config=config,
    )


class PerceptionRuntime:
    """串接严格 RGB 适配、模型调用和 typed candidate 解码。"""

    def __init__(
        self,
        model: DensePerceptionModel,
        config: PerceptionConfig,
        *,
        device: torch.device | str = "cpu",
        amp: bool = False,
    ) -> None:
        if not isinstance(config, PerceptionConfig):
            raise TypeError("config 必须是 PerceptionConfig")
        self._model = model
        self._config = config
        self._device = torch.device(device)
        if not isinstance(amp, bool):
            raise TypeError("amp 必须是 bool")
        if amp and self._device.type != "cuda":
            raise ValueError("PerceptionRuntime AMP 只允许用于 CUDA")
        self._amp = amp
        if isinstance(model, nn.Module):
            model.to(self._device)
            model.eval()

    def infer(self, frame: RuntimeFrame) -> tuple[CandidateObservation, ...]:
        """执行单帧无监督信息注入的感知推理。"""

        tensor_frame = runtime_frame_to_tensor(frame, self._config)
        with torch.inference_mode():
            with torch.autocast(
                device_type=self._device.type,
                enabled=self._amp,
            ):
                output = self._model(tensor_frame.image.to(self._device))
        return decode_runtime_output(tensor_frame, output, self._config)


__all__ = (
    "DensePerceptionModel",
    "PerceptionRuntime",
    "RuntimeTensorFrame",
    "decode_runtime_output",
    "runtime_frame_to_tensor",
)
