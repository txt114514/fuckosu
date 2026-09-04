"""只在张量进入训练或推理链路时执行的集中契约。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import torch
from torch import Tensor


ShapeDimension: TypeAlias = int | None
ShapeSpec: TypeAlias = tuple[ShapeDimension, ...]
AllowedDType: TypeAlias = torch.dtype | tuple[torch.dtype, ...]


@dataclass(frozen=True, slots=True)
class TensorSpec:
    """一个边界张量的 shape、dtype、device 与数值契约。

    ``shape`` 中的 ``None`` 是单维通配符。默认检查非空和有限性；这些可能扫描完整
    张量的检查只应在边界调用，内部热路径通过 ``Trusted`` 对象继续传递。
    """

    shape: ShapeSpec | None = None
    dtype: AllowedDType | None = None
    device: torch.device | str | None = None
    requires_grad: bool | None = None
    allow_empty: bool = False
    finite: bool = True

    def __post_init__(self) -> None:
        if self.shape is not None:
            for dimension in self.shape:
                if dimension is not None and (
                    isinstance(dimension, bool)
                    or not isinstance(dimension, int)
                    or dimension < 0
                ):
                    raise ValueError(
                        "TensorSpec shape dimensions must be nonnegative or None"
                    )
        if self.dtype is not None:
            dtypes = self.dtype if isinstance(self.dtype, tuple) else (self.dtype,)
            if not dtypes or any(
                not isinstance(dtype, torch.dtype) for dtype in dtypes
            ):
                raise TypeError("TensorSpec dtype must contain torch.dtype values")
        if self.device is not None:
            torch.device(self.device)
        if not isinstance(self.allow_empty, bool) or not isinstance(self.finite, bool):
            raise TypeError("TensorSpec allow_empty and finite must be booleans")
        if self.requires_grad is not None and not isinstance(self.requires_grad, bool):
            raise TypeError("TensorSpec requires_grad must be a boolean or None")

    def validate(self, value: object, name: str = "tensor") -> Tensor:
        """验证并返回同一个 Tensor，不复制数据也不改变 device/dtype。"""

        tensor = _require_tensor_type(value, name)
        if self.shape is not None:
            require_shape(tensor, self.shape, name)
        if self.dtype is not None:
            allowed = self.dtype if isinstance(self.dtype, tuple) else (self.dtype,)
            if tensor.dtype not in allowed:
                expected = ", ".join(str(dtype) for dtype in allowed)
                raise TypeError(
                    f"{name} dtype must be one of ({expected}); got {tensor.dtype}"
                )
        if self.device is not None and not _device_matches(
            tensor.device,
            torch.device(self.device),
        ):
            raise ValueError(
                f"{name} device must be {torch.device(self.device)}; got {tensor.device}"
            )
        if (
            self.requires_grad is not None
            and tensor.requires_grad is not self.requires_grad
        ):
            raise ValueError(
                f"{name}.requires_grad must be {self.requires_grad}; "
                f"got {tensor.requires_grad}"
            )
        if not self.allow_empty and tensor.numel() == 0:
            raise ValueError(f"{name} must not be empty")
        if self.finite and not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} must contain only finite values")
        return tensor


def require_tensor(
    value: object,
    name: str = "tensor",
    *,
    spec: TensorSpec | None = None,
) -> Tensor:
    """返回 Tensor，并在提供规格时执行一次完整边界校验。"""

    tensor = _require_tensor_type(value, name)
    return tensor if spec is None else spec.validate(tensor, name)


def require_shape(
    tensor: Tensor,
    expected: ShapeSpec,
    name: str = "tensor",
) -> tuple[int, ...]:
    """验证 Tensor 形状；``None`` 接受该位置的任意维度。"""

    _require_tensor_type(tensor, name)
    actual = tuple(tensor.shape)
    if len(actual) != len(expected) or any(
        wanted is not None and got != wanted
        for got, wanted in zip(actual, expected, strict=True)
    ):
        raise ValueError(f"{name} shape must match {expected}; got {actual}")
    return actual


def require_same_device(
    *tensors: Tensor,
    name: str = "tensors",
) -> torch.device:
    """验证一组非空 Tensor 位于完全相同的设备并返回该设备。"""

    checked = _require_tensor_group(tensors, name)
    device = checked[0].device
    if any(tensor.device != device for tensor in checked[1:]):
        devices = ", ".join(str(tensor.device) for tensor in checked)
        raise ValueError(f"{name} must use one device; got {devices}")
    return device


def require_same_dtype(
    *tensors: Tensor,
    name: str = "tensors",
) -> torch.dtype:
    """验证一组非空 Tensor 使用完全相同的 dtype 并返回该 dtype。"""

    checked = _require_tensor_group(tensors, name)
    dtype = checked[0].dtype
    if any(tensor.dtype != dtype for tensor in checked[1:]):
        dtypes = ", ".join(str(tensor.dtype) for tensor in checked)
        raise TypeError(f"{name} must use one dtype; got {dtypes}")
    return dtype


def _require_tensor_type(value: object, name: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    return value


def _require_tensor_group(
    tensors: tuple[Tensor, ...],
    name: str,
) -> tuple[Tensor, ...]:
    if not tensors:
        raise ValueError(f"{name} must contain at least one tensor")
    return tuple(
        _require_tensor_type(tensor, f"{name}[{index}]")
        for index, tensor in enumerate(tensors)
    )


def _device_matches(actual: torch.device, expected: torch.device) -> bool:
    if actual.type != expected.type:
        return False
    return expected.index is None or actual.index == expected.index


__all__ = [
    "AllowedDType",
    "ShapeDimension",
    "ShapeSpec",
    "TensorSpec",
    "require_same_device",
    "require_same_dtype",
    "require_shape",
    "require_tensor",
]
