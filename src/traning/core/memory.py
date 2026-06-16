from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Iterator

import torch
from torch import nn


@dataclass(frozen=True)
class MemorySnapshot:
    cuda_available: bool
    max_allocated_gib: float | None
    max_reserved_gib: float | None
    current_allocated_gib: float | None
    current_reserved_gib: float | None


@dataclass(frozen=True)
class CudaRuntimeConfig:
    allow_tf32: bool = True
    cudnn_benchmark: bool = True
    matmul_float32_precision: str = "high"
    channels_last: bool = True


@dataclass(frozen=True)
class CudaRuntimeState:
    device: str
    amp_dtype: str
    channels_last: bool
    allow_tf32: bool
    cudnn_benchmark: bool
    matmul_float32_precision: str
    grad_scaler_enabled: bool


def resolve_amp_dtype(device: torch.device, amp_dtype: str) -> torch.dtype | None:
    if device.type != "cuda" or amp_dtype == "float32":
        return None
    if amp_dtype == "float16":
        return torch.float16
    if amp_dtype == "bfloat16":
        return torch.bfloat16
    if amp_dtype != "auto":
        raise ValueError(f"unsupported amp dtype: {amp_dtype}")
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


@contextmanager
def autocast_context(device: torch.device, amp_dtype: str) -> Iterator[None]:
    dtype = resolve_amp_dtype(device, amp_dtype)
    if dtype is None:
        with nullcontext():
            yield
    else:
        with torch.autocast(device_type=device.type, dtype=dtype):
            yield


def configure_torch_runtime(
    *,
    device: torch.device,
    amp_dtype: str,
    runtime: CudaRuntimeConfig = CudaRuntimeConfig(),
) -> CudaRuntimeState:
    """Apply CUDA runtime defaults used by training and smoke tests."""

    if device.type == "cuda":
        precision = "tf32" if runtime.allow_tf32 else "ieee"
        if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
            torch.backends.cuda.matmul.fp32_precision = precision
        else:  # pragma: no cover - old PyTorch compatibility.
            torch.set_float32_matmul_precision(runtime.matmul_float32_precision)
            torch.backends.cuda.matmul.allow_tf32 = runtime.allow_tf32
        if hasattr(torch.backends.cudnn, "fp32_precision"):
            torch.backends.cudnn.fp32_precision = precision
        else:  # pragma: no cover - old PyTorch compatibility.
            torch.backends.cudnn.allow_tf32 = runtime.allow_tf32
        if hasattr(torch.backends.cudnn, "conv") and hasattr(
            torch.backends.cudnn.conv,
            "fp32_precision",
        ):
            torch.backends.cudnn.conv.fp32_precision = precision
        torch.backends.cudnn.benchmark = runtime.cudnn_benchmark
    return CudaRuntimeState(
        device=str(device),
        amp_dtype=str(resolve_amp_dtype(device, amp_dtype)),
        channels_last=runtime.channels_last and device.type == "cuda",
        allow_tf32=runtime.allow_tf32,
        cudnn_benchmark=runtime.cudnn_benchmark if device.type == "cuda" else False,
        matmul_float32_precision=runtime.matmul_float32_precision,
        grad_scaler_enabled=amp_uses_grad_scaler(device, amp_dtype),
    )


def amp_uses_grad_scaler(device: torch.device, amp_dtype: str) -> bool:
    return (
        device.type == "cuda" and resolve_amp_dtype(device, amp_dtype) == torch.float16
    )


def create_grad_scaler(
    *,
    device: torch.device,
    amp_dtype: str,
    mode: str = "auto",
) -> torch.amp.GradScaler:
    if mode not in {"auto", "enabled", "disabled"}:
        raise ValueError("grad scaler mode must be auto, enabled, or disabled")
    enabled = {
        "auto": amp_uses_grad_scaler(device, amp_dtype),
        "enabled": device.type == "cuda",
        "disabled": False,
    }[mode]
    return torch.amp.GradScaler(device.type, enabled=enabled)


def module_to_device(
    module: nn.Module,
    device: torch.device,
    *,
    channels_last: bool,
) -> nn.Module:
    module = module.to(device)
    if channels_last and device.type == "cuda":
        module = module.to(memory_format=torch.channels_last)
    return module


def maybe_compile_module(
    module: nn.Module,
    *,
    enabled: bool,
    mode: str = "default",
) -> nn.Module:
    if not enabled:
        return module
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is not available in this PyTorch build")
    return torch.compile(module, mode=mode)


def tensor_to_device(
    tensor: torch.Tensor,
    device: torch.device,
    *,
    channels_last: bool,
    non_blocking: bool = True,
) -> torch.Tensor:
    if channels_last and device.type == "cuda" and tensor.ndim == 4:
        return tensor.to(
            device=device,
            non_blocking=non_blocking,
            memory_format=torch.channels_last,
        )
    return tensor.to(device=device, non_blocking=non_blocking)


def collect_memory_snapshot() -> MemorySnapshot:
    if not torch.cuda.is_available():
        return MemorySnapshot(
            cuda_available=False,
            max_allocated_gib=None,
            max_reserved_gib=None,
            current_allocated_gib=None,
            current_reserved_gib=None,
        )
    return MemorySnapshot(
        cuda_available=True,
        max_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
        max_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3,
        current_allocated_gib=torch.cuda.memory_allocated() / 1024**3,
        current_reserved_gib=torch.cuda.memory_reserved() / 1024**3,
    )


def format_oom_guidance(
    *,
    patch_size: tuple[int, int],
    global_size: tuple[int, int],
    batch_size: int,
    amp_dtype: str,
    config_path: str | None,
) -> str:
    snapshot = collect_memory_snapshot()
    allocated = (
        "unknown"
        if snapshot.max_allocated_gib is None
        else f"{snapshot.max_allocated_gib:.2f} GiB"
    )
    reserved = (
        "unknown"
        if snapshot.max_reserved_gib is None
        else f"{snapshot.max_reserved_gib:.2f} GiB"
    )
    return "\n".join(
        (
            "CUDA out of memory during small-VRAM training.",
            f"patch_size: {patch_size[0]}x{patch_size[1]}",
            f"global_input: {global_size[0]}x{global_size[1]}",
            f"batch_size: {batch_size}",
            f"amp_dtype: {amp_dtype}",
            f"max_allocated: {allocated}",
            f"max_reserved: {reserved}",
            f"config: {config_path or 'default'}",
            "Suggested order: feature_channels 48->32; global 640x360->512x288; "
            "fusion hidden_dim 96->64; fusion layers 2->1; sampling points 4->2; "
            "increase checkpointing; reduce history frames; shrink patch size last.",
        )
    )


__all__ = [
    "CudaRuntimeConfig",
    "CudaRuntimeState",
    "MemorySnapshot",
    "amp_uses_grad_scaler",
    "autocast_context",
    "collect_memory_snapshot",
    "configure_torch_runtime",
    "create_grad_scaler",
    "format_oom_guidance",
    "maybe_compile_module",
    "module_to_device",
    "resolve_amp_dtype",
    "tensor_to_device",
]
