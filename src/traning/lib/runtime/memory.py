"""统一训练与冒烟测试的内存预算、CUDA 性能开关和 AMP 设备操作。"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Iterator

import psutil
import torch
from torch import nn

AmpDType = str | torch.dtype | None


@dataclass(frozen=True)
class MemorySnapshot:
    """PyTorch CUDA allocator 的峰值与当前 allocated/reserved 快照。"""

    cuda_available: bool
    max_allocated_gib: float | None
    max_reserved_gib: float | None
    current_allocated_gib: float | None
    current_reserved_gib: float | None


@dataclass(frozen=True)
class RuntimeMemoryBudget:
    """预算检查后得到的主存、显存和 CUDA allocator 上限记录。"""

    device: str
    ram_total_gib: float
    ram_available_gib: float
    ram_process_rss_gib: float
    ram_budget_gib: float
    ram_reserved_for_system_gib: float
    vram_total_gib: float | None
    vram_free_gib: float | None
    vram_current_reserved_gib: float | None
    vram_budget_gib: float | None
    vram_reserved_for_system_gib: float | None
    cuda_memory_fraction: float | None

    def as_dict(self) -> dict[str, float | str | None]:
        return {
            "device": self.device,
            "ram_total_gib": self.ram_total_gib,
            "ram_available_gib": self.ram_available_gib,
            "ram_process_rss_gib": self.ram_process_rss_gib,
            "ram_budget_gib": self.ram_budget_gib,
            "ram_reserved_for_system_gib": self.ram_reserved_for_system_gib,
            "vram_total_gib": self.vram_total_gib,
            "vram_free_gib": self.vram_free_gib,
            "vram_current_reserved_gib": self.vram_current_reserved_gib,
            "vram_budget_gib": self.vram_budget_gib,
            "vram_reserved_for_system_gib": self.vram_reserved_for_system_gib,
            "cuda_memory_fraction": self.cuda_memory_fraction,
        }


@dataclass(frozen=True)
class CudaRuntimeConfig:
    """CUDA 数值性能开关；CPU 设备会忽略 CUDA 专属项。"""

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


def enforce_runtime_memory_budget(
    *,
    device: torch.device,
    max_vram_gib: float,
    reserve_vram_gib: float,
    max_ram_gib: float | None,
    reserve_ram_gib: float,
    set_cuda_fraction: bool = True,
) -> RuntimeMemoryBudget:
    """验证 CPU/CUDA 预算，并为宿主系统保留不可占用的余量。"""

    if max_vram_gib <= 0 or not _finite(max_vram_gib):
        raise ValueError("max_vram_gib must be finite and positive")
    if reserve_vram_gib < 0 or not _finite(reserve_vram_gib):
        raise ValueError("reserve_vram_gib must be finite and nonnegative")
    if max_ram_gib is not None and (max_ram_gib <= 0 or not _finite(max_ram_gib)):
        raise ValueError("max_ram_gib must be finite and positive when set")
    if reserve_ram_gib < 0 or not _finite(reserve_ram_gib):
        raise ValueError("reserve_ram_gib must be finite and nonnegative")

    ram = psutil.virtual_memory()
    process_rss_gib = psutil.Process().memory_info().rss / 1024**3
    ram_total_gib = ram.total / 1024**3
    ram_available_gib = ram.available / 1024**3
    ram_system_budget = max(ram_total_gib - reserve_ram_gib, 0.0)
    # available 不含进程已占用内存，因此加回 RSS 得到“进程最终可达到”的上限。
    ram_available_budget = process_rss_gib + max(
        ram_available_gib - reserve_ram_gib,
        0.0,
    )
    ram_budget_gib = min(
        max_ram_gib if max_ram_gib is not None else ram_system_budget,
        ram_system_budget,
        ram_available_budget,
    )
    if ram_budget_gib <= 0:
        raise RuntimeError(
            "CPU RAM budget is zero after system reserve; lower reserve_ram_gib"
        )
    if ram_available_gib <= reserve_ram_gib:
        raise RuntimeError(
            f"available CPU RAM {ram_available_gib:.2f} GiB is at or below "
            f"system reserve {reserve_ram_gib:.2f} GiB"
        )
    if process_rss_gib > ram_budget_gib:
        raise RuntimeError(
            f"current process RSS {process_rss_gib:.2f} GiB exceeds CPU RAM "
            f"budget {ram_budget_gib:.2f} GiB"
        )

    vram_total_gib = None
    vram_free_gib = None
    vram_current_reserved_gib = None
    vram_budget_gib = None
    cuda_fraction = None
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but CUDA is not available")
        device_index = device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
        vram_total_gib = total_bytes / 1024**3
        vram_free_gib = free_bytes / 1024**3
        vram_current_reserved_gib = torch.cuda.memory_reserved(device_index) / 1024**3
        vram_system_budget = max(vram_total_gib - reserve_vram_gib, 0.0)
        vram_budget_gib = min(max_vram_gib, vram_system_budget)
        if vram_budget_gib <= 0:
            raise RuntimeError(
                "CUDA VRAM budget is zero after system reserve; lower reserve_vram_gib"
            )
        if vram_free_gib <= reserve_vram_gib:
            raise RuntimeError(
                f"free CUDA VRAM {vram_free_gib:.2f} GiB is at or below "
                f"system reserve {reserve_vram_gib:.2f} GiB"
            )
        if vram_current_reserved_gib > vram_budget_gib:
            raise RuntimeError(
                f"current CUDA reserved memory {vram_current_reserved_gib:.2f} GiB "
                f"exceeds budget {vram_budget_gib:.2f} GiB"
            )
        cuda_fraction = min(max(vram_budget_gib / vram_total_gib, 0.01), 1.0)
        if set_cuda_fraction:
            # 这是 PyTorch allocator 的进程上限，不代表立即预留相应物理显存。
            torch.cuda.set_per_process_memory_fraction(
                cuda_fraction,
                device=device_index,
            )

    return RuntimeMemoryBudget(
        device=str(device),
        ram_total_gib=ram_total_gib,
        ram_available_gib=ram_available_gib,
        ram_process_rss_gib=process_rss_gib,
        ram_budget_gib=ram_budget_gib,
        ram_reserved_for_system_gib=reserve_ram_gib,
        vram_total_gib=vram_total_gib,
        vram_free_gib=vram_free_gib,
        vram_current_reserved_gib=vram_current_reserved_gib,
        vram_budget_gib=vram_budget_gib,
        vram_reserved_for_system_gib=(
            reserve_vram_gib if device.type == "cuda" else None
        ),
        cuda_memory_fraction=cuda_fraction,
    )


def resolve_amp_dtype(device: torch.device, amp_dtype: AmpDType) -> torch.dtype | None:
    """解析 AMP 精度；非 CUDA 或 float32 返回 ``None`` 表示禁用 autocast。"""

    if device.type != "cuda" or amp_dtype is None:
        return None
    if isinstance(amp_dtype, torch.dtype):
        if amp_dtype == torch.float32:
            return None
        if amp_dtype in {torch.float16, torch.bfloat16}:
            return amp_dtype
        raise ValueError(f"unsupported amp dtype: {amp_dtype}")
    normalized = amp_dtype.removeprefix("torch.").lower()
    if normalized in {"none", "float32"}:
        return None
    if normalized == "float16":
        return torch.float16
    if normalized == "bfloat16":
        return torch.bfloat16
    if normalized != "auto":
        raise ValueError(f"unsupported amp dtype: {amp_dtype}")
    # auto 优先 bfloat16 的动态范围；硬件不支持时才退到 float16。
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


@contextmanager
def autocast_context(device: torch.device, amp_dtype: AmpDType) -> Iterator[None]:
    """提供统一上下文；AMP 关闭时退化为无操作上下文。"""

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
    amp_dtype: AmpDType,
    runtime: CudaRuntimeConfig = CudaRuntimeConfig(),
) -> CudaRuntimeState:
    """应用训练与冒烟测试共用的 CUDA 数值和卷积运行时设置。"""

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


def amp_uses_grad_scaler(device: torch.device, amp_dtype: AmpDType) -> bool:
    """仅 float16 CUDA 路径需要 GradScaler；bfloat16 通常无需缩放。"""

    return (
        device.type == "cuda" and resolve_amp_dtype(device, amp_dtype) == torch.float16
    )


def create_grad_scaler(
    *,
    device: torch.device,
    amp_dtype: AmpDType,
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
    """搬运模块，并仅在 CUDA 请求时切为 channels-last 内存格式。"""

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
    """搬运张量；只有四维 CUDA 图像张量应用 channels-last。"""

    if channels_last and device.type == "cuda" and tensor.ndim == 4:
        return tensor.to(
            device=device,
            non_blocking=non_blocking,
            memory_format=torch.channels_last,
        )
    return tensor.to(device=device, non_blocking=non_blocking)


def collect_memory_snapshot() -> MemorySnapshot:
    """采集当前默认 CUDA 设备的 PyTorch allocator 统计。"""

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


def _finite(value: float) -> bool:
    return value == value and value not in {float("inf"), float("-inf")}


__all__ = [
    "CudaRuntimeConfig",
    "CudaRuntimeState",
    "MemorySnapshot",
    "RuntimeMemoryBudget",
    "amp_uses_grad_scaler",
    "autocast_context",
    "collect_memory_snapshot",
    "configure_torch_runtime",
    "create_grad_scaler",
    "enforce_runtime_memory_budget",
    "format_oom_guidance",
    "maybe_compile_module",
    "module_to_device",
    "resolve_amp_dtype",
    "tensor_to_device",
]
