from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Iterator

import torch


@dataclass(frozen=True)
class MemorySnapshot:
    cuda_available: bool
    max_allocated_gib: float | None
    max_reserved_gib: float | None
    current_allocated_gib: float | None
    current_reserved_gib: float | None


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
    "MemorySnapshot",
    "autocast_context",
    "collect_memory_snapshot",
    "format_oom_guidance",
    "resolve_amp_dtype",
]
