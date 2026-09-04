"""CUDA/CPU 内存预算、AMP、设备搬运与运行时配置公共入口。"""

from traning.lib.runtime.memory import (
    CudaRuntimeConfig,
    CudaRuntimeState,
    MemoryReport,
    MemorySnapshot,
    RuntimeMemoryBudget,
    amp_uses_grad_scaler,
    autocast_context,
    collect_memory_snapshot,
    configure_torch_runtime,
    create_grad_scaler,
    enforce_runtime_memory_budget,
    format_oom_guidance,
    maybe_compile_module,
    module_to_device,
    resolve_amp_dtype,
    tensor_to_device,
)

__all__ = [
    "CudaRuntimeConfig",
    "CudaRuntimeState",
    "MemoryReport",
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
