from __future__ import annotations

import unittest

import psutil
import torch
from torch import nn

from traning.Lib.runtime import (
    CudaRuntimeConfig,
    amp_uses_grad_scaler,
    configure_torch_runtime,
    create_grad_scaler,
    enforce_runtime_memory_budget,
    module_to_device,
    tensor_to_device,
)


class CudaOptimizationTests(unittest.TestCase):
    def test_cpu_runtime_keeps_cuda_only_options_inactive(self) -> None:
        state = configure_torch_runtime(
            device=torch.device("cpu"),
            amp_dtype="auto",
            runtime=CudaRuntimeConfig(channels_last=True),
        )
        self.assertEqual(state.device, "cpu")
        self.assertFalse(state.channels_last)
        self.assertFalse(state.cudnn_benchmark)
        self.assertFalse(state.grad_scaler_enabled)

    def test_grad_scaler_auto_is_disabled_without_fp16_cuda(self) -> None:
        scaler = create_grad_scaler(
            device=torch.device("cpu"),
            amp_dtype="auto",
            mode="auto",
        )
        self.assertFalse(scaler.is_enabled())
        self.assertFalse(amp_uses_grad_scaler(torch.device("cpu"), "auto"))

    def test_tensor_to_device_preserves_cpu_contiguous_layout(self) -> None:
        tensor = torch.randn(1, 3, 16, 16)
        moved = tensor_to_device(
            tensor,
            torch.device("cpu"),
            channels_last=True,
        )
        self.assertTrue(moved.is_contiguous())

    def test_cpu_memory_budget_reports_system_reserve(self) -> None:
        budget = enforce_runtime_memory_budget(
            device=torch.device("cpu"),
            max_vram_gib=1.0,
            reserve_vram_gib=0.0,
            max_ram_gib=None,
            reserve_ram_gib=0.1,
        )
        self.assertEqual(budget.device, "cpu")
        self.assertGreater(budget.ram_budget_gib, 0.0)
        self.assertEqual(budget.ram_reserved_for_system_gib, 0.1)
        self.assertIsNone(budget.vram_budget_gib)

    def test_cpu_memory_budget_rejects_unavailable_reserve(self) -> None:
        total_gib = psutil.virtual_memory().total / 1024**3
        with self.assertRaises(RuntimeError):
            enforce_runtime_memory_budget(
                device=torch.device("cpu"),
                max_vram_gib=1.0,
                reserve_vram_gib=0.0,
                max_ram_gib=None,
                reserve_ram_gib=total_gib + 1.0,
            )

    def test_cuda_channels_last_when_available(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")
        tensor = torch.randn(1, 3, 16, 16)
        moved = tensor_to_device(
            tensor,
            torch.device("cuda"),
            channels_last=True,
        )
        self.assertTrue(moved.is_contiguous(memory_format=torch.channels_last))
        module = module_to_device(
            nn.Conv2d(3, 4, kernel_size=3),
            torch.device("cuda"),
            channels_last=True,
        )
        weight = next(module.parameters())
        self.assertTrue(weight.is_contiguous(memory_format=torch.channels_last))


if __name__ == "__main__":
    unittest.main()
