from __future__ import annotations

import unittest

import torch

from traning.lib.runtime import autocast_context, collect_memory_snapshot
from traning.lib.runtime import (
    CudaRuntimeConfig,
    configure_torch_runtime,
    create_grad_scaler,
    module_to_device,
    tensor_to_device,
)
from traning.lib.data import PatchMeta
from traning.lib.models import (
    GatedSparseFusion,
    SmallLocalEncoder,
    SpatialPredictionHead,
)


class MemorySmokeTests(unittest.TestCase):
    def run_smoke(self, device: torch.device) -> None:
        runtime = configure_torch_runtime(
            device=device,
            amp_dtype="auto",
            runtime=CudaRuntimeConfig(channels_last=True),
        )
        local = module_to_device(
            SmallLocalEncoder(stem_channels=4, feature_channels=8),
            device,
            channels_last=runtime.channels_last,
        )
        fusion = module_to_device(
            GatedSparseFusion(
                local_channels=8,
                global_channels=8,
                hidden_dim=16,
                heads=4,
                sampling_points=2,
                layers=1,
            ),
            device,
            channels_last=runtime.channels_last,
        )
        head = module_to_device(
            SpatialPredictionHead(8, embedding_dim=8),
            device,
            channels_last=runtime.channels_last,
        )
        optimizer = torch.optim.AdamW(
            list(local.parameters())
            + list(fusion.parameters())
            + list(head.parameters()),
            lr=1e-4,
        )
        scaler = create_grad_scaler(device=device, amp_dtype="auto")
        optimizer.zero_grad(set_to_none=True)
        patch = tensor_to_device(
            torch.randn(1, 3, 64, 64),
            device,
            channels_last=runtime.channels_last,
        )
        global_feature = tensor_to_device(
            torch.randn(1, 8, 8, 8),
            device,
            channels_last=runtime.channels_last,
        )
        meta = PatchMeta(0, 0, 0, 64, 64, 64, 64, 64, 64)
        with autocast_context(device, "auto"):
            local_features = local(patch)
            fused = fusion(
                local_features=local_features,
                global_features=global_feature,
                patch_meta=meta,
            )
            prediction = head(fused.dense)
            loss = prediction.center_heatmap.mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    def test_cpu_forward_backward_step(self) -> None:
        self.run_smoke(torch.device("cpu"))

    def test_cuda_forward_backward_step_when_available(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")
        torch.cuda.reset_peak_memory_stats()
        self.run_smoke(torch.device("cuda"))
        snapshot = collect_memory_snapshot()
        self.assertIsNotNone(snapshot.max_allocated_gib)


if __name__ == "__main__":
    unittest.main()
