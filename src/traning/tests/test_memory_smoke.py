from __future__ import annotations

import unittest

import torch

from traning.core.memory import autocast_context, collect_memory_snapshot
from traning.data import PatchMeta
from traning.models import GatedSparseFusion, SmallLocalEncoder, SpatialPredictionHead


class MemorySmokeTests(unittest.TestCase):
    def run_smoke(self, device: torch.device) -> None:
        local = SmallLocalEncoder(stem_channels=4, feature_channels=8).to(device)
        fusion = GatedSparseFusion(
            local_channels=8,
            global_channels=8,
            hidden_dim=16,
            heads=4,
            sampling_points=2,
            layers=1,
        ).to(device)
        head = SpatialPredictionHead(8, embedding_dim=8).to(device)
        optimizer = torch.optim.AdamW(
            list(local.parameters())
            + list(fusion.parameters())
            + list(head.parameters()),
            lr=1e-4,
        )
        optimizer.zero_grad(set_to_none=True)
        patch = torch.randn(1, 3, 64, 64, device=device)
        global_feature = torch.randn(1, 8, 8, 8, device=device)
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
        loss.backward()
        optimizer.step()

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
