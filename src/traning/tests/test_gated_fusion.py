from __future__ import annotations

import unittest

import torch

from traning.data import PatchMeta
from traning.models import GatedSparseFusion
from traning.models.local_encoder import LocalFeatures


class GatedFusionTests(unittest.TestCase):
    def test_forward_and_backward(self) -> None:
        fusion = GatedSparseFusion(
            local_channels=8,
            global_channels=8,
            hidden_dim=16,
            heads=4,
            sampling_points=2,
            layers=1,
        )
        local_dense = torch.randn(1, 8, 6, 6, requires_grad=True)
        global_dense = torch.randn(1, 8, 8, 8, requires_grad=True)
        meta = PatchMeta(0, 0, 0, 48, 48, 96, 96, 48, 48)
        output = fusion(
            local_features=LocalFeatures(local_dense, {"stride8": local_dense}, 8),
            global_features=global_dense,
            patch_meta=meta,
        )
        self.assertEqual(tuple(output.dense.shape), (1, 8, 6, 6))
        output.dense.mean().backward()
        self.assertIsNotNone(local_dense.grad)
        self.assertIsNotNone(global_dense.grad)


if __name__ == "__main__":
    unittest.main()
