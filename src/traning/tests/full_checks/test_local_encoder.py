from __future__ import annotations

import unittest

import torch

from traning.lib.models import SmallLocalEncoder


class LocalEncoderTests(unittest.TestCase):
    def test_forward_shapes_and_backward(self) -> None:
        model = SmallLocalEncoder(
            stem_channels=4,
            feature_channels=16,
            gradient_checkpointing=True,
        )
        patch = torch.randn(1, 3, 64, 64, requires_grad=True)
        features = model(patch)
        self.assertEqual(features.stride, 8)
        self.assertEqual(tuple(features.dense.shape), (1, 16, 8, 8))
        self.assertIn("stride2", features.pyramid)
        features.dense.mean().backward()
        self.assertIsNotNone(patch.grad)


if __name__ == "__main__":
    unittest.main()
