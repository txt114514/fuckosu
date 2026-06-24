from __future__ import annotations

import unittest

import torch

from traning.lib.models import GlobalStructureHead, LightweightGlobalEncoder


class GlobalEncoderTests(unittest.TestCase):
    def test_lightweight_encoder_and_structure_head(self) -> None:
        encoder = LightweightGlobalEncoder(
            input_height=64,
            input_width=96,
            feature_channels=32,
        )
        frame = torch.randn(1, 3, 80, 120)
        features = encoder(frame)
        self.assertEqual(features.dense.shape[1], 32)
        self.assertEqual(features.tokens.shape[0], 1)
        prediction = GlobalStructureHead(32)(features.dense)
        self.assertEqual(prediction.objectness.shape[1], 1)
        self.assertEqual(prediction.context_tokens.shape[0], 1)

    def test_non_default_backbone_requires_external_setup(self) -> None:
        with self.assertRaises(NotImplementedError):
            LightweightGlobalEncoder(backbone="convnext_atto", pretrained=False)


if __name__ == "__main__":
    unittest.main()
