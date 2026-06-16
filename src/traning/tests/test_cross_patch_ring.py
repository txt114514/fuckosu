from __future__ import annotations

import unittest

import torch.nn.functional as F

from traning.data import PatchStream, make_cross_patch_ring
from traning.models import sample_global_feature


class CrossPatchRingTests(unittest.TestCase):
    def test_ring_is_visible_from_multiple_patches_with_global_context(self) -> None:
        sample = make_cross_patch_ring()
        stream = PatchStream(
            patch_width=512, patch_height=512, overlap_x=128, overlap_y=128
        )
        metas = stream.metas(
            frame_width=sample.image.shape[-1], frame_height=sample.image.shape[-2]
        )
        visible = [
            meta
            for meta in metas
            if bool(sample.mask[meta.y0 : meta.y1, meta.x0 : meta.x1].any())
        ]
        self.assertGreaterEqual(len(visible), 4)
        global_mask = F.interpolate(
            sample.mask.float().view(1, 1, *sample.mask.shape),
            size=(24, 24),
            mode="bilinear",
            align_corners=False,
        )
        means = [
            float(sample_global_feature(global_mask, meta, (8, 8)).mean())
            for meta in visible[:4]
        ]
        self.assertTrue(all(value > 0.0 for value in means))


if __name__ == "__main__":
    unittest.main()
