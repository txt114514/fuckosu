from __future__ import annotations

import unittest

import torch.nn.functional as F

from traning.lib.data import PatchStream, make_cross_patch_slider
from traning.lib.models import sample_global_feature


class CrossPatchSliderTests(unittest.TestCase):
    def test_slider_spans_multiple_patches_with_shared_global_context(self) -> None:
        sample = make_cross_patch_slider(thickness=80.0)
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
        self.assertGreaterEqual(len(visible), 3)
        global_mask = F.interpolate(
            sample.mask.float().view(1, 1, *sample.mask.shape),
            size=(16, 36),
            mode="bilinear",
            align_corners=False,
        )
        responses = [
            float(sample_global_feature(global_mask, meta, (8, 8)).max())
            for meta in visible
        ]
        self.assertGreater(max(responses), 0.0)


if __name__ == "__main__":
    unittest.main()
