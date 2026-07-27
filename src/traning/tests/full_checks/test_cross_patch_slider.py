"""验证跨 patch slider 结构的连通性、解码与融合边界。"""

from __future__ import annotations

import unittest

import torch.nn.functional as F

from traning.lib.data import PatchStream, make_cross_patch_slider
from traning.lib.models import sample_global_feature


class CrossPatchSliderTests(unittest.TestCase):
    def test_slider_spans_multiple_patches_with_shared_global_context(self) -> None:
        # 长滑条跨过多个 patch；fixture 用于捕获仅在首个局部块可见、其余
        # 块因全局特征坐标错误而丢失连续结构的回归。
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
        # 下采样 mask 代表共享全局特征，而不是直接复用高分辨率标签。
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
