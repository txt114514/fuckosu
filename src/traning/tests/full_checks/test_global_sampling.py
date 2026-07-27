"""验证全局特征在局部 patch 查询点上的采样坐标和边界。"""

from __future__ import annotations

import unittest

import torch

from traning.lib.data import PatchMeta
from traning.lib.models import sample_global_feature


class GlobalSamplingTests(unittest.TestCase):
    def test_patch_position_changes_sampled_context(self) -> None:
        # 用 x 坐标本身编码特征值，使左右 patch 的期望关系无需硬编码
        # 插值后的每个数值，也能捕获归一化坐标方向或 offset 错误。
        x_values = torch.linspace(0.0, 1.0, 10).view(1, 1, 1, 10)
        global_feature = x_values.expand(1, 1, 6, 10).contiguous()
        left = PatchMeta(0, 0, 0, 50, 50, 100, 60, 50, 50)
        right = PatchMeta(1, 50, 0, 100, 50, 100, 60, 50, 50)
        left_context = sample_global_feature(global_feature, left, (4, 4))
        right_context = sample_global_feature(global_feature, right, (4, 4))
        self.assertLess(float(left_context.mean()), float(right_context.mean()))


if __name__ == "__main__":
    unittest.main()
