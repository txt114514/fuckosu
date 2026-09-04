"""集中设置 Python、NumPy 与 PyTorch 的可复现随机种子。"""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """设置所有已使用随机源；仅在 CUDA 可用时触碰 CUDA API。"""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed 必须是 int")
    if not 0 <= seed < 2**32:
        raise ValueError("seed 必须位于 [0, 2**32) 范围内")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


__all__ = ("seed_everything",)
