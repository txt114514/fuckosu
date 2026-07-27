"""把帧样本整理成既支持等尺寸张量又支持变尺寸图像的 batch。"""

from __future__ import annotations

from typing import Any

import torch


def collate_frame_samples(
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    if not samples:
        raise ValueError("samples must not be empty")
    images = [sample["image"] for sample in samples]
    same_shape = len({tuple(image.shape) for image in images}) == 1
    # 等尺寸时堆成 BCHW；变尺寸帧保留 list，避免隐式 resize 改变像素坐标契约。
    return {
        "images": torch.stack(images) if same_shape else images,
        "samples": tuple(
            {key: value for key, value in sample.items() if key != "image"}
            for sample in samples
        ),
    }


__all__ = ["collate_frame_samples"]
