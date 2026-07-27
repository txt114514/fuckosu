"""验证帧采样、分组和可复现随机顺序的边界条件。"""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import Dataset

from traning.core.dataset_import import loader as loader_module


class _IdDataset(Dataset[dict[str, object]]):
    def __init__(self, size: int = 12) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict[str, object]:
        # sample_id 直接暴露原始索引，使测试观察 DataLoader 顺序而不受图像内容影响。
        return {"image": torch.zeros((1, 1, 1)), "sample_id": index}


def _settings(*, seed: int, shuffle: bool) -> SimpleNamespace:
    return SimpleNamespace(
        runtime=SimpleNamespace(seed=seed),
        loader=SimpleNamespace(
            batch_size=1,
            num_workers=0,
            shuffle=shuffle,
            pin_memory=False,
            persistent_workers=False,
            prefetch_factor=None,
            drop_last=False,
        ),
    )


def _sample_order(settings: SimpleNamespace) -> tuple[int, ...]:
    # 只替换数据发现边界，实际 DataLoader、generator 与 collate 路径保持不变。
    with patch.object(loader_module, "build_dataset", return_value=_IdDataset()):
        dataloader = loader_module.build_dataloader(settings, split="train")
        return tuple(int(batch["samples"][0]["sample_id"]) for batch in dataloader)


class DataSamplingTests(unittest.TestCase):
    def test_training_shuffle_is_seeded_and_not_sequential(self) -> None:
        first = _sample_order(_settings(seed=2026, shuffle=True))
        second = _sample_order(_settings(seed=2026, shuffle=True))
        different = _sample_order(_settings(seed=99, shuffle=True))

        self.assertEqual(first, second)
        self.assertNotEqual(first, tuple(range(12)))
        self.assertNotEqual(first, different)

    def test_evaluation_order_is_deterministic_when_shuffle_is_disabled(self) -> None:
        first = _sample_order(_settings(seed=2026, shuffle=False))
        second = _sample_order(_settings(seed=99, shuffle=False))

        self.assertEqual(first, tuple(range(12)))
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
