from __future__ import annotations

import unittest

from pydantic import ValidationError

from traning.conf import LoaderSettings, MemoryConfig


class CudaConfigTests(unittest.TestCase):
    def test_memory_defaults_enable_cuda_optimized_runtime(self) -> None:
        config = MemoryConfig()
        self.assertEqual(config.amp_dtype, "auto")
        self.assertTrue(config.channels_last)
        self.assertTrue(config.allow_tf32)
        self.assertTrue(config.cudnn_benchmark)
        self.assertEqual(config.grad_scaler, "auto")
        self.assertFalse(config.compile_model)
        self.assertEqual(config.max_vram_gib, 6.5)
        self.assertEqual(config.reserve_vram_gib, 1.0)
        self.assertEqual(config.max_ram_gib, 24.0)
        self.assertEqual(config.reserve_ram_gib, 4.0)

    def test_loader_worker_options_require_workers(self) -> None:
        with self.assertRaises(ValidationError):
            LoaderSettings(num_workers=0, persistent_workers=True)
        with self.assertRaises(ValidationError):
            LoaderSettings(num_workers=0, prefetch_factor=2)
        config = LoaderSettings(
            num_workers=2, persistent_workers=True, prefetch_factor=2
        )
        self.assertTrue(config.persistent_workers)
        self.assertEqual(config.prefetch_factor, 2)


if __name__ == "__main__":
    unittest.main()
