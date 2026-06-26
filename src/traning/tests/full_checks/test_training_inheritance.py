from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from traning.conf import load_settings
from traning.core.training_inheritance import (
    create_inheritance_package,
    load_inheritance_package,
)


class TrainingInheritanceTests(unittest.TestCase):
    def test_create_and_load_inheritance_package(self) -> None:
        settings = load_settings(Path("configs/model_small_vram.yaml"))
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint = root / "checkpoint.pt"
            torch.save({"model_state": {"weight": torch.ones(1)}}, checkpoint)

            package = create_inheritance_package(
                output_dir=root,
                settings=settings,
                resolved_config_path=Path("configs/model_small_vram.yaml"),
                latest_checkpoint_path=checkpoint,
                best_checkpoint_path=checkpoint,
                stage_checkpoints={"spatial": checkpoint, "temporal": checkpoint},
                training_state={"global_step": 3},
                score_state={"score": 0.7},
            )
            self.assertTrue(package.manifest_path.exists())
            self.assertTrue((package.path / "latest_checkpoint.pt").exists())
            self.assertTrue(
                (package.path / "stage_checkpoints" / "spatial_checkpoint.pt").exists()
            )

            loaded = load_inheritance_package(
                inherit_from=package.path,
                current_settings=settings,
                policy="auto",
            )
            self.assertEqual(loaded.status, "loaded")
            self.assertTrue(loaded.compatible)
            self.assertTrue(loaded.loaded_checkpoint_path.exists())
            self.assertEqual(
                set(loaded.stage_checkpoint_paths),
                {"spatial", "temporal"},
            )
            self.assertIn("spatial_checkpoint", loaded.restored_fields)
            self.assertIn("temporal_checkpoint", loaded.restored_fields)

    def test_strict_rejects_incompatible_dataset(self) -> None:
        settings = load_settings(Path("configs/model_small_vram.yaml"))
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            package = create_inheritance_package(
                output_dir=root,
                settings=settings,
                resolved_config_path=Path("configs/model_small_vram.yaml"),
            )
            manifest = package.manifest_path.read_text(encoding="utf-8")
            package.manifest_path.write_text(
                manifest.replace(str(settings.data_input.dataset_root), "/changed"),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_inheritance_package(
                    inherit_from=package.path,
                    current_settings=settings,
                    policy="strict",
                )


if __name__ == "__main__":
    unittest.main()
