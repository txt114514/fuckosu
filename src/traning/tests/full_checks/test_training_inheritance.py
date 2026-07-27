"""验证训练继承包的创建、兼容判断和自动降级策略。"""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from traning.conf import Settings, load_settings
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
            # 仅篡改继承包记录的数据集路径，保持 checkpoint 和当前设置
            # 不变，以隔离 strict 策略对 dataset identity 的拒绝分支。
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

    def test_auto_downgrades_when_transform_equation_changes(self) -> None:
        """确认只改仿射偏移量也会使 auto 继承降级为仅加载权重。"""

        settings = load_settings(Path("configs/model_small_vram.yaml"))
        changed_payload = settings.model_dump(mode="python")
        changed_matrix = [
            list(row) for row in changed_payload["coordinate_transform"]["matrix"]
        ]
        # 保持协议版本与其他配置不变，隔离验证 fingerprint 对方程内容的检测能力。
        changed_matrix[0][2] += 1.0
        changed_payload["coordinate_transform"]["matrix"] = changed_matrix
        changed_settings = Settings.model_validate(changed_payload)

        with tempfile.TemporaryDirectory() as temp_dir:
            package = create_inheritance_package(
                output_dir=Path(temp_dir),
                settings=settings,
                resolved_config_path=Path("configs/model_small_vram.yaml"),
            )
            loaded = load_inheritance_package(
                inherit_from=package.path,
                current_settings=changed_settings,
                policy="auto",
            )

        self.assertEqual(loaded.policy, "weights-only")
        self.assertFalse(loaded.compatible)
        self.assertIn("transform_fingerprint", loaded.downgrade_reasons)
        self.assertIn(
            "auto_downgraded_to_weights_only",
            loaded.downgrade_reasons,
        )


if __name__ == "__main__":
    unittest.main()
