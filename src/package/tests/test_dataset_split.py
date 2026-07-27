"""验证 item 级数据划分的引导、冻结和增量同步规则。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from package.dataset_split import (
    SplitRatios,
    load_split_manifest,
    sync_dataset_split_manifest,
)


def _segment(root: Path, item_name: str, segment_id: str) -> None:
    directory = root / item_name / "atomic" / segment_id
    directory.mkdir(parents=True)
    (directory / "beatmap.json").write_text("{}", encoding="utf-8")


class DatasetSplitTests(unittest.TestCase):
    def test_sync_bootstraps_existing_config_and_freezes_assignments(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "video_segments"
            _segment(root, "item_000001", "segment_000001")
            _segment(root, "item_000002", "segment_000001")
            manifest_path = Path(tmpdir) / "splits" / "dataset_split_manifest.json"

            first = sync_dataset_split_manifest(
                root,
                manifest_path=manifest_path,
                ratios=SplitRatios(train=0.8, validation=0.1, test=0.1),
                bootstrap_splits={
                    "item_000001": "train",
                    "item_000002": "validation",
                },
            )

            self.assertTrue(first.created)
            self.assertTrue(manifest_path.is_file())
            self.assertEqual(first.manifest.split_items("train"), ("item_000001",))
            self.assertEqual(
                first.manifest.split_items("validation"),
                ("item_000002",),
            )

            _segment(root, "item_000003", "segment_000001")
            second = sync_dataset_split_manifest(
                root,
                manifest_path=manifest_path,
                ratios=SplitRatios(train=0.8, validation=0.1, test=0.1),
            )

            loaded = load_split_manifest(manifest_path)
            self.assertIsNotNone(loaded)
            self.assertEqual(second.manifest.items["item_000001"].split, "train")
            self.assertEqual(second.manifest.items["item_000002"].split, "validation")
            self.assertIn(second.new_items[0].split, {"train", "validation"})

    def test_dry_run_does_not_write_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "video_segments"
            _segment(root, "item_000001", "segment_000001")
            manifest_path = Path(tmpdir) / "splits" / "dataset_split_manifest.json"

            result = sync_dataset_split_manifest(
                root,
                manifest_path=manifest_path,
                dry_run=True,
            )

            self.assertTrue(result.changed)
            self.assertFalse(manifest_path.exists())

    def test_seeded_initial_assignment_is_reproducible_and_seed_sensitive(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "video_segments"
            for index in range(12):
                _segment(root, f"item_{index:06d}", "segment_000001")

            first = sync_dataset_split_manifest(
                root,
                manifest_path=Path(tmpdir) / "splits_a" / "dataset_split_manifest.json",
                seed=2026,
                ratios=SplitRatios(train=0.75, validation=0.25, test=0.0),
            )
            second = sync_dataset_split_manifest(
                root,
                manifest_path=Path(tmpdir) / "splits_b" / "dataset_split_manifest.json",
                seed=2026,
                ratios=SplitRatios(train=0.75, validation=0.25, test=0.0),
            )
            different = sync_dataset_split_manifest(
                root,
                manifest_path=Path(tmpdir) / "splits_c" / "dataset_split_manifest.json",
                seed=99,
                ratios=SplitRatios(train=0.75, validation=0.25, test=0.0),
            )

        first_assignments = {
            name: item.split for name, item in first.manifest.items.items()
        }
        second_assignments = {
            name: item.split for name, item in second.manifest.items.items()
        }
        different_assignments = {
            name: item.split for name, item in different.manifest.items.items()
        }
        self.assertEqual(first_assignments, second_assignments)
        self.assertNotEqual(first_assignments, different_assignments)


if __name__ == "__main__":
    unittest.main()
