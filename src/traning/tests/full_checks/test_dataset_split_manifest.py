from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from package.dataset_split import SplitRatios, sync_dataset_split_manifest
from traning.conf import Settings
from traning.core.dataset_import.preflight import discover_data_input
from traning.lib.data.models import DiscoveryResult


def _segment(root: Path, item_name: str, segment_id: str) -> None:
    directory = root / item_name / "atomic" / segment_id
    directory.mkdir(parents=True)
    (directory / "beatmap.json").write_text("{}", encoding="utf-8")


class DatasetSplitManifestTests(unittest.TestCase):
    def test_discovery_uses_split_manifest_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "video_segments"
            _segment(root, "item_000001", "segment_000001")
            _segment(root, "item_000002", "segment_000001")
            manifest_path = Path(tmpdir) / "splits" / "dataset_split_manifest.json"
            sync_dataset_split_manifest(
                root,
                manifest_path=manifest_path,
                ratios=SplitRatios(),
                bootstrap_splits={
                    "item_000001": "train",
                    "item_000002": "validation",
                },
            )
            settings = Settings(
                data_input={
                    "dataset_root": root,
                    "split_manifest_path": manifest_path,
                    "train_items": ("legacy_train",),
                    "validation_items": ("legacy_validation",),
                }
            )

            with patch(
                "traning.core.dataset_import.preflight.discover_segments",
                return_value=DiscoveryResult(records=(), issues=()),
            ) as discover_mock:
                discover_data_input(settings, split="validation")

            discover_mock.assert_called_once()
            self.assertEqual(
                discover_mock.call_args.kwargs["include_items"],
                ("item_000002",),
            )


if __name__ == "__main__":
    unittest.main()
