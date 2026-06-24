from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from traning.lib.data import discover_segments


def _write_segment(root: Path, item_name: str, segment_id: str) -> None:
    segment = root / item_name / "single_point" / segment_id
    segment.mkdir(parents=True)
    (segment / "video.mp4").write_bytes(b"placeholder")
    (segment / "beatmap.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "segment_id": segment_id,
                "dataset_dimension": "atomic",
                "category": "single_point",
                "difficulty": {
                    "approach_preempt_ms": 600.0,
                    "circle_radius_osu_pixels": 32.0,
                },
                "source": {
                    "folder_name": item_name,
                    "osu_filename": "map.osu",
                    "clip_start_ms": 0,
                    "clip_end_ms": 1000,
                },
                "hit_objects": [],
            }
        ),
        encoding="utf-8",
    )


class DiscoverySplitTests(unittest.TestCase):
    def test_include_items_filters_records_before_loading(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_segment(root, "item_000001", "segment_a")
            _write_segment(root, "item_000002", "segment_b")

            result = discover_segments(root, include_items=("item_000002",))

        self.assertEqual(result.issues, ())
        self.assertEqual([record.item_name for record in result.records], ["item_000002"])

    def test_exclude_items_removes_records(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_segment(root, "item_000001", "segment_a")
            _write_segment(root, "item_000002", "segment_b")

            result = discover_segments(root, exclude_items=("item_000001",))

        self.assertEqual(result.issues, ())
        self.assertEqual([record.item_name for record in result.records], ["item_000002"])


if __name__ == "__main__":
    unittest.main()
