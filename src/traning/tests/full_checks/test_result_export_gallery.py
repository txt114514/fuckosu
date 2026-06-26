from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch

from traning.lib.visualization.gallery import save_best_trial_gallery
from traning.state import (
    BatchGalleryRequest,
    FrameEvaluation,
    TrialGalleryEvaluation,
    TrialParameters,
)


class _FakeSegmentFrameDataset:
    def __init__(self) -> None:
        self.records = (
            SimpleNamespace(
                key="item_a/long_sequence_0001",
                category="long_sequence",
                dataset_dimension="long_sequence",
            ),
            SimpleNamespace(
                key="item_b/long_sequence_0002",
                category="long_sequence",
                dataset_dimension="long_sequence",
            ),
        )
        self.references = (
            SimpleNamespace(record_index=0, frame_index=1, timestamp_ms=100.0),
            SimpleNamespace(record_index=0, frame_index=2, timestamp_ms=200.0),
            SimpleNamespace(record_index=0, frame_index=3, timestamp_ms=300.0),
            SimpleNamespace(record_index=1, frame_index=1, timestamp_ms=100.0),
        )

    def __getitem__(self, index: int) -> dict[str, object]:
        reference = self.references[index]
        record = self.records[reference.record_index]
        return {
            "image": torch.zeros((3, 96, 128), dtype=torch.float32),
            "sample_key": record.key,
            "item_name": record.key,
            "segment_id": "segment",
            "dataset_dimension": record.dataset_dimension,
            "category": record.category,
            "frame_index": reference.frame_index,
            "timestamp_ms": reference.timestamp_ms,
            "hit_objects": (
                {
                    "type": "circle",
                    "x": 256.0,
                    "y": 192.0,
                    "source_index": 10 + reference.frame_index,
                },
            ),
            "visible_hit_objects": (
                {
                    "type": "circle",
                    "x": 256.0,
                    "y": 192.0,
                    "source_index": 10 + reference.frame_index,
                },
            ),
            "circle_radius_osu_pixels": 32.0,
        }


def _request(frames: tuple[FrameEvaluation, ...]) -> BatchGalleryRequest:
    return BatchGalleryRequest(
        batch_id="gallery_test",
        random_seed=2026,
        trials=(
            TrialGalleryEvaluation(
                trial_id="pg-0001",
                score=0.9,
                parameters=TrialParameters(),
                frames=frames,
            ),
        ),
    )


class ResultExportGalleryTests(unittest.TestCase):
    def test_outputs_one_folder_per_selected_sample_group(self) -> None:
        dataset = _FakeSegmentFrameDataset()
        request = _request(
            (
                FrameEvaluation(
                    sample_key="item_a/long_sequence_0001",
                    frame_index=1,
                    passed=True,
                    target_source_index=11,
                    predicted_video_xy=(64.0, 48.0),
                ),
                FrameEvaluation(
                    sample_key="item_a/long_sequence_0001",
                    frame_index=2,
                    passed=True,
                    target_source_index=12,
                    predicted_video_xy=(72.0, 48.0),
                ),
                FrameEvaluation(
                    sample_key="item_a/long_sequence_0001",
                    frame_index=3,
                    passed=True,
                ),
            )
        )

        with tempfile.TemporaryDirectory() as temporary:
            output_dir, saved_count, issues = save_best_trial_gallery(
                dataset,
                request,
                output_root=Path(temporary),
                samples_per_group=10,
            )

            manifest = json.loads(
                (output_dir / "manifest.json").read_text(encoding="utf-8")
            )
            sample_dirs = tuple(
                (output_dir / "passed" / "long_sequence").iterdir()
            )
            sample_dir_count = len(sample_dirs)
            sample_png_count = len(tuple(sample_dirs[0].glob("*.png")))

        self.assertEqual(issues, ())
        self.assertEqual(saved_count, 2)
        self.assertEqual(manifest["selected_sample_group_count"], 1)
        self.assertEqual(manifest["saved_frame_count"], 2)
        self.assertEqual(manifest["sample_groups"][0]["frame_count"], 2)
        self.assertEqual(sample_dir_count, 1)
        self.assertEqual(sample_png_count, 2)

    def test_samples_per_group_limits_sample_folders_not_frames(self) -> None:
        dataset = _FakeSegmentFrameDataset()
        request = _request(
            (
                FrameEvaluation(
                    sample_key="item_a/long_sequence_0001",
                    frame_index=1,
                    passed=True,
                    target_source_index=11,
                    predicted_video_xy=(64.0, 48.0),
                ),
                FrameEvaluation(
                    sample_key="item_a/long_sequence_0001",
                    frame_index=2,
                    passed=True,
                    target_source_index=12,
                    predicted_video_xy=(72.0, 48.0),
                ),
                FrameEvaluation(
                    sample_key="item_b/long_sequence_0002",
                    frame_index=1,
                    passed=True,
                    target_source_index=11,
                    predicted_video_xy=(80.0, 48.0),
                ),
            )
        )

        with tempfile.TemporaryDirectory() as temporary:
            output_dir, saved_count, issues = save_best_trial_gallery(
                dataset,
                request,
                output_root=Path(temporary),
                samples_per_group=1,
            )

            manifest = json.loads(
                (output_dir / "manifest.json").read_text(encoding="utf-8")
            )
            sample_dirs = tuple(
                (output_dir / "passed" / "long_sequence").iterdir()
            )
            sample_dir_count = len(sample_dirs)

        self.assertEqual(issues, ())
        self.assertEqual(manifest["selected_sample_group_count"], 1)
        self.assertEqual(sample_dir_count, 1)
        self.assertEqual(
            saved_count,
            manifest["sample_groups"][0]["frame_count"],
        )


if __name__ == "__main__":
    unittest.main()
