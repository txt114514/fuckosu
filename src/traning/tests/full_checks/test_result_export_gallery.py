from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

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


def _multi_trial_request(
    trials: tuple[tuple[str, float, tuple[FrameEvaluation, ...]], ...],
) -> BatchGalleryRequest:
    return BatchGalleryRequest(
        batch_id="batch_0003",
        random_seed=2026,
        metadata={"curriculum_stage": "level_a", "batch_id": "batch_0003"},
        trials=tuple(
            TrialGalleryEvaluation(
                trial_id=trial_id,
                score=score,
                parameters=TrialParameters(training={"trial_id": trial_id}),
                frames=frames,
            )
            for trial_id, score, frames in trials
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

    def test_best_trial_exports_even_below_promotion_threshold(self) -> None:
        dataset = _FakeSegmentFrameDataset()
        best_frame = FrameEvaluation(
            sample_key="item_a/long_sequence_0001",
            frame_index=1,
            passed=False,
            target_source_index=11,
            predicted_video_xy=(64.0, 48.0),
            primary_error="spatial",
        )
        request = _multi_trial_request(
            (
                ("trial_0001", 0.51, ()),
                ("trial_0002", 0.63, (best_frame,)),
                ("trial_0003", 0.58, ()),
            )
        )

        with tempfile.TemporaryDirectory() as temporary:
            output_dir, saved_count, issues = save_best_trial_gallery(
                dataset,
                request,
                output_root=Path(temporary),
                samples_per_group=10,
            )
            best = json.loads(
                (output_dir / "best_parameters.json").read_text(encoding="utf-8")
            )
            manifest = json.loads(
                (output_dir / "manifest.json").read_text(encoding="utf-8")
            )

        self.assertEqual(issues, ())
        self.assertEqual(saved_count, 1)
        self.assertEqual(best["trial_id"], "trial_0002")
        self.assertEqual(best["score"], 0.63)
        self.assertEqual(manifest["selected_trial_id"], "trial_0002")

    def test_failed_samples_export_without_any_passed_sample(self) -> None:
        dataset = _FakeSegmentFrameDataset()
        request = _multi_trial_request(
            (
                (
                    "trial_failed",
                    0.42,
                    (
                        FrameEvaluation(
                            sample_key="item_a/long_sequence_0001",
                            frame_index=1,
                            passed=False,
                            target_source_index=11,
                            predicted_video_xy=(96.0, 72.0),
                            primary_error="decision",
                            error_tags=("missing_hit",),
                        ),
                    ),
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
            failed_pngs = tuple((output_dir / "failed").rglob("*.png"))
            passed_pngs = tuple((output_dir / "passed").rglob("*.png"))

        self.assertEqual(issues, ())
        self.assertEqual(saved_count, 1)
        self.assertEqual(len(failed_pngs), 1)
        self.assertEqual(len(passed_pngs), 0)

    def test_score_tie_selects_lexicographically_first_trial_id(self) -> None:
        dataset = _FakeSegmentFrameDataset()
        frame = FrameEvaluation(
            sample_key="item_a/long_sequence_0001",
            frame_index=1,
            passed=False,
            target_source_index=11,
            predicted_video_xy=(64.0, 48.0),
            primary_error="spatial",
        )
        request = _multi_trial_request(
            (
                ("trial_0002", 0.63, ()),
                ("trial_0001", 0.63, (frame,)),
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

        self.assertEqual(issues, ())
        self.assertEqual(saved_count, 1)
        self.assertEqual(manifest["selected_trial_id"], "trial_0001")

    def test_export_failure_does_not_commit_counter_or_formal_artifact(self) -> None:
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
            )
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch(
                "visualization.core.gallery.exporter.render_annotated_frame",
                side_effect=RuntimeError("render exploded"),
            ):
                with self.assertRaisesRegex(RuntimeError, "render exploded"):
                    save_best_trial_gallery(
                        dataset,
                        request,
                        output_root=root,
                        samples_per_group=10,
                    )

            self.assertFalse((root / ".output_counter").exists())
            self.assertEqual(tuple(root.glob("output_*")), ())


if __name__ == "__main__":
    unittest.main()
