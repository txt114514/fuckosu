from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest
from unittest.mock import patch

import torch

from traning.conf import Settings
from traning.core.decision import (
    CANDIDATE_CACHE_VERSION,
    build_candidate_cache_record,
    generate_candidate_cache,
)
from traning.lib.training import SliderPathCandidate
from traning.lib.training.spatial_decode import SpatialCandidate


def _candidate(
    *,
    score: float = 0.55,
    object_type: str = "slider_head",
) -> SpatialCandidate:
    return SpatialCandidate(
        x=16.0,
        y=20.0,
        score=score,
        object_type=object_type,
        object_type_id=3,
        center_score=0.8,
        visible_score=0.9,
        type_score=0.7,
        ring_score=0.1,
        ring_radius_px=12.0,
        slider_score=0.8,
        spinner_score=0.1,
        embedding=(0.1, 0.2, 0.3),
    )


def _slider_path(*, ambiguous: bool = False) -> SliderPathCandidate:
    return SliderPathCandidate(
        component_id=4,
        score=0.85,
        continuity=0.9,
        ambiguous=ambiguous,
        ambiguity_reasons=("branch_points",) if ambiguous else (),
        bbox=(8.0, 16.0, 48.0, 24.0),
        head=(16.0, 20.0),
        tail=(44.0, 20.0),
        polyline=((16.0, 20.0), (30.0, 20.0), (44.0, 20.0)),
        cell_count=8,
        branch_points=1 if ambiguous else 0,
        endpoint_count=2,
    )


class CandidateCacheTests(unittest.TestCase):
    def test_record_keeps_embedding_and_candidate_ambiguity(self) -> None:
        candidates = (_candidate(score=0.55), _candidate(score=0.53))
        record = build_candidate_cache_record(
            {
                "sample_key": "sample-a",
                "frame_index": 2,
                "timestamp_ms": 100.0,
                "hit_objects": (
                    {
                        "type": "circle",
                        "start_ms": 100,
                        "end_ms": 100,
                        "x": 64.0,
                        "y": 64.0,
                        "source_index": 0,
                    },
                ),
            },
            candidates,
            (_slider_path(ambiguous=True),),
            frame_width=128,
            frame_height=96,
            device="cpu",
            patches_processed=3,
            frame_channels=6,
            save_dtype="float16",
            low_confidence_threshold=0.60,
            close_score_margin=0.05,
            slider_attach_distance_px=48.0,
        )

        self.assertEqual(record["version"], CANDIDATE_CACHE_VERSION)
        self.assertEqual(record["candidates"][0]["slider_path_id"], 4)
        self.assertIn("low_confidence", record["candidates"][0]["ambiguity_reasons"])
        self.assertIn("close_score", record["candidates"][0]["ambiguity_reasons"])
        self.assertIn(
            "slider_path_ambiguous",
            record["candidates"][0]["ambiguity_reasons"],
        )
        self.assertEqual(record["temporal_target"]["action"], "press")
        self.assertEqual(record["temporal_target"]["action_id"], 1)
        self.assertEqual(len(record["candidates"][0]["embedding"]), 3)
        self.assertEqual(record["slider_paths"][0]["component_id"], 4)

    def test_generate_candidate_cache_writes_manifest_and_jsonl(self) -> None:
        sample = {
            "image": torch.zeros((3, 24, 32)),
            "sample_key": "sample-a",
            "frame_index": 0,
            "timestamp_ms": 0.0,
        }
        fake_result = SimpleNamespace(
            candidates=(_candidate(score=0.8, object_type="hit_circle"),),
            slider_paths=(_slider_path(),),
            patches_processed=1,
            frame_channels=3,
        )
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            with patch(
                "traning.core.decision.generator.run_spatial_frame_inference",
                return_value=fake_result,
            ) as inference_mock:
                checkpoint_path = output_dir / "spatial_model.pt"
                result = generate_candidate_cache(
                    Settings(),
                    output_dir=output_dir,
                    device=torch.device("cpu"),
                    spatial_checkpoint_path=checkpoint_path,
                    dataset=[sample],
                    max_frames=1,
                )

            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            records = result.records_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(manifest["version"], CANDIDATE_CACHE_VERSION)
            self.assertEqual(manifest["frames"], 1)
            self.assertEqual(manifest["spatial_checkpoint_path"], str(checkpoint_path))
            self.assertEqual(len(records), 1)
            self.assertEqual(json.loads(records[0])["sample_key"], "sample-a")
            self.assertEqual(
                inference_mock.call_args.kwargs["checkpoint_path"],
                checkpoint_path,
            )

    def test_local_consistency_review_resolves_supported_ambiguity(self) -> None:
        settings = Settings()
        settings.candidate_cache.ambiguity_review_enabled = True
        candidates = (_candidate(score=0.58), _candidate(score=0.56))

        record = build_candidate_cache_record(
            {
                "sample_key": "sample-review",
                "frame_index": 1,
                "timestamp_ms": 100.0,
                "hit_objects": (),
            },
            candidates,
            (_slider_path(ambiguous=True),),
            frame_width=128,
            frame_height=96,
            device="cpu",
            patches_processed=1,
            frame_channels=3,
            save_dtype="float32",
            low_confidence_threshold=0.60,
            close_score_margin=0.05,
            slider_attach_distance_px=48.0,
            settings=settings,
        )

        review = record["candidates"][0]["ambiguity_review"]
        self.assertEqual(review["strategy"], "local_consistency_model_v1")
        self.assertTrue(review["resolved"])
        self.assertFalse(record["candidates"][0]["ambiguous"])


if __name__ == "__main__":
    unittest.main()
