from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch

from traning.core.decision import CANDIDATE_CACHE_VERSION
from traning.core.temporal import (
    TemporalCandidateWindowDataset,
    TemporalFeatureSpec,
    load_candidate_cache_records,
)


def _record(
    sample_key: str,
    frame_index: int,
    *,
    candidates: list[dict] | None = None,
    temporal_target: dict | None = None,
) -> dict:
    record = {
        "version": CANDIDATE_CACHE_VERSION,
        "sample_key": sample_key,
        "frame_index": frame_index,
        "timestamp_ms": frame_index * 16.0,
        "frame_width": 100,
        "frame_height": 50,
        "candidates": candidates or [],
        "slider_paths": [],
    }
    if temporal_target is not None:
        record["temporal_target"] = temporal_target
    return record


def _candidate(
    score: float,
    *,
    x: float = 25.0,
    y: float = 10.0,
    candidate_id: int = 0,
) -> dict:
    return {
        "candidate_id": candidate_id,
        "x": x,
        "y": y,
        "score": score,
        "object_type": "hit_circle",
        "object_type_id": 1,
        "center_score": score,
        "visible_score": 0.9,
        "type_score": 0.8,
        "ring_score": 0.1,
        "ring_radius_px": 8.0,
        "slider_score": 0.0,
        "spinner_score": 0.0,
        "embedding": [0.25, 0.5],
        "slider_path_id": None,
        "ambiguous": False,
        "ambiguity_reasons": [],
    }


def _write_cache(path: Path, records: list[dict]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "version": CANDIDATE_CACHE_VERSION,
                "records": "frames.jsonl",
            }
        ),
        encoding="utf-8",
    )
    (path / "frames.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )


class TemporalDatasetTests(unittest.TestCase):
    def test_loads_candidate_cache_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            records = [_record("a", 0, candidates=[_candidate(0.8)])]
            _write_cache(cache_dir, records)
            loaded = load_candidate_cache_records(cache_dir)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["sample_key"], "a")

    def test_encodes_fixed_windows_without_crossing_samples(self) -> None:
        records = [
            _record("a", 0, candidates=[_candidate(0.7)]),
            _record("a", 1),
            _record("b", 0, candidates=[_candidate(0.9, x=50.0)]),
        ]
        dataset = TemporalCandidateWindowDataset(
            records,
            sequence_length=2,
            feature_spec=TemporalFeatureSpec(candidate_slots=2, embedding_dim=2),
        )
        self.assertEqual(len(dataset), 2)
        first = dataset[0]
        second = dataset[1]
        self.assertEqual(first.sample_keys, ("a", "a"))
        self.assertEqual(second.sample_keys, ("b", None))
        self.assertEqual(first.features.shape[0], 2)
        self.assertEqual(first.candidate_features.shape[:2], (2, 2))
        self.assertTrue(first.candidate_mask[0, 0])
        self.assertFalse(first.candidate_mask[1, 0])
        self.assertEqual(first.action_target.tolist(), [1, 0])
        self.assertEqual(first.selected_candidate_target.tolist(), [0, -100])
        self.assertTrue(torch.allclose(first.xy_target[0], torch.tensor([0.25, 0.2])))

    def test_uses_explicit_temporal_target_when_present(self) -> None:
        records = [
            _record(
                "a",
                0,
                candidates=[
                    _candidate(0.7, candidate_id=7),
                    _candidate(0.9, x=60.0, candidate_id=3),
                ],
                temporal_target={
                    "target_strategy": "beatmap_action_v1",
                    "action": "hold",
                    "action_id": 2,
                    "selected_candidate_id": 7,
                    "target_video_xy": [25.0, 10.0],
                    "time_offset_ms": 3.0,
                },
            )
        ]
        dataset = TemporalCandidateWindowDataset(
            records,
            sequence_length=1,
            feature_spec=TemporalFeatureSpec(candidate_slots=2, embedding_dim=2),
        )
        window = dataset[0]
        self.assertEqual(window.action_target.tolist(), [2])
        self.assertEqual(window.selected_candidate_target.tolist(), [1])
        self.assertEqual(window.target_strategy, "beatmap_action_v1")
        self.assertTrue(torch.allclose(window.time_offset_target[0], torch.tensor([3.0])))


if __name__ == "__main__":
    unittest.main()
