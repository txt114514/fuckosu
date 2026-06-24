from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch

from traning.conf import Settings
from traning.core.decision import CANDIDATE_CACHE_VERSION
from traning.core.temporal import run_temporal_training


def _record(frame_index: int) -> dict:
    return {
        "version": CANDIDATE_CACHE_VERSION,
        "sample_key": "sample",
        "frame_index": frame_index,
        "timestamp_ms": frame_index * 16.0,
        "frame_width": 100,
        "frame_height": 50,
        "candidates": [
            {
                "candidate_id": 0,
                "x": 20.0 + frame_index,
                "y": 10.0,
                "score": 0.9,
                "object_type": "hit_circle",
                "object_type_id": 1,
                "center_score": 0.9,
                "visible_score": 0.8,
                "type_score": 0.7,
                "ring_score": 0.1,
                "ring_radius_px": 8.0,
                "slider_score": 0.0,
                "spinner_score": 0.0,
                "embedding": [0.1, 0.2, 0.3],
                "slider_path_id": None,
                "ambiguous": False,
                "ambiguity_reasons": [],
            }
        ],
        "slider_paths": [],
    }


def _write_cache(path: Path) -> None:
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
        "\n".join(json.dumps(_record(index)) for index in range(3)) + "\n",
        encoding="utf-8",
    )


class TemporalTrainerTests(unittest.TestCase):
    def test_cpu_temporal_training_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_dir = root / "cache"
            run_dir = root / "run"
            _write_cache(cache_dir)
            settings = Settings()
            settings.temporal.hidden_size = 8
            settings.temporal.layers = 1
            result = run_temporal_training(
                settings,
                cache_dir=cache_dir,
                device=torch.device("cpu"),
                run_dir=run_dir,
                max_steps=2,
                learning_rate=1e-3,
                sequence_length=2,
                candidate_slots=2,
            )
            self.assertEqual(result.steps, 2)
            self.assertEqual(result.sequence_length, 2)
            self.assertEqual(result.candidate_slots, 2)
            self.assertGreater(result.input_size, 0)
            self.assertTrue((run_dir / "summary.json").is_file())
            self.assertTrue(result.checkpoint_path.is_file())


if __name__ == "__main__":
    unittest.main()
