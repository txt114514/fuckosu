from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch

from traning.conf import Settings
from traning.core.decision import CANDIDATE_CACHE_VERSION, run_temporal_decision
from traning.core.temporal import run_temporal_training


def _record(frame_index: int) -> dict:
    return {
        "version": CANDIDATE_CACHE_VERSION,
        "sample_key": "sample",
        "frame_index": frame_index,
        "timestamp_ms": frame_index * 16.0,
        "frame_width": 100,
        "frame_height": 50,
        "temporal_target": {
            "target_strategy": "beatmap_action_v1",
            "action": "press",
            "action_id": 1,
            "selected_candidate_id": 0,
            "target_video_xy": [20.0, 10.0],
            "time_offset_ms": 0.0,
        },
        "candidates": [
            {
                "candidate_id": 0,
                "x": 20.0,
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
                "embedding": [0.1, 0.2],
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
        json.dumps({"version": CANDIDATE_CACHE_VERSION, "records": "frames.jsonl"}),
        encoding="utf-8",
    )
    (path / "frames.jsonl").write_text(
        "\n".join(json.dumps(_record(index)) for index in range(2)) + "\n",
        encoding="utf-8",
    )


class TemporalDecisionTests(unittest.TestCase):
    def test_train_then_run_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_dir = root / "cache"
            train_dir = root / "train"
            decision_dir = root / "decision"
            _write_cache(cache_dir)
            settings = Settings()
            settings.temporal.hidden_size = 8
            settings.temporal.layers = 1
            trained = run_temporal_training(
                settings,
                cache_dir=cache_dir,
                device=torch.device("cpu"),
                run_dir=train_dir,
                max_steps=1,
                sequence_length=2,
                candidate_slots=1,
            )
            result = run_temporal_decision(
                settings,
                cache_dir=cache_dir,
                checkpoint_path=trained.checkpoint_path,
                output_dir=decision_dir,
                device=torch.device("cpu"),
            )
            rows = [
                json.loads(line)
                for line in result.decisions_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(result.frames, 2)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["version"], "temporal-decision-v1")
            self.assertIn(rows[0]["action"], {"no_op", "press", "hold", "release"})
            self.assertTrue(result.manifest_path.is_file())


if __name__ == "__main__":
    unittest.main()
