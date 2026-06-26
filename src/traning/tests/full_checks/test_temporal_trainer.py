from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch

from traning.conf import Settings
from traning.core.decision import CANDIDATE_CACHE_VERSION
from traning.core.temporal import run_temporal_training
from traning.core.training_inheritance import load_training_checkpoint


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
        "\n".join(json.dumps(_record(index)) for index in range(6)) + "\n",
        encoding="utf-8",
    )


def _assert_nested_close(
    case: unittest.TestCase,
    left,
    right,
    *,
    path: str = "root",
) -> None:
    if torch.is_tensor(left):
        case.assertTrue(
            torch.equal(left.cpu(), right.cpu()),
            f"tensor mismatch at {path}",
        )
        return
    if isinstance(left, dict):
        case.assertEqual(set(left), set(right), f"keys mismatch at {path}")
        for key, value in left.items():
            _assert_nested_close(case, value, right[key], path=f"{path}.{key}")
        return
    if isinstance(left, (list, tuple)):
        case.assertEqual(len(left), len(right), f"length mismatch at {path}")
        for index, value in enumerate(left):
            _assert_nested_close(case, value, right[index], path=f"{path}[{index}]")
        return
    case.assertEqual(left, right, f"value mismatch at {path}")


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

    def test_resume_matches_continuous_temporal_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_dir = root / "cache"
            _write_cache(cache_dir)
            settings = Settings()
            settings.runtime.seed = 123
            settings.temporal.hidden_size = 8
            settings.temporal.layers = 1

            continuous = run_temporal_training(
                settings,
                cache_dir=cache_dir,
                device=torch.device("cpu"),
                run_dir=root / "continuous",
                max_steps=4,
                learning_rate=1e-3,
                sequence_length=2,
                candidate_slots=2,
            )
            partial = run_temporal_training(
                settings,
                cache_dir=cache_dir,
                device=torch.device("cpu"),
                run_dir=root / "partial",
                max_steps=2,
                learning_rate=1e-3,
                sequence_length=2,
                candidate_slots=2,
            )
            resumed = run_temporal_training(
                settings,
                cache_dir=cache_dir,
                device=torch.device("cpu"),
                run_dir=root / "resumed",
                max_steps=4,
                learning_rate=1e-3,
                sequence_length=2,
                candidate_slots=2,
                resume_checkpoint_path=partial.checkpoint_path,
                resume_policy="strict",
            )

            self.assertEqual(resumed.steps, continuous.steps)
            partial_checkpoint = load_training_checkpoint(partial.checkpoint_path)
            self.assertEqual(
                partial_checkpoint["training_position"]["temporal_step"],
                2,
            )
            continuous_checkpoint = load_training_checkpoint(continuous.checkpoint_path)
            resumed_checkpoint = load_training_checkpoint(resumed.checkpoint_path)
            self.assertEqual(
                resumed_checkpoint["training_position"]["temporal_step"],
                4,
            )
            _assert_nested_close(
                self,
                continuous_checkpoint["model_state"],
                resumed_checkpoint["model_state"],
            )
            _assert_nested_close(
                self,
                continuous_checkpoint["optimizer"],
                resumed_checkpoint["optimizer"],
            )


if __name__ == "__main__":
    unittest.main()
