"""验证候选缓存窗口的连续性、mask、特征槽位和监督编码。"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
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
    # 100×50 使候选 (25, 10) 恰好编码为 (0.25, 0.2)，便于同时验证
    # 候选槽位和整帧归一化坐标契约。
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


def _write_cache(
    path: Path,
    records: list[dict],
    *,
    version: str = CANDIDATE_CACHE_VERSION,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    versioned_records = [dict(record, version=version) for record in records]
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "version": version,
                "records": "frames.jsonl",
            }
        ),
        encoding="utf-8",
    )
    (path / "frames.jsonl").write_text(
        "\n".join(json.dumps(record) for record in versioned_records) + "\n",
        encoding="utf-8",
    )


class TemporalDatasetTests(unittest.TestCase):
    def test_temporal_package_imports_in_fresh_interpreter(self) -> None:
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[3])
        environment["PYTHONDONTWRITEBYTECODE"] = "1"

        result = subprocess.run(
            [sys.executable, "-c", "import traning.core.temporal"],
            capture_output=True,
            check=False,
            env=environment,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_legacy_v1_cache_requires_explicit_diagnostic_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            _write_cache(
                cache_dir,
                [_record("legacy", 0, candidates=[_candidate(0.8)])],
                version="spatial-candidate-cache-v1",
            )

            with self.assertRaisesRegex(ValueError, "diagnostic-only"):
                load_candidate_cache_records(cache_dir)
            loaded = load_candidate_cache_records(cache_dir, allow_legacy=True)

        self.assertEqual(loaded[0]["version"], "spatial-candidate-cache-v1")

    def test_loads_candidate_cache_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            records = [_record("a", 0, candidates=[_candidate(0.8)])]
            _write_cache(cache_dir, records)
            loaded = load_candidate_cache_records(cache_dir)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["sample_key"], "a")

    def test_encodes_fixed_windows_without_crossing_samples(self) -> None:
        # a 正好填满窗口，b 需要一个 padding 帧；若窗口跨 sample 拼接，
        # b 会错误消费 a 的时序状态且 mask/sentinel 均会变化。
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
                # 目标候选 ID=7 不是分数最高项，约束监督按显式 ID 而非 argmax。
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
        self.assertTrue(
            torch.allclose(window.time_offset_target[0], torch.tensor([3.0]))
        )

    def test_preserves_selected_target_candidate_when_outside_top_scores(self) -> None:
        # candidate_slots=2 但监督候选按分数排第三；数据集必须替换掉普通
        # top-k 的末槽，否则 candidate loss 会得到 ignore sentinel。
        records = [
            _record(
                "a",
                0,
                candidates=[
                    _candidate(0.95, candidate_id=1, x=10.0),
                    _candidate(0.90, candidate_id=2, x=20.0),
                    _candidate(0.10, candidate_id=7, x=70.0),
                ],
                temporal_target={
                    "target_strategy": "beatmap_action_v1",
                    "action": "press",
                    "action_id": 1,
                    "selected_candidate_id": 7,
                    "target_video_xy": [70.0, 10.0],
                    "time_offset_ms": 0.0,
                },
            )
        ]
        dataset = TemporalCandidateWindowDataset(
            records,
            sequence_length=1,
            feature_spec=TemporalFeatureSpec(candidate_slots=2, embedding_dim=2),
        )
        window = dataset[0]

        self.assertEqual(window.candidate_ids[0], (1, 7))
        self.assertEqual(window.selected_candidate_target.tolist(), [1])
        self.assertTrue(torch.allclose(window.xy_target[0], torch.tensor([0.7, 0.2])))


if __name__ == "__main__":
    unittest.main()
