"""验证候选缓存的生成、筛选、歧义标记和持久化契约。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json
import math
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch

from package.coordinates import AffineOsuVideoTransform
from traning.conf import Settings
from traning.core.decision import (
    CANDIDATE_CACHE_VERSION,
    build_candidate_cache_record,
    generate_candidate_cache,
)
from traning.lib.training import SliderPathCandidate
from traning.lib.training.spatial_decode import SpatialCandidate


class _GroupedSampleDataset:
    def __init__(self, group_count: int = 6, frames_per_group: int = 3) -> None:
        # 每组提供连续帧，但所有组交错共享一个扁平索引；该 fixture 用来
        # 捕获 max_frames 被错误实现为简单截取前 N 帧的回归。
        self.records = tuple(
            SimpleNamespace(key=f"sample-{group_index}")
            for group_index in range(group_count)
        )
        references = []
        for group_index in range(group_count):
            for frame_index in range(frames_per_group):
                references.append(
                    SimpleNamespace(
                        record_index=group_index,
                        frame_index=frame_index,
                    )
                )
        self.references = tuple(references)

    def __len__(self) -> int:
        return len(self.references)

    def __getitem__(self, index: int) -> dict[str, object]:
        reference = self.references[index]
        record = self.records[reference.record_index]
        return {
            "image": torch.zeros((3, 24, 32)),
            "sample_key": record.key,
            "frame_index": reference.frame_index,
            "timestamp_ms": float(index),
        }


def _candidate(
    *,
    score: float = 0.55,
    object_type: str = "slider_head",
    x: float = 16.0,
    y: float = 20.0,
) -> SpatialCandidate:
    return SpatialCandidate(
        x=x,
        y=y,
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
    def test_target_matching_uses_actual_circle_radius_after_affine_mapping(
        self,
    ) -> None:
        """CS3 的真实半径应接纳指定回归样本中距中心 74px 的候选。"""

        transform = AffineOsuVideoTransform(
            (
                (2.115860914627143, 0.0011971920855575358, 242.59057485632047),
                (0.0003418231662923798, 2.1166805757239477, 16.12108357719331),
            )
        )
        target_osu = (40.0, 262.0)
        target_video = transform.osu_to_video(*target_osu)
        candidate_video = (328.6796875, 496.6796875)
        actual_radius_osu = 40.9767936
        actual_radius_video = transform.osu_radius_to_video(actual_radius_osu)
        self.assertGreater(math.dist(candidate_video, target_video), 64.0)
        self.assertLess(math.dist(candidate_video, target_video), actual_radius_video)

        record = build_candidate_cache_record(
            {
                "sample_key": "item_000001/long_sequence_000008",
                "frame_index": 105,
                "timestamp_ms": 1750.0,
                "circle_radius_osu_pixels": actual_radius_osu,
                "coordinate_transform": transform.spec(
                    source="test.affine",
                    status="calibrated",
                ).as_dict(),
                "hit_objects": (
                    {
                        "type": "slider",
                        "start_ms": 1754.0,
                        "end_ms": 2216.0,
                        "path": (target_osu, (96.0, 295.0), (159.0, 280.0)),
                        "source_index": 85,
                    },
                ),
            },
            (_candidate(x=candidate_video[0], y=candidate_video[1]),),
            (),
            frame_width=1484,
            frame_height=846,
            device="cpu",
            patches_processed=1,
            frame_channels=6,
            save_dtype="float16",
            low_confidence_threshold=0.60,
            close_score_margin=0.05,
            slider_attach_distance_px=48.0,
        )

        target = record["temporal_target"]
        self.assertEqual(record["circle_radius_osu_pixels"], actual_radius_osu)
        self.assertAlmostEqual(
            record["circle_radius_video_pixels"],
            actual_radius_video,
        )
        self.assertAlmostEqual(target["candidate_match_radius_px"], actual_radius_video)
        self.assertEqual(target["candidate_match_status"], "matched")
        self.assertEqual(target["selected_candidate_id"], 0)

    def test_record_keeps_embedding_and_candidate_ambiguity(self) -> None:
        # 两个低分且分数接近的候选再挂接一条有分支路径，使三种歧义来源
        # 同时出现，验证序列化不会丢失其中任一诊断标签。
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
            runner = SimpleNamespace(infer_frame=Mock(return_value=fake_result))
            # 只 mock 空间推理边界，保留候选记录构建和磁盘提交的真实路径。
            with patch(
                "traning.core.decision.generator.prepare_spatial_frame_inference",
                return_value=runner,
            ) as prepare_mock:
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
                prepare_mock.call_args.kwargs["checkpoint_path"],
                checkpoint_path,
            )
            self.assertEqual(runner.infer_frame.call_count, 1)

    def test_candidate_cache_max_frames_samples_across_groups(self) -> None:
        dataset = _GroupedSampleDataset(group_count=6, frames_per_group=3)
        fake_result = SimpleNamespace(
            candidates=(_candidate(score=0.8, object_type="hit_circle"),),
            slider_paths=(),
            patches_processed=1,
            frame_channels=3,
        )
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            runner = SimpleNamespace(infer_frame=Mock(return_value=fake_result))
            with patch(
                "traning.core.decision.generator.prepare_spatial_frame_inference",
                return_value=runner,
            ):
                result = generate_candidate_cache(
                    Settings(runtime={"seed": 99}, temporal={"history_frames": 3}),
                    output_dir=output_dir,
                    device=torch.device("cpu"),
                    dataset=dataset,
                    max_frames=6,
                )

            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            records = [
                json.loads(line)
                for line in result.records_path.read_text(encoding="utf-8").splitlines()
            ]

        selected_sample_keys = tuple(record["sample_key"] for record in records)
        self.assertEqual(
            manifest["sampling"]["mode"],
            "seeded_group_contiguous_round_robin",
        )
        self.assertEqual(
            manifest["spatial_inference_context"], "reused_per_candidate_cache"
        )
        self.assertEqual(manifest["sampling"]["contiguous_block_frames"], 3)
        self.assertEqual(manifest["sampling"]["unique_sample_groups"], 2)
        self.assertEqual(len(set(selected_sample_keys)), 2)
        for sample_key in set(selected_sample_keys):
            frame_indices = [
                int(record["frame_index"])
                for record in records
                if record["sample_key"] == sample_key
            ]
            self.assertEqual(
                frame_indices,
                list(range(frame_indices[0], frame_indices[0] + len(frame_indices))),
            )
        self.assertEqual(runner.infer_frame.call_count, 6)

    def test_local_consistency_review_resolves_supported_ambiguity(self) -> None:
        settings = Settings()
        settings.candidate_cache.ambiguity_review_enabled = True
        # 分数仍落在基础歧义阈值内，测试本地一致性复核确实能覆盖初判，
        # 而不是因为输入本身已不再 ambiguous 才得到 resolved。
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
