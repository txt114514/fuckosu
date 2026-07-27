"""验证最佳结果图集的筛选、仿射渲染、目录布局与故障隔离。"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch

from package.coordinates import AffineOsuVideoTransform, OsuVideoTransform
from traning.lib.visualization import render_annotated_frame
from traning.lib.visualization.render import _annotation_font
from traning.lib.visualization.gallery import save_best_trial_gallery
from traning.state import (
    BatchGalleryRequest,
    FrameEvaluation,
    TrialGalleryEvaluation,
    TrialParameters,
)
from visualization.core.gallery.exporter import MAX_SAFE_NAME_LENGTH, _safe_name


# 与正式配置一致，用真实非居中映射检查渲染链路没有回退到 legacy_centered。
AFFINE_RENDER_MATRIX = (
    (2.115860914627143, 0.0011971920855575358, 242.59057485632047),
    (0.0003418231662923798, 2.1166805757239477, 16.12108357719331),
)
AFFINE_MARKER_OSU_XY = (256.0, 183.0)
TARGET_MARKER_COLOR = (255, 72, 72)
PREDICTED_MARKER_COLOR = (255, 80, 255)


class _FakeSegmentFrameDataset:
    def __init__(self) -> None:
        # item_a 有三个帧、item_b 只有一个帧，用同一 fixture 区分
        # “sample group 数量上限”和“组内帧数量”两层选择语义。
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


class _DiverseFakeSegmentFrameDataset(_FakeSegmentFrameDataset):
    def __init__(self, size: int = 6) -> None:
        self.records = tuple(
            SimpleNamespace(
                key=f"item_{index}/long_sequence_{index:04d}",
                category="long_sequence",
                dataset_dimension="long_sequence",
            )
            for index in range(size)
        )
        self.references = tuple(
            SimpleNamespace(record_index=index, frame_index=1, timestamp_ms=100.0)
            for index in range(size)
        )


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
    def test_safe_name_caps_long_tokens_with_stable_collision_resistant_hash(
        self,
    ) -> None:
        short = "trial-ramp-a__r01"
        long_prefix = "trial-" + "parameter-group-" * 12
        first = _safe_name(long_prefix + "alpha")
        second = _safe_name(long_prefix + "beta")

        self.assertEqual(_safe_name(short), short)
        self.assertLessEqual(len(first), MAX_SAFE_NAME_LENGTH)
        self.assertEqual(first, _safe_name(long_prefix + "alpha"))
        self.assertNotEqual(first, second)

    def test_annotation_font_distinguishes_chinese_diagnostic_text(self) -> None:
        font = _annotation_font(846)

        self.assertNotEqual(
            bytes(font.getmask("真值目标")), bytes(font.getmask("模型点击"))
        )

    def test_failed_gallery_persists_no_op_reason_and_action(self) -> None:
        dataset = _FakeSegmentFrameDataset()
        request = _request(
            (
                FrameEvaluation(
                    sample_key="item_a/long_sequence_0001",
                    frame_index=1,
                    passed=False,
                    target_source_index=11,
                    action="no_op",
                    action_probability=0.8981025218963623,
                    primary_error="spatial",
                    error_tags=(
                        "unresolved_target",
                        "candidate_match_failed",
                        "nearest_candidate_outside_radius",
                    ),
                    failure_reason=(
                        "target-candidate matching failed: "
                        "nearest_candidate_outside_radius"
                    ),
                ),
            )
        )

        with tempfile.TemporaryDirectory() as temporary:
            output_dir, saved_count, issues = save_best_trial_gallery(
                dataset,
                request,
                output_root=Path(temporary),
            )
            manifest = json.loads(
                (output_dir / "manifest.json").read_text(encoding="utf-8")
            )
            with (output_dir / "index.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                index_rows = list(csv.DictReader(handle))

        frame = manifest["frames"][0]
        self.assertEqual(issues, ())
        self.assertEqual(saved_count, 1)
        self.assertTrue(frame["path"].startswith("failed/long_sequence/"))
        self.assertEqual(frame["action"], "no_op")
        self.assertIsNone(frame["predicted_video_xy"])
        self.assertEqual(frame["primary_error"], "spatial")
        self.assertEqual(index_rows[0]["action_type"], "no_op")
        self.assertIn("target-candidate", index_rows[0]["failure_reason"])

    def test_affine_circle_slider_and_spinner_render_at_affine_marker(self) -> None:
        """验证三类对象都消费样本携带的仿射规格，而非仅修正某一渲染分支。"""

        transform = AffineOsuVideoTransform(AFFINE_RENDER_MATRIX)
        transform_spec = transform.spec(source="test.affine", status="calibrated")
        expected_marker = tuple(
            round(value) for value in transform.osu_to_video(*AFFINE_MARKER_OSU_XY)
        )
        legacy_marker = tuple(
            round(value)
            for value in OsuVideoTransform.fit_centered(1484, 846).osu_to_video(
                *AFFINE_MARKER_OSU_XY
            )
        )
        # 三类对象走不同绘制路径；spinner 通过预测标记显式检查变换后的中心。
        objects = {
            "circle": {
                "type": "circle",
                "x": AFFINE_MARKER_OSU_XY[0],
                "y": AFFINE_MARKER_OSU_XY[1],
                "source_index": 1,
            },
            "slider": {
                "type": "slider",
                "x": AFFINE_MARKER_OSU_XY[0],
                "y": AFFINE_MARKER_OSU_XY[1],
                "path": (
                    AFFINE_MARKER_OSU_XY,
                    (AFFINE_MARKER_OSU_XY[0] + 48.0, AFFINE_MARKER_OSU_XY[1]),
                ),
                "curve_type": "L",
                "source_index": 1,
            },
            "spinner": {
                "type": "spinner",
                "start_ms": 0.0,
                "end_ms": 1000.0,
                "source_index": 1,
            },
        }

        for object_type, hit_object in objects.items():
            with self.subTest(object_type=object_type):
                sample = {
                    "image": torch.zeros((3, 846, 1484), dtype=torch.float32),
                    "sample_key": f"item/affine_{object_type}",
                    "frame_index": 0,
                    "timestamp_ms": 0.0,
                    "hit_objects": (hit_object,),
                    "visible_hit_objects": (hit_object,),
                    "circle_radius_osu_pixels": 16.0,
                    "coordinate_transform": transform_spec.as_dict(),
                }
                image = render_annotated_frame(
                    sample,
                    target_source_index=1,
                    predicted_osu_xy=(
                        AFFINE_MARKER_OSU_XY if object_type == "spinner" else None
                    ),
                )

                marker_color = (
                    PREDICTED_MARKER_COLOR
                    if object_type == "spinner"
                    else TARGET_MARKER_COLOR
                )
                # 同时断言旧中心没有目标颜色，可捕获悄然回退旧映射的回归。
                self.assertEqual(image.getpixel(expected_marker), marker_color)
                self.assertNotEqual(image.getpixel(legacy_marker), marker_color)

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
            sample_dirs = tuple((output_dir / "passed" / "long_sequence").iterdir())
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
            sample_dirs = tuple((output_dir / "passed" / "long_sequence").iterdir())
            sample_dir_count = len(sample_dirs)

        self.assertEqual(issues, ())
        self.assertEqual(manifest["selected_sample_group_count"], 1)
        self.assertEqual(sample_dir_count, 1)
        self.assertEqual(
            saved_count,
            manifest["sample_groups"][0]["frame_count"],
        )

    def test_gallery_samples_diverse_groups_by_seed_not_first_n(self) -> None:
        dataset = _DiverseFakeSegmentFrameDataset(size=6)
        request = BatchGalleryRequest(
            batch_id="gallery_diversity",
            random_seed=99,
            trials=(
                TrialGalleryEvaluation(
                    trial_id="pg-diverse",
                    score=0.9,
                    parameters=TrialParameters(),
                    frames=tuple(
                        FrameEvaluation(
                            sample_key=record.key,
                            frame_index=1,
                            passed=True,
                            target_source_index=11,
                            predicted_video_xy=(64.0, 48.0),
                        )
                        for record in dataset.records
                    ),
                ),
            ),
        )

        with tempfile.TemporaryDirectory() as temporary:
            output_dir, saved_count, issues = save_best_trial_gallery(
                dataset,
                request,
                output_root=Path(temporary),
                samples_per_group=2,
            )
            manifest = json.loads(
                (output_dir / "manifest.json").read_text(encoding="utf-8")
            )

        # 不固定具体抽样结果，只约束可复现随机策略没有退化为目录前 N 项。
        selected = tuple(group["sample_key"] for group in manifest["sample_groups"])
        self.assertEqual(issues, ())
        self.assertEqual(saved_count, 2)
        self.assertEqual(len(selected), 2)
        self.assertNotEqual(
            selected,
            tuple(record.key for record in dataset.records[:2]),
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
            # 在首帧渲染边界注入失败，验证计数器和正式 output_* 目录都只在
            # 整批成功后提交，不遗留看似有效的半成品。
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
