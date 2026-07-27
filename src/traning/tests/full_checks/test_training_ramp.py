"""验证渐进训练级别、门限、恢复和晋级决策。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest
from unittest.mock import patch

import torch
import yaml

from traning.core.training_ramp import (
    RampLevelSpec,
    RampEvaluationGateError,
    RampGateError,
    RampSearchExhausted,
    RampTarget,
    _gate_level,
    _record_ramp_interrupted,
    _report_level_finished,
    _report_level_started,
    _run_level,
    _run_preflight,
    _report_ramp_failed,
    _report_ramp_started,
    _trial_runtime_overrides,
    _write_level_config,
    build_ramp_levels,
    ensure_full_target_config,
)
from traning.main import CliParameterError, run_training_job_spec
from visualization.lib import DashboardReporter, ResourceState


class TrainingRampTests(unittest.TestCase):
    def test_build_ramp_levels_clips_and_reaches_target(self) -> None:
        # 目标刻意不对齐内置 level 模板，验证中间级别会被裁剪且最后一级
        # 精确到达目标，而不是越过目标或停在最近模板值。
        target = RampTarget(
            spatial_steps=350,
            temporal_steps=325,
            patch_limit=3,
            cache_frames=1600,
            sequence_length=80,
            candidate_slots=20,
            gallery_samples_per_group=3,
        )

        levels = build_ramp_levels(target)

        self.assertGreaterEqual(len(levels), 2)
        self.assertEqual(levels[0].key, "a")
        self.assertEqual(levels[-1].spatial_steps, target.spatial_steps)
        self.assertEqual(levels[-1].temporal_steps, target.temporal_steps)
        self.assertEqual(levels[-1].patch_limit, target.patch_limit)
        self.assertEqual(levels[-1].cache_frames, target.cache_frames)
        self.assertEqual(levels[-1].sequence_length, target.sequence_length)
        self.assertEqual(levels[-1].candidate_slots, target.candidate_slots)
        for previous, current in zip(levels, levels[1:]):
            self.assertLessEqual(previous.spatial_steps, current.spatial_steps)
            self.assertLessEqual(previous.temporal_steps, current.temporal_steps)
            self.assertLessEqual(previous.cache_frames, current.cache_frames)

    def test_ensure_full_target_config_writes_target_and_absolutizes_paths(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "configs"
            output_dir = root / "out"
            config_dir.mkdir()
            source_config = config_dir / "small.yaml"
            target_config = config_dir / "full.yaml"
            # 所有持久化输出路径都使用相对值，捕获 resolved config 被移动到
            # output_dir 后相对基准意外改变的回归。
            source_config.write_text(
                yaml.safe_dump(
                    {
                        "data_input": {
                            "dataset_root": "../training_package/video_segments",
                            "split_manifest_path": "../training_package/splits/dataset_split_manifest.json",
                        },
                        "candidate_cache": {"output_root": "../runs/candidate_cache"},
                        "visualization": {"output_dir": "../runs/gallery"},
                        "optimization": {"trial_store_path": "../runs/trials.jsonl"},
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            resolved, target = ensure_full_target_config(
                source_config=source_config,
                target_config=target_config,
                output_dir=output_dir,
            )

            self.assertEqual(resolved, output_dir / "resolved_target_config.yaml")
            self.assertTrue(target_config.exists())
            self.assertEqual(target, RampTarget())
            raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
            self.assertEqual(raw["training_ramp"]["target"], RampTarget().__dict__)
            self.assertTrue(Path(raw["data_input"]["dataset_root"]).is_absolute())
            self.assertTrue(Path(raw["candidate_cache"]["output_root"]).is_absolute())
            self.assertTrue(Path(raw["visualization"]["output_dir"]).is_absolute())
            self.assertTrue(Path(raw["optimization"]["trial_store_path"]).is_absolute())

    def test_level_config_isolates_jsonl_and_sqlite_trial_history(self) -> None:
        level = RampLevelSpec("a", "level_a", 1, 1, 1, 1, 1, 1, 1)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "config.yaml"
            level_dir = root / "level"
            source.write_text(
                yaml.safe_dump(
                    {
                        "optimization": {
                            "trial_store_backend": "sqlite",
                            "trial_store_path": "global/trials.jsonl",
                            "trial_store_sqlite_path": "global/trials.sqlite",
                        }
                    }
                ),
                encoding="utf-8",
            )

            resolved = _write_level_config(source, level_dir, level)
            raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))

        self.assertEqual(
            Path(raw["optimization"]["trial_store_path"]),
            (level_dir / "metrics" / "trials.jsonl").resolve(),
        )
        self.assertEqual(
            Path(raw["optimization"]["trial_store_sqlite_path"]),
            (level_dir / "metrics" / "trials.sqlite").resolve(),
        )

    def test_ramp_reporter_tracks_level_pass_and_failure(self) -> None:
        level = RampLevelSpec("a", "level_a", 3, 2, 1, 5, 2, 1, 1)
        target = RampTarget(
            spatial_steps=3,
            temporal_steps=2,
            patch_limit=1,
            cache_frames=5,
            sequence_length=2,
            candidate_slots=1,
            gallery_samples_per_group=1,
        )
        record = {
            "status": "passed",
            "evaluation": {
                "quality_score": 0.75,
                "gallery_output_dir": "traning_example/example_gallery",
            },
            "steps_per_second": 1.25,
            "frames_per_second": 2.5,
            "artifact_manifest": "artifact/manifest.json",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            reporter = DashboardReporter(
                run_id="ramp-ui-test",
                output_dir=Path(temp_dir) / "dashboard",
            )

            _report_ramp_started(
                reporter,
                levels=[level],
                target=target,
                auto_launch_full=True,
            )
            state = reporter.snapshot()
            self.assertEqual(state.status, "running")
            self.assertEqual(state.total_levels, 1)
            self.assertEqual(
                state.pipeline_stages["training_ramp"].status,
                "running",
            )

            _report_level_started(reporter, level=level, index=1, total_levels=1)
            state = reporter.snapshot()
            self.assertEqual(state.current_level, "a")
            self.assertEqual(state.current_trial_id, "ramp-a")
            self.assertEqual(state.pipeline_stages["level_a"].status, "running")

            _report_level_finished(
                reporter,
                level=level,
                index=1,
                total_levels=1,
                record=record,
            )
            state = reporter.snapshot()
            self.assertEqual(state.completed_levels, 1)
            self.assertEqual(state.metrics.score, 0.75)
            self.assertEqual(state.metrics.level_best_score, 0.75)
            self.assertEqual(state.pipeline_stages["level_a"].status, "passed")
            self.assertEqual(state.promotion_status, "Level A 已通过 gate")

            failure_reporter = DashboardReporter(
                run_id="ramp-ui-failure-test",
                output_dir=Path(temp_dir) / "failure_dashboard",
            )
            _report_ramp_failed(
                failure_reporter,
                error=RuntimeError("boom"),
                active_level=level,
                active_index=1,
                completed_levels=0,
                total_levels=1,
            )
            failed = failure_reporter.snapshot()
            self.assertEqual(failed.status, "failed")
            self.assertEqual(failed.stop_state.reason, "RAMP_FAILED")
            self.assertEqual(failed.pipeline_stages["level_a"].status, "failed")

    def test_preflight_marks_gpu_bridge_passed_when_cuda_is_visible(self) -> None:
        env = SimpleNamespace(
            python_version="3.11",
            torch=SimpleNamespace(
                version="2.9.0",
                torch_cuda="13.0",
                cuda_available=True,
                gpu_name="NVIDIA GPU",
                total_vram_gib=8.0,
                free_vram_gib=7.5,
            ),
        )
        data_report = SimpleNamespace(
            ok=True,
            segment_count=2,
            frame_count_estimate=20,
            category_counts={},
            dimension_counts={},
            distribution={"data_quality_issues": ()},
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            reporter = DashboardReporter(
                run_id="preflight-gpu",
                output_dir=Path(temp_dir) / "dashboard",
            )
            # 环境、数据、资源和磁盘探针全部固定，仅验证预检结果如何映射
            # 到 dashboard gate 状态，不依赖执行机器是否真的具有 GPU。
            with (
                patch(
                    "traning.core.training_ramp.collect_environment_report",
                    return_value=env,
                ),
                patch(
                    "traning.core.training_ramp.load_settings", return_value=object()
                ),
                patch(
                    "traning.core.training_ramp.inspect_data_input",
                    return_value=data_report,
                ),
                patch(
                    "traning.core.training_ramp.collect_resource_state",
                    return_value=ResourceState(
                        gpu_name="NVIDIA GPU",
                        gpu_utilization=23.0,
                        gpu_monitor_source="nvidia-smi",
                    ),
                ),
                patch(
                    "traning.core.training_ramp._free_disk_bytes",
                    return_value=20 * 1024**3,
                ),
            ):
                _run_preflight(
                    config_path=Path("config.yaml"),
                    device="cuda",
                    output_dir=Path(temp_dir),
                    run_full_checks=False,
                    reporter=reporter,
                )

            state = reporter.snapshot()
            self.assertEqual(state.pipeline_stages["gpu_bridge"].status, "passed")
            self.assertEqual(state.pipeline_stages["output_disk"].status, "passed")
            self.assertEqual(state.pipeline_stages["gpu_bridge"].processed, 1)
            self.assertEqual(state.resources.gpu_utilization, 23.0)

    def test_preflight_reports_disk_space_gate_failure(self) -> None:
        env = SimpleNamespace(
            python_version="3.11",
            torch=SimpleNamespace(
                version="2.9.0",
                torch_cuda="13.0",
                cuda_available=False,
                gpu_name=None,
                total_vram_gib=None,
                free_vram_gib=None,
            ),
        )
        data_report = SimpleNamespace(
            ok=True,
            segment_count=2,
            frame_count_estimate=20,
            category_counts={},
            dimension_counts={},
            distribution={"data_quality_issues": ()},
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            reporter = DashboardReporter(
                run_id="preflight-disk",
                output_dir=Path(temp_dir) / "dashboard",
            )
            with (
                patch(
                    "traning.core.training_ramp.collect_environment_report",
                    return_value=env,
                ),
                patch(
                    "traning.core.training_ramp.load_settings", return_value=object()
                ),
                patch(
                    "traning.core.training_ramp.inspect_data_input",
                    return_value=data_report,
                ),
                patch(
                    "traning.core.training_ramp._free_disk_bytes",
                    return_value=2 * 1024**3,
                ),
                patch.dict(
                    "os.environ",
                    {"OSU_AI_MIN_RAMP_OUTPUT_FREE_GIB": "10"},
                ),
            ):
                with self.assertRaisesRegex(RampGateError, "free space"):
                    _run_preflight(
                        config_path=Path("config.yaml"),
                        device="cpu",
                        output_dir=Path(temp_dir),
                        run_full_checks=False,
                        reporter=reporter,
                    )

            state = reporter.snapshot()
            self.assertEqual(state.pipeline_stages["output_disk"].status, "failed")
            self.assertTrue(state.pipeline_stages["output_disk"].blocks_training)

    def test_gate_rejects_quality_score_below_threshold(self) -> None:
        level = RampLevelSpec("a", "level_a", 1, 1, 1, 1, 1, 1, 1)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spatial_checkpoint = root / "spatial.pt"
            temporal_checkpoint = root / "temporal.pt"
            report_path = root / "report.json"
            next_job_path = root / "next_job.json"
            spatial_checkpoint.write_bytes(b"checkpoint")
            temporal_checkpoint.write_bytes(b"checkpoint")
            report_path.write_text('{"samples": []}\n', encoding="utf-8")
            next_job_path.write_text("{}\n", encoding="utf-8")
            # 产物均完整且 smoke 通过，只让 quality_score 低于阈值，隔离
            # “分数不足”这一条 gate 原因。
            result = SimpleNamespace(
                spatial=SimpleNamespace(
                    steps=1,
                    last_loss=1.0,
                    checkpoint_path=spatial_checkpoint,
                    as_dict=lambda: {},
                    cuda_max_reserved_gib=0.1,
                ),
                temporal=SimpleNamespace(
                    steps=1,
                    final_loss=1.0,
                    checkpoint_path=temporal_checkpoint,
                    as_dict=lambda: {},
                    cuda_max_reserved_gib=0.2,
                ),
                evaluation=SimpleNamespace(
                    quality_score=0.634,
                    pass_threshold=0.8,
                    passed=False,
                    hit_count=0,
                    miss_count=0,
                    unresolved_count=1,
                    gallery_status="saved",
                    gallery_saved_frame_count=1,
                    report_path=report_path,
                    next_job_path=next_job_path,
                    as_dict=lambda: {
                        "quality_score": 0.634,
                        "pass_threshold": 0.8,
                        "passed": False,
                    },
                ),
                candidate_cache=SimpleNamespace(frames=1, as_dict=lambda: {}),
                decision=SimpleNamespace(as_dict=lambda: {}),
            )

            with patch("traning.core.training_ramp.torch.load", return_value={}):
                with self.assertRaisesRegex(RampGateError, "below pass threshold"):
                    _gate_level(
                        level=level,
                        result=result,
                        elapsed=1.0,
                        artifact_path=root / "artifact.json",
                        artifact_issues=(),
                        artifact_smoke={"finite": True},
                        dry_run={"returncode": 0},
                    )

    def test_gate_reports_unresolved_evaluation_when_score_is_above_threshold(
        self,
    ) -> None:
        level = RampLevelSpec("a", "level_a", 1, 1, 1, 1, 1, 1, 1)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spatial_checkpoint = root / "spatial.pt"
            temporal_checkpoint = root / "temporal.pt"
            report_path = root / "report.json"
            next_job_path = root / "next_job.json"
            spatial_checkpoint.write_bytes(b"checkpoint")
            temporal_checkpoint.write_bytes(b"checkpoint")
            report_path.write_text('{"samples": []}\n', encoding="utf-8")
            next_job_path.write_text("{}\n", encoding="utf-8")
            # 分数高于阈值但 passed=False 且 unresolved 很高，确保 gate 不会
            # 只看聚合分数而错误晋级。
            result = SimpleNamespace(
                spatial=SimpleNamespace(
                    steps=1,
                    last_loss=1.0,
                    checkpoint_path=spatial_checkpoint,
                    as_dict=lambda: {},
                    cuda_max_reserved_gib=0.1,
                ),
                temporal=SimpleNamespace(
                    steps=1,
                    final_loss=1.0,
                    checkpoint_path=temporal_checkpoint,
                    as_dict=lambda: {},
                    cuda_max_reserved_gib=0.2,
                ),
                evaluation=SimpleNamespace(
                    quality_score=0.824,
                    pass_threshold=0.8,
                    passed=False,
                    hit_count=0,
                    miss_count=0,
                    unresolved_count=88,
                    gallery_status="saved",
                    gallery_saved_frame_count=1,
                    report_path=report_path,
                    next_job_path=next_job_path,
                    as_dict=lambda: {
                        "quality_score": 0.824,
                        "pass_threshold": 0.8,
                        "passed": False,
                        "hit_count": 0,
                        "miss_count": 0,
                        "unresolved_count": 88,
                    },
                ),
                candidate_cache=SimpleNamespace(frames=1, as_dict=lambda: {}),
                decision=SimpleNamespace(as_dict=lambda: {}),
            )

            with patch("traning.core.training_ramp.torch.load", return_value={}):
                with self.assertRaisesRegex(
                    RampGateError,
                    "evaluation report did not pass.*unresolved=88",
                ):
                    _gate_level(
                        level=level,
                        result=result,
                        elapsed=1.0,
                        artifact_path=root / "artifact.json",
                        artifact_issues=(),
                        artifact_smoke={"finite": True},
                        dry_run={"returncode": 0},
                    )

    def test_gate_rejects_background_only_evaluation(self) -> None:
        level = RampLevelSpec("a", "level_a", 1, 1, 1, 1, 1, 1, 1)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spatial_checkpoint = root / "spatial.pt"
            temporal_checkpoint = root / "temporal.pt"
            report_path = root / "report.json"
            next_job_path = root / "next_job.json"
            spatial_checkpoint.write_bytes(b"checkpoint")
            temporal_checkpoint.write_bytes(b"checkpoint")
            report_path.write_text('{"samples": []}\n', encoding="utf-8")
            next_job_path.write_text("{}\n", encoding="utf-8")
            result = SimpleNamespace(
                spatial=SimpleNamespace(
                    steps=1,
                    last_loss=1.0,
                    checkpoint_path=spatial_checkpoint,
                    as_dict=lambda: {},
                    cuda_max_reserved_gib=0.1,
                ),
                temporal=SimpleNamespace(
                    steps=1,
                    final_loss=1.0,
                    checkpoint_path=temporal_checkpoint,
                    as_dict=lambda: {},
                    cuda_max_reserved_gib=0.2,
                ),
                evaluation=SimpleNamespace(
                    quality_score=1.0,
                    pass_threshold=0.8,
                    passed=False,
                    target_count=0,
                    hit_count=0,
                    miss_count=0,
                    unresolved_count=0,
                    gallery_status="saved",
                    gallery_saved_frame_count=1,
                    report_path=report_path,
                    next_job_path=next_job_path,
                    as_dict=lambda: {},
                ),
                candidate_cache=SimpleNamespace(frames=1, as_dict=lambda: {}),
                decision=SimpleNamespace(as_dict=lambda: {}),
            )

            with patch("traning.core.training_ramp.torch.load", return_value={}):
                with self.assertRaisesRegex(
                    RampGateError,
                    "no target frames were evaluated",
                ):
                    _gate_level(
                        level=level,
                        result=result,
                        elapsed=1.0,
                        artifact_path=root / "artifact.json",
                        artifact_issues=(),
                        artifact_smoke={"finite": True},
                        dry_run={"returncode": 0},
                    )

    def test_level_training_uses_configured_gallery_output_root(self) -> None:
        level = RampLevelSpec("a", "level_a", 1, 1, 1, 1, 1, 1, 1)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            gallery_root = root / "traning_example"
            config_path = root / "config.yaml"
            config_path.write_text("visualization: {}\n", encoding="utf-8")
            reporter = DashboardReporter(
                run_id="ramp-gallery-root",
                output_dir=root / "dashboard",
            )
            checkpoint = root / "checkpoint.pt"
            checkpoint.write_bytes(b"checkpoint")
            score_report = root / "score.json"
            score_report.write_text('{"samples": []}\n', encoding="utf-8")
            next_job = root / "next_job.json"
            next_job.write_text("{}\n", encoding="utf-8")
            result = SimpleNamespace(
                spatial=SimpleNamespace(
                    steps=1,
                    last_loss=1.0,
                    checkpoint_path=checkpoint,
                    cuda_max_reserved_gib=0.1,
                    as_dict=lambda: {},
                ),
                temporal=SimpleNamespace(
                    steps=1,
                    final_loss=1.0,
                    checkpoint_path=checkpoint,
                    cuda_max_reserved_gib=0.1,
                    as_dict=lambda: {},
                ),
                evaluation=SimpleNamespace(
                    quality_score=1.0,
                    pass_threshold=0.8,
                    passed=True,
                    gallery_status="saved",
                    gallery_saved_frame_count=1,
                    gallery_output_dir=gallery_root / "output_000001__batch__trial",
                    report_path=score_report,
                    next_job_path=next_job,
                    gallery_request_path=root / "gallery_request.json",
                    as_dict=lambda: {
                        "quality_score": 1.0,
                        "pass_threshold": 0.8,
                        "passed": True,
                        "gallery_output_dir": str(gallery_root),
                        "gallery_status": "saved",
                        "report_path": str(score_report),
                        "asha_action": None,
                        "asha_reasons": (),
                    },
                ),
                candidate_cache=SimpleNamespace(
                    frames=1,
                    manifest_path=root / "candidate_manifest.json",
                    as_dict=lambda: {},
                ),
                decision=SimpleNamespace(as_dict=lambda: {}),
                summary_path=root / "summary.json",
                as_summary=lambda: {"quality_score": 1.0},
            )
            (root / "candidate_manifest.json").write_text("{}\n", encoding="utf-8")
            (root / "summary.json").write_text("{}\n", encoding="utf-8")

            with (
                patch(
                    "traning.core.training_ramp._write_level_config",
                    return_value=config_path,
                ),
                patch(
                    "traning.core.training_ramp.load_settings", return_value=object()
                ),
                patch(
                    "traning.core.training_ramp.run_full_training_pipeline",
                    return_value=result,
                ) as pipeline_mock,
                patch(
                    "traning.core.training_ramp.export_model_artifact",
                    return_value=SimpleNamespace(manifest_path=root / "artifact.json"),
                ),
                patch(
                    "traning.core.training_ramp.validate_model_artifact",
                    return_value=(),
                ),
                patch(
                    "traning.core.training_ramp.smoke_test_model_artifact",
                    return_value={"finite": True},
                ),
                patch(
                    "traning.core.training_ramp._run_job_dry_run",
                    return_value={"returncode": 0},
                ),
                patch("traning.core.training_ramp.torch.load", return_value={}),
            ):
                _run_level(
                    level=level,
                    base_config=config_path,
                    level_dir=root / "level",
                    device="cpu",
                    reporter=reporter,
                    resume_policy="none",
                    resume_stage_checkpoints={},
                    gallery_output_root=gallery_root,
                    gallery_samples_per_group=1,
                )

        full_config = pipeline_mock.call_args.kwargs["config"]
        self.assertEqual(full_config.gallery_output_root, gallery_root)

    def test_trial_runtime_consumes_optimizer_parameters_and_resume_budget(
        self,
    ) -> None:
        level = RampLevelSpec("a", "level_a", 100, 100, 2, 500, 32, 16, 2)
        settings = SimpleNamespace(
            candidate_cache=SimpleNamespace(
                score_threshold=0.05,
                max_candidates_per_frame=32,
                nms_radius_px=24.0,
                slider_threshold=0.35,
                max_slider_paths=16,
            )
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "temporal.pt"
            torch.save(
                {
                    "training_position": {
                        "global_step": 100,
                        "temporal_step": 100,
                        "last_committed_step": 100,
                    }
                },
                checkpoint,
            )

            # job 中的参数已经由 optimizer 解析成绝对值；checkpoint 提供
            # 已消费步数，runner 只负责绝对值消费和累计 max_steps。
            runtime = _trial_runtime_overrides(
                settings=settings,
                level=level,
                trial_index=1,
                budget_steps=100,
                trial_job={
                    "parameters": {
                        "training": {
                            "spatial_learning_rate": 0.0002,
                            "temporal_learning_rate": 0.0003,
                            "patch_limit": 3,
                            "cache_max_frames": 700,
                        },
                        "inference": {
                            "score_threshold": 0.02,
                            "max_candidates": 36,
                        },
                    }
                },
                parent_checkpoint_path=checkpoint,
            )

        self.assertAlmostEqual(runtime["score_threshold"], 0.02)
        self.assertEqual(runtime["max_candidates"], 36)
        self.assertEqual(runtime["parent_temporal_step"], 100)
        self.assertEqual(runtime["temporal_max_steps"], 100)
        self.assertEqual(runtime["patch_limit"], 3)
        self.assertEqual(runtime["cache_max_frames"], 700)
        self.assertAlmostEqual(runtime["spatial_learning_rate"], 0.0002)
        self.assertAlmostEqual(runtime["temporal_learning_rate"], 0.0003)

    def test_trial_runtime_clamps_legacy_negative_absolute_threshold(self) -> None:
        level = RampLevelSpec("a", "level_a", 1, 1, 1, 1, 8, 8, 1)
        settings = SimpleNamespace(
            candidate_cache=SimpleNamespace(
                score_threshold=0.05,
                max_candidates_per_frame=32,
                nms_radius_px=24.0,
                slider_threshold=0.35,
                max_slider_paths=16,
            )
        )

        runtime = _trial_runtime_overrides(
            settings=settings,
            level=level,
            trial_index=1,
            budget_steps=1,
            trial_job={
                "parameters": {
                    "training": {
                        "patch_limit": 0,
                        "cache_max_frames": 0,
                    },
                    "inference": {
                        "score_threshold": -0.01,
                        "max_candidates": 4,
                    },
                }
            },
            parent_checkpoint_path=None,
        )

        self.assertEqual(runtime["score_threshold"], 0.0)
        self.assertEqual(runtime["max_candidates"], 4)
        self.assertIsNone(runtime["patch_limit"])
        self.assertIsNone(runtime["cache_max_frames"])

    def test_unbounded_level_consumes_jobs_until_strict_pass(self) -> None:
        level = RampLevelSpec("a", "level_a", 1, 1, 1, 1, 8, 8, 1)
        passing_record = self._passing_level_record("trial-3")
        jobs = (
            {"trial_id": "trial-2", "curriculum_stage": "basic", "rung": 0},
            {"trial_id": "trial-3", "curriculum_stage": "basic", "rung": 0},
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "config.yaml"
            config_path.write_text("optimization: {}\n", encoding="utf-8")
            reporter = DashboardReporter(
                run_id="unbounded-ramp",
                output_dir=root / "dashboard",
            )
            with (
                patch(
                    "traning.core.training_ramp._write_level_config",
                    return_value=config_path,
                ),
                patch(
                    "traning.core.training_ramp.load_settings",
                    return_value=SimpleNamespace(
                        optimization=SimpleNamespace(
                            max_trials=None,
                            execute_generated_jobs=True,
                        )
                    ),
                ),
                patch(
                    "traning.core.training_ramp._run_level_trial",
                    side_effect=(
                        RampEvaluationGateError("trial-1 failed"),
                        RampEvaluationGateError("trial-2 failed"),
                        passing_record,
                    ),
                ) as trial_mock,
                patch(
                    "traning.core.training_ramp._load_next_job",
                    side_effect=jobs,
                ),
            ):
                record = _run_level(
                    level=level,
                    base_config=config_path,
                    level_dir=root / "level",
                    device="cpu",
                    reporter=reporter,
                    resume_policy="none",
                    resume_stage_checkpoints={},
                    gallery_output_root=None,
                    gallery_samples_per_group=1,
                )

            state = json.loads(
                (root / "level" / "search_state.json").read_text(encoding="utf-8")
            )

        self.assertEqual(trial_mock.call_count, 3)
        self.assertEqual(record["evaluation"]["parameter_group_id"], "trial-3")
        self.assertEqual(state["status"], "passed")
        self.assertEqual(state["attempted_trials"], 3)

    def test_finite_trial_budget_preserves_pending_job(self) -> None:
        level = RampLevelSpec("a", "level_a", 1, 1, 1, 1, 8, 8, 1)
        pending_job = {
            "trial_id": "trial-3",
            "curriculum_stage": "basic",
            "rung": 0,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "config.yaml"
            config_path.write_text("optimization: {}\n", encoding="utf-8")
            reporter = DashboardReporter(
                run_id="finite-ramp",
                output_dir=root / "dashboard",
            )
            with (
                patch(
                    "traning.core.training_ramp._write_level_config",
                    return_value=config_path,
                ),
                patch(
                    "traning.core.training_ramp.load_settings",
                    return_value=SimpleNamespace(
                        optimization=SimpleNamespace(
                            max_trials=2,
                            execute_generated_jobs=True,
                        )
                    ),
                ),
                patch(
                    "traning.core.training_ramp._run_level_trial",
                    side_effect=(
                        RampEvaluationGateError("trial-1 failed"),
                        RampEvaluationGateError("trial-2 failed"),
                    ),
                ) as trial_mock,
                patch(
                    "traning.core.training_ramp._load_next_job",
                    side_effect=(
                        {"trial_id": "trial-2"},
                        pending_job,
                    ),
                ),
            ):
                with self.assertRaisesRegex(RampSearchExhausted, "pending next job"):
                    _run_level(
                        level=level,
                        base_config=config_path,
                        level_dir=root / "level",
                        device="cpu",
                        reporter=reporter,
                        resume_policy="none",
                        resume_stage_checkpoints={},
                        gallery_output_root=None,
                        gallery_samples_per_group=1,
                    )

            state = json.loads(
                (root / "level" / "search_state.json").read_text(encoding="utf-8")
            )

        self.assertEqual(trial_mock.call_count, 2)
        self.assertEqual(state["status"], "search_exhausted")
        self.assertEqual(state["pending_trial_id"], "trial-3")

    def test_level_resumes_persisted_pending_job_for_same_run(self) -> None:
        level = RampLevelSpec("b", "level_b", 300, 300, 4, 1500, 64, 16, 4)
        passing_record = self._passing_level_record("trial-2")
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            level_dir = root / "level"
            pending_path = (
                level_dir / "training" / "evaluation" / "next_training_job.json"
            )
            pending_path.parent.mkdir(parents=True)
            pending_path.write_text(
                json.dumps(
                    {
                        "trial_id": "trial-2",
                        "curriculum_stage": "basic",
                        "rung": 0,
                        "budget_steps": 300,
                        "parameters": {
                            "training": {"cache_max_frames": 1500},
                        },
                    }
                ),
                encoding="utf-8",
            )
            (level_dir / "search_state.json").write_text(
                json.dumps(
                    {
                        "status": "pending",
                        "attempted_trials": 1,
                        "pending_next_job": str(pending_path),
                        "last_error": "trial-1 failed",
                    }
                ),
                encoding="utf-8",
            )
            config_path = root / "config.yaml"
            config_path.write_text("optimization: {}\n", encoding="utf-8")
            reporter = DashboardReporter(
                run_id="resume-pending-ramp",
                output_dir=root / "dashboard",
            )
            with (
                patch(
                    "traning.core.training_ramp._write_level_config",
                    return_value=config_path,
                ),
                patch(
                    "traning.core.training_ramp.load_settings",
                    return_value=SimpleNamespace(
                        optimization=SimpleNamespace(
                            max_trials=None,
                            execute_generated_jobs=True,
                        )
                    ),
                ),
                patch(
                    "traning.core.training_ramp._run_level_trial",
                    return_value=passing_record,
                ) as trial_mock,
            ):
                record = _run_level(
                    level=level,
                    base_config=config_path,
                    level_dir=level_dir,
                    device="cpu",
                    reporter=reporter,
                    resume_policy="none",
                    resume_stage_checkpoints={},
                    gallery_output_root=None,
                    gallery_samples_per_group=4,
                )

        self.assertEqual(record["evaluation"]["parameter_group_id"], "trial-2")
        self.assertEqual(trial_mock.call_args.kwargs["trial_index"], 1)
        self.assertEqual(
            trial_mock.call_args.kwargs["trial_job"]["trial_id"], "trial-2"
        )

    def test_disabled_generated_job_execution_keeps_first_pending_job(self) -> None:
        level = RampLevelSpec("a", "level_a", 1, 1, 1, 1, 8, 8, 1)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "config.yaml"
            config_path.write_text("optimization: {}\n", encoding="utf-8")
            reporter = DashboardReporter(
                run_id="disabled-ramp",
                output_dir=root / "dashboard",
            )
            with (
                patch(
                    "traning.core.training_ramp._write_level_config",
                    return_value=config_path,
                ),
                patch(
                    "traning.core.training_ramp.load_settings",
                    return_value=SimpleNamespace(
                        optimization=SimpleNamespace(
                            max_trials=None,
                            execute_generated_jobs=False,
                        )
                    ),
                ),
                patch(
                    "traning.core.training_ramp._run_level_trial",
                    side_effect=RampEvaluationGateError("trial-1 failed"),
                ) as trial_mock,
                patch(
                    "traning.core.training_ramp._load_next_job",
                    return_value={"trial_id": "trial-2"},
                ),
            ):
                with self.assertRaisesRegex(
                    RampSearchExhausted,
                    "execution is disabled",
                ):
                    _run_level(
                        level=level,
                        base_config=config_path,
                        level_dir=root / "level",
                        device="cpu",
                        reporter=reporter,
                        resume_policy="none",
                        resume_stage_checkpoints={},
                        gallery_output_root=None,
                        gallery_samples_per_group=1,
                    )

            state = json.loads(
                (root / "level" / "search_state.json").read_text(encoding="utf-8")
            )

        self.assertEqual(trial_mock.call_count, 1)
        self.assertEqual(state["status"], "execution_disabled")
        self.assertEqual(state["pending_trial_id"], "trial-2")

    def test_run_job_validates_and_consumes_resolved_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint = root / "temporal.pt"
            checkpoint.write_bytes(b"checkpoint")
            job_path = root / "job.json"
            job_path.write_text(
                json.dumps(
                    {
                        "trial_id": "trial-job",
                        "curriculum_stage": "multi_object",
                        "rung": 2,
                        "budget_steps": 7,
                        "parent_checkpoint_path": str(checkpoint),
                        "parameters": {
                            "training": {
                                "spatial_learning_rate": 0.0002,
                                "temporal_learning_rate": 0.0003,
                                "patch_limit": 3,
                                "cache_max_frames": 123,
                                "sequence_length": 24,
                                "candidate_slots": 12,
                            },
                            "inference": {
                                "score_threshold": -0.01,
                                "max_candidates": 40,
                                "nms_radius_px": 30.0,
                                "slider_threshold": 0.4,
                                "max_slider_paths": 8,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )

            dry_run = run_training_job_spec(job=job_path, execute=False)
            with patch(
                "traning.main.run_training",
                return_value=SimpleNamespace(as_summary=lambda: {"status": "ok"}),
            ) as run_mock:
                run_training_job_spec(job=job_path, execute=True)

        self.assertEqual(dry_run["parameters"]["inference"]["score_threshold"], 0.0)
        kwargs = run_mock.call_args.kwargs
        self.assertEqual(kwargs["spatial_max_steps"], 7)
        self.assertEqual(kwargs["temporal_max_steps"], 7)
        self.assertEqual(kwargs["candidate_slots"], 12)
        self.assertEqual(kwargs["cache_max_frames"], 123)
        self.assertEqual(kwargs["score_threshold"], 0.0)
        self.assertEqual(kwargs["optimization_stage"].value, "multi_object")
        self.assertEqual(kwargs["optimization_rung"], 2)
        self.assertEqual(kwargs["direct_stage_checkpoints"], {"temporal": checkpoint})

    def test_run_job_dry_run_rejects_missing_parent_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            job_path = Path(temp_dir) / "job.json"
            job_path.write_text(
                json.dumps(
                    {
                        "trial_id": "missing-parent",
                        "curriculum_stage": "basic",
                        "rung": 0,
                        "budget_steps": 1,
                        "parent_checkpoint_path": str(Path(temp_dir) / "missing.pt"),
                        "parameters": {},
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(CliParameterError, "checkpoint is unavailable"):
                run_training_job_spec(job=job_path, execute=False)

    def test_user_interrupt_persists_manifest_and_readiness(self) -> None:
        level = RampLevelSpec("a", "level_a", 1, 1, 1, 1, 1, 1, 1)
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            reporter = DashboardReporter(
                run_id="interrupt-ramp",
                output_dir=output_dir / "dashboard",
            )
            manifest = {"run_id": "interrupt-ramp", "levels": [], "status": "running"}

            _record_ramp_interrupted(
                manifest=manifest,
                output_dir=output_dir,
                target=RampTarget(spatial_steps=1, temporal_steps=1),
                levels=[level],
                auto_launch_full=False,
                reporter=reporter,
                active_index=1,
            )

            persisted = json.loads(
                (output_dir / "manifest.json").read_text(encoding="utf-8")
            )
            readiness = json.loads(
                (output_dir / "final_readiness.json").read_text(encoding="utf-8")
            )

        self.assertEqual(persisted["status"], "interrupted")
        self.assertIn("interrupted_at_utc", persisted)
        self.assertEqual(readiness["status"], "interrupted")
        self.assertEqual(reporter.snapshot().stop_state.reason, "USER_INTERRUPTED")

    @staticmethod
    def _passing_level_record(trial_id: str) -> dict[str, object]:
        return {
            "steps_per_second": 1.0,
            "frames_per_second": 1.0,
            "peak_vram_gib": 0.0,
            "slider_score": None,
            "slider_sample_count": 0,
            "evaluation": {
                "parameter_group_id": trial_id,
                "quality_score": 1.0,
                "pass_threshold": 0.8,
                "passed": True,
                "gallery_status": "saved",
                "report_path": "score.json",
            },
            "artifact_manifest": "artifact.json",
            "artifact_smoke": {"finite": True},
            "dry_run": {"returncode": 0},
        }


if __name__ == "__main__":
    unittest.main()
