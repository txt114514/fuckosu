from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import yaml

from traning.core.training_ramp import (
    RampLevelSpec,
    RampGateError,
    RampTarget,
    _gate_level,
    _report_level_finished,
    _report_level_started,
    _run_preflight,
    _report_ramp_failed,
    _report_ramp_started,
    build_ramp_levels,
    ensure_full_target_config,
)
from visualization.lib import DashboardReporter, ResourceState


class TrainingRampTests(unittest.TestCase):
    def test_build_ramp_levels_clips_and_reaches_target(self) -> None:
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

    def test_ensure_full_target_config_writes_target_and_absolutizes_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "configs"
            output_dir = root / "out"
            config_dir.mkdir()
            source_config = config_dir / "small.yaml"
            target_config = config_dir / "full.yaml"
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
            with (
                patch(
                    "traning.core.training_ramp.collect_environment_report",
                    return_value=env,
                ),
                patch("traning.core.training_ramp.load_settings", return_value=object()),
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
            self.assertEqual(state.pipeline_stages["gpu_bridge"].processed, 1)
            self.assertEqual(state.resources.gpu_utilization, 23.0)

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


if __name__ == "__main__":
    unittest.main()
