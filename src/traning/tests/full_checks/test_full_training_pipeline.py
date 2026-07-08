from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import ANY, patch

import torch

from start.checks import StartupCheckReport, TrainingStartupCheckReport
from traning.conf import Settings
from traning.core.dataset_import import DataInputReport
from traning.core.decision import (
    CandidateCacheBuildResult,
    FullTrainingRunConfig,
    TemporalDecisionRunResult,
    run_full_training_pipeline,
)
from traning.core.spatial import SpatialTrainingResult
from traning.core.temporal import TemporalTrainingResult
from traning.lib.visualization import GalleryResult
from visualization.lib import NullReporter, PipelinePhase, PipelineStageState


class FullTrainingPipelineTests(unittest.TestCase):
    def test_pipeline_runs_all_training_steps_and_writes_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            reporter = _RecordingReporter()
            data_report = DataInputReport(
                split="train",
                segment_count=1,
                frame_count_estimate=3,
                item_counts={"item": 1},
                category_counts={"single_point": 1},
                dimension_counts={"atomic": 1},
                issue_count=0,
                issues=(),
            )
            spatial_result = SpatialTrainingResult(
                run_dir=run_dir / "spatial",
                device="cpu",
                steps=1,
                samples_seen=1,
                last_loss=1.25,
                last_patch_count=1,
                amp_dtype="float32",
                channels_last=False,
                ram_budget_gib=1.0,
                ram_reserved_for_system_gib=0.0,
                vram_budget_gib=None,
                vram_reserved_for_system_gib=None,
                cuda_max_allocated_gib=None,
                cuda_max_reserved_gib=None,
            )
            cache_result = CandidateCacheBuildResult(
                output_dir=run_dir / "candidate_cache",
                manifest_path=run_dir / "candidate_cache" / "manifest.json",
                records_path=run_dir / "candidate_cache" / "frames.jsonl",
                device="cpu",
                split="train",
                frames=1,
                candidates=2,
                slider_paths=0,
                ambiguous_candidates=0,
                ambiguous_slider_paths=0,
            )
            temporal_result = TemporalTrainingResult(
                run_dir=run_dir / "temporal",
                checkpoint_path=run_dir / "temporal" / "temporal_model.pt",
                device="cpu",
                steps=1,
                windows=1,
                sequence_length=2,
                candidate_slots=2,
                input_size=8,
                final_loss=0.75,
                action_loss=0.5,
                candidate_loss=0.1,
                xy_loss=0.1,
                time_loss=0.05,
                target_strategy="beatmap_action_v1",
                cuda_max_allocated_gib=None,
                cuda_max_reserved_gib=None,
            )
            decision_result = TemporalDecisionRunResult(
                output_dir=run_dir / "decision",
                manifest_path=run_dir / "decision" / "manifest.json",
                decisions_path=run_dir / "decision" / "decisions.jsonl",
                checkpoint_path=temporal_result.checkpoint_path,
                device="cpu",
                frames=1,
                sequence_length=2,
                candidate_slots=2,
            )
            cache_result.output_dir.mkdir(parents=True)
            cache_result.records_path.write_text(
                json.dumps(
                    {
                        "sample_key": "item_0001/single_point_0001",
                        "frame_index": 0,
                        "timestamp_ms": 0.0,
                        "frame_width": 640,
                        "frame_height": 480,
                        "temporal_target": {
                            "target_strategy": "beatmap_action_v1",
                            "action": "no_op",
                            "action_id": 0,
                            "selected_candidate_id": None,
                            "time_offset_ms": 0.0,
                        },
                        "candidates": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            decision_result.output_dir.mkdir(parents=True)
            decision_result.decisions_path.write_text(
                json.dumps(
                    {
                        "sample_key": "item_0001/single_point_0001",
                        "frame_index": 0,
                        "timestamp_ms": 0.0,
                        "action": "no_op",
                        "action_id": 0,
                        "action_probability": 1.0,
                        "selected_candidate_id": None,
                        "time_offset_ms": 0.0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            startup_report = TrainingStartupCheckReport(
                report=StartupCheckReport(scope="test", results=()),
                data_input=data_report,
            )

            with (
                patch(
                    "traning.core.decision.pipeline.run_training_startup_checks",
                    return_value=startup_report,
                ) as startup_mock,
                patch(
                    "traning.core.decision.pipeline.run_spatial_training",
                    return_value=spatial_result,
                ) as spatial_mock,
                patch(
                    "traning.core.decision.pipeline.generate_candidate_cache",
                    return_value=cache_result,
                ) as cache_mock,
                patch(
                    "traning.core.decision.pipeline.run_temporal_training",
                    return_value=temporal_result,
                ) as temporal_mock,
                patch(
                    "traning.core.decision.pipeline.run_temporal_decision",
                    return_value=decision_result,
                ) as decision_mock,
            ):
                result = run_full_training_pipeline(
                    Settings(),
                    config=FullTrainingRunConfig(
                        run_dir=run_dir,
                        device=torch.device("cpu"),
                        sequence_length=2,
                        candidate_slots=2,
                        render_gallery=False,
                        reporter=reporter,
                        resume_policy="strict",
                        resume_stage_checkpoints={
                            "spatial": run_dir / "resume_spatial.pt",
                            "temporal": run_dir / "resume_temporal.pt",
                        },
                    ),
                )

            self.assertEqual(result.as_summary()["decision_frames"], 1)
            self.assertEqual(result.as_summary()["parameter_group_id"], "pg-0001")
            self.assertEqual(result.as_summary()["gallery_status"], "skipped")
            self.assertTrue(result.summary_path.is_file())
            summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
            self.assertTrue(summary["startup_checks"]["report"]["ok"])
            self.assertEqual(summary["candidate_cache"]["frames"], 1)
            self.assertEqual(summary["evaluation"]["quality_score"], 1.0)
            self.assertEqual(
                summary["temporal"]["checkpoint_path"],
                str(temporal_result.checkpoint_path),
            )
            parameter_updates = [
                update["current_parameters"]
                for update in reporter.metric_updates
                if "current_parameters" in update
            ]
            phase_updates = [
                update["pipeline_phase"]
                for update in reporter.metric_updates
                if "pipeline_phase" in update
            ]
            trial_status_updates = [
                update["trial_status"]
                for update in reporter.metric_updates
                if "trial_status" in update
            ]
            self.assertTrue(parameter_updates)
            self.assertIn(PipelinePhase.TRAINING.value, phase_updates)
            self.assertIn("evaluating", trial_status_updates)
            planned_parameters = parameter_updates[0]
            self.assertEqual(planned_parameters["parameter_group_id"], "pg-0001")
            self.assertIsNone(planned_parameters["evaluation"]["quality_score"])
            current_parameters = parameter_updates[-1]
            self.assertEqual(current_parameters["parameter_group_id"], "pg-0001")
            self.assertEqual(current_parameters["training"]["spatial_max_steps"], 1)
            self.assertEqual(current_parameters["temporal"]["sequence_length"], 2)
            self.assertEqual(current_parameters["evaluation"]["quality_score"], 1.0)
            self.assertEqual(current_parameters["evaluation"]["pass_threshold"], 0.8)
            self.assertEqual(current_parameters["outputs"]["decision_frames"], 1)
            evaluation_stages = [
                stage
                for stage in reporter.stage_updates
                if stage.stage_id == "evaluation"
            ]
            self.assertEqual(evaluation_stages[-1].score, 1.0)
            self.assertEqual(evaluation_stages[-1].threshold, 0.8)
            startup_mock.assert_called_once()
            spatial_mock.assert_called_once()
            cache_mock.assert_called_once_with(
                ANY,
                output_dir=run_dir / "candidate_cache",
                device=torch.device("cpu"),
                spatial_checkpoint_path=spatial_result.checkpoint_path,
                split="train",
                max_frames=1,
                patch_limit=1,
                max_candidates=None,
                score_threshold=None,
                nms_radius_px=None,
                slider_threshold=None,
                max_slider_paths=None,
            )
            temporal_mock.assert_called_once()
            self.assertEqual(
                spatial_mock.call_args.kwargs["resume_checkpoint_path"],
                run_dir / "resume_spatial.pt",
            )
            self.assertEqual(spatial_mock.call_args.kwargs["resume_policy"], "strict")
            self.assertEqual(
                temporal_mock.call_args.kwargs["resume_checkpoint_path"],
                run_dir / "resume_temporal.pt",
            )
            self.assertEqual(temporal_mock.call_args.kwargs["resume_policy"], "strict")
            decision_mock.assert_called_once()

    def test_gallery_export_failure_emits_fail_event(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            reporter = _RecordingReporter()
            data_report = DataInputReport(
                split="train",
                segment_count=1,
                frame_count_estimate=1,
                item_counts={"item": 1},
                category_counts={"single_point": 1},
                dimension_counts={"atomic": 1},
                issue_count=0,
                issues=(),
            )
            spatial_result = SpatialTrainingResult(
                run_dir=run_dir / "spatial",
                device="cpu",
                steps=1,
                samples_seen=1,
                last_loss=1.0,
                last_patch_count=1,
                amp_dtype="float32",
                channels_last=False,
                ram_budget_gib=1.0,
                ram_reserved_for_system_gib=0.0,
                vram_budget_gib=None,
                vram_reserved_for_system_gib=None,
                cuda_max_allocated_gib=None,
                cuda_max_reserved_gib=None,
            )
            cache_result = CandidateCacheBuildResult(
                output_dir=run_dir / "candidate_cache",
                manifest_path=run_dir / "candidate_cache" / "manifest.json",
                records_path=run_dir / "candidate_cache" / "frames.jsonl",
                device="cpu",
                split="train",
                frames=1,
                candidates=0,
                slider_paths=0,
                ambiguous_candidates=0,
                ambiguous_slider_paths=0,
            )
            temporal_result = TemporalTrainingResult(
                run_dir=run_dir / "temporal",
                checkpoint_path=run_dir / "temporal" / "temporal_model.pt",
                device="cpu",
                steps=1,
                windows=1,
                sequence_length=2,
                candidate_slots=2,
                input_size=8,
                final_loss=0.75,
                action_loss=0.5,
                candidate_loss=0.1,
                xy_loss=0.1,
                time_loss=0.05,
                target_strategy="beatmap_action_v1",
                cuda_max_allocated_gib=None,
                cuda_max_reserved_gib=None,
            )
            decision_result = TemporalDecisionRunResult(
                output_dir=run_dir / "decision",
                manifest_path=run_dir / "decision" / "manifest.json",
                decisions_path=run_dir / "decision" / "decisions.jsonl",
                checkpoint_path=temporal_result.checkpoint_path,
                device="cpu",
                frames=1,
                sequence_length=2,
                candidate_slots=2,
            )
            cache_result.output_dir.mkdir(parents=True)
            cache_result.records_path.write_text(
                json.dumps(
                    {
                        "sample_key": "item_0001/single_point_0001",
                        "frame_index": 0,
                        "timestamp_ms": 0.0,
                        "frame_width": 640,
                        "frame_height": 480,
                        "temporal_target": {"action": "no_op"},
                        "candidates": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            decision_result.output_dir.mkdir(parents=True)
            decision_result.decisions_path.write_text(
                json.dumps(
                    {
                        "sample_key": "item_0001/single_point_0001",
                        "frame_index": 0,
                        "timestamp_ms": 0.0,
                        "action": "no_op",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            startup_report = TrainingStartupCheckReport(
                report=StartupCheckReport(scope="test", results=()),
                data_input=data_report,
            )

            with (
                patch("traning.core.decision.pipeline.run_training_startup_checks", return_value=startup_report),
                patch("traning.core.decision.pipeline.run_spatial_training", return_value=spatial_result),
                patch("traning.core.decision.pipeline.generate_candidate_cache", return_value=cache_result),
                patch("traning.core.decision.pipeline.run_temporal_training", return_value=temporal_result),
                patch("traning.core.decision.pipeline.run_temporal_decision", return_value=decision_result),
                patch(
                    "traning.core.decision.pipeline.save_annotation_gallery",
                    return_value=GalleryResult(
                        status="failed",
                        selected_trial_id="pg-0001",
                        warning="render exploded",
                    ),
                ),
            ):
                result = run_full_training_pipeline(
                    Settings(),
                    config=FullTrainingRunConfig(
                        run_dir=run_dir,
                        device=torch.device("cpu"),
                        sequence_length=2,
                        candidate_slots=2,
                        render_gallery=True,
                        reporter=reporter,
                    ),
                )

        self.assertEqual(result.evaluation.gallery_status, "failed")
        self.assertTrue(
            any("[GALLERY][FAIL] render exploded" == event.raw_message for event in reporter.events),
            [event.raw_message for event in reporter.events],
        )


class _RecordingReporter(NullReporter):
    def __init__(self) -> None:
        self.metric_updates: list[dict[str, object]] = []
        self.stage_updates: list[PipelineStageState] = []
        self.events = []

    def update_metrics(self, **metrics: object) -> None:
        self.metric_updates.append(metrics)

    def update_pipeline_stage(self, stage: PipelineStageState) -> None:
        self.stage_updates.append(stage)

    def emit_event(self, event) -> None:
        self.events.append(event)


if __name__ == "__main__":
    unittest.main()
