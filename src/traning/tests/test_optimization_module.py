from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from traning.core.optimization import (
    AGGREGATE_SCORE_VERSION,
    OptimizationExecutorConfig,
    ParameterSearchConfig,
    SampleScoringInput,
    TrialHistoryEntry,
    analyze_trial_attribution,
    build_batch_gallery_request,
    build_hard_example_sampling_plan,
    evaluate_curriculum_gate,
    execute_optimization_plan,
    plan_next_trial,
    score_sample,
    score_trial,
)
from traning.core.optimization.parameter_search import DEFAULT_CURRICULUM_RULES
from traning.lib.metrics import PredictedClick, TargetObject
from traning.state import CurriculumStage


def _circle_target(target_id: str = "circle-1") -> TargetObject:
    return TargetObject(
        target_id=target_id,
        target_type="circle",
        start_ms=1000.0,
        end_ms=1000.0,
        x=100.0,
        y=100.0,
    )


class OptimizationModuleTests(unittest.TestCase):
    def test_score_trial_aggregates_point_slider_sequence_rules(self) -> None:
        sample = SampleScoringInput(
            sample_key="item_000001/segment_000001",
            subproject="single_point",
            targets=(_circle_target(),),
            predictions=(PredictedClick(time_ms=1000.0, x=100.0, y=100.0),),
            circle_radius=10.0,
        )

        report = score_trial("trial-perfect", (sample,))

        self.assertEqual(report.score_version, AGGREGATE_SCORE_VERSION)
        self.assertEqual(report.hit_count, 1)
        self.assertEqual(report.miss_count, 0)
        self.assertEqual(report.unresolved_count, 0)
        self.assertTrue(report.passed)
        self.assertAlmostEqual(report.quality_score, 1.0)

    def test_attribution_groups_temporal_and_decision_errors(self) -> None:
        sample = SampleScoringInput(
            sample_key="item_000001/segment_early",
            subproject="single_point",
            targets=(_circle_target(),),
            predictions=(PredictedClick(time_ms=800.0, x=100.0, y=100.0),),
            circle_radius=10.0,
            frame_index=12,
        )
        report = score_trial("trial-early", (sample,))

        attribution = analyze_trial_attribution(report)

        self.assertEqual(attribution.domain_counts["temporal"], 1)
        self.assertEqual(attribution.domain_counts["decision"], 1)
        self.assertEqual(attribution.tag_counts["early_click"], 1)
        self.assertEqual(attribution.tag_counts["unresolved_target"], 1)
        self.assertTrue(attribution.hard_examples)

    def test_parameter_plan_uses_attribution_and_asha_thresholds(self) -> None:
        sample = SampleScoringInput(
            sample_key="item_000001/segment_late",
            subproject="single_point",
            targets=(_circle_target(),),
            predictions=(PredictedClick(time_ms=1225.0, x=100.0, y=100.0),),
            circle_radius=10.0,
        )
        report = score_trial(
            "trial-low",
            (sample,),
            metrics={"peak_vram_mb": 9000.0},
        )
        attribution = analyze_trial_attribution(report)
        history = (
            TrialHistoryEntry("a", 0, CurriculumStage.BASIC, 0.30),
            TrialHistoryEntry("b", 0, CurriculumStage.BASIC, 0.45),
            TrialHistoryEntry("c", 0, CurriculumStage.BASIC, 0.60),
        )

        plan = plan_next_trial(
            report,
            attribution,
            history=history,
            config=ParameterSearchConfig(target_peak_vram_mb=8000.0),
        )

        self.assertEqual(plan.asha_action, "prune")
        self.assertIn("temporal", plan.priority_domains)
        self.assertEqual(plan.parameter_updates["search"]["sampler"], "tpe")
        self.assertEqual(
            plan.parameter_updates["training"]["temporal_loss_weight_multiplier"],
            1.20,
        )
        self.assertEqual(plan.parameter_updates["training"]["patch_limit_delta"], -1)
        self.assertTrue(plan.hard_example_keys)

    def test_gallery_request_is_built_from_trial_score_report(self) -> None:
        sample = SampleScoringInput(
            sample_key="item_000001/segment_gallery",
            subproject="single_point",
            targets=(_circle_target(),),
            predictions=(PredictedClick(time_ms=1000.0, x=100.0, y=100.0),),
            circle_radius=10.0,
            frame_index=7,
        )
        report = score_trial("trial-gallery", (sample,))

        request = build_batch_gallery_request(report, batch_id="batch-gallery")

        self.assertEqual(request.batch_id, "batch-gallery")
        self.assertEqual(request.best_trial.trial_id, "trial-gallery")
        self.assertEqual(request.best_trial.frames[0].sample_key, sample.sample_key)
        self.assertTrue(request.best_trial.frames[0].passed)

    def test_curriculum_gate_and_hard_example_sampling(self) -> None:
        passing_samples = tuple(
            score_sample(
                SampleScoringInput(
                    sample_key=f"single/{index}",
                    subproject="single_point",
                    targets=(_circle_target(f"circle-{index}"),),
                    predictions=(
                        PredictedClick(time_ms=1000.0, x=100.0, y=100.0),
                    ),
                    circle_radius=10.0,
                )
            )
            for index in range(15)
        )
        gate = evaluate_curriculum_gate(
            passing_samples,
            rules={"single_point": DEFAULT_CURRICULUM_RULES["single_point"]},
        )

        self.assertTrue(gate.passed)
        self.assertEqual(
            gate.subprojects["single_point"].longest_pass_streak,
            15,
        )

        failing = SampleScoringInput(
            sample_key="single/hard",
            subproject="single_point",
            targets=(_circle_target("hard"),),
            predictions=(PredictedClick(time_ms=800.0, x=100.0, y=100.0),),
            circle_radius=10.0,
        )
        report = score_trial("trial-hard", (failing,))
        attribution = analyze_trial_attribution(report)
        sampling = build_hard_example_sampling_plan(attribution)

        self.assertGreater(sampling.sample_weights["single/hard"], 1.0)

    def test_execute_optimization_plan_records_trial_and_job(self) -> None:
        sample = SampleScoringInput(
            sample_key="item_000001/segment_executor",
            subproject="single_point",
            targets=(_circle_target(),),
            predictions=(PredictedClick(time_ms=800.0, x=100.0, y=100.0),),
            circle_radius=10.0,
        )
        report = score_trial("trial-executor", (sample,))
        attribution = analyze_trial_attribution(report)
        plan = plan_next_trial(report, attribution)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "parent.pt"
            checkpoint.write_bytes(b"checkpoint")
            execution = execute_optimization_plan(
                report,
                attribution,
                plan,
                parent_checkpoint_path=checkpoint,
                config=OptimizationExecutorConfig(
                    output_dir=Path(tmpdir),
                    base_budget_steps=5,
                ),
            )
            records_path = Path(tmpdir) / "trials.jsonl"

            self.assertTrue(records_path.exists())
            self.assertEqual(execution.source_trial_id, "trial-executor")
            self.assertEqual(execution.job.parent_checkpoint_path, checkpoint)
            self.assertGreaterEqual(execution.job.budget_steps, 5)


if __name__ == "__main__":
    unittest.main()
