"""验证评分归因、参数规划、trial store 与多目标排序闭环。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from traning.core.optimization import (
    AGGREGATE_SCORE_VERSION,
    OptimizationExecutorConfig,
    ParameterSearchConfig,
    SQLiteTrialStore,
    SampleScoringInput,
    TrialHistoryEntry,
    analyze_trial_attribution,
    build_batch_gallery_request,
    build_hard_example_sampling_plan,
    evaluate_curriculum_gate,
    execute_optimization_plan,
    plan_next_trial,
    score_trial_objectives,
    score_sample,
    score_trial,
    trial_history_from_records,
    training_job_from_dict,
)
from traning.core.optimization.parameter_search import DEFAULT_CURRICULUM_RULES
from traning.lib.metrics import PredictedClick, TargetObject
from traning.state import CurriculumStage, SearchMethod, TrialParameters


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
    def test_parameter_search_rejects_unimplemented_method(self) -> None:
        self.assertEqual(
            ParameterSearchConfig().search_method,
            SearchMethod.RULE_BASED,
        )
        with self.assertRaisesRegex(ValueError, "only rule_based"):
            ParameterSearchConfig(search_method=SearchMethod.TPE)

    def test_training_job_rejects_unwired_hard_example_weights(self) -> None:
        with self.assertRaisesRegex(ValueError, "hard_example_weights"):
            training_job_from_dict(
                {
                    "trial_id": "trial-weighted",
                    "curriculum_stage": "basic",
                    "rung": 0,
                    "budget_steps": 1,
                    "parameters": {},
                    "hard_example_weights": {"sample-hard": 2.0},
                }
            )

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
        # 三个同 rung 历史分数建立明确的 ASHA 比较分布；当前 trial 同时
        # 超显存且表现较差，应触发 prune 和对应参数收缩。
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
        self.assertTrue(plan.asha_reasons)
        self.assertIn("asha_reasons", plan.as_dict())
        self.assertLess(plan.objective_score, report.quality_score)
        self.assertIn("temporal", plan.priority_domains)
        self.assertNotIn("search", plan.parameter_updates)
        self.assertGreater(
            plan.parameter_updates["training"]["temporal_learning_rate_multiplier"],
            1.15,
        )
        self.assertEqual(plan.parameter_updates["training"]["patch_limit_delta"], -1)
        self.assertTrue(plan.hard_example_keys)

    def test_asha_does_not_promote_until_every_sample_passes(self) -> None:
        passing = tuple(
            SampleScoringInput(
                sample_key=f"single/pass-{index}",
                subproject="single_point",
                targets=(_circle_target(f"pass-{index}"),),
                predictions=(PredictedClick(time_ms=1000.0, x=100.0, y=100.0),),
                circle_radius=10.0,
            )
            for index in range(9)
        )
        failing = SampleScoringInput(
            sample_key="single/unresolved",
            subproject="single_point",
            targets=(_circle_target("unresolved"),),
            predictions=(),
            circle_radius=10.0,
        )
        report = score_trial("trial-mixed", (*passing, failing))

        plan = plan_next_trial(report, analyze_trial_attribution(report))

        self.assertGreater(report.quality_score, 0.70)
        self.assertFalse(report.passed)
        self.assertEqual(plan.asha_action, "continue")
        self.assertEqual(plan.next_stage, CurriculumStage.BASIC)
        self.assertEqual(plan.next_rung, 0)

    def test_promotion_preserves_stage_cap_and_advances_rung(self) -> None:
        sample = SampleScoringInput(
            sample_key="single/perfect",
            subproject="single_point",
            targets=(_circle_target("perfect"),),
            predictions=(PredictedClick(time_ms=1000.0, x=100.0, y=100.0),),
            circle_radius=10.0,
        )
        report = score_trial("trial-perfect-stage", (sample,))

        plan = plan_next_trial(
            report,
            analyze_trial_attribution(report),
            current_stage=CurriculumStage.BASIC,
            rung=2,
            config=ParameterSearchConfig(max_stage=CurriculumStage.BASIC),
        )

        self.assertEqual(plan.asha_action, "promote")
        self.assertEqual(plan.current_stage, CurriculumStage.BASIC)
        self.assertEqual(plan.next_stage, CurriculumStage.BASIC)
        self.assertEqual(plan.current_rung, 2)
        self.assertEqual(plan.next_rung, 3)

    def test_executor_clamps_resolved_absolute_parameters(self) -> None:
        sample = SampleScoringInput(
            sample_key="single/spatial-hard",
            subproject="single_point",
            targets=(_circle_target("spatial-hard"),),
            predictions=(),
            circle_radius=10.0,
            metadata={
                "candidate_match_status": "unmatched",
                "candidate_match_unmatched_reason": "nearest_candidate_outside_radius",
            },
        )
        report = score_trial("trial-clamp", (sample,))
        attribution = analyze_trial_attribution(report)
        plan = plan_next_trial(report, attribution)
        base = TrialParameters(
            training={
                "spatial_learning_rate": 1e-4,
                "temporal_learning_rate": 1e-4,
                "patch_limit": 2,
                "cache_max_frames": 500,
                "sequence_length": 32,
                "candidate_slots": 16,
            },
            inference={
                "score_threshold": 0.02,
                "max_candidates": 36,
                "nms_radius_px": 32.0,
                "slider_threshold": 0.5,
                "max_slider_paths": 16,
            },
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            execution = execute_optimization_plan(
                report,
                attribution,
                plan,
                base_parameters=base,
                config=OptimizationExecutorConfig(output_dir=Path(temp_dir)),
            )

        self.assertEqual(execution.job.parameters.inference["score_threshold"], 0.0)
        self.assertEqual(execution.job.parameters.inference["max_candidates"], 40)
        self.assertEqual(execution.job.parameters.training["cache_max_frames"], 500)
        self.assertEqual(execution.job.curriculum_stage, CurriculumStage.BASIC)
        self.assertLessEqual(len(execution.job.trial_id), 72)

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

    def test_gallery_reuses_spatial_attribution_for_unresolved_target(self) -> None:
        sample = SampleScoringInput(
            sample_key="item_000001/long_sequence_000008",
            subproject="long_sequence",
            targets=(
                TargetObject(
                    target_id="item_000001/long_sequence_000008:105:85",
                    target_type="circle",
                    start_ms=1754.0,
                    end_ms=1754.0,
                    x=40.0,
                    y=262.0,
                    source_index=85,
                ),
            ),
            predictions=(),
            circle_radius=40.9767936,
            frame_index=105,
            metadata={
                "action": "no_op",
                "action_probability": 0.8981025218963623,
                "candidate_count": 11,
                "candidate_match_status": "unmatched",
                "candidate_match_unmatched_reason": (
                    "nearest_candidate_outside_radius"
                ),
                "transform_status": "calibrated",
            },
        )

        report = score_trial("trial-unresolved", (sample,))
        attribution = analyze_trial_attribution(report)
        frame = build_batch_gallery_request(report).best_trial.frames[0]

        self.assertEqual(attribution.primary_domain, "spatial")
        self.assertEqual(frame.primary_error, "spatial")
        self.assertEqual(
            frame.error_tags,
            (
                "unresolved_target",
                "candidate_match_failed",
                "nearest_candidate_outside_radius",
            ),
        )
        self.assertEqual(frame.target_source_index, 85)
        self.assertEqual(frame.action, "no_op")
        self.assertAlmostEqual(frame.action_probability or 0.0, 0.8981025218963623)
        self.assertIsNone(frame.predicted_osu_xy)
        self.assertIn("target-candidate matching failed", frame.failure_reason or "")

    def test_curriculum_gate_and_hard_example_sampling(self) -> None:
        # 连续 15 个通过样本刚好形成可检查的 streak，随后单独加入早点击
        # 失败样本，验证 curriculum gate 与 hard-example 权重互不混淆。
        passing_samples = tuple(
            score_sample(
                SampleScoringInput(
                    sample_key=f"single/{index}",
                    subproject="single_point",
                    targets=(_circle_target(f"circle-{index}"),),
                    predictions=(PredictedClick(time_ms=1000.0, x=100.0, y=100.0),),
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

    def test_sqlite_trial_store_records_execution(self) -> None:
        sample = SampleScoringInput(
            sample_key="item_000001/segment_sqlite",
            subproject="single_point",
            targets=(_circle_target(),),
            predictions=(PredictedClick(time_ms=800.0, x=100.0, y=100.0),),
            circle_radius=10.0,
        )
        report = score_trial(
            "trial-sqlite",
            (sample,),
            metrics={"peak_vram_mb": 2048.0, "latency_ms": 12.0},
        )
        attribution = analyze_trial_attribution(report)
        plan = plan_next_trial(report, attribution)

        with tempfile.TemporaryDirectory() as tmpdir:
            store = SQLiteTrialStore(Path(tmpdir) / "trials.sqlite")
            execute_optimization_plan(
                report,
                attribution,
                plan,
                config=OptimizationExecutorConfig(output_dir=Path(tmpdir)),
                store=store,
            )
            records = store.load()

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["source_trial_id"], "trial-sqlite")
        self.assertIn("objective_score", records[0]["plan"])

    def test_trial_history_uses_source_stage_rung_and_score_version(self) -> None:
        records = (
            {
                "source_trial_id": "source-fallback",
                "score": {
                    "trial_id": "trial-history",
                    "score_version": AGGREGATE_SCORE_VERSION,
                    "quality_score": 0.73,
                    "metrics": {"latency_ms": 12.0},
                },
                "plan": {
                    "current_stage": "multi_object",
                    "current_rung": 2,
                    "next_stage": "complex",
                    "next_rung": 3,
                },
            },
            {
                "score": {
                    "trial_id": "old-score",
                    "score_version": "obsolete-score-v0",
                    "quality_score": 0.99,
                },
                "plan": {"current_stage": "basic", "current_rung": 0},
            },
        )

        history = trial_history_from_records(
            records,
            score_version=AGGREGATE_SCORE_VERSION,
        )

        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].trial_id, "trial-history")
        self.assertEqual(history[0].curriculum_stage, CurriculumStage.MULTI_OBJECT)
        self.assertEqual(history[0].rung, 2)
        self.assertEqual(history[0].metrics["latency_ms"], 12.0)

    def test_multi_objective_score_uses_quality_vram_and_latency(self) -> None:
        sample = SampleScoringInput(
            sample_key="item_000001/segment_objective",
            subproject="single_point",
            targets=(_circle_target(),),
            predictions=(PredictedClick(time_ms=1000.0, x=100.0, y=100.0),),
            circle_radius=10.0,
        )
        report = score_trial(
            "trial-objective",
            (sample,),
            metrics={"peak_vram_mb": 1000.0, "latency_ms": 20.0},
        )

        # 空间/时间为满分但资源指标非零，复合分必须低于纯质量分，防止
        # 多目标排序退化为只比较 accuracy。
        objective = score_trial_objectives(report)

        self.assertEqual(objective.values["quality_score"], 1.0)
        self.assertLess(objective.composite_score, 1.0)


if __name__ == "__main__":
    unittest.main()
