from __future__ import annotations

import re
from pathlib import Path
import tempfile
import unittest

from visualization.core.view_router import dashboard_view_kind
from visualization.lib import DashboardReporter, PipelinePhase, PipelineStageState


class VisualizationRealtimeStateTests(unittest.TestCase):
    def test_trial_switch_replaces_current_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            reporter = DashboardReporter(
                run_id="trial-switch",
                output_dir=Path(temporary),
            )
            reporter.update_metrics(
                pipeline_phase=PipelinePhase.TRAINING.value,
                current_trial_id="trial_a",
                trial_status="training",
                current_parameters={"learning_rate": 0.1, "channels": 32},
            )
            self.assertEqual(reporter.snapshot().current_trial_id, "trial_a")

            reporter.update_metrics(
                current_trial_id="trial_b",
                trial_status="training",
                current_parameters={"learning_rate": 0.01, "temporal_state": 192},
            )
            state = reporter.snapshot()
            status = state.current_parameter_status
            self.assertEqual(status["trial_id"], "trial_b")
            self.assertEqual(status["parameter_values"]["learning_rate"], 0.01)
            self.assertNotIn("channels", status["parameter_values"])

    def test_test_score_updates_through_pending_running_passed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            reporter = DashboardReporter(
                run_id="score-flow",
                output_dir=Path(temporary),
            )
            reporter.update_metrics(
                pipeline_phase=PipelinePhase.TRAINING.value,
                current_trial_id="trial_score",
                current_parameters={"learning_rate": 0.001},
            )
            reporter.update_pipeline_stage(
                PipelineStageState("single_click", "单点检测", status="pending")
            )
            self.assertIn(
                "单点检测",
                reporter.snapshot().current_parameter_status["pending_tests"],
            )

            reporter.update_pipeline_stage(
                PipelineStageState("single_click", "单点检测", status="running")
            )
            self.assertEqual(
                reporter.snapshot().current_parameter_status["test_statuses"][
                    "单点检测"
                ],
                "running",
            )

            reporter.update_pipeline_stage(
                PipelineStageState(
                    "single_click",
                    "单点检测",
                    status="passed",
                    score=0.91,
                    threshold=0.8,
                )
            )
            status = reporter.snapshot().current_parameter_status
            self.assertIn("单点检测", status["passed_tests"])
            self.assertEqual(status["test_scores"]["单点检测"], 0.91)
            self.assertEqual(status["test_thresholds"]["单点检测"], 0.8)

    def test_reporter_notifies_refresh_callback_after_state_updates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            reporter = DashboardReporter(
                run_id="refresh-callback",
                output_dir=Path(temporary),
            )
            refresh_count = 0

            def refresh() -> None:
                nonlocal refresh_count
                refresh_count += 1

            reporter.add_refresh_callback(refresh)
            reporter.update_pipeline_stage(
                PipelineStageState("startup", "启动检查", status="running")
            )
            self.assertEqual(refresh_count, 1)

            reporter.remove_refresh_callback(refresh)
            reporter.update_pipeline_stage(
                PipelineStageState("startup", "启动检查", status="passed")
            )
            self.assertEqual(refresh_count, 1)

    def test_consecutive_passes_and_prune_state_are_real_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            reporter = DashboardReporter(
                run_id="passes-prune",
                output_dir=Path(temporary),
            )
            for passes in (0, 1, 2, 3):
                reporter.update_metrics(
                    pipeline_phase=PipelinePhase.TRAINING.value,
                    current_trial_id="trial_pass",
                    current_parameters={"learning_rate": 0.001},
                    consecutive_passes=passes,
                    required_passes=3,
                    trial_status="promoted" if passes == 3 else "training",
                )
                status = reporter.snapshot().current_parameter_status
                self.assertEqual(status["consecutive_passes"], passes)
                self.assertEqual(status["required_consecutive_passes"], 3)
            self.assertEqual(status["trial_status"], "promoted")

            reporter.update_metrics(
                current_trial_id="trial_pruned",
                current_parameters={"learning_rate": 0.5},
                trial_status="pruned",
                current_grade="pruned",
                prune_reason="quality below ASHA prune floor",
            )
            status = reporter.snapshot().current_parameter_status
            self.assertEqual(status["trial_status"], "pruned")
            self.assertEqual(
                status["prune_reason"],
                "quality below ASHA prune floor",
            )

    def test_pipeline_phase_transition_switches_view(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            reporter = DashboardReporter(
                run_id="phase-switch",
                output_dir=Path(temporary),
            )
            self.assertEqual(dashboard_view_kind(reporter.snapshot()), "startup")
            reporter.update_metrics(
                pipeline_phase=PipelinePhase.DATA_PREPARATION.value,
                phase="数据准备",
            )
            self.assertEqual(dashboard_view_kind(reporter.snapshot()), "startup")
            reporter.update_metrics(
                pipeline_phase=PipelinePhase.PRETRAIN_CHECK.value,
                phase="训练预检",
            )
            self.assertEqual(dashboard_view_kind(reporter.snapshot()), "startup")
            reporter.update_metrics(
                pipeline_phase=PipelinePhase.TRAINING.value,
                phase="正式训练",
                current_trial_id="trial_live",
                current_parameters={"learning_rate": 0.001},
            )
            state = reporter.snapshot()
            self.assertEqual(dashboard_view_kind(state), "training")
            self.assertEqual(state.current_parameter_status["trial_id"], "trial_live")

    def test_no_placeholder_state_in_production_visualization_code(self) -> None:
        pattern = re.compile(
            r"CURRENT_PARAMS|example_trial|get_current_status|DEFAULT_STATUS|"
            r"dummy score|demo current|fake passed|score\s*=\s*80"
        )
        for path in Path("src/visualization").rglob("*.py"):
            if "tests" in path.parts:
                continue
            source = path.read_text(encoding="utf-8")
            self.assertIsNone(pattern.search(source), str(path))


if __name__ == "__main__":
    unittest.main()
