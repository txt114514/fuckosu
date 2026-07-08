from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets  # noqa: E402

from visualization.core.gui import MainWindow, VisualizationApplication
from visualization.lib import (
    DashboardReporter,
    PipelinePhase,
    PipelineStageState,
    ResourceState,
    TrainingDashboardState,
    TrainingEvent,
)


def _app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class QtGuiDashboardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = _app()

    def test_page_switches_from_startup_to_training_widget(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            reporter = DashboardReporter(run_id="qt-switch", output_dir=Path(temporary))
            window = MainWindow()
            window.apply_state(reporter.snapshot())
            self.assertIs(window.currentWidget(), window.startup_page.widget)

            reporter.update_metrics(
                pipeline_phase=PipelinePhase.TRAINING.value,
                phase="正式训练",
                current_trial_id="trial_live",
            )
            window.apply_state(reporter.snapshot())

        self.assertIs(window.currentWidget(), window.training_page.widget)

    def test_realtime_trial_state_updates_visible_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            reporter = DashboardReporter(run_id="qt-live", output_dir=Path(temporary))
            window = MainWindow()
            reporter.update_metrics(
                pipeline_phase=PipelinePhase.TRAINING.value,
                current_trial_id="trial_live",
                phase="temporal",
                global_step=35,
                target_global_steps=100,
                loss=10.655584,
                current_parameters={"training": {"lr": 0.001}},
            )
            window.apply_state(reporter.snapshot())
            self.assertIn("35/100", window.training_page.trial_labels["step"].text())
            self.assertEqual(
                window.training_page.trial_labels["loss"].text(), "10.655584"
            )

            reporter.update_metrics(global_step=36, loss=9.9, score=0.72)
            window.apply_state(reporter.snapshot())

        self.assertIn("36/100", window.training_page.trial_labels["step"].text())
        self.assertEqual(window.training_page.trial_labels["loss"].text(), "9.900000")
        self.assertEqual(window.training_page.trial_labels["score"].text(), "0.720000")

    def test_parameter_tree_keeps_large_parameter_set(self) -> None:
        window = MainWindow()
        with tempfile.TemporaryDirectory() as temporary:
            reporter = DashboardReporter(run_id="qt-params", output_dir=Path(temporary))
            reporter.update_metrics(
                pipeline_phase=PipelinePhase.TRAINING.value,
                current_trial_id="trial_params",
                current_parameters={
                    "candidate_cache": {f"p_{index:03d}": index for index in range(100)}
                },
            )
            window.apply_state(reporter.snapshot())

        root = window.training_page.parameters_model.item(0, 0)
        self.assertEqual(root.text(), "candidate_cache")
        self.assertEqual(root.rowCount(), 100)
        self.assertEqual(root.child(99, 0).text(), "p_099")

    def test_long_event_log_is_bounded_and_keeps_latest_event(self) -> None:
        state = TrainingDashboardState(
            run_id="qt-events",
            pipeline_phase=PipelinePhase.TRAINING.value,
            recent_events=[
                TrainingEvent.create(
                    event_type="test",
                    severity="info",
                    message_key=f"event_{index:04d}",
                )
                for index in range(1100)
            ],
        )
        window = MainWindow()
        window.apply_state(state)

        lines = window.training_page.events.toPlainText().splitlines()
        self.assertLessEqual(len(lines), 1000)
        self.assertIn("event_1099", lines[-1])

    def test_state_flow_from_reporter_store_to_gui(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            reporter = DashboardReporter(run_id="qt-flow", output_dir=output)
            reporter.update_pipeline_stage(
                PipelineStageState(
                    stage_id="evaluation",
                    name="固定评估评分",
                    status="warning",
                    score=0.634,
                    threshold=0.8,
                    warning_count=183,
                )
            )
            reporter.update_metrics(
                pipeline_phase=PipelinePhase.TRAINING.value,
                current_trial_id="trial_score",
                score=0.634,
                consecutive_passes=1,
                required_passes=3,
            )
            reporter.report_resource(
                ResourceState(
                    gpu_name="Test GPU",
                    gpu_utilization=42.0,
                    gpu_memory_used_gb=3.0,
                    gpu_total_gb=8.0,
                    cpu_percent=12.5,
                )
            )
            window = MainWindow()
            window.apply_state(reporter.snapshot())
            self.assertTrue((output / "dashboard_state.json").exists())

        self.assertEqual(window.training_page.trial_labels["score"].text(), "0.634000")
        self.assertEqual(
            window.training_page.trial_labels["passes"].text(), "1/3 (33.3%)"
        )
        self.assertEqual(window.training_page.resource_labels["gpu"].text(), "Test GPU")
        self.assertEqual(window.training_page.tests.item(0, 0).text(), "固定评估评分")

    def test_application_close_stops_timer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            reporter = DashboardReporter(run_id="qt-life", output_dir=Path(temporary))
            application = VisualizationApplication(reporter, refresh_ms=50)
            self.assertTrue(application.timer.isActive())
            application.stop()

        self.assertFalse(application.timer.isActive())


if __name__ == "__main__":
    unittest.main()
