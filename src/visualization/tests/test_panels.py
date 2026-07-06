from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from rich.console import Console

from visualization.core.multi_terminal import launch_panel_terminals
from visualization.core.panels import (
    render_best_parameters_panel,
    render_current_learning_panel,
    render_events_panel,
    render_overall_progress_panel,
    render_pipeline_panel,
    render_resources_panel,
)
from visualization.core.view_router import render_dashboard_view
from visualization.lib import PipelineStageState, TrainingDashboardState


class VisualizationPanelTests(unittest.TestCase):
    def test_all_required_panels_render_from_state_only(self) -> None:
        state = TrainingDashboardState(run_id="panel-test")
        state.pipeline_stages["readiness"] = PipelineStageState(
            stage_id="readiness",
            name="训练 readiness",
            status="passed",
        )
        for render in (
            render_pipeline_panel,
            render_best_parameters_panel,
            render_current_learning_panel,
            render_overall_progress_panel,
            render_resources_panel,
            render_events_panel,
        ):
            panel = render(state)
            self.assertIsNotNone(panel)

    def test_compact_training_pages_render(self) -> None:
        state = TrainingDashboardState(
            run_id="compact-test",
            pipeline_phase="training",
            current_trial_id="trial-1",
            current_parameters={
                "training": {"spatial_max_steps": 100},
                "device": "cuda",
            },
        )
        state.pipeline_stages["temporal"] = PipelineStageState(
            stage_id="temporal",
            name="时序训练",
            status="running",
            processed=8,
            total=100,
        )
        for page in ("overview", "parameters", "tests", "scores", "resources", "events"):
            with self.subTest(page=page):
                view = render_dashboard_view(state, compact=True, page=page)
                self.assertIsNotNone(view)

    def test_current_trial_panel_translates_runtime_status(self) -> None:
        state = TrainingDashboardState(
            run_id="status-translation",
            pipeline_phase="training",
            current_trial_id="trial-zh",
            trial_status="evaluating",
        )
        state.current_parameter_status = {"trial_status": "evaluating"}
        console = Console(record=True, width=100)
        console.print(render_dashboard_view(state, compact=True, page="overview"))
        output = console.export_text()

        self.assertIn("正式训练", output)
        self.assertIn("评估中", output)
        self.assertNotIn("evaluating", output)

    def test_compact_startup_pages_render(self) -> None:
        state = TrainingDashboardState(run_id="compact-startup")
        state.pipeline_stages["startup"] = PipelineStageState(
            stage_id="startup",
            name="启动检测",
            status="running",
        )
        for page in ("overview", "resources", "events"):
            with self.subTest(page=page):
                view = render_dashboard_view(state, compact=True, page=page)
                self.assertIsNotNone(view)

    def test_multi_terminal_launcher_degrades_without_tmux(self) -> None:
        with patch("visualization.core.multi_terminal.shutil.which", return_value=None):
            result = launch_panel_terminals(
                run_id="no-tmux",
                dashboard_dir=Path("/tmp/no-tmux-dashboard"),
                cwd=Path.cwd(),
            )

        self.assertEqual(result.status, "unavailable")


if __name__ == "__main__":
    unittest.main()
