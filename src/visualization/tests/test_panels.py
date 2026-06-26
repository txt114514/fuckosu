from __future__ import annotations

import unittest

from visualization.core.panels import (
    render_best_parameters_panel,
    render_current_learning_panel,
    render_events_panel,
    render_overall_progress_panel,
    render_pipeline_panel,
    render_resources_panel,
)
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


if __name__ == "__main__":
    unittest.main()
