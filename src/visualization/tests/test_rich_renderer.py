from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from rich.console import Console

from visualization.conf import DashboardSettings
from visualization.core.renderers.rich_renderer import RichDashboardRenderer
from visualization.core.view_router import render_dashboard_page
from visualization.lib import DashboardReporter, PipelineStageState, TrainingDashboardState


class RichRendererPaginationTests(unittest.TestCase):
    def test_arrow_keys_change_pages_without_wrapping(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            reporter = DashboardReporter(
                run_id="keyboard",
                output_dir=Path(temporary),
            )
            renderer = RichDashboardRenderer(
                reporter,
                settings=DashboardSettings(),
            )
            renderer._page_count = 3

            self.assertTrue(renderer._handle_key("\x1b[B"))
            self.assertEqual(renderer._page_index, 1)
            self.assertTrue(renderer._handle_key("\x1b[B"))
            self.assertEqual(renderer._page_index, 2)
            self.assertFalse(renderer._handle_key("\x1b[B"))
            self.assertEqual(renderer._page_index, 2)
            self.assertTrue(renderer._handle_key("\x1b[A"))
            self.assertEqual(renderer._page_index, 1)

    def test_dashboard_page_splits_complete_panels(self) -> None:
        state = TrainingDashboardState(
            run_id="paged",
            pipeline_phase="training",
            current_trial_id="trial-1",
        )
        for index in range(12):
            state.pipeline_stages[f"stage_{index}"] = PipelineStageState(
                stage_id=f"stage_{index}",
                name=f"阶段 {index}",
                status="running",
                processed=index,
                total=12,
            )

        renderable, page_count = render_dashboard_page(
            state,
            page_index=0,
            terminal_height=12,
            terminal_width=80,
        )
        console = Console(record=True, width=80)
        console.print(renderable)
        output = console.export_text()

        self.assertGreater(page_count, 1)
        self.assertIn("页面：1/", output)
        self.assertIn("下方向键下一页", output)


if __name__ == "__main__":
    unittest.main()
