from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from visualization.conf import DashboardSettings
from visualization.core.display_overrides import apply_display_overrides
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
            terminal_height=80,
            terminal_width=80,
        )
        console = Console(record=True, width=80)
        console.print(renderable)
        output = console.export_text()

        self.assertGreater(page_count, 1)
        self.assertIn("页面：1/", output)
        self.assertIn("固定页，不自动切换", output)
        self.assertNotIn("自动轮播", output)

    def test_compact_dashboard_uses_semantic_pages_on_small_terminal(self) -> None:
        state = TrainingDashboardState(
            run_id="compact-paged",
            pipeline_phase="training",
            current_trial_id="trial-1",
        )
        renderable, page_count = render_dashboard_page(
            state,
            page_index=1,
            terminal_height=12,
            terminal_width=80,
        )
        console = Console(record=True, width=80)
        console.print(renderable)
        output = console.export_text()

        self.assertEqual(page_count, 6)
        self.assertIn("页面：2/6", output)
        self.assertIn("参数", output)

    def test_display_overrides_translate_raw_rich_renderables(self) -> None:
        table = Table(expand=True)
        table.add_column("stage")
        table.add_column("status")
        table.add_row("GPU bridge", "running")
        renderable = apply_display_overrides(Panel(table, title="checkpoint"))

        console = Console(record=True, width=100)
        console.print(renderable)
        output = console.export_text()

        self.assertIn("检查点", output)
        self.assertIn("阶段", output)
        self.assertIn("状态", output)
        self.assertIn("GPU 桥接检查", output)
        self.assertIn("正在运行", output)
        self.assertNotIn("stage", output)
        self.assertNotIn("status", output)
        self.assertNotIn("GPU bridge", output)


if __name__ == "__main__":
    unittest.main()
