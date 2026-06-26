from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

from typer.testing import CliRunner

from before_traning.conf import Settings
from before_traning import main as before_main


class BeforeTrainingCliAdapterTests(unittest.TestCase):
    def test_business_workflow_calls_pipeline_without_typer(self) -> None:
        settings = Settings()
        with (
            patch("before_traning.main.load_settings", return_value=settings),
            patch(
                "before_traning.main.run_training_pipeline",
                return_value={"clip": True},
            ) as pipeline,
        ):
            result = before_main.run_data_workflow(
                config=Path("before.yaml"),
                overwrite=True,
                skip_clip=True,
                global_offset_ms=12.5,
            )

        self.assertEqual(result, {"clip": True})
        pipeline.assert_called_once()
        called_settings = pipeline.call_args.args[0]
        self.assertTrue(called_settings.runtime.overwrite)
        self.assertEqual(called_settings.video_clip.global_offset_ms, 12.5)
        self.assertFalse(pipeline.call_args.kwargs["run_clip_stage"])

    def test_run_cli_passes_arguments_to_business_function(self) -> None:
        runner = CliRunner()
        with patch(
            "before_traning.main.run_data_workflow",
            return_value={"stage": True},
        ) as business:
            result = runner.invoke(
                before_main.app,
                [
                    "run",
                    "--config",
                    "before.yaml",
                    "--overwrite",
                    "--skip-clip",
                    "--global-offset-ms",
                    "12.5",
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        kwargs = business.call_args.kwargs
        self.assertEqual(kwargs["config"], Path("before.yaml"))
        self.assertTrue(kwargs["overwrite"])
        self.assertTrue(kwargs["skip_clip"])
        self.assertEqual(kwargs["global_offset_ms"], 12.5)

    def test_run_cli_uses_pipeline_result_as_exit_code(self) -> None:
        runner = CliRunner()
        with patch(
            "before_traning.main.run_data_workflow",
            return_value={"stage": False},
        ):
            result = runner.invoke(before_main.app, ["run"])

        self.assertEqual(result.exit_code, 1, result.output)


if __name__ == "__main__":
    unittest.main()
