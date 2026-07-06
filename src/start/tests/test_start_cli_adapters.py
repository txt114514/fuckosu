from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch, sentinel

import torch
from typer.testing import CliRunner

from start import main as start_main


class StartCliAdapterTests(unittest.TestCase):
    def test_business_start_flow_builds_startup_config(self) -> None:
        with patch(
            "start.main.run_startup_flow",
            return_value=sentinel.result,
        ) as flow:
            result = start_main.run_training_startup_flow(
                training_config=Path("configs/test.yaml"),
                device="cpu",
                dry_run=True,
                test_level="none",
                patch_limit=0,
                cache_max_frames=0,
            )

        self.assertIs(result, sentinel.result)
        config = flow.call_args.args[0]
        self.assertEqual(config.training_config, Path("configs/test.yaml"))
        self.assertEqual(config.device, torch.device("cpu"))
        self.assertTrue(config.dry_run)
        self.assertEqual(config.test_level, "none")
        self.assertIsNone(config.patch_limit)
        self.assertIsNone(config.cache_max_frames)

    def test_run_cli_alias_passes_arguments_to_ui_full_flow_function(self) -> None:
        runner = CliRunner()
        fake_result = SimpleNamespace(status="passed")
        with (
            patch(
                "start.main.run_training_ui_flow",
                return_value=fake_result,
            ) as business,
            patch("start.main._render_full_flow_table"),
        ):
            result = runner.invoke(
                start_main.app,
                [
                    "run",
                    "--config",
                    "configs/test.yaml",
                    "--device",
                    "cpu",
                    "--dry-run",
                    "--test-level",
                    "none",
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        kwargs = business.call_args.kwargs
        self.assertEqual(kwargs["training_config"], Path("configs/test.yaml"))
        self.assertEqual(kwargs["device"], "cpu")
        self.assertTrue(kwargs["dry_run"])
        self.assertEqual(kwargs["test_level"], "none")
        self.assertTrue(kwargs["auto_launch_full"])
        self.assertEqual(kwargs["gallery_output_root"], Path("traning_example"))

    def test_no_args_defaults_to_ui_full_flow(self) -> None:
        runner = CliRunner()
        fake_result = SimpleNamespace(status="passed")
        with (
            patch(
                "start.main.run_training_ui_flow",
                return_value=fake_result,
            ) as business,
            patch("start.main._render_full_flow_table"),
        ):
            result = runner.invoke(start_main.app, [])

        self.assertEqual(result.exit_code, 0, result.output)
        kwargs = business.call_args.kwargs
        self.assertEqual(
            kwargs["training_config"],
            start_main.DEFAULT_UI_TRAINING_CONFIG,
        )
        self.assertEqual(kwargs["gallery_output_root"], Path("traning_example"))

    def test_run_help_is_available(self) -> None:
        runner = CliRunner()
        result = runner.invoke(start_main.app, ["run", "--help"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Usage:", result.output)
        self.assertIn("--device", result.output)


if __name__ == "__main__":
    unittest.main()
