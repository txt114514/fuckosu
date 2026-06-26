from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch, sentinel

import torch
from typer.testing import CliRunner

from traning.core.decision import FullTrainingRunConfig
from traning import main as training_main


class TrainingCliAdapterTests(unittest.TestCase):
    def test_business_run_training_calls_pipeline_without_typer(self) -> None:
        with (
            patch("traning.main.load_settings", return_value=sentinel.settings),
            patch("traning.main._run_dir", return_value=Path("runs/full")),
            patch(
                "traning.main.run_full_training_pipeline",
                return_value=sentinel.result,
            ) as pipeline,
        ):
            result = training_main.run_training(
                config=Path("configs/test.yaml"),
                device="cpu",
                patch_limit=0,
                cache_max_frames=0,
                sequence_length=4,
                candidate_slots=8,
            )

        self.assertIs(result, sentinel.result)
        pipeline.assert_called_once()
        self.assertIs(pipeline.call_args.args[0], sentinel.settings)
        run_config = pipeline.call_args.kwargs["config"]
        self.assertIsInstance(run_config, FullTrainingRunConfig)
        self.assertEqual(run_config.device, torch.device("cpu"))
        self.assertIsNone(run_config.patch_limit)
        self.assertIsNone(run_config.cache_max_frames)
        self.assertEqual(run_config.sequence_length, 4)
        self.assertEqual(run_config.candidate_slots, 8)

    def test_run_cli_passes_arguments_to_business_function(self) -> None:
        runner = CliRunner()
        fake_result = SimpleNamespace(
            as_summary=lambda: {"ok": True},
            evaluation=sentinel.evaluation,
        )
        with (
            patch(
                "traning.main.run_training",
                return_value=fake_result,
            ) as business,
            patch("traning.main._render_dict_table"),
            patch("traning.main._render_parameter_group_score"),
        ):
            result = runner.invoke(
                training_main.app,
                [
                    "run",
                    "--config",
                    "configs/test.yaml",
                    "--device",
                    "cpu",
                    "--patch-limit",
                    "0",
                    "--cache-max-frames",
                    "0",
                    "--sequence-length",
                    "4",
                    "--candidate-slots",
                    "8",
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        kwargs = business.call_args.kwargs
        self.assertEqual(kwargs["config"], Path("configs/test.yaml"))
        self.assertEqual(kwargs["device"], "cpu")
        self.assertEqual(kwargs["patch_limit"], 0)
        self.assertEqual(kwargs["cache_max_frames"], 0)
        self.assertEqual(kwargs["sequence_length"], 4)
        self.assertEqual(kwargs["candidate_slots"], 8)

    def test_cli_parameter_error_maps_to_typer_exit(self) -> None:
        runner = CliRunner()
        with patch(
            "traning.main.run_training",
            side_effect=training_main.CliParameterError("device must be cpu"),
        ):
            result = runner.invoke(training_main.app, ["run", "--device", "bad"])

        self.assertEqual(result.exit_code, 2, result.output)
        self.assertIn("device must be cpu", result.output)


if __name__ == "__main__":
    unittest.main()
