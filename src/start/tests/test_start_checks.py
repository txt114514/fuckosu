from __future__ import annotations

import unittest
from pathlib import Path

import torch

from start.checks import run_startup_checks, run_training_startup_checks
from start.modules import source_module_entry, source_module_entries
from traning.conf import load_settings


class StartEntryTests(unittest.TestCase):
    def test_src_module_entries_are_importable(self) -> None:
        keys = {entry.key for entry in source_module_entries(include_start=True)}

        self.assertTrue({"start", "package", "before_traning", "traning"} <= keys)
        self.assertTrue(source_module_entry("traning").importable)

    def test_global_startup_checks_pass_without_cuda_requirement(self) -> None:
        report = run_startup_checks(require_cuda=False)

        self.assertTrue(report.ok)
        self.assertIn("environment", {result.key for result in report.results})


class TrainingStartupCheckTests(unittest.TestCase):
    def test_training_checks_include_data_input_report(self) -> None:
        settings = load_settings(Path("configs/model_small_vram.yaml"))
        training_report = run_training_startup_checks(
            settings,
            split="train",
            device=torch.device("cpu"),
            require_cuda=False,
        )

        self.assertTrue(training_report.ok)
        self.assertGreater(training_report.data_input.segment_count, 0)
        self.assertIn(
            "training:data_input",
            {result.key for result in training_report.report.results},
        )


if __name__ == "__main__":
    unittest.main()
