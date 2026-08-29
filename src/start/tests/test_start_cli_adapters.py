"""验证总 CLI 只适配参数，并且不再暴露 legacy/V2 双入口。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from typer.testing import CliRunner

from start import main as start_main


def test_run_command_forwards_single_v2_config_and_device_override() -> None:
    runner = CliRunner()
    fake = SimpleNamespace(ok=True, stages=(), run_id="run", report_path=Path("report"))
    with (
        patch("start.main.run_training_flow", return_value=fake) as business,
        patch("start.main._render_result"),
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
                "--run-id",
                "test-run",
            ],
        )

    assert result.exit_code == 0, result.output
    assert business.call_args.kwargs["config"] == Path("configs/test.yaml")
    assert business.call_args.kwargs["device"] is start_main.DeviceOption.CPU
    assert business.call_args.kwargs["dry_run"] is True
    assert business.call_args.kwargs["run_id"] == "test-run"


def test_no_args_runs_same_total_flow() -> None:
    runner = CliRunner()
    fake = SimpleNamespace(ok=True, stages=(), run_id="run", report_path=Path("report"))
    with (
        patch("start.main.run_training_flow", return_value=fake) as business,
        patch("start.main._render_result"),
    ):
        result = runner.invoke(start_main.app, [])

    assert result.exit_code == 0, result.output
    business.assert_called_once_with()


def test_help_has_no_legacy_evaluator_or_v2_namespace() -> None:
    runner = CliRunner()
    result = runner.invoke(start_main.app, ["--help"])

    assert result.exit_code == 0, result.output
    assert "--evaluator" not in result.output
    assert " v2 " not in result.output
    assert "run" in result.output
