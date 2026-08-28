"""V2 独立 CLI 的配置与失败语义验收。"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from typer.testing import CliRunner

from traning.app.cli import app
from traning.config import TelemetryConfig, V2Config, v2_config_to_dict
from traning.telemetry import StateStore
from traning.training import ParameterVector, TrialAcceptance, TrialObservation


class _CliPassesOnThirdTrial:
    """验证 CLI 确实保持进程并执行第三个 proposal。"""

    def evaluate(
        self,
        parameters: ParameterVector,
        trial_index: int,
    ) -> TrialObservation:
        """前两轮返回普通 gate 失败，第三轮返回完整通过。"""

        passed = trial_index == 2
        return TrialObservation(
            trial_index,
            parameters,
            float(trial_index),
            TrialAcceptance(*(passed for _ in range(7))),
        )


def test_config_check_loads_the_formal_v2_config() -> None:
    """工程默认 V2 配置必须能从真实 CLI 边界严格加载。"""

    result = CliRunner().invoke(
        app,
        ("config-check", "--config", "configs/traning.yaml"),
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == 1
    assert payload["cache"]["schema_version"] == 2
    assert payload["optimization"]["max_trials"] is None
    assert payload["coordinates"]["source_width"] == 1484
    assert payload["coordinates"]["calibration_evidence_path"].endswith(
        "traning_coordinate_evidence.json"
    )


def test_coordinate_audit_reports_validation_only_provenance() -> None:
    """默认审计通过控制点，但必须公开原拟合集不可重放。"""

    result = CliRunner().invoke(
        app,
        ("coordinate-audit", "--config", "configs/traning.yaml"),
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["fit_reproducible"] is False
    assert payload["control_count"] == 5
    assert payload["max_error_px"] == pytest.approx(1.3936472394, abs=1e-9)
    assert payload["limitation"]


def test_coordinate_audit_can_require_missing_refit_provenance() -> None:
    """调用方要求完整拟合复现时，validation-only 证据必须非零退出。"""

    result = CliRunner().invoke(
        app,
        (
            "coordinate-audit",
            "--config",
            "configs/traning.yaml",
            "--require-refit-provenance",
        ),
    )

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["fit_reproducible"] is False


def test_config_check_rejects_missing_file() -> None:
    """配置不存在时 CLI 必须失败，不能退回隐式默认值。"""

    result = CliRunner().invoke(
        app,
        ("config-check", "--config", "missing-v2-config.yaml"),
    )

    assert result.exit_code == 1
    assert "配置失败" in result.output


def test_config_check_rejects_legacy_candidate_cache_schema(tmp_path) -> None:
    """CLI 不得将无坐标指纹的 cache schema 1 静默迁移到 2。"""

    config_path = tmp_path / "legacy-cache.json"
    config_path.write_text(
        json.dumps({"schema_version": 1, "cache": {"schema_version": 1}}),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        ("config-check", "--config", str(config_path)),
    )

    assert result.exit_code == 1
    assert "cache.schema_version 仅支持 2" in result.output


def test_env_check_can_report_without_hiding_failure(monkeypatch) -> None:
    """非 strict 模式仍输出 CUDA 真实状态，只是不改变进程退出码。"""

    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    result = CliRunner().invoke(
        app,
        ("env-check", "--config", "configs/traning.yaml", "--no-strict"),
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert any(item["status"] == "failed" for item in payload["results"])


def test_repository_start_entry_exposes_namespaced_v2_cli() -> None:
    """用户从原 start/main.py 出发也只能显式进入独立 V2 边界。"""

    workspace = Path(__file__).resolve().parents[4]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = "src:."
    result = subprocess.run(
        (
            sys.executable,
            "src/start/main.py",
            "v2",
            "config-check",
            "--config",
            "configs/traning.yaml",
        ),
        cwd=workspace,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == 1
    assert payload["cache"]["schema_version"] == 2
    assert payload["coordinates"]["transform_identity"] == "legacy-control-validated-v1"


def test_train_cli_continues_until_real_evaluator_passes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """V2 CLI 不得在前两轮 gate 失败后静默停止。"""

    config_path = tmp_path / "v2.json"
    telemetry_root = tmp_path / "telemetry"
    config_path.write_text(
        json.dumps(
            v2_config_to_dict(
                V2Config(telemetry=TelemetryConfig(directory=telemetry_root))
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "traning.app.cli.load_trial_evaluator",
        lambda _factory_spec, _config: _CliPassesOnThirdTrial(),
    )

    result = CliRunner().invoke(
        app,
        (
            "train",
            "--config",
            str(config_path),
            "--evaluator",
            "test_module:factory",
            "--run-id",
            "cli-run",
            "--no-check-environment",
        ),
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["status"] == "passed"
    assert payload["trial_index"] == 2
    event_types = tuple(
        event.event_type
        for event in StateStore(telemetry_root / "cli-run").history().events
    )
    assert event_types == (
        "search.trial.completed",
        "search.trial.completed",
        "search.trial.completed",
        "search.passed",
    )
