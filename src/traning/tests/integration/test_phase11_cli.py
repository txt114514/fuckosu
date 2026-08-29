"""V2 独立 CLI 的配置与失败语义验收。"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from traning.app.cli import app
from traning.training import ParameterVector


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


def test_repository_start_entry_exposes_model_diagnostics_without_namespace() -> None:
    """总入口直接暴露当前模型诊断，不再保留 v2 兼容命名空间。"""

    workspace = Path(__file__).resolve().parents[4]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = "src:."
    result = subprocess.run(
        (
            sys.executable,
            "src/start/main.py",
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


def test_train_cli_uses_production_trainer_without_external_evaluator(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """train 命令只装配内建生产服务，不接受 module:factory。"""

    observation = SimpleNamespace(
        trial_index=2,
        objective=0.9,
        parameters=ParameterVector(0.001, 0.05, 64, 0.1, 0.0, 0.0),
    )
    production_result = SimpleNamespace(
        observation=observation,
        checkpoint_directory=tmp_path / "checkpoint",
        resumed=False,
    )
    trainer = Mock()
    trainer.run.return_value = production_result
    trainer_factory = Mock(return_value=trainer)
    monkeypatch.setattr("traning.app.cli.build_training_datasets", Mock())
    monkeypatch.setattr("traning.app.cli.ProductionTrainer", trainer_factory)

    result = CliRunner().invoke(
        app,
        (
            "train",
            "--config",
            "configs/traning.yaml",
            "--run-id",
            "cli-run",
            "--output-root",
            str(tmp_path),
            "--no-check-environment",
        ),
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["status"] == "passed"
    assert payload["trial_index"] == 2
    assert "--evaluator" not in CliRunner().invoke(app, ("train", "--help")).output
    trainer.run.assert_called_once_with(
        run_dir=tmp_path / "cli-run",
        run_id="cli-run",
        resume=True,
    )
