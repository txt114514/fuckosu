"""OSU V2 的独立配置与环境检查命令行入口。"""

from __future__ import annotations

from dataclasses import asdict
import importlib
import json
from pathlib import Path
import time

import typer

from traning.app.environment import check_v2_environment, require_v2_environment
from traning.app.factory import build_frame_coordinate_transform
from traning.config import V2Config, load_v2_config, v2_config_to_dict
from traning.data import (
    audit_affine_calibration,
    load_affine_calibration_evidence,
)
from traning.telemetry import StateStore, TelemetryReporter
from traning.training import SearchExhaustedError, TrialEvaluator

from .training import run_configured_search


app = typer.Typer(help="OSU Decision Model V2 启动与环境检查。")


def load_checked_config(path: Path) -> V2Config:
    """加载严格 V2 配置，保留原始异常供 CLI 显式报告。"""

    if not isinstance(path, Path):
        raise TypeError("path 必须是 pathlib.Path")
    return load_v2_config(path)


@app.command("config-check")
def config_check(
    config: Path = typer.Option(Path("configs/traning.yaml"), "--config"),
) -> None:
    """验证 schema、坐标标定和搜索预算并输出规范化配置。"""

    try:
        checked = load_checked_config(config)
    except (OSError, TypeError, ValueError) as exc:
        typer.echo(f"V2 配置失败：{exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(
        json.dumps(
            v2_config_to_dict(checked),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


@app.command("env-check")
def env_check(
    config: Path = typer.Option(Path("configs/traning.yaml"), "--config"),
    strict: bool = typer.Option(True, "--strict/--no-strict"),
) -> None:
    """检查实际设备与正式坐标标定；strict 模式以失败码退出。"""

    try:
        checked = load_checked_config(config)
        report = check_v2_environment(checked)
    except (OSError, TypeError, ValueError) as exc:
        typer.echo(f"V2 环境检查失败：{exc}", err=True)
        raise typer.Exit(1) from exc
    payload = {
        "ok": report.ok,
        "results": [
            {
                "name": item.name,
                "status": item.status.value,
                "message": item.message,
            }
            for item in report.results
        ],
    }
    typer.echo(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    if strict and not report.ok:
        raise typer.Exit(1)


@app.command("coordinate-audit")
def coordinate_audit(
    config: Path = typer.Option(Path("configs/traning.yaml"), "--config"),
    require_refit_provenance: bool = typer.Option(
        False,
        "--require-refit-provenance/--allow-validation-only",
    ),
) -> None:
    """复算坐标控制残差，并显式报告原始拟合集是否可重放。"""

    try:
        checked = load_checked_config(config)
        evidence_path = checked.coordinates.calibration_evidence_path
        if evidence_path is None:
            raise ValueError("V2 config 未配置 calibration_evidence_path")
        transform = build_frame_coordinate_transform(checked)
        report = audit_affine_calibration(
            transform,
            load_affine_calibration_evidence(evidence_path),
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        typer.echo(f"V2 坐标审计失败：{exc}", err=True)
        raise typer.Exit(1) from exc
    payload = {
        "ok": report.ok,
        "fit_reproducible": report.fit_reproducible,
        "identity_matches": report.identity_matches,
        "frame_size_matches": report.frame_size_matches,
        "matrix_matches": report.matrix_matches,
        "transform_fingerprint": report.transform_fingerprint,
        "evidence_artifact_sha256": report.evidence_artifact_sha256,
        "control_set_sha256": report.control_set_sha256,
        "control_count": report.control_count,
        "mean_error_px": report.mean_error_px,
        "rmse_error_px": report.rmse_error_px,
        "max_error_px": report.max_error_px,
        "max_allowed_error_px": report.max_allowed_error_px,
        "worst_control_id": report.worst_control_id,
        "limitation": (
            None
            if report.fit_reproducible
            else "原始拟合观测未入库；当前只能复核独立控制点，不能重放拟合"
        ),
    }
    typer.echo(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    if not report.ok or (require_refit_provenance and not report.fit_reproducible):
        raise typer.Exit(1)


def load_trial_evaluator(factory_spec: str, config: V2Config) -> TrialEvaluator:
    """从显式 ``module:factory`` 边界加载用户的真实 typed evaluator。"""

    if not isinstance(factory_spec, str):
        raise TypeError("factory_spec 必须是字符串")
    module_name, separator, attribute_name = factory_spec.partition(":")
    if (
        not separator
        or not module_name
        or not attribute_name
        or module_name != module_name.strip()
        or attribute_name != attribute_name.strip()
    ):
        raise ValueError("--evaluator 必须采用 module:factory 格式")
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute_name)
    if not callable(factory):
        raise TypeError("evaluator factory 必须可调用")
    evaluator = factory(config)
    if not callable(getattr(evaluator, "evaluate", None)):
        raise TypeError("evaluator factory 必须返回实现 evaluate 的 TrialEvaluator")
    return evaluator


@app.command("train")
def train(
    evaluator: str = typer.Option(..., "--evaluator", help="module:factory"),
    config: Path = typer.Option(Path("configs/traning.yaml"), "--config"),
    run_id: str | None = typer.Option(None, "--run-id"),
    check_environment: bool = typer.Option(
        True,
        "--check-environment/--no-check-environment",
    ),
) -> None:
    """运行真实 V2 evaluator；普通 gate 失败会继续选择新参数。"""

    try:
        checked = load_checked_config(config)
        if check_environment:
            require_v2_environment(checked)
        selected_run_id = run_id or f"v2-search-{time.time_ns()}"
        store = StateStore(
            checked.telemetry.directory / selected_run_id,
            schema_version=checked.telemetry.schema_version,
        )
        reporter = TelemetryReporter(selected_run_id, store)
        selected_evaluator = load_trial_evaluator(evaluator, checked)
        result = run_configured_search(
            checked,
            selected_evaluator,
            reporter=reporter,
        )
    except SearchExhaustedError as exc:
        typer.echo(
            json.dumps(
                {
                    "status": "exhausted",
                    "trial_count": exc.decision.trial_count,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            err=True,
        )
        raise typer.Exit(2) from exc
    except Exception as exc:
        typer.echo(f"V2 训练失败：{exc}", err=True)
        raise typer.Exit(1) from exc

    typer.echo(
        json.dumps(
            {
                "status": "passed",
                "trial_index": result.trial_index,
                "objective": result.objective,
                "parameters": asdict(result.parameters),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


__all__ = (
    "app",
    "config_check",
    "coordinate_audit",
    "env_check",
    "load_checked_config",
    "load_trial_evaluator",
    "train",
)
