"""traning 模型的严格配置、环境、坐标审计与生产训练 CLI。"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import time

import typer

from traning.app.environment import check_v2_environment, require_v2_environment
from traning.app.factory import build_frame_coordinate_transform
from traning.config import V2Config, load_v2_config, v2_config_to_dict
from traning.data import (
    audit_affine_calibration,
    build_training_datasets,
    load_affine_calibration_evidence,
)
from traning.training import ProductionTrainer, SearchExhaustedError


app = typer.Typer(help="OSU Decision Model 的配置、环境、坐标和生产训练入口。")


def load_checked_config(path: Path) -> V2Config:
    """加载严格配置，保留原始异常供 CLI 显式报告。"""

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
        typer.echo(f"配置失败：{exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(_json(v2_config_to_dict(checked)))


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
        typer.echo(f"环境检查失败：{exc}", err=True)
        raise typer.Exit(1) from exc
    payload = {
        "ok": report.ok,
        "results": tuple(
            {
                "name": item.name,
                "status": item.status.value,
                "message": item.message,
            }
            for item in report.results
        ),
    }
    typer.echo(_json(payload))
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
    """复算控制点残差，并明确报告原始拟合集能否重放。"""

    try:
        checked = load_checked_config(config)
        evidence_path = checked.coordinates.calibration_evidence_path
        if evidence_path is None:
            raise ValueError("config 未配置 calibration_evidence_path")
        report = audit_affine_calibration(
            build_frame_coordinate_transform(checked),
            load_affine_calibration_evidence(evidence_path),
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        typer.echo(f"坐标审计失败：{exc}", err=True)
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
    typer.echo(_json(payload))
    if not report.ok or (require_refit_provenance and not report.fit_reproducible):
        raise typer.Exit(1)


@app.command("train")
def train(
    config: Path = typer.Option(Path("configs/traning.yaml"), "--config"),
    output_root: Path = typer.Option(
        Path("artifacts/training_runs"),
        "--output-root",
    ),
    run_id: str | None = typer.Option(None, "--run-id"),
    resume: bool = typer.Option(True, "--resume/--no-resume"),
    check_environment: bool = typer.Option(
        True,
        "--check-environment/--no-check-environment",
    ),
) -> None:
    """运行真实数据生产训练；普通门禁失败会持续提出未重复参数。"""

    try:
        checked = load_checked_config(config)
        if check_environment:
            require_v2_environment(checked)
        selected_run_id = run_id or f"traning-search-{time.time_ns()}"
        datasets = build_training_datasets(checked)
        result = ProductionTrainer(checked, datasets).run(
            run_dir=output_root / selected_run_id,
            run_id=selected_run_id,
            resume=resume,
        )
    except SearchExhaustedError as exc:
        typer.echo(
            _json(
                {
                    "status": "exhausted",
                    "trial_count": exc.decision.trial_count,
                }
            ),
            err=True,
        )
        raise typer.Exit(2) from exc
    except Exception as exc:
        typer.echo(f"训练失败：{exc}", err=True)
        raise typer.Exit(1) from exc

    typer.echo(
        _json(
            {
                "status": "passed",
                "trial_index": result.observation.trial_index,
                "objective": result.observation.objective,
                "parameters": asdict(result.observation.parameters),
                "checkpoint": result.checkpoint_directory,
                "resumed": result.resumed,
            }
        )
    )


def _json(value: object) -> str:
    """以稳定键序和 UTF-8 文本编码 CLI JSON。"""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


__all__ = (
    "app",
    "config_check",
    "coordinate_audit",
    "env_check",
    "load_checked_config",
    "train",
)
