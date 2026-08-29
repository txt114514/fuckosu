"""仓库唯一启动 CLI：数据准备、划分、检查、训练搜索与报告。"""

from __future__ import annotations

from dataclasses import replace
from enum import Enum
import json
from pathlib import Path
import sys

# 直接执行 ``python src/start/main.py`` 时显式补齐仓库与 src；模块执行路径
# ``python -m start`` 不经过该分支，二者随后共享完全相同的业务入口。
if __package__ in {None, ""}:
    _repository_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(_repository_root))
    sys.path.insert(0, str(_repository_root / "src"))

import typer
from rich.console import Console
from rich.table import Table

from package import DataSplit
from start.executor import ProductionTrainingExecutor
from start.flow import StartFlowConfig, StartFlowResult, run_start_flow
from start.modules import source_module_entries
from start.samples import DEFAULT_MATCHED_MANIFEST
from traning.app.cli import (
    config_check as model_config_check,
    coordinate_audit as model_coordinate_audit,
    env_check as model_env_check,
)
from traning.config import RuntimeDevice


DEFAULT_TRAINING_CONFIG = Path("configs/traning.yaml")
DEFAULT_OUTPUT_ROOT = Path("artifacts/training_runs")


class DeviceOption(str, Enum):
    """CLI 的 auto 表示尊重配置文件，不在启动层猜测设备。"""

    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"


app = typer.Typer(
    help="统一执行 raw scan → before_traning → split → checks → training → report。",
    no_args_is_help=False,
)
console = Console()


def run_training_flow(
    *,
    config: Path = DEFAULT_TRAINING_CONFIG,
    before_config: Path | None = None,
    split: DataSplit = DataSplit.TRAIN,
    device: DeviceOption = DeviceOption.AUTO,
    matched_manifest: Path = DEFAULT_MATCHED_MANIFEST,
    split_manifest: Path | None = None,
    split_seed: int | None = None,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    allow_test_growth: bool = False,
    skip_before_traning: bool = False,
    before_match_probe: bool = True,
    before_min_match_score: float = 0.1,
    dry_run: bool = False,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    run_id: str | None = None,
    resume: bool = True,
) -> StartFlowResult:
    """把 CLI 值适配为唯一 StartFlowConfig，并注入正式训练执行器。"""

    requested_device = (
        None if device is DeviceOption.AUTO else RuntimeDevice(device.value)
    )
    flow_config = StartFlowConfig(
        training_config=config,
        before_config=before_config,
        split=split,
        requested_device=requested_device,
        matched_manifest_path=matched_manifest,
        run_before_traning=not skip_before_traning,
        before_match_probe=before_match_probe,
        before_min_match_score=before_min_match_score,
        split_manifest_path=split_manifest,
        split_seed=split_seed,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        allow_test_growth=allow_test_growth,
        dry_run=dry_run,
        output_root=output_root,
        resume=resume,
    )
    if run_id is not None:
        flow_config = replace(flow_config, run_id=run_id)
    return run_start_flow(
        flow_config,
        executor=ProductionTrainingExecutor(),
    )


def _render_and_exit(result: StartFlowResult) -> None:
    """渲染稳定阶段表，并让失败流程返回非零退出码。"""

    _render_result(result)
    if not result.ok:
        raise typer.Exit(1)


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """无子命令时保持原有总启动语义，直接运行完整流程。"""

    if ctx.invoked_subcommand is None:
        _render_and_exit(run_training_flow())


@app.command("run")
def run_command(
    config: Path = typer.Option(DEFAULT_TRAINING_CONFIG, "--config"),
    before_config: Path | None = typer.Option(None, "--before-config"),
    split: DataSplit = typer.Option(DataSplit.TRAIN, "--split"),
    device: DeviceOption = typer.Option(DeviceOption.AUTO, "--device"),
    matched_manifest: Path = typer.Option(
        DEFAULT_MATCHED_MANIFEST,
        "--matched-manifest",
    ),
    split_manifest: Path | None = typer.Option(None, "--split-manifest"),
    split_seed: int | None = typer.Option(None, "--split-seed"),
    train_ratio: float = typer.Option(0.8, "--train-ratio"),
    validation_ratio: float = typer.Option(0.1, "--validation-ratio"),
    test_ratio: float = typer.Option(0.1, "--test-ratio"),
    allow_test_growth: bool = typer.Option(False, "--allow-test-growth"),
    skip_before_traning: bool = typer.Option(False, "--skip-before-traning"),
    before_match_probe: bool = typer.Option(
        True,
        "--before-match-probe/--no-before-match-probe",
    ),
    before_min_match_score: float = typer.Option(0.1, "--before-min-match-score"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    output_root: Path = typer.Option(DEFAULT_OUTPUT_ROOT, "--output-root"),
    run_id: str | None = typer.Option(None, "--run-id"),
    resume: bool = typer.Option(True, "--resume/--no-resume"),
) -> None:
    """执行完整启动生命周期；普通 trial 未通过会继续选择新参数。"""

    result = run_training_flow(
        config=config,
        before_config=before_config,
        split=split,
        device=device,
        matched_manifest=matched_manifest,
        split_manifest=split_manifest,
        split_seed=split_seed,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        allow_test_growth=allow_test_growth,
        skip_before_traning=skip_before_traning,
        before_match_probe=before_match_probe,
        before_min_match_score=before_min_match_score,
        dry_run=dry_run,
        output_root=output_root,
        run_id=run_id,
        resume=resume,
    )
    _render_and_exit(result)


@app.command("modules")
def modules_command() -> None:
    """输出当前唯一活动源码模块及其公开入口。"""

    typer.echo(
        json.dumps(
            tuple(item.as_dict() for item in source_module_entries(include_start=True)),
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


# 配置、环境和坐标审计是同一模型的诊断命令，直接挂在总入口，不再保留
# ``v2`` 兼容命名空间或外部 evaluator 工厂。
app.command("config-check")(model_config_check)
app.command("env-check")(model_env_check)
app.command("coordinate-audit")(model_coordinate_audit)


def _render_result(result: StartFlowResult) -> None:
    """以最小 Rich 表展示阶段终态；JSON 报告仍是权威审计产物。"""

    table = Table(title=f"训练启动流程 · {result.run_id}")
    table.add_column("阶段")
    table.add_column("状态")
    table.add_column("说明")
    colors = {"passed": "green", "skipped": "yellow", "failed": "red"}
    for stage in result.stages:
        color = colors[stage.status]
        table.add_row(stage.stage_id, f"[{color}]{stage.status}[/{color}]", stage.message)
    console.print(table)
    console.print(f"报告：{result.report_path}")


if __name__ == "__main__":
    app()


__all__ = (
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_TRAINING_CONFIG",
    "DeviceOption",
    "app",
    "main",
    "run_command",
    "run_training_flow",
)
