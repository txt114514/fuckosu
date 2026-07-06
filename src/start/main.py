from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
import json
import os
from pathlib import Path
import shlex
import sys

if __package__ in {None, ""}:
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "src"))

import torch
import typer
from rich.console import Console
from rich.table import Table

from start.checks import run_startup_checks, run_training_startup_checks
from start.flow import StartupFlowConfig, run_startup_flow
from start.samples import DEFAULT_MATCHED_MANIFEST
from start.modules import source_module_entries
from traning.conf import DataSplit, load_settings
from traning.core.full_flow import FullFlowConfig, FullFlowResult, run_full_flow
from visualization.core.multi_terminal import launch_attached_training_terminals


app = typer.Typer(help="Unified src startup entry and preflight checks.")
console = Console()
DEFAULT_UI_TRAINING_CONFIG = Path("configs") / "model_full_small_vram.yaml"
DEFAULT_UI_OUTPUT_ROOT = Path("artifacts") / "training_runs"
DEFAULT_UI_GALLERY_ROOT = Path("traning_example")
TMUX_UI_ENV = "OSU_AI_TMUX_UI_ATTACHED"


class TestLevelOption(str, Enum):
    none = "none"
    quick = "quick"
    full = "full"


class CliParameterError(ValueError):
    """Raised when a plain business entry receives an invalid CLI-like value."""


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is not None:
        return
    try:
        result = run_training_ui_flow(
            training_config=DEFAULT_UI_TRAINING_CONFIG,
            output_root=DEFAULT_UI_OUTPUT_ROOT,
            gallery_output_root=DEFAULT_UI_GALLERY_ROOT,
        )
    except CliParameterError as error:
        raise typer.BadParameter(str(error)) from error
    _render_full_flow_table(result)
    if result.status not in {"passed", "dry-run-passed", "planned"}:
        raise typer.Exit(1)


def select_device(device: str) -> torch.device:
    selected = "cuda" if device == "auto" and torch.cuda.is_available() else device
    if selected == "auto":
        selected = "cpu"
    if selected not in {"cpu", "cuda"}:
        raise CliParameterError("device must be cpu, cuda, or auto")
    return torch.device(selected)


def collect_startup_check_result(
    *,
    config: Path | None = None,
    split: DataSplit = "train",
    device: str = "auto",
    require_cuda: bool = False,
) -> tuple[object, dict]:
    if config is None:
        report = run_startup_checks(require_cuda=require_cuda)
        return report, report.as_dict()

    settings = load_settings(config)
    selected = select_device(device)
    training_report = run_training_startup_checks(
        settings,
        split=split,
        device=selected,
        require_cuda=require_cuda or selected.type == "cuda",
    )
    return training_report.report, training_report.as_dict()


def run_training_startup_flow(
    *,
    training_config: Path,
    before_config: Path | None = None,
    split: DataSplit = "train",
    device: str = "auto",
    require_cuda: bool | None = None,
    matched_manifest: Path = DEFAULT_MATCHED_MANIFEST,
    dry_run: bool = False,
    skip_before_traning: bool = False,
    before_match_probe: bool = True,
    before_min_match_score: float = 0.1,
    split_manifest: Path | None = None,
    split_seed: int = 2026,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    allow_test_growth: bool = False,
    test_level: str = "quick",
    spatial_max_steps: int = 1,
    temporal_max_steps: int = 1,
    spatial_learning_rate: float = 1e-4,
    temporal_learning_rate: float = 1e-4,
    patch_limit: int = 1,
    cache_max_frames: int = 1,
    sequence_length: int | None = None,
    candidate_slots: int | None = None,
    parameter_group_id: str = "pg-0001",
    render_gallery: bool = True,
    gallery_output_root: Path | None = None,
    gallery_samples_per_group: int | None = None,
):
    selected = select_device(device)
    return run_startup_flow(
        StartupFlowConfig(
            training_config=training_config,
            before_config=before_config,
            split=split,
            device=selected,
            require_cuda=require_cuda,
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
            test_level=test_level,
            dry_run=dry_run,
            spatial_max_steps=spatial_max_steps,
            temporal_max_steps=temporal_max_steps,
            spatial_learning_rate=spatial_learning_rate,
            temporal_learning_rate=temporal_learning_rate,
            patch_limit=None if patch_limit == 0 else patch_limit,
            cache_max_frames=None if cache_max_frames == 0 else cache_max_frames,
            sequence_length=sequence_length,
            candidate_slots=candidate_slots,
            parameter_group_id=parameter_group_id,
            render_gallery=render_gallery,
            gallery_output_root=gallery_output_root,
            gallery_samples_per_group=gallery_samples_per_group,
        )
    )


def run_training_ui_flow(
    *,
    training_config: Path,
    target_config: Path | None = None,
    before_config: Path | None = None,
    split: DataSplit = "train",
    device: str = "auto",
    matched_manifest: Path = DEFAULT_MATCHED_MANIFEST,
    dry_run: bool = False,
    skip_before_traning: bool = False,
    before_match_probe: bool = True,
    before_min_match_score: float = 0.1,
    split_manifest: Path | None = None,
    split_seed: int = 2026,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    allow_test_growth: bool = False,
    test_level: str = "quick",
    progress_ui: str = "auto",
    progress_language: str = "zh-CN",
    output_root: Path = Path("artifacts") / "training_runs",
    run_id: str | None = None,
    auto_launch_full: bool = True,
    force_level: bool = False,
    max_levels: int | None = None,
    inherit_from: Path | str | None = None,
    resume_policy: str = "auto",
    resume: bool = True,
    gallery_output_root: Path | None = Path("traning_example"),
    gallery_samples_per_group: int | None = None,
) -> FullFlowResult:
    if progress_ui not in {"auto", "rich", "plain", "off"}:
        raise CliParameterError("progress-ui must be auto, rich, plain, or off")
    if resume_policy not in {"strict", "auto", "weights-only", "none"}:
        raise CliParameterError("resume-policy must be strict, auto, weights-only, or none")
    run_id = run_id or _new_cli_run_id()
    _maybe_attach_tmux_ui(
        training_config=training_config,
        target_config=target_config,
        before_config=before_config,
        split=split,
        device=device,
        matched_manifest=matched_manifest,
        dry_run=dry_run,
        skip_before_traning=skip_before_traning,
        before_match_probe=before_match_probe,
        before_min_match_score=before_min_match_score,
        split_manifest=split_manifest,
        split_seed=split_seed,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        allow_test_growth=allow_test_growth,
        test_level=test_level,
        progress_ui=progress_ui,
        progress_language=progress_language,
        output_root=output_root,
        run_id=run_id,
        auto_launch_full=auto_launch_full,
        force_level=force_level,
        max_levels=max_levels,
        inherit_from=inherit_from,
        resume_policy=resume_policy,
        resume=resume,
        gallery_output_root=gallery_output_root,
        gallery_samples_per_group=gallery_samples_per_group,
    )
    return run_full_flow(
        FullFlowConfig(
            config_path=training_config,
            target_config_path=target_config,
            before_config=before_config,
            split=split,
            device=device,
            mode="dry-run" if dry_run else "execute",
            output_root=output_root,
            run_id=run_id,
            auto_launch_full=auto_launch_full,
            force_level=force_level,
            max_levels=max_levels,
            run_full_checks=test_level != "none",
            progress_ui=progress_ui,
            progress_language=progress_language,
            inherit_from=inherit_from,
            resume_policy=resume_policy,
            resume_requested=resume,
            matched_manifest_path=matched_manifest,
            skip_before_traning=skip_before_traning,
            before_match_probe=before_match_probe,
            before_min_match_score=before_min_match_score,
            split_manifest=split_manifest,
            split_seed=split_seed,
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
            test_ratio=test_ratio,
            allow_test_growth=allow_test_growth,
            test_level=test_level,
            gallery_output_root=gallery_output_root,
            gallery_samples_per_group=gallery_samples_per_group,
        )
    )


def _new_cli_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _maybe_attach_tmux_ui(**kwargs: object) -> None:
    progress_ui = str(kwargs["progress_ui"])
    if progress_ui in {"off", "plain"}:
        return
    if os.environ.get(TMUX_UI_ENV) == "1":
        return
    if os.environ.get("TMUX"):
        return
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return
    output_root = Path(str(kwargs["output_root"]))
    run_id = str(kwargs["run_id"])
    command = _training_tmux_command(kwargs)
    result = launch_attached_training_terminals(
        run_id=run_id,
        dashboard_dir=output_root / run_id / "dashboard",
        cwd=Path.cwd(),
        training_command=command,
    )
    if result.status in {"attached", "launched"}:
        raise typer.Exit(0)
    console.print(f"[yellow]{result.message}，继续使用单终端 UI[/yellow]")


def _training_tmux_command(values: dict[str, object]) -> str:
    env = f"{TMUX_UI_ENV}=1 PYTHONPATH=src:."
    args = [
        sys.executable,
        str(Path(__file__).resolve()),
        "run",
        "--config",
        str(values["training_config"]),
        "--split",
        str(values["split"]),
        "--device",
        str(values["device"]),
        "--matched-manifest",
        str(values["matched_manifest"]),
        "--before-min-match-score",
        str(values["before_min_match_score"]),
        "--split-seed",
        str(values["split_seed"]),
        "--train-ratio",
        str(values["train_ratio"]),
        "--validation-ratio",
        str(values["validation_ratio"]),
        "--test-ratio",
        str(values["test_ratio"]),
        "--test-level",
        str(values["test_level"]),
        "--progress-ui",
        "rich",
        "--progress-language",
        str(values["progress_language"]),
        "--output-root",
        str(values["output_root"]),
        "--run-id",
        str(values["run_id"]),
        "--resume-policy",
        str(values["resume_policy"]),
    ]
    _append_optional_path(args, "--target-config", values["target_config"])
    _append_optional_path(args, "--before-config", values["before_config"])
    _append_optional_path(args, "--split-manifest", values["split_manifest"])
    _append_optional_path(args, "--inherit-from", values["inherit_from"])
    _append_optional_path(args, "--gallery-output-root", values["gallery_output_root"])
    _append_optional_int(args, "--max-levels", values["max_levels"])
    _append_optional_int(
        args,
        "--gallery-samples-per-group",
        values["gallery_samples_per_group"],
    )
    if bool(values["dry_run"]):
        args.append("--dry-run")
    if bool(values["skip_before_traning"]):
        args.append("--skip-before-traning")
    if not bool(values["before_match_probe"]):
        args.append("--no-before-match-probe")
    if bool(values["allow_test_growth"]):
        args.append("--allow-test-growth")
    if not bool(values["auto_launch_full"]):
        args.append("--no-auto-launch-full")
    if bool(values["force_level"]):
        args.append("--force-level")
    if bool(values["resume"]):
        args.append("--resume")
    else:
        args.append("--no-resume")
    return f"{env} " + " ".join(shlex.quote(str(arg)) for arg in args)


def _append_optional_path(args: list[str], option: str, value: object) -> None:
    if value is not None:
        args.extend((option, str(value)))


def _append_optional_int(args: list[str], option: str, value: object) -> None:
    if value is not None:
        args.extend((option, str(value)))


def _render_modules_table() -> None:
    table = Table(title="src module entries")
    table.add_column("Key")
    table.add_column("Import")
    table.add_column("Public entry")
    table.add_column("CLI")
    table.add_column("Importable")
    for entry in source_module_entries(include_start=True):
        table.add_row(
            entry.key,
            entry.import_name,
            entry.public_entry,
            entry.cli_entry or "",
            "yes" if entry.importable else "no",
        )
    console.print(table)


def _render_check_table(report) -> None:
    table = Table(title=f"startup checks: {report.scope}")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Message")
    for result in report.results:
        table.add_row(result.key, result.status, result.message)
    console.print(table)


def _render_check_detail_table(title: str, report) -> None:
    table = Table(title=title)
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Message")
    table.add_column("Details")
    for item in report.results:
        details = _compact_details(getattr(item, "details", {}) or {})
        table.add_row(item.key, item.status, item.message, details)
    console.print(table)


def _render_parameter_group_score(evaluation) -> None:
    table = Table(title="parameter group score")
    table.add_column("Group")
    table.add_column("Quality")
    table.add_column("Passed")
    table.add_column("Targets")
    table.add_column("Hits")
    table.add_column("Misses")
    table.add_column("Unresolved")
    table.add_column("Freq limited")
    table.add_column("No-op")
    table.add_column("Actions")
    table.add_column("Gallery")
    table.add_row(
        str(evaluation.parameter_group_id),
        f"{evaluation.quality_score:.6f}",
        "yes" if evaluation.passed else "no",
        str(evaluation.target_count),
        str(evaluation.hit_count),
        str(evaluation.miss_count),
        str(evaluation.unresolved_count),
        str(evaluation.frequency_limited_count),
        str(evaluation.no_op_frame_count),
        str(evaluation.action_frame_count),
        (
            f"{evaluation.gallery_status}: {evaluation.gallery_output_dir}"
            if evaluation.gallery_output_dir is not None
            else evaluation.gallery_status
        ),
    )
    console.print(table)
    if evaluation.gallery_warning:
        console.print(f"[yellow]{evaluation.gallery_warning}[/yellow]")


def _render_flow_table(result) -> None:
    table = Table(title="startup flow")
    table.add_column("Step")
    table.add_column("Status")
    table.add_column("Detail")
    raw_data = _raw_data_details(result.before_startup)
    before_status = (
        "run"
        if raw_data.get("should_run_before_traning")
        else "skip"
    )
    table.add_row(
        "before_traning scan",
        before_status,
        str(raw_data.get("reason") or "raw-data check not available"),
    )
    table.add_row(
        "before_traning build",
        result.before_run.status,
        result.before_run.message,
    )
    table.add_row(
        "dataset split sync",
        "changed" if result.split_sync.changed else "no-op",
        (
            f"new={len(result.split_sync.new_items)}, "
            f"counts={result.split_sync.manifest.counts()}"
        ),
    )
    table.add_row(
        "traning self-check",
        "passed" if result.training_startup.ok else "failed",
        f"{len(result.training_startup.results)} checks",
    )
    table.add_row(
        "progressive tests",
        result.tests.status,
        f"level={result.tests.level}, returncode={result.tests.returncode}",
    )
    table.add_row(
        "full training",
        "dry-run" if result.dry_run else "passed",
        (
            str(result.full_training.run_dir)
            if result.full_training is not None
            else "not executed"
        ),
    )
    console.print(table)
    _render_check_detail_table(
        "before_traning startup checks",
        result.before_startup,
    )
    _render_check_detail_table(
        "traning startup checks",
        result.training_startup,
    )
    printed_scopes = {
        result.before_startup.scope,
        result.training_startup.scope,
    }
    for report in result.tests.reports:
        if report.scope in printed_scopes:
            continue
        printed_scopes.add(report.scope)
        _render_check_detail_table(
            f"progressive checks: {report.scope}",
            report,
        )
    if result.full_training is not None:
        _render_parameter_group_score(result.full_training.evaluation)


def _render_full_flow_table(result: FullFlowResult) -> None:
    table = Table(title="full-flow startup, tests, and training")
    table.add_column("Stage")
    table.add_column("Status")
    table.add_column("Detail")
    for stage in result.stages:
        detail = stage.error or ""
        if not detail and stage.artifacts:
            detail = stage.artifacts[0]
        if not detail and stage.result:
            detail = _compact_details(dict(stage.result))
        table.add_row(stage.display_name, stage.status, detail)
    console.print(table)
    console.print(f"[green]run[/green]: {result.output_dir}")
    if result.ramp_manifest_path is not None:
        console.print(f"[green]ramp[/green]: {result.ramp_manifest_path}")


def _write_full_flow_report(result: FullFlowResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.as_dict(), ensure_ascii=False, indent=2, default=str)
        + "\n",
        encoding="utf-8",
    )


def _raw_data_details(report) -> dict:
    for item in report.results:
        if item.key == "before_traning:raw_data":
            return dict(item.details)
    return {}


def _compact_details(details: dict) -> str:
    if not details:
        return ""
    rendered = []
    for key, value in details.items():
        if isinstance(value, (dict, list, tuple)):
            rendered_value = json.dumps(value, ensure_ascii=False, default=str)
        else:
            rendered_value = str(value)
        if len(rendered_value) > 80:
            rendered_value = rendered_value[:77] + "..."
        rendered.append(f"{key}={rendered_value}")
    return ", ".join(rendered)


@app.command("modules")
def modules() -> None:
    _render_modules_table()


@app.command("check")
def check(
    config: Path | None = typer.Option(None, "--config"),
    split: DataSplit = typer.Option("train", "--split"),
    device: str = typer.Option("auto", "--device"),
    require_cuda: bool = typer.Option(False, "--require-cuda/--no-require-cuda"),
    json_output: Path | None = typer.Option(None, "--json-output"),
) -> None:
    try:
        report, output = collect_startup_check_result(
            config=config,
            split=split,
            device=device,
            require_cuda=require_cuda,
        )
    except CliParameterError as error:
        raise typer.BadParameter(str(error)) from error
    _render_check_table(report)
    if json_output is not None:
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(
            json.dumps(output, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        console.print(f"[green]saved[/green]: {json_output}")
    if not report.ok:
        raise typer.Exit(1)


@app.command("run")
def run(
    training_config: Path = typer.Option(
        DEFAULT_UI_TRAINING_CONFIG,
        "--training-config",
        "--config",
    ),
    target_config: Path | None = typer.Option(None, "--target-config"),
    before_config: Path | None = typer.Option(None, "--before-config"),
    split: DataSplit = typer.Option("train", "--split"),
    device: str = typer.Option("auto", "--device"),
    require_cuda: bool | None = typer.Option(None, "--require-cuda/--no-require-cuda"),
    matched_manifest: Path = typer.Option(
        DEFAULT_MATCHED_MANIFEST,
        "--matched-manifest",
    ),
    dry_run: bool = typer.Option(False, "--dry-run/--no-dry-run"),
    skip_before_traning: bool = typer.Option(False, "--skip-before-traning"),
    before_match_probe: bool = typer.Option(
        True,
        "--before-match-probe/--no-before-match-probe",
    ),
    before_min_match_score: float = typer.Option(
        0.1,
        "--before-min-match-score",
        min=0.0,
    ),
    split_manifest: Path | None = typer.Option(None, "--split-manifest"),
    split_seed: int = typer.Option(2026, "--split-seed"),
    train_ratio: float = typer.Option(0.8, "--train-ratio", min=0.0),
    validation_ratio: float = typer.Option(0.1, "--validation-ratio", min=0.0),
    test_ratio: float = typer.Option(0.1, "--test-ratio", min=0.0),
    allow_test_growth: bool = typer.Option(
        False,
        "--allow-test-growth/--freeze-test-growth",
    ),
    test_level: TestLevelOption = typer.Option(TestLevelOption.quick, "--test-level"),
    json_output: Path | None = typer.Option(None, "--json-output"),
    progress_ui: str = typer.Option("auto", "--progress-ui"),
    progress_language: str = typer.Option("zh-CN", "--progress-language"),
    output_root: Path = typer.Option(
        DEFAULT_UI_OUTPUT_ROOT,
        "--output-root",
    ),
    run_id: str | None = typer.Option(None, "--run-id"),
    auto_launch_full: bool = typer.Option(
        True,
        "--auto-launch-full/--no-auto-launch-full",
        help="Automatically start formal training after progressive tests pass.",
    ),
    force_level: bool = typer.Option(False, "--force-level"),
    max_levels: int | None = typer.Option(None, "--max-levels", min=1),
    inherit_from: str | None = typer.Option(None, "--inherit-from"),
    resume_policy: str = typer.Option("auto", "--resume-policy"),
    resume: bool = typer.Option(True, "--resume/--no-resume"),
    gallery_output_root: Path | None = typer.Option(
        DEFAULT_UI_GALLERY_ROOT,
        "--gallery-output-root",
        help="Formal training gallery output root.",
    ),
    gallery_samples_per_group: int | None = typer.Option(
        None,
        "--gallery-samples-per-group",
        min=1,
    ),
) -> None:
    if require_cuda and device == "auto":
        device = "cuda"
    try:
        result = run_training_ui_flow(
            training_config=training_config,
            target_config=target_config,
            before_config=before_config,
            split=split,
            device=device,
            matched_manifest=matched_manifest,
            dry_run=dry_run,
            skip_before_traning=skip_before_traning,
            before_match_probe=before_match_probe,
            before_min_match_score=before_min_match_score,
            split_manifest=split_manifest,
            split_seed=split_seed,
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
            test_ratio=test_ratio,
            allow_test_growth=allow_test_growth,
            test_level=test_level.value,
            progress_ui=progress_ui,
            progress_language=progress_language,
            output_root=output_root,
            run_id=run_id,
            auto_launch_full=auto_launch_full,
            force_level=force_level,
            max_levels=max_levels,
            inherit_from=inherit_from,
            resume_policy=resume_policy,
            resume=resume,
            gallery_output_root=gallery_output_root,
            gallery_samples_per_group=gallery_samples_per_group,
        )
    except CliParameterError as error:
        raise typer.BadParameter(str(error)) from error
    _render_full_flow_table(result)
    if json_output is not None:
        _write_full_flow_report(result, json_output)
        console.print(f"[green]saved[/green]: {json_output}")
    if result.status not in {"passed", "dry-run-passed", "planned"}:
        raise typer.Exit(1)

if __name__ == "__main__":
    app()
