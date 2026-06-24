from __future__ import annotations

from enum import Enum
import json
from pathlib import Path

import torch
import typer
from rich.console import Console
from rich.table import Table

from start.checks import run_startup_checks, run_training_startup_checks
from start.flow import StartupFlowConfig, run_startup_flow, write_startup_flow_report
from start.samples import DEFAULT_MATCHED_MANIFEST
from start.modules import source_module_entries
from traning.conf import DataSplit, load_settings


app = typer.Typer(help="Unified src startup entry and preflight checks.")
console = Console()


class TestLevelOption(str, Enum):
    none = "none"
    quick = "quick"
    full = "full"


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


def _raw_data_details(report) -> dict:
    for item in report.results:
        if item.key == "before_traning:raw_data":
            return dict(item.details)
    return {}


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
    if config is None:
        report = run_startup_checks(require_cuda=require_cuda)
        output = report.as_dict()
    else:
        settings = load_settings(config)
        selected = _select_device(device)
        training_report = run_training_startup_checks(
            settings,
            split=split,
            device=selected,
            require_cuda=require_cuda or selected.type == "cuda",
        )
        report = training_report.report
        output = training_report.as_dict()
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
        Path("configs/model_small_vram.yaml"),
        "--training-config",
        "--config",
    ),
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
    spatial_max_steps: int = typer.Option(1, "--spatial-max-steps", min=1),
    temporal_max_steps: int = typer.Option(1, "--temporal-max-steps", min=1),
    spatial_learning_rate: float = typer.Option(1e-4, "--spatial-lr", min=1e-8),
    temporal_learning_rate: float = typer.Option(1e-4, "--temporal-lr", min=1e-8),
    patch_limit: int = typer.Option(
        1,
        "--patch-limit",
        min=0,
        help="0 means process all patches in each frame.",
    ),
    cache_max_frames: int = typer.Option(
        1,
        "--cache-max-frames",
        min=0,
        help="0 means no frame limit for candidate cache generation.",
    ),
    sequence_length: int | None = typer.Option(None, "--sequence-length", min=1),
    candidate_slots: int | None = typer.Option(None, "--candidate-slots", min=1),
) -> None:
    selected = _select_device(device)
    result = run_startup_flow(
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
            test_level=test_level.value,
            dry_run=dry_run,
            spatial_max_steps=spatial_max_steps,
            temporal_max_steps=temporal_max_steps,
            spatial_learning_rate=spatial_learning_rate,
            temporal_learning_rate=temporal_learning_rate,
            patch_limit=None if patch_limit == 0 else patch_limit,
            cache_max_frames=None if cache_max_frames == 0 else cache_max_frames,
            sequence_length=sequence_length,
            candidate_slots=candidate_slots,
        )
    )
    _render_flow_table(result)
    if json_output is not None:
        write_startup_flow_report(result, json_output)
        console.print(f"[green]saved[/green]: {json_output}")
    if not result.ok:
        raise typer.Exit(1)


def _select_device(device: str) -> torch.device:
    selected = "cuda" if device == "auto" and torch.cuda.is_available() else device
    if selected == "auto":
        selected = "cpu"
    if selected not in {"cpu", "cuda"}:
        raise typer.BadParameter("device must be cpu, cuda, or auto")
    return torch.device(selected)


if __name__ == "__main__":
    app()
