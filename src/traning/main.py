from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import typer
from rich.console import Console
from rich.table import Table

from traning.Lib.data import build_patch_windows
from traning.conf import load_settings
from traning.core.data_input import build_dataset, inspect_data_input
from traning.core.pipeline import run_pipeline
from traning.core.visualization import (
    save_annotation_gallery,
    visualize_click_label,
)
from traning.state import load_batch_gallery_request


app = typer.Typer(help="osu! video model training commands.")
console = Console()


def _render_report(report) -> None:
    table = Table(title="Data input")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("segments", str(report.segment_count))
    table.add_row("estimated frames", str(report.frame_count_estimate))
    table.add_row("issues", str(report.issue_count))
    table.add_row("dimensions", str(report.dimension_counts))
    table.add_row("categories", str(report.category_counts))
    console.print(table)
    for issue in report.issues[:20]:
        console.print(f"[red]- {issue}[/red]")


@app.command("data-check")
def data_check(
    config: Path | None = typer.Option(None, "--config"),
) -> None:
    settings = load_settings(config)
    report = inspect_data_input(settings)
    _render_report(report)
    if settings.data_input.strict and not report.ok:
        raise typer.Exit(1)


@app.command("data-preview")
def data_preview(
    index: int = typer.Option(0, "--index", min=0),
    config: Path | None = typer.Option(None, "--config"),
) -> None:
    settings = load_settings(config)
    dataset = build_dataset(settings)
    if index >= len(dataset):
        raise typer.BadParameter(
            f"index {index} is outside dataset length {len(dataset)}"
        )
    sample = dataset[index]
    _, height, width = sample["image"].shape
    data = settings.data_input
    windows = build_patch_windows(
        width,
        height,
        patch_width=data.patch_width,
        patch_height=data.patch_height,
        overlap_x=data.overlap_x,
        overlap_y=data.overlap_y,
    )
    console.print(
        {
            "sample_key": sample["sample_key"],
            "frame_index": sample["frame_index"],
            "timestamp_ms": sample["timestamp_ms"],
            "image_shape": tuple(sample["image"].shape),
            "visible_objects": len(sample["visible_hit_objects"]),
            "patch_count": len(windows),
            "first_patch": windows[0],
        }
    )


@app.command("run")
def run(
    config: Path | None = typer.Option(None, "--config"),
) -> None:
    results = run_pipeline(load_settings(config))
    _render_report(results["data_input"])


@app.command("visualize-label")
def visualize_label(
    segment_index: int = typer.Option(0, "--segment-index", min=0),
    object_index: int = typer.Option(0, "--object-index", min=0),
    output: Path | None = typer.Option(None, "--output"),
    show: bool = typer.Option(False, "--show/--no-show"),
    config: Path | None = typer.Option(None, "--config"),
) -> None:
    result = visualize_click_label(
        load_settings(config),
        segment_index=segment_index,
        object_index=object_index,
        output_path=output,
        show_window=show,
    )
    if result.output_path is not None:
        console.print(f"[green]{result.status}[/green]: {result.output_path}")
    else:
        console.print(result.status)
    if result.warning:
        console.print(f"[yellow]warning:[/yellow] {result.warning}")


@app.command("save-annotation-gallery")
def save_gallery(
    results: Path = typer.Option(
        ...,
        "--results",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    output_root: Path | None = typer.Option(None, "--output-root"),
    samples_per_group: int | None = typer.Option(
        None,
        "--samples-per-group",
        min=1,
    ),
    config: Path | None = typer.Option(None, "--config"),
) -> None:
    result = save_annotation_gallery(
        load_settings(config),
        load_batch_gallery_request(results),
        output_root=output_root,
        samples_per_group=samples_per_group,
    )
    if result.output_dir is not None:
        console.print(
            f"[green]{result.status}[/green]: {result.output_dir} "
            f"({result.saved_frame_count} frames, "
            f"trial={result.selected_trial_id})"
        )
    else:
        console.print(result.status)
    if result.warning:
        console.print(f"[yellow]warning:[/yellow] {result.warning}")
    if not result.succeeded:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
