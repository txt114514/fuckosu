from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import typer
from rich.console import Console
from rich.table import Table

from Traning.conf import Settings, ensure_prefect_home, load_settings
from Traning.core.flows.pipeline import train_pipeline, train_pipeline_direct


ensure_prefect_home()

app = typer.Typer(help="Training data workflow commands.", no_args_is_help=False)
console = Console()


def _resolve(default: bool, override: bool | None) -> bool:
    return default if override is None else override


def _skip(default: bool, skip_flag: bool) -> bool:
    return False if skip_flag else default


def _settings(
    config: Path | None,
    overwrite: bool | None = None,
    continue_on_error: bool | None = None,
    use_audio_match_experiment: bool | None = None,
    global_offset_ms: float | None = None,
) -> Settings:
    settings = load_settings(config)
    runtime = settings.runtime.model_copy(
        update={
            "overwrite": _resolve(settings.overwrite, overwrite),
            "continue_on_error": _resolve(settings.continue_on_error, continue_on_error),
        }
    )
    video_clip = settings.video_clip.model_copy(
        update={
            "use_audio_match_experiment": _resolve(
                settings.video_clip.use_audio_match_experiment,
                use_audio_match_experiment,
            ),
            "global_offset_ms": (
                settings.global_offset_ms
                if global_offset_ms is None
                else global_offset_ms
            ),
        }
    )
    return settings.model_copy(update={"runtime": runtime, "video_clip": video_clip})


def _render(results: dict[str, bool], elapsed: float):
    table = Table(title="Training pipeline result")
    table.add_column("Stage")
    table.add_column("Status")
    for stage_name, success in results.items():
        status = "[green]success[/green]" if success else "[red]failed[/red]"
        table.add_row(stage_name, status)
    console.print()
    console.print(table)
    console.print(f"[bold]Total:[/bold] {elapsed:.2f}s")


def _run(settings: Settings, **stages: bool | None) -> int:
    started_at = perf_counter()
    runner = (
        train_pipeline
        if os.environ.get("TRAINING_PREFECT_ENGINE") == "1"
        else train_pipeline_direct
    )
    results = runner(settings=settings, **stages)
    _render(results, perf_counter() - started_at)
    return 1 if any(not success for success in results.values()) else 0


@app.command("run")
def run_command(
    config: Path | None = typer.Option(None, "--config", help="config.yaml/json path."),
    overwrite: bool | None = typer.Option(None, "--overwrite/--no-overwrite"),
    continue_on_error: bool | None = typer.Option(None, "--continue-on-error/--stop-on-error"),
    skip_get_files: bool = typer.Option(False, "--skip-get-files"),
    skip_verify_export: bool = typer.Option(False, "--skip-verify-export"),
    skip_difficulty_export: bool = typer.Option(False, "--skip-difficulty-export"),
    skip_video_match: bool = typer.Option(False, "--skip-video-match"),
    skip_av_correspondence: bool = typer.Option(False, "--skip-av-correspondence"),
    skip_clip: bool = typer.Option(False, "--skip-clip"),
    use_audio_match_experiment: bool | None = typer.Option(
        None,
        "--use-audio-match-experiment/--disable-audio-match-experiment",
    ),
    global_offset_ms: float | None = typer.Option(None, "--global-offset-ms"),
):
    settings = _settings(
        config,
        overwrite,
        continue_on_error,
        use_audio_match_experiment,
        global_offset_ms,
    )
    raise typer.Exit(
        _run(
            settings,
            run_get_files=_skip(settings.check_data.run_get_files, skip_get_files),
            run_verify_export=_skip(settings.check_data.run_verify_export, skip_verify_export),
            run_difficulty_export=_skip(
                settings.check_data.run_difficulty_export,
                skip_difficulty_export,
            ),
            run_video_match=_skip(settings.video_clip.run_video_match, skip_video_match),
            run_av_correspondence=_skip(
                settings.video_clip.run_av_correspondence,
                skip_av_correspondence,
            ),
            run_clip_stage=_skip(settings.video_clip.run_clip_stage, skip_clip),
        )
    )


@app.command("verify")
def verify_command(
    config: Path | None = typer.Option(None, "--config", help="config.yaml/json path."),
    overwrite: bool | None = typer.Option(None, "--overwrite/--no-overwrite"),
    continue_on_error: bool | None = typer.Option(None, "--continue-on-error/--stop-on-error"),
):
    raise typer.Exit(
        _run(
            _settings(config, overwrite, continue_on_error),
            run_get_files=False,
            run_verify_export=True,
            run_difficulty_export=False,
            run_video_match=False,
            run_av_correspondence=False,
            run_clip_stage=False,
        )
    )


@app.command("match")
def match_command(
    config: Path | None = typer.Option(None, "--config", help="config.yaml/json path."),
    overwrite: bool | None = typer.Option(None, "--overwrite/--no-overwrite"),
    continue_on_error: bool | None = typer.Option(None, "--continue-on-error/--stop-on-error"),
    use_audio_match_experiment: bool | None = typer.Option(
        None,
        "--use-audio-match-experiment/--disable-audio-match-experiment",
    ),
):
    raise typer.Exit(
        _run(
            _settings(config, overwrite, continue_on_error, use_audio_match_experiment),
            run_get_files=False,
            run_verify_export=False,
            run_difficulty_export=False,
            run_video_match=True,
            run_av_correspondence=False,
            run_clip_stage=False,
        )
    )


@app.command("clip")
def clip_command(
    config: Path | None = typer.Option(None, "--config", help="config.yaml/json path."),
    overwrite: bool | None = typer.Option(None, "--overwrite/--no-overwrite"),
    continue_on_error: bool | None = typer.Option(None, "--continue-on-error/--stop-on-error"),
    global_offset_ms: float | None = typer.Option(None, "--global-offset-ms"),
):
    raise typer.Exit(
        _run(
            _settings(config, overwrite, continue_on_error, global_offset_ms=global_offset_ms),
            run_get_files=False,
            run_verify_export=False,
            run_difficulty_export=False,
            run_video_match=False,
            run_av_correspondence=True,
            run_clip_stage=True,
        )
    )


@app.callback(invoke_without_command=True)
def default_command(ctx: typer.Context):
    if ctx.invoked_subcommand is None:
        raise typer.Exit(_run(load_settings()))


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        app()
        return 0
    try:
        app(args=argv, standalone_mode=False)
    except typer.Exit as e:
        return int(e.exit_code or 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
