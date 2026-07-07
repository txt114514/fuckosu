from __future__ import annotations

import json
from pathlib import Path
import time
import typer

from rich.console import Console
from rich.live import Live

from visualization.core.panels.best_parameters_panel import render_best_parameters_panel
from visualization.core.panels.current_learning_panel import render_current_learning_panel
from visualization.core.panels.events_panel import render_events_panel
from visualization.core.panels.overall_progress_panel import render_overall_progress_panel
from visualization.core.panels.pipeline_panel import render_pipeline_panel
from visualization.core.panels.resources_panel import render_resources_panel
from visualization.core.training_view import (
    render_current_trial_panel,
    render_parameters_panel,
    render_score_summary_panel,
    render_tests_panel,
)
from visualization.lib.models import (
    BestParameterRecord,
    CurrentTrainingMetrics,
    DatasetUsageState,
    PipelineStageState,
    ResourceState,
    TrainingDashboardState,
    TrainingEvent,
    TrainingStopState,
)


app = typer.Typer(help="从 dashboard_state.json 监看一个可视化面板。")


@app.command("watch")
def watch_panel(
    state_path: Path = typer.Argument(..., help="dashboard_state.json 路径。"),
    panel: str = typer.Argument(..., help="要显示的面板名称。"),
    interval: float = typer.Option(1.0, "--interval", min=0.1, help="刷新间隔秒数。"),
) -> None:
    console = Console()
    with Live(
        _render_panel(state_path, panel),
        console=console,
        refresh_per_second=max(1.0 / interval, 1.0),
        transient=False,
    ) as live:
        while True:
            live.update(_render_panel(state_path, panel))
            time.sleep(interval)


def _render_panel(state_path: Path, panel: str):
    state = _load_state(state_path)
    renderers = {
        "startup": render_pipeline_panel,
        "current": render_current_trial_panel,
        "parameters": render_parameters_panel,
        "tests": render_tests_panel,
        "scores": render_score_summary_panel,
        "best": render_best_parameters_panel,
        "progress": render_overall_progress_panel,
        "resources": render_resources_panel,
        "events": render_events_panel,
        "learning": render_current_learning_panel,
    }
    render = renderers.get(panel)
    if render is None:
        return f"未知面板：{panel}"
    return render(state)


def _load_state(path: Path) -> TrainingDashboardState:
    if not path.exists():
        return TrainingDashboardState(run_id="waiting", status="等待状态文件")
    raw = json.loads(path.read_text(encoding="utf-8"))
    state = TrainingDashboardState(
        run_id=str(raw.get("run_id") or "unknown"),
        phase=str(raw.get("phase") or "初始化"),
        pipeline_phase=str(raw.get("pipeline_phase") or "startup"),
        status=str(raw.get("status") or "pending"),
        metrics=CurrentTrainingMetrics(**dict(raw.get("metrics") or {})),
        dataset_usage=DatasetUsageState(**dict(raw.get("dataset_usage") or {})),
        resources=ResourceState(**dict(raw.get("resources") or {})),
        best_parameters=BestParameterRecord(**dict(raw.get("best_parameters") or {})),
        current_parameters=dict(raw.get("current_parameters") or {}),
        current_parameter_status=dict(raw.get("current_parameter_status") or {}),
        recent_events=[
            TrainingEvent(**dict(item)) for item in raw.get("recent_events") or ()
        ],
    )
    for key, value in dict(raw.get("pipeline_stages") or {}).items():
        state.pipeline_stages[str(key)] = PipelineStageState(**dict(value))
    for key in (
        "current_level",
        "completed_levels",
        "total_levels",
        "epoch",
        "total_epochs",
        "global_step",
        "target_global_steps",
        "spatial_step",
        "spatial_target",
        "temporal_step",
        "temporal_target",
        "current_trial_id",
        "trial_status",
        "prune_reason",
        "current_grade",
        "best_grade",
        "promotion_status",
        "consecutive_passes",
        "required_passes",
        "category_scores",
        "frames_per_second",
        "steps_per_second",
        "elapsed_seconds",
        "eta_seconds",
        "latest_checkpoint",
        "best_checkpoint",
        "inheritance_path",
    ):
        if key in raw:
            setattr(state, key, raw[key])
    if raw.get("stop_state") is not None:
        state.stop_state = TrainingStopState(**dict(raw["stop_state"]))
    return state


if __name__ == "__main__":
    app()
