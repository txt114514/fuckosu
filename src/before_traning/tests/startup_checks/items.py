from __future__ import annotations

from pathlib import Path

from before_traning.Lib.beatmap.hit_objects import Circle
from before_traning.Lib.tools.ffmpeg import build_segment_video_args
from before_traning.Lib.video.segmentation.planner import build_segment_plans
from before_traning.conf import Settings, load_settings
from before_traning.core.beatmap.pipeline import TRAINING_TASKS
from before_traning.tests.startup_checks.samples import (
    DEFAULT_MATCHED_MANIFEST,
    inspect_before_training_samples,
)
from package.checks import StartupCheckResult


def check_settings_load(
    config_path: Path | None = None,
) -> tuple[StartupCheckResult, Settings]:
    settings = load_settings(config_path)
    return (
        StartupCheckResult(
            key="before_traning:settings",
            status="passed",
            message="before_traning settings loaded",
            details={
                "target_root": settings.file_management.target_root,
                "video_root": settings.file_management.video_root,
                "segment_root": settings.file_management.segment_root,
                "keyword": settings.file_formats.keyword,
            },
        ),
        settings,
    )


def check_pipeline_tasks(_settings: Settings | None = None) -> tuple[StartupCheckResult, None]:
    task_names = tuple(task.key for task in TRAINING_TASKS)
    expected = (
        "import_beatmaps",
        "verify_export",
        "difficulty_export",
        "video_match",
        "av_correspondence",
        "clip",
        "video_segment",
    )
    missing = tuple(name for name in expected if name not in task_names)
    if missing:
        return (
            StartupCheckResult(
                key="before_traning:pipeline_tasks",
                status="failed",
                message="before_traning pipeline is missing required stages",
                details={"missing": missing, "tasks": task_names},
            ),
            None,
        )
    return (
        StartupCheckResult(
            key="before_traning:pipeline_tasks",
            status="passed",
            message="before_traning pipeline task order is available",
            details={"tasks": task_names},
        ),
        None,
    )


def check_segment_planner_contract(_settings: Settings | None = None) -> tuple[StartupCheckResult, None]:
    objects = (
        Circle(3000, 3000, 100.0, 100.0),
        Circle(5000, 5000, 120.0, 100.0),
        Circle(7000, 7000, 140.0, 100.0),
        Circle(9000, 9000, 160.0, 100.0),
    )
    plans = build_segment_plans(
        objects,
        approach_preempt_ratio=0.5,
        circle_size=5.0,
        min_circle_overlap_ratio=0.5,
        priority_merge_window_ms=0,
        use_priority_merge=False,
        approach_preempt_seconds=1.0,
        pre_context_jitter_seconds=0.2,
        post_context_seconds=0.4,
        video_duration_seconds=12.0,
    )
    args = build_segment_video_args(
        Path("source.mp4"),
        Path("segment.mp4"),
        trim_start_seconds=1.0,
        trim_duration_seconds=2.0,
    )
    if len(plans) != len(objects):
        return (
            StartupCheckResult(
                key="before_traning:segment_planner",
                status="failed",
                message="segment planner did not produce one plan per object",
                details={"plan_count": len(plans), "object_count": len(objects)},
            ),
            None,
        )
    if "-an" not in args:
        return (
            StartupCheckResult(
                key="before_traning:segment_planner",
                status="failed",
                message="segment video command should strip audio by default",
                details={"ffmpeg_args": args},
            ),
            None,
        )
    return (
        StartupCheckResult(
            key="before_traning:segment_planner",
            status="passed",
            message="segment planner and ffmpeg argument contracts passed",
            details={"plan_count": len(plans), "ffmpeg_args": args},
        ),
        None,
    )


def check_raw_training_inputs(
    settings: Settings,
    *,
    matched_manifest_path: Path = DEFAULT_MATCHED_MANIFEST,
    run_match_probe: bool = True,
    min_match_score: float = 0.1,
) -> tuple[StartupCheckResult, None]:
    inspection = inspect_before_training_samples(
        settings,
        matched_manifest_path=matched_manifest_path,
        run_match_probe=run_match_probe,
        min_match_score=min_match_score,
    )
    if inspection.issues:
        status = "warning"
        message = "raw data scan completed with non-blocking issues"
    elif inspection.should_run_before_traning:
        status = "passed"
        message = "new unmatched raw data can update the training dataset"
    else:
        status = "passed"
        message = "no new matchable raw data found"
    return (
        StartupCheckResult(
            key="before_traning:raw_data",
            status=status,
            message=message,
            details=inspection.as_dict(),
        ),
        None,
    )


__all__ = [
    "check_pipeline_tasks",
    "check_raw_training_inputs",
    "check_segment_planner_contract",
    "check_settings_load",
]
