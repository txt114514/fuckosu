from __future__ import annotations

import os
import traceback
from pathlib import Path
from typing import Callable

from loguru import logger

os.environ.setdefault("PREFECT_HOME", str(Path(__file__).resolve().parents[3] / ".prefect"))

from prefect import flow

from Traning.conf import Settings, load_settings
from Traning.core.beatmap import export_difficulty, export_verify, import_beatmaps
from Traning.core.video import av_correspondence, crop_video, match_videos
from Traning.core.tasks.av import av_correspondence_task
from Traning.core.tasks.clip import crop_video_task
from Traning.core.tasks.difficulty import export_difficulty_task
from Traning.core.tasks.importer import import_beatmaps_task
from Traning.core.tasks.match import match_videos_task
from Traning.core.tasks.verify import export_verify_task


StageCall = Callable[[], bool]


def _call_stage(label: str, stage_func: StageCall, continue_on_error: bool) -> bool:
    try:
        return stage_func()
    except Exception as e:
        logger.error("失败 {}: {}", label, e)
        traceback.print_exc()
        if not continue_on_error:
            raise
        return False


def _enabled(override: bool | None, default: bool) -> bool:
    return default if override is None else override


@flow(name="train_pipeline", log_prints=True)
def train_pipeline(
    settings: Settings | None = None,
    run_get_files: bool | None = None,
    run_verify_export: bool | None = None,
    run_difficulty_export: bool | None = None,
    run_video_match: bool | None = None,
    run_av_correspondence: bool | None = None,
    run_clip_stage: bool | None = None,
) -> dict[str, bool]:
    cfg = settings or load_settings()
    results: dict[str, bool] = {}

    if _enabled(run_get_files, cfg.check_data.run_get_files):
        results["import_beatmaps"] = _call_stage(
            "import_beatmaps",
            lambda: import_beatmaps_task(cfg),
            cfg.continue_on_error,
        )
    if _enabled(run_verify_export, cfg.check_data.run_verify_export):
        results["verify_export"] = _call_stage(
            "verify_export",
            lambda: export_verify_task(cfg),
            cfg.continue_on_error,
        )
    if _enabled(run_difficulty_export, cfg.check_data.run_difficulty_export):
        results["difficulty_export"] = _call_stage(
            "difficulty_export",
            lambda: export_difficulty_task(cfg),
            cfg.continue_on_error,
        )
    if _enabled(run_video_match, cfg.video_clip.run_video_match):
        results["video_match"] = _call_stage(
            "video_match",
            lambda: match_videos_task(cfg),
            cfg.continue_on_error,
        )
    if _enabled(run_av_correspondence, cfg.video_clip.run_av_correspondence):
        results["av_correspondence"] = _call_stage(
            "av_correspondence",
            lambda: av_correspondence_task(cfg),
            cfg.continue_on_error,
        )
    if _enabled(run_clip_stage, cfg.video_clip.run_clip_stage):
        results["clip"] = _call_stage(
            "clip",
            lambda: crop_video_task(cfg),
            cfg.continue_on_error,
        )

    return results


def train_pipeline_direct(
    settings: Settings | None = None,
    run_get_files: bool | None = None,
    run_verify_export: bool | None = None,
    run_difficulty_export: bool | None = None,
    run_video_match: bool | None = None,
    run_av_correspondence: bool | None = None,
    run_clip_stage: bool | None = None,
) -> dict[str, bool]:
    cfg = settings or load_settings()
    results: dict[str, bool] = {}

    if _enabled(run_get_files, cfg.check_data.run_get_files):
        results["import_beatmaps"] = _call_stage(
            "import_beatmaps",
            lambda: import_beatmaps(cfg),
            cfg.continue_on_error,
        )
    if _enabled(run_verify_export, cfg.check_data.run_verify_export):
        results["verify_export"] = _call_stage(
            "verify_export",
            lambda: export_verify(cfg),
            cfg.continue_on_error,
        )
    if _enabled(run_difficulty_export, cfg.check_data.run_difficulty_export):
        results["difficulty_export"] = _call_stage(
            "difficulty_export",
            lambda: export_difficulty(cfg),
            cfg.continue_on_error,
        )
    if _enabled(run_video_match, cfg.video_clip.run_video_match):
        results["video_match"] = _call_stage(
            "video_match",
            lambda: match_videos(cfg),
            cfg.continue_on_error,
        )
    if _enabled(run_av_correspondence, cfg.video_clip.run_av_correspondence):
        results["av_correspondence"] = _call_stage(
            "av_correspondence",
            lambda: av_correspondence(cfg),
            cfg.continue_on_error,
        )
    if _enabled(run_clip_stage, cfg.video_clip.run_clip_stage):
        results["clip"] = _call_stage(
            "clip",
            lambda: crop_video(cfg),
            cfg.continue_on_error,
        )

    return results


class TemporaryTrainingRunner:
    def __init__(self, config_path: Path | None = None):
        self.settings = load_settings(config_path)

    def run(
        self,
        overwrite: bool = False,
        run_check_data: bool = True,
        run_get_files: bool = True,
        run_verify_export: bool = True,
        run_difficulty_export: bool = True,
        run_video_clip: bool = True,
        run_video_match: bool = True,
        run_av_correspondence: bool = True,
        run_clip_stage: bool = True,
        use_audio_match_experiment: bool = True,
        global_offset_ms: float | None = None,
        continue_on_error: bool = False,
    ) -> dict[str, bool]:
        runtime = self.settings.runtime.model_copy(
            update={"overwrite": overwrite, "continue_on_error": continue_on_error}
        )
        video_clip = self.settings.video_clip.model_copy(
            update={
                "use_audio_match_experiment": use_audio_match_experiment,
                "global_offset_ms": (
                    self.settings.global_offset_ms
                    if global_offset_ms is None
                    else global_offset_ms
                ),
            }
        )
        settings = self.settings.model_copy(
            update={"runtime": runtime, "video_clip": video_clip}
        )
        return train_pipeline_direct(
            settings=settings,
            run_get_files=run_check_data and run_get_files,
            run_verify_export=run_check_data and run_verify_export,
            run_difficulty_export=run_check_data and run_difficulty_export,
            run_video_match=run_video_clip and run_video_match,
            run_av_correspondence=run_video_clip and run_av_correspondence,
            run_clip_stage=run_video_clip and run_clip_stage,
        )
