from __future__ import annotations

import traceback
from dataclasses import dataclass
from functools import partial
from typing import Callable

from loguru import logger

from Traning.conf import ensure_prefect_home

ensure_prefect_home()

from prefect import flow

from Traning.Lib.common.failures import format_exception
from Traning.conf import Settings, load_settings
from Traning.core.beatmap import export_difficulty, export_verify, import_beatmaps
from Traning.core.video import (
    av_correspondence,
    crop_video,
    match_videos,
    segment_videos,
)
from Traning.core.tasks.av import av_correspondence_task
from Traning.core.tasks.clip import crop_video_task
from Traning.core.tasks.difficulty import export_difficulty_task
from Traning.core.tasks.importer import import_beatmaps_task
from Traning.core.tasks.match import match_videos_task
from Traning.core.tasks.segment import segment_videos_task
from Traning.core.tasks.verify import export_verify_task


StageCall = Callable[[], bool]
SettingsStageCall = Callable[[Settings], bool]


@dataclass(frozen=True)
class PipelineStage:
    key: str
    override_key: str
    settings_group: str
    settings_field: str
    direct_call: SettingsStageCall
    prefect_call: SettingsStageCall


PIPELINE_STAGES = (
    PipelineStage(
        "import_beatmaps",
        "run_get_files",
        "check_data",
        "run_get_files",
        import_beatmaps,
        import_beatmaps_task,
    ),
    PipelineStage(
        "verify_export",
        "run_verify_export",
        "check_data",
        "run_verify_export",
        export_verify,
        export_verify_task,
    ),
    PipelineStage(
        "difficulty_export",
        "run_difficulty_export",
        "check_data",
        "run_difficulty_export",
        export_difficulty,
        export_difficulty_task,
    ),
    PipelineStage(
        "video_match",
        "run_video_match",
        "video_clip",
        "run_video_match",
        match_videos,
        match_videos_task,
    ),
    PipelineStage(
        "av_correspondence",
        "run_av_correspondence",
        "video_clip",
        "run_av_correspondence",
        av_correspondence,
        av_correspondence_task,
    ),
    PipelineStage(
        "clip",
        "run_clip_stage",
        "video_clip",
        "run_clip_stage",
        crop_video,
        crop_video_task,
    ),
    PipelineStage(
        "video_segment",
        "run_segment_stage",
        "video_clip",
        "run_segment_stage",
        segment_videos,
        segment_videos_task,
    ),
)


def _call_stage(label: str, stage_func: StageCall, continue_on_error: bool) -> bool:
    try:
        return stage_func()
    except Exception as e:
        logger.error("失败 {}: {}", label, format_exception(e))
        traceback.print_exc()
        if not continue_on_error:
            raise
        return False


def _enabled(override: bool | None, default: bool) -> bool:
    return default if override is None else override


def _run_stages(
    settings: Settings,
    overrides: dict[str, bool | None],
    *,
    use_prefect: bool,
) -> dict[str, bool]:
    results: dict[str, bool] = {}

    for stage in PIPELINE_STAGES:
        settings_group = getattr(settings, stage.settings_group)
        default_enabled = bool(getattr(settings_group, stage.settings_field))
        if not _enabled(overrides[stage.override_key], default_enabled):
            continue

        stage_func = stage.prefect_call if use_prefect else stage.direct_call
        results[stage.key] = _call_stage(
            stage.key,
            partial(stage_func, settings),
            settings.continue_on_error,
        )

    return results


@flow(name="train_pipeline", log_prints=True)
def train_pipeline(
    settings: Settings | None = None,
    run_get_files: bool | None = None,
    run_verify_export: bool | None = None,
    run_difficulty_export: bool | None = None,
    run_video_match: bool | None = None,
    run_av_correspondence: bool | None = None,
    run_clip_stage: bool | None = None,
    run_segment_stage: bool | None = None,
) -> dict[str, bool]:
    cfg = settings or load_settings()
    return _run_stages(
        cfg,
        {
            "run_get_files": run_get_files,
            "run_verify_export": run_verify_export,
            "run_difficulty_export": run_difficulty_export,
            "run_video_match": run_video_match,
            "run_av_correspondence": run_av_correspondence,
            "run_clip_stage": run_clip_stage,
            "run_segment_stage": run_segment_stage,
        },
        use_prefect=True,
    )


def train_pipeline_direct(
    settings: Settings | None = None,
    run_get_files: bool | None = None,
    run_verify_export: bool | None = None,
    run_difficulty_export: bool | None = None,
    run_video_match: bool | None = None,
    run_av_correspondence: bool | None = None,
    run_clip_stage: bool | None = None,
    run_segment_stage: bool | None = None,
) -> dict[str, bool]:
    cfg = settings or load_settings()
    return _run_stages(
        cfg,
        {
            "run_get_files": run_get_files,
            "run_verify_export": run_verify_export,
            "run_difficulty_export": run_difficulty_export,
            "run_video_match": run_video_match,
            "run_av_correspondence": run_av_correspondence,
            "run_clip_stage": run_clip_stage,
            "run_segment_stage": run_segment_stage,
        },
        use_prefect=False,
    )


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
        run_segment_stage: bool = True,
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
            run_segment_stage=run_video_clip and run_segment_stage,
        )
