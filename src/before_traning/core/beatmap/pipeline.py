from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from before_traning.conf import Settings, ensure_prefect_home, load_settings


ensure_prefect_home()

from before_traning.Lib.tasks import TaskSpec, build_task_pipeline
from before_traning.core.beatmap.difficulty import export_difficulty
from before_traning.core.beatmap.importer import import_beatmaps
from before_traning.core.beatmap.verify import export_verify
from before_traning.core.video.av import av_correspondence
from before_traning.core.video.clip import crop_video
from before_traning.core.video.match import match_videos
from before_traning.core.video.segment import segment_videos


BEATMAP_TASK_KEYS = (
    "import_beatmaps",
    "verify_export",
    "difficulty_export",
)

BEATMAP_OVERRIDE_KEYS = (
    "run_get_files",
    "run_verify_export",
    "run_difficulty_export",
)

VIDEO_TASK_KEYS = (
    "video_match",
    "av_correspondence",
    "clip",
    "video_segment",
)

VIDEO_OVERRIDE_KEYS = (
    "run_video_match",
    "run_av_correspondence",
    "run_clip_stage",
    "run_segment_stage",
)

TRAINING_TASKS = (
    TaskSpec(
        "import_beatmaps",
        import_beatmaps,
        "run_get_files",
        ("check_data", "run_get_files"),
    ),
    TaskSpec(
        "verify_export",
        export_verify,
        "run_verify_export",
        ("check_data", "run_verify_export"),
    ),
    TaskSpec(
        "difficulty_export",
        export_difficulty,
        "run_difficulty_export",
        ("check_data", "run_difficulty_export"),
    ),
    TaskSpec(
        "video_match",
        match_videos,
        "run_video_match",
        ("video_clip", "run_video_match"),
    ),
    TaskSpec(
        "av_correspondence",
        av_correspondence,
        "run_av_correspondence",
        ("video_clip", "run_av_correspondence"),
    ),
    TaskSpec(
        "clip",
        crop_video,
        "run_clip_stage",
        ("video_clip", "run_clip_stage"),
    ),
    TaskSpec(
        "video_segment",
        segment_videos,
        "run_segment_stage",
        ("video_clip", "run_segment_stage"),
    ),
)

TRAINING_PIPELINE = build_task_pipeline(
    TRAINING_TASKS,
    settings_loader=load_settings,
    continue_on_error=lambda settings: settings.continue_on_error,
    flow_name="train_pipeline",
)

train_pipeline = TRAINING_PIPELINE.run_prefect
train_pipeline_direct = TRAINING_PIPELINE.run_direct

LEGACY_STAGE_GROUPS = {
    "run_check_data": BEATMAP_OVERRIDE_KEYS,
    "run_video_clip": VIDEO_OVERRIDE_KEYS,
}


def prepare_beatmaps(settings: Settings) -> dict[str, bool]:
    return TRAINING_PIPELINE.run_direct(
        settings,
        only=BEATMAP_TASK_KEYS,
    )


class TemporaryTrainingRunner:
    def __init__(self, config_path: Path | None = None):
        self.settings = load_settings(config_path)

    def run(
        self,
        *,
        overwrite: bool = False,
        use_audio_match_experiment: bool = True,
        global_offset_ms: float | None = None,
        continue_on_error: bool = False,
        stage_overrides: Mapping[str, bool | None] | None = None,
        **legacy_stage_options: bool,
    ) -> dict[str, bool]:
        runtime = self.settings.runtime.model_copy(
            update={
                "overwrite": overwrite,
                "continue_on_error": continue_on_error,
            }
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
        overrides = dict(stage_overrides or {})
        for group_key, stage_keys in LEGACY_STAGE_GROUPS.items():
            group_supplied = group_key in legacy_stage_options
            group_enabled = legacy_stage_options.pop(group_key, True)
            for stage_key in stage_keys:
                stage_supplied = stage_key in legacy_stage_options
                stage_enabled = legacy_stage_options.pop(stage_key, True)
                if group_supplied or stage_supplied:
                    overrides[stage_key] = group_enabled and stage_enabled
        if legacy_stage_options:
            unknown = ", ".join(sorted(legacy_stage_options))
            raise TypeError(f"未知运行参数: {unknown}")
        return TRAINING_PIPELINE.run_direct(
            settings,
            overrides=overrides,
        )


__all__ = [
    "BEATMAP_TASK_KEYS",
    "BEATMAP_OVERRIDE_KEYS",
    "TRAINING_PIPELINE",
    "TRAINING_TASKS",
    "TemporaryTrainingRunner",
    "VIDEO_OVERRIDE_KEYS",
    "VIDEO_TASK_KEYS",
    "prepare_beatmaps",
    "train_pipeline",
    "train_pipeline_direct",
]
