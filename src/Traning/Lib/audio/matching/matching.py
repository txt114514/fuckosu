from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from Traning.Lib.audio.matching.preflight import AudioMatchPreflightMixin
from Traning.Lib.audio.matching.steps import AudioMatchStepsMixin
from Traning.Lib.audio.matching.wrapup import AudioMatchWrapUpMixin
from Traning.Lib.beatmap.folder_store import BeatmapFolderStore
from Traning.Lib.beatmap.order import OrderFolderWalker
from Traning.Lib.common.batch import read_config_values
from Traning.Lib.common.pathspec import suffix_spec
from Traning.Lib.defaults import DEFAULT_SETTINGS as DEFAULTS
from Traning.Lib.video.av_processing import VideoAVProcessor
from Traning.conf import Settings
from Traning.conf.legacy_config import (
    AUDIO_MATCH_EXPERIMENT_CONFIG_SPECS,
    ConfigReader,
    assign_group,
    build_from_config_or_default,
    forward_kwargs,
    settings_namespace,
)
from Traning.state.process_status import ProcessStatusManager


def _load_audio_match_experiment_config(config: ConfigReader) -> dict[str, object]:
    return read_config_values(config, AUDIO_MATCH_EXPERIMENT_CONFIG_SPECS)


def build_audio_match_experiment_from_config_or_default(
    config_path: Path | None = None,
) -> "AudioMatchExperiment":
    return build_from_config_or_default(
        AudioMatchExperiment,
        [_load_audio_match_experiment_config],
        config_path=config_path,
        default_builder=AudioMatchExperiment,
    )


class AudioMatchExperiment(
    AudioMatchWrapUpMixin,
    AudioMatchStepsMixin,
    AudioMatchPreflightMixin,
):
    def __init__(
        self,
        settings: Settings = DEFAULTS,
        status_manager: ProcessStatusManager | None = None,
        **overrides: object,
    ):
        if not isinstance(settings, Settings):
            overrides = {"video_root": settings, **overrides}
            settings = DEFAULTS

        config = settings_namespace(settings, processor="audio_match", overrides=overrides)

        if config.top_k <= 0:
            raise ValueError("top_k 必须大于 0")

        assign_group(self, config, "audio_match")
        self.video_root = Path(self.video_root)
        self.target_root = Path(self.target_root)
        self.video_suffixes = {suffix.lower() for suffix in config.video_suffixes}
        self.video_file_spec = suffix_spec(self.video_suffixes)
        self.walker = OrderFolderWalker(
            target_root=str(self.target_root),
            order_filename=self.order_filename,
        )
        self.store = BeatmapFolderStore(
            target_root=str(self.target_root),
            order_filename=self.order_filename,
        )
        self.status_manager = status_manager or ProcessStatusManager(
            target_root=str(self.target_root),
            order_filename=self.order_filename,
        )
        aligner_kwargs = forward_kwargs(config, "audio_match_to_av")
        aligner_kwargs["target_root"] = str(self.target_root)
        aligner_kwargs["video_suffixes"] = self.video_suffixes
        self.aligner = VideoAVProcessor(**aligner_kwargs)


class AudioMatchProcessor(AudioMatchExperiment):
    """Task-aligned name for the audio-based video matching processor."""


def main():
    experiment = build_audio_match_experiment_from_config_or_default()
    experiment.run()


if __name__ == "__main__":
    main()
