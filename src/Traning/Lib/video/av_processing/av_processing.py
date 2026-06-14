from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from Traning.Lib.beatmap.folder_store import BeatmapFolderStore
from Traning.Lib.common.batch import FolderBatchProcessor, read_config_values
from Traning.Lib.common.pathspec import suffix_spec
from Traning.Lib.defaults import DEFAULT_SETTINGS as DEFAULTS
from Traning.Lib.video.av_processing.preflight import AVPreflightMixin
from Traning.Lib.video.av_processing.steps import AVCoreStepsMixin
from Traning.Lib.video.av_processing.wrapup import AVWrapUpMixin
from Traning.conf import Settings
from Traning.conf.legacy_config import (
    AV_CORRESPONDENCE_PROCESSOR_CONFIG_SPECS,
    ConfigReader,
    assign_group,
    build_from_config_or_default,
    settings_namespace,
)
from Traning.state.process_status import ProcessStatusManager


def _load_av_correspondence_processor_config(config: ConfigReader) -> dict[str, object]:
    return read_config_values(config, AV_CORRESPONDENCE_PROCESSOR_CONFIG_SPECS)


def build_av_correspondence_processor_from_config_or_default(
    config_path: Path | None = None,
) -> "AVCorrespondenceProcessor":
    return build_from_config_or_default(
        AVCorrespondenceProcessor,
        [_load_av_correspondence_processor_config],
        config_path=config_path,
        default_builder=AVCorrespondenceProcessor,
    )


class AVCorrespondenceProcessor(
    AVWrapUpMixin,
    AVCoreStepsMixin,
    AVPreflightMixin,
    FolderBatchProcessor,
):
    def __init__(
        self,
        settings: Settings = DEFAULTS,
        status_manager: ProcessStatusManager | None = None,
        **overrides: object,
    ):
        if not isinstance(settings, Settings):
            overrides = {"target_root": settings, **overrides}
            settings = DEFAULTS

        config = settings_namespace(settings, processor="av", overrides=overrides)
        self._validate_config(config)

        assign_group(self, config, "av")
        self.target_root = Path(self.target_root)
        super().__init__()
        self.status_step = self.status_step.strip()
        self.required_steps = tuple(
            step.strip()
            for step in config.required_steps
            if step.strip()
        )
        self.refine_hz = min(config.sample_rate, config.refine_hz)
        self.global_offset_ms = float(config.global_offset_ms)
        self.video_suffixes = {suffix.lower() for suffix in config.video_suffixes}
        self.video_file_spec = suffix_spec(self.video_suffixes)
        self.store = BeatmapFolderStore(
            target_root=str(self.target_root),
            manifest_filename=self.manifest_filename,
        )
        self.walker = self.store.walker
        self.status_manager = status_manager or ProcessStatusManager(
            target_root=str(self.target_root),
            manifest_filename=self.manifest_filename,
        )

        self._ensure_status_steps_registered()


class VideoAVProcessor(AVCorrespondenceProcessor):
    """Task-aligned name for the video AV correspondence processor."""


def main():
    processor = build_av_correspondence_processor_from_config_or_default()
    processor.run(overwrite=False)


if __name__ == "__main__":
    main()
