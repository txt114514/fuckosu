from __future__ import annotations

from Traning.Lib.defaults import DEFAULT_SETTINGS as DEFAULTS
from Traning.Lib.video.matching.renamer import VideoMatchRenamer
from Traning.conf import Settings
from Traning.conf.legacy_config import assign_group, forward_kwargs, settings_namespace


class VideoMatchProcessor:
    """Video matching entry point used by the video/match task."""

    def __init__(
        self,
        settings: Settings = DEFAULTS,
        **overrides: object,
    ):
        if not isinstance(settings, Settings):
            overrides = {"video_root": settings, **overrides}
            settings = DEFAULTS

        config = settings_namespace(settings, processor="video_match", overrides=overrides)
        assign_group(self, config, "video_match")
        self.video_suffixes = tuple(config.video_suffixes)

    def run(self) -> None:
        if self.use_audio_match_experiment:
            from Traning.Lib.audio.matching import AudioMatchProcessor

            AudioMatchProcessor(
                **forward_kwargs(self, "video_match_to_audio_match")
            ).run(apply_matches=True, allow_fallback_videos=False)
            return

        VideoMatchRenamer(
            video_root=self.video_root,
            target_root=self.target_root,
            order_filename=self.order_filename,
            video_suffixes=self.video_suffixes,
        ).run()
