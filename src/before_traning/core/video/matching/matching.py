"""在音频内容匹配与录像时间顺序匹配策略之间选择业务入口。"""

from __future__ import annotations

from before_traning.conf.defaults import DEFAULT_SETTINGS as DEFAULTS
from before_traning.core.video.matching.renamer import VideoMatchRenamer
from before_traning.conf import Settings
from before_traning.conf.legacy_config import assign_group, forward_kwargs, settings_namespace


class VideoMatchProcessor:
    """供 video/match 业务阶段调用的录像匹配策略入口。"""

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
            from before_traning.core.audio.matching import AudioMatchProcessor

            AudioMatchProcessor(
                **forward_kwargs(self, "video_match_to_audio_match")
            ).run(apply_matches=True, allow_fallback_videos=False)
            return

        VideoMatchRenamer(
            video_root=self.video_root,
            target_root=self.target_root,
            manifest_filename=self.manifest_filename,
            video_suffixes=self.video_suffixes,
        ).run()
