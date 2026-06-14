from __future__ import annotations

from time import perf_counter

from loguru import logger

from Traning.Lib.video.matching import VideoMatchProcessor
from Traning.conf import Settings


def match_videos(settings: Settings) -> bool:
    logger.info("开始 video_match")
    started_at = perf_counter()
    try:
        VideoMatchProcessor(
            video_root=str(settings.file_management.video_root),
            target_root=str(settings.file_management.target_root),
            manifest_filename=settings.file_management.manifest_filename,
            audio_filename=settings.file_management.audio_filename,
            video_suffixes=settings.file_formats.video_suffixes,
            use_audio_match_experiment=settings.video_clip.use_audio_match_experiment,
            sample_rate=settings.av.sample_rate,
            envelope_hz=settings.av.envelope_hz,
            refine_hz=settings.av.refine_hz,
            refine_search_seconds=settings.av.refine_search_seconds,
            music_lowpass_hz=settings.av.music_lowpass_hz,
        ).run()
    except ValueError as e:
        if "无需继续处理" not in str(e):
            raise
        logger.info("video_match 无需继续处理: {}", e)
    logger.info("完成 video_match ({:.2f}s)", perf_counter() - started_at)
    return True
