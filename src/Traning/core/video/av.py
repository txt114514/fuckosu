from __future__ import annotations

from time import perf_counter

from loguru import logger

from Traning.Lib.video.av_processing import (
    VideoAVProcessor,
)
from Traning.conf import Settings


def av_correspondence(settings: Settings) -> bool:
    logger.info("开始 av_correspondence")
    started_at = perf_counter()
    VideoAVProcessor(
        target_root=str(settings.file_management.target_root),
        order_filename=settings.file_management.order_filename,
        audio_filename=settings.file_management.audio_filename,
        verify_filename=settings.file_management.verify_filename,
        output_filename=settings.file_management.output_filename,
        failed_filename=settings.file_management.av_correspondence_failed_filename,
        status_step=settings.progress.av_status_step,
        required_steps=settings.progress.av_required_steps,
        sample_rate=settings.av.sample_rate,
        envelope_hz=settings.av.envelope_hz,
        refine_hz=settings.av.refine_hz,
        refine_search_seconds=settings.av.refine_search_seconds,
        music_lowpass_hz=settings.av.music_lowpass_hz,
        global_offset_ms=settings.global_offset_ms,
        video_suffixes=settings.file_formats.video_suffixes,
    ).run(overwrite=settings.overwrite)
    logger.info("完成 av_correspondence ({:.2f}s)", perf_counter() - started_at)
    return True
