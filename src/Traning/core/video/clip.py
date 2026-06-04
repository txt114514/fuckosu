from __future__ import annotations

from time import perf_counter

from loguru import logger

from Traning.Lib.video.clip import VideoClipProcessor
from Traning.conf import Settings


def crop_video(settings: Settings) -> bool:
    logger.info("开始 clip")
    started_at = perf_counter()
    VideoClipProcessor(
        target_root=str(settings.file_management.target_root),
        order_filename=settings.file_management.order_filename,
        output_filename=settings.file_management.output_filename,
        failed_filename=settings.file_management.clip_failed_filename,
        status_step=settings.clip.status_step,
        required_steps=settings.clip.required_steps,
        crop_reference_width=settings.clip.crop_reference_width,
        crop_reference_height=settings.clip.crop_reference_height,
        crop_left=settings.clip.crop_left,
        crop_top=settings.clip.crop_top,
        crop_right=settings.clip.crop_right,
        crop_bottom=settings.clip.crop_bottom,
    ).run(overwrite=settings.overwrite)
    logger.info("完成 clip ({:.2f}s)", perf_counter() - started_at)
    return True
