from __future__ import annotations

from time import perf_counter

from loguru import logger

from Traning.Lib.video.clipping import VideoClipProcessor
from Traning.conf import Settings


def crop_video(settings: Settings) -> bool:
    logger.info("开始 clip")
    started_at = perf_counter()
    VideoClipProcessor.from_settings(settings).run(overwrite=settings.overwrite)
    logger.info("完成 clip ({:.2f}s)", perf_counter() - started_at)
    return True
