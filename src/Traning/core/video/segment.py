from __future__ import annotations

from time import perf_counter

from loguru import logger

from Traning.Lib.video.segmentation import VideoSegmentationProcessor
from Traning.conf import Settings


def segment_videos(settings: Settings) -> bool:
    logger.info("开始 video_segment")
    started_at = perf_counter()
    success = VideoSegmentationProcessor(settings).run(overwrite=settings.overwrite)
    logger.info("完成 video_segment ({:.2f}s)", perf_counter() - started_at)
    return success
