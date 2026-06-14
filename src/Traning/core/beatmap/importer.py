from __future__ import annotations

from time import perf_counter

from loguru import logger

from Traning.Lib.beatmap.importing import BeatmapImportProcessor
from Traning.conf import Settings


def import_beatmaps(settings: Settings) -> bool:
    logger.info("开始 import_beatmaps")
    started_at = perf_counter()
    success = BeatmapImportProcessor(
        export_dir=str(settings.file_management.export_dir),
        target_root=str(settings.file_management.target_root),
        keyword=settings.file_formats.keyword,
        manifest_filename=settings.file_management.manifest_filename,
        audio_filename=settings.file_management.audio_filename,
    ).run()
    logger.info("完成 import_beatmaps ({:.2f}s)", perf_counter() - started_at)
    return success
