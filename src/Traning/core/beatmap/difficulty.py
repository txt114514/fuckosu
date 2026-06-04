from __future__ import annotations

from time import perf_counter

from loguru import logger

from Traning.Lib.beatmap.difficulty import BeatmapDifficultyProcessor
from Traning.conf import Settings

from .verify import build_store


def export_difficulty(settings: Settings) -> bool:
    logger.info("开始 difficulty_export")
    started_at = perf_counter()
    store = build_store(settings)
    BeatmapDifficultyProcessor(
        walker=store.walker,
        store=store,
        difficulty_filename=settings.file_management.difficulty_filename,
        failed_filename=settings.file_management.difficulty_failed_filename,
    ).run(overwrite=settings.overwrite)
    logger.info("完成 difficulty_export ({:.2f}s)", perf_counter() - started_at)
    return True
