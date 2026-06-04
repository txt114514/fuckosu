from __future__ import annotations

from time import perf_counter

from loguru import logger

from Traning.Lib.beatmap.verify import BeatmapVerifyExporter
from Traning.Lib.beatmap.folder_store import BeatmapFolderStore
from Traning.conf import Settings


def build_store(settings: Settings) -> BeatmapFolderStore:
    return BeatmapFolderStore(
        target_root=str(settings.file_management.target_root),
        order_filename=settings.file_management.order_filename,
    )


def export_verify(settings: Settings) -> bool:
    logger.info("开始 verify_export")
    started_at = perf_counter()
    store = build_store(settings)
    BeatmapVerifyExporter(
        walker=store.walker,
        store=store,
        verify_filename=settings.file_management.verify_filename,
        failed_filename=settings.file_management.verify_failed_filename,
    ).run(overwrite=settings.overwrite)
    logger.info("完成 verify_export ({:.2f}s)", perf_counter() - started_at)
    return True
