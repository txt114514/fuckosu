from __future__ import annotations

from time import perf_counter

from loguru import logger

from Traning.Lib.beatmap.verification import BeatmapVerifyExporter
from Traning.Lib.beatmap.folder_store import BeatmapFolderStore
from Traning.conf import Settings


def build_store(settings: Settings) -> BeatmapFolderStore:
    return BeatmapFolderStore(
        target_root=str(settings.file_management.target_root),
        manifest_filename=settings.file_management.manifest_filename,
    )


def export_verify(settings: Settings) -> bool:
    logger.info("开始 verify_export")
    started_at = perf_counter()
    store = build_store(settings)
    success = BeatmapVerifyExporter(
        walker=store.walker,
        store=store,
    ).run(overwrite=settings.overwrite)
    logger.info("完成 verify_export ({:.2f}s)", perf_counter() - started_at)
    return success
