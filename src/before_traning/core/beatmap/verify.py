from __future__ import annotations

from pathlib import Path
from time import perf_counter

from loguru import logger

from before_traning.Lib.beatmap.folder_store import BeatmapFolderStore
from before_traning.Lib.beatmap.osu_parser import VerifyOsuParser
from before_traning.Lib.beatmap.standard import load_standard_beatmap
from before_traning.Lib.common.batch import BatchProcessResult, FolderBatchProcessor
from before_traning.Lib.common.processing import ProcessingGuard
from before_traning.conf import DEFAULT_SETTINGS, Settings, VERIFY_FILENAME, load_settings
from before_traning.state.process_status import ProcessStatusManager


def build_store(settings: Settings) -> BeatmapFolderStore:
    return BeatmapFolderStore(
        target_root=str(settings.file_management.target_root),
        manifest_filename=settings.file_management.manifest_filename,
    )


class BeatmapVerifyExporter(FolderBatchProcessor):
    def __init__(
        self,
        store: BeatmapFolderStore,
        *,
        status_manager: ProcessStatusManager | None = None,
    ):
        super().__init__()
        self.store = store
        self.walker = store.walker
        self.verify_filename = VERIFY_FILENAME
        self.parser = VerifyOsuParser()
        self.status_manager = status_manager or ProcessStatusManager(
            target_root=str(store.target_root),
            manifest_filename=store.manifest_filename,
        )
        self.guard = ProcessingGuard(
            self.store,
            self.status_manager,
            "verify_exported",
            required_steps=("osu_imported",),
        )

    def process_one(
        self,
        folder_name: str,
        overwrite: bool = False,
    ) -> BatchProcessResult:
        files = self.guard.prepare_folder(
            folder_name,
            required_patterns=("*.osu",),
        )
        if files is None:
            return "skip"
        if self.guard.is_complete(
            folder_name,
            overwrite=overwrite,
            output_files=(self.verify_filename,),
        ):
            return "skip"

        _osu_path, beatmap = load_standard_beatmap(
            self.store,
            folder_name,
            refresh=True,
            parser=self.parser,
        )
        result = self.store.write_lines(
            folder_name=folder_name,
            filename=self.verify_filename,
            lines=self.parser.objects_to_lines(beatmap.hit_objects),
            mode="overwrite" if overwrite else "skip_if_exists",
        )
        detail = {
            "filename": self.verify_filename,
            "hit_object_count": len(beatmap.hit_objects),
            "beatmap_data_cached": True,
        }
        was_done = self.guard.step_done(folder_name)
        self.guard.mark_done(folder_name, detail=detail)
        return "skip" if result == "skipped" and was_done else "success"

    def handle_failure(self, folder_name: str, error: Exception) -> None:
        self.guard.record_failure(folder_name, error)


VerifyExporter = BeatmapVerifyExporter


def build_verify_exporter_from_config_or_default(
    config_path: Path | None = None,
) -> BeatmapVerifyExporter:
    settings = load_settings(config_path)
    return BeatmapVerifyExporter(build_store(settings))


def build_beatmap_verify_exporter_from_config_or_default(
    config_path: Path | None = None,
) -> BeatmapVerifyExporter:
    return build_verify_exporter_from_config_or_default(config_path)


def export_verify(settings: Settings = DEFAULT_SETTINGS) -> bool:
    logger.info("开始 verify_export")
    started_at = perf_counter()
    success = BeatmapVerifyExporter(
        build_store(settings),
    ).run(overwrite=settings.overwrite)
    logger.info("完成 verify_export ({:.2f}s)", perf_counter() - started_at)
    return success


__all__ = [
    "BeatmapVerifyExporter",
    "VerifyExporter",
    "build_beatmap_verify_exporter_from_config_or_default",
    "build_store",
    "build_verify_exporter_from_config_or_default",
    "export_verify",
]
