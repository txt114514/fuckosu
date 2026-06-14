from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from loguru import logger

from before_traning.Lib.beatmap.folder_store import BeatmapFolderStore
from before_traning.Lib.beatmap.osu_metadata import read_overall_difficulty
from before_traning.Lib.common.batch import BatchProcessResult, FolderBatchProcessor
from before_traning.Lib.common.processing import ProcessingGuard
from before_traning.conf import Settings
from before_traning.core.beatmap.verify import build_store
from before_traning.state.process_status import ProcessStatusManager


@dataclass(frozen=True)
class DifficultyEntry:
    folder_name: str
    difficulty_value: float


class BeatmapDifficultyProcessor(FolderBatchProcessor):
    def __init__(
        self,
        store: BeatmapFolderStore,
        *,
        status_manager: ProcessStatusManager | None = None,
    ):
        super().__init__()
        self.store = store
        self.walker = store.walker
        self.manifest = self.walker.manifest
        self.status_manager = status_manager or ProcessStatusManager(
            target_root=str(store.target_root),
            manifest_filename=store.manifest_filename,
        )
        self.guard = ProcessingGuard(
            self.store,
            self.status_manager,
            "difficulty_exported",
            required_steps=("osu_imported",),
        )

    def write_difficulty(
        self,
        folder_name: str,
        difficulty_value: float,
    ) -> None:
        self.manifest.set_difficulty(folder_name, difficulty_value)

    def read_difficulty(self, folder_name: str) -> float:
        difficulty_value = self.manifest.difficulty_for(folder_name)
        if difficulty_value is None:
            raise ValueError(f"{folder_name} 尚未生成 difficulty")
        return difficulty_value

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

        existing_value = self.manifest.difficulty_for(folder_name)
        reconciled = self.guard.reconcile_existing(
            folder_name,
            overwrite=overwrite,
            artifact_exists=existing_value is not None,
            detail=(
                {"difficulty_value": existing_value}
                if existing_value is not None
                else None
            ),
        )
        if reconciled is not None:
            return reconciled

        difficulty_value = read_overall_difficulty(files["*.osu"][0])
        self.write_difficulty(folder_name, difficulty_value)
        self.guard.mark_done(
            folder_name,
            detail={"difficulty_value": difficulty_value},
        )
        return "success"

    def handle_failure(self, folder_name: str, error: Exception) -> None:
        self.guard.record_failure(folder_name, error)

    def list_difficulties(
        self,
        min_difficulty: float | None = None,
        max_difficulty: float | None = None,
    ) -> list[DifficultyEntry]:
        if (
            min_difficulty is not None
            and max_difficulty is not None
            and min_difficulty > max_difficulty
        ):
            raise ValueError("min_difficulty 不能大于 max_difficulty")

        matched: list[DifficultyEntry] = []
        for folder_name in self.walker.read_folder_names():
            value = self.read_difficulty(folder_name)
            if min_difficulty is not None and value < min_difficulty:
                continue
            if max_difficulty is not None and value > max_difficulty:
                continue
            matched.append(DifficultyEntry(folder_name, value))
        return matched


DifficultyFileManager = BeatmapDifficultyProcessor


def export_difficulty(settings: Settings) -> bool:
    logger.info("开始 difficulty_export")
    started_at = perf_counter()
    success = BeatmapDifficultyProcessor(
        build_store(settings),
    ).run(overwrite=settings.overwrite)
    logger.info(
        "完成 difficulty_export ({:.2f}s)",
        perf_counter() - started_at,
    )
    return success


__all__ = [
    "BeatmapDifficultyProcessor",
    "DifficultyEntry",
    "DifficultyFileManager",
    "export_difficulty",
]
