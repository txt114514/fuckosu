from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from Traning.Lib.beatmap.folder_store import (
    BeatmapFolderStore,
    WriteMode,
)
from Traning.Lib.beatmap.difficulty_batch import DifficultyBatchMixin
from Traning.Lib.beatmap.osu_metadata import read_overall_difficulty
from Traning.Lib.beatmap.order import OrderFolderWalker
from Traning.conf import Settings
from Traning.conf.legacy_config import settings_namespace
from Traning.Lib.defaults import DEFAULT_SETTINGS as DEFAULTS
from Traning.state.process_status import (
    ProcessStatusManager,
)


@dataclass(frozen=True)
class DifficultyEntry:
    folder_name: str
    difficulty_value: float


class DifficultyFileManager(DifficultyBatchMixin):
    def __init__(
        self,
        store: BeatmapFolderStore,
        walker: OrderFolderWalker | None = None,
        settings: Settings = DEFAULTS,
        status_manager: ProcessStatusManager | None = None,
        **overrides: object,
    ):
        if not isinstance(settings, Settings):
            overrides = {"difficulty_filename": settings, **overrides}
            settings = DEFAULTS

        config = settings_namespace(settings, processor="difficulty", overrides=overrides)
        self.store = store
        self.walker = walker or store.walker
        self.difficulty_filename = config.difficulty_filename
        self.failed_filename = config.failed_filename
        self.status_manager = status_manager or ProcessStatusManager(
            target_root=str(store.target_root),
            order_filename=store.order_filename,
        )

        self.success_count = 0
        self.skip_count = 0
        self.fail_count = 0
        self.failed_cases: List[tuple[str, str]] = []

    def write_difficulty(
        self,
        folder_name: str,
        difficulty_value: float,
        mode: WriteMode = "overwrite",
    ) -> str:
        return self.store.write_text(
            folder_name=folder_name,
            filename=self.difficulty_filename,
            content=f"{difficulty_value}\n",
            mode=mode,
        )

    def read_difficulty(self, folder_name: str) -> float:
        content = self.store.read_text(
            folder_name=folder_name,
            filename=self.difficulty_filename,
        )
        return float(content.strip())

    def export_one(self, folder_name: str, overwrite: bool = False) -> str:
        if not self.store.folder_exists(folder_name):
            self.skip_count += 1
            return "skip"

        self.status_manager.ensure_status_file(folder_name)
        difficulty_exists = self.store.file_exists(folder_name, self.difficulty_filename)
        difficulty_done = self.status_manager.is_step_done(
            folder_name,
            "difficulty_exported",
        )
        if not overwrite and difficulty_exists and difficulty_done:
            self.skip_count += 1
            return "skip"

        osu_files = self.store.find_osu_files(folder_name)
        if not osu_files:
            self.skip_count += 1
            return "skip"

        difficulty_value = read_overall_difficulty(osu_files[0])
        write_mode: WriteMode = "overwrite" if overwrite else "skip_if_exists"

        result = self.write_difficulty(
            folder_name=folder_name,
            difficulty_value=difficulty_value,
            mode=write_mode,
        )

        if result == "skipped":
            self.status_manager.mark_step_done(
                folder_name,
                "difficulty_exported",
                detail={"filename": self.difficulty_filename},
            )
            if not difficulty_done:
                self.success_count += 1
                return "success"
            self.skip_count += 1
            return "skip"

        self.status_manager.mark_step_done(
            folder_name,
            "difficulty_exported",
            detail={"filename": self.difficulty_filename},
        )
        self.success_count += 1
        return "success"

    def list_difficulties(
        self,
        min_difficulty: float | None = None,
        max_difficulty: float | None = None,
    ) -> List[DifficultyEntry]:
        if (
            min_difficulty is not None
            and max_difficulty is not None
            and min_difficulty > max_difficulty
        ):
            raise ValueError("min_difficulty 不能大于 max_difficulty")

        matched_entries: List[DifficultyEntry] = []

        for folder_name in self.walker.read_folder_names():
            difficulty_value = self.read_difficulty(folder_name)

            if min_difficulty is not None and difficulty_value < min_difficulty:
                continue
            if max_difficulty is not None and difficulty_value > max_difficulty:
                continue

            matched_entries.append(
                DifficultyEntry(
                    folder_name=folder_name,
                    difficulty_value=difficulty_value,
                )
            )

        return matched_entries

class BeatmapDifficultyProcessor(DifficultyFileManager):
    """Task-aligned name for the beatmap difficulty export processor."""
