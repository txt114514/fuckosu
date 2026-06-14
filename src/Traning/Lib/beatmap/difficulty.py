from __future__ import annotations

from dataclasses import dataclass
from typing import List

from Traning.Lib.beatmap.folder_store import BeatmapFolderStore
from Traning.Lib.beatmap.difficulty_batch import DifficultyBatchMixin
from Traning.Lib.beatmap.osu_metadata import read_overall_difficulty
from Traning.Lib.beatmap.manifest import ManifestFolderWalker
from Traning.conf import Settings
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
        walker: ManifestFolderWalker | None = None,
        settings: Settings = DEFAULTS,
        status_manager: ProcessStatusManager | None = None,
        **_overrides: object,
    ):
        self.store = store
        self.walker = walker or store.walker
        self.manifest = self.walker.manifest
        self.status_manager = status_manager or ProcessStatusManager(
            target_root=str(store.target_root),
            manifest_filename=store.manifest_filename,
        )

        self.success_count = 0
        self.skip_count = 0
        self.fail_count = 0

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

    def export_one(self, folder_name: str, overwrite: bool = False) -> str:
        if not self.store.folder_exists(folder_name):
            self.skip_count += 1
            return "skip"

        self.status_manager.ensure_status_file(folder_name)
        difficulty_value = self.manifest.difficulty_for(folder_name)
        status = self.status_manager.load_status(folder_name)
        difficulty_status = status["steps"]["difficulty_exported"]
        difficulty_done = bool(difficulty_status["done"])
        if not overwrite and difficulty_value is not None:
            if difficulty_done:
                expected_detail = {"difficulty_value": difficulty_value}
                if difficulty_status["detail"] != expected_detail:
                    self.status_manager.mark_step_done(
                        folder_name,
                        "difficulty_exported",
                        detail=expected_detail,
                    )
                self.skip_count += 1
                return "skip"
            self.status_manager.mark_step_done(
                folder_name,
                "difficulty_exported",
                detail={"difficulty_value": difficulty_value},
            )
            self.success_count += 1
            return "success"

        osu_files = self.store.find_osu_files(folder_name)
        if not osu_files:
            self.skip_count += 1
            return "skip"

        difficulty_value = read_overall_difficulty(osu_files[0])
        self.write_difficulty(
            folder_name=folder_name,
            difficulty_value=difficulty_value,
        )

        self.status_manager.mark_step_done(
            folder_name,
            "difficulty_exported",
            detail={"difficulty_value": difficulty_value},
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
