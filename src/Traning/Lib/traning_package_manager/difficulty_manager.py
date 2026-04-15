from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from Traning.Lib.traning_package_manager.files_manager import (
    BeatmapFolderStore,
    WriteMode,
)
from Traning.Lib.traning_package_manager.order_walker import OrderFolderWalker


@dataclass(frozen=True)
class DifficultyEntry:
    folder_name: str
    difficulty_value: float


class DifficultyFileManager:
    def __init__(
        self,
        store: BeatmapFolderStore,
        walker: OrderFolderWalker | None = None,
        difficulty_filename: str = "difficulty.txt",
        failed_filename: str = "difficulty_failed.txt",
    ):
        self.store = store
        self.walker = walker or store.walker
        self.difficulty_filename = difficulty_filename
        self.failed_filename = failed_filename

        self.success_count = 0
        self.skip_count = 0
        self.fail_count = 0
        self.failed_cases: List[tuple[str, str]] = []

    def _extract_difficulty_from_osu(self, osu_path: Path) -> float:
        in_difficulty_section = False

        with osu_path.open("r", encoding="utf-8-sig") as f:
            for raw_line in f:
                line = raw_line.strip()

                if not line or line.startswith("//"):
                    continue

                if line.startswith("[") and line.endswith("]"):
                    in_difficulty_section = (line == "[Difficulty]")
                    continue

                if not in_difficulty_section or ":" not in line:
                    continue

                key, value = line.split(":", 1)
                if key.strip() == "OverallDifficulty":
                    return float(value.strip())

        raise ValueError(f"{osu_path} 缺少 OverallDifficulty")

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

        osu_files = self.store.find_osu_files(folder_name)
        if not osu_files:
            self.skip_count += 1
            return "skip"

        difficulty_value = self._extract_difficulty_from_osu(osu_files[0])
        write_mode: WriteMode = "overwrite" if overwrite else "skip_if_exists"

        result = self.write_difficulty(
            folder_name=folder_name,
            difficulty_value=difficulty_value,
            mode=write_mode,
        )

        if result == "skipped":
            self.skip_count += 1
            return "skip"

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

    def run(self, overwrite: bool = False):
        folder_names = self.walker.read_folder_names()

        for folder_name in folder_names:
            try:
                result = self.export_one(folder_name, overwrite=overwrite)
                if result == "success":
                    print(f"[完成] {folder_name}")
                else:
                    print(f"[跳过] {folder_name}")
            except Exception as e:
                self.fail_count += 1
                self.failed_cases.append((folder_name, str(e)))
                print(f"[失败] {folder_name}: {e}")

        failed_path = self.store.write_failed_report(
            self.failed_cases,
            failed_filename=self.failed_filename,
        )

        print()
        print(
            f"处理完成：成功 {self.success_count} 个，跳过 {self.skip_count} 个，失败 {self.fail_count} 个"
        )
        print(f"失败名单：{failed_path}")
