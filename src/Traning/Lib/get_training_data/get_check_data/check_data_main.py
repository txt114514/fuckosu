from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from Traning.Lib.get_training_data.config_loader import (
    build_from_get_check_data_config_or_default,
)
from Traning.Lib.get_training_data.get_check_data.export_verify import VerifyExporter
from Traning.Lib.get_training_data.get_check_data.get_files import OsuOszProcessor
from Traning.Lib.traning_package_manager.difficulty_manager import (
    DifficultyEntry,
    DifficultyFileManager,
)
from Traning.Lib.traning_package_manager.files_manager import BeatmapFolderStore
from Traning.Lib.traning_package_manager.package_update import PackageUpdater


DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_EXPORT_DIR = DEFAULT_REPO_ROOT / "osu-lazer" / "exports"
DEFAULT_TARGET_ROOT = DEFAULT_REPO_ROOT / "training_package" / "match-completed_package"
DEFAULT_KEYWORD = "normal"
DEFAULT_ORDER_FILENAME = "order.txt"
DEFAULT_AUDIO_FILENAME = "audio.mp3"
DEFAULT_VERIFY_FILENAME = "verify.txt"
DEFAULT_DIFFICULTY_FILENAME = "difficulty.txt"
DEFAULT_VERIFY_FAILED_FILENAME = "verify_failed.txt"
DEFAULT_DIFFICULTY_FAILED_FILENAME = "difficulty_failed.txt"

# 默认值保留在当前文件；config.json 里的合法参数只用于覆盖这些默认值。

class CheckDataPipeline:
    def __init__(
        self,
        export_dir: str = str(DEFAULT_EXPORT_DIR),
        target_root: str = str(DEFAULT_TARGET_ROOT),
        keyword: str = DEFAULT_KEYWORD,
        order_filename: str = DEFAULT_ORDER_FILENAME,
        audio_filename: str = DEFAULT_AUDIO_FILENAME,
        verify_filename: str = DEFAULT_VERIFY_FILENAME,
        difficulty_filename: str = DEFAULT_DIFFICULTY_FILENAME,
        verify_failed_filename: str = DEFAULT_VERIFY_FAILED_FILENAME,
        difficulty_failed_filename: str = DEFAULT_DIFFICULTY_FAILED_FILENAME,
    ):
        self.export_dir = export_dir
        self.target_root = target_root
        self.keyword = keyword
        self.order_filename = order_filename
        self.audio_filename = audio_filename
        self.verify_filename = verify_filename
        self.difficulty_filename = difficulty_filename
        self.verify_failed_filename = verify_failed_filename
        self.difficulty_failed_filename = difficulty_failed_filename
        self.updater = PackageUpdater(
            target_root=self.target_root,
            order_filename=self.order_filename,
        )
        self.store = BeatmapFolderStore(
            target_root=self.target_root,
            order_filename=self.order_filename,
        )
        self.walker = self.store.walker

    def _build_verify_exporter(self) -> VerifyExporter:
        return VerifyExporter(
            walker=self.walker,
            store=self.store,
            verify_filename=self.verify_filename,
            failed_filename=self.verify_failed_filename,
        )

    def _build_difficulty_manager(self) -> DifficultyFileManager:
        return DifficultyFileManager(
            walker=self.walker,
            store=self.store,
            difficulty_filename=self.difficulty_filename,
            failed_filename=self.difficulty_failed_filename,
        )

    def run(
        self,
        overwrite: bool = False,
        run_get_files: bool = True,
        run_verify_export: bool = True,
        run_difficulty_export: bool = True,
    ):
        if run_get_files:
            print("[阶段] 导入目标 .osu 文件")
            processor = OsuOszProcessor(
                export_dir=self.export_dir,
                target_root=self.target_root,
                keyword=self.keyword,
                order_filename=self.order_filename,
                audio_filename=self.audio_filename,
            )
            processor.run()

        if run_verify_export:
            print()
            print("[阶段] 导出 verify.txt")
            self._build_verify_exporter().run(overwrite=overwrite)

        if run_difficulty_export:
            print()
            print("[阶段] 导出 difficulty.txt")
            self._build_difficulty_manager().run(overwrite=overwrite)

    def list_difficulties(
        self,
        min_difficulty: float | None = None,
        max_difficulty: float | None = None,
    ) -> list[DifficultyEntry]:
        return self._build_difficulty_manager().list_difficulties(
            min_difficulty=min_difficulty,
            max_difficulty=max_difficulty,
        )


def main():
    pipeline = build_from_get_check_data_config_or_default(
        CheckDataPipeline,
        default_builder=CheckDataPipeline,
    )
    pipeline.run(
        overwrite=False,
        run_get_files=True,
        run_verify_export=True,
        run_difficulty_export=True,
    )


if __name__ == "__main__":
    main()
