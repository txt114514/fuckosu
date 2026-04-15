from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from Traning.Lib.get_training_data.config_loader import CONFIG_PATH, load_check_data_config
from Traning.Lib.get_training_data.get_check_data.export_verify import VerifyExporter
from Traning.Lib.get_training_data.get_check_data.get_files import (
    DEFAULT_EXPORT_DIR,
    DEFAULT_TARGET_ROOT,
    OsuOszProcessor,
)
from Traning.Lib.traning_package_manager.difficulty_manager import (
    DifficultyEntry,
    DifficultyFileManager,
)
from Traning.Lib.traning_package_manager.files_manager import BeatmapFolderStore
from Traning.Lib.traning_package_manager.order_walker import OrderFolderWalker
from Traning.Lib.traning_package_manager.package_update import PackageUpdater


class CheckDataPipeline:
    def __init__(
        self,
        export_dir: str = str(DEFAULT_EXPORT_DIR),
        target_root: str = str(DEFAULT_TARGET_ROOT),
        keyword: str = "normal",
        order_filename: str = "order.txt",
        verify_filename: str = "verify.txt",
        difficulty_filename: str = "difficulty.txt",
        verify_failed_filename: str = "verify_failed.txt",
        difficulty_failed_filename: str = "difficulty_failed.txt",
    ):
        self.export_dir = export_dir
        self.target_root = target_root
        self.keyword = keyword
        self.order_filename = order_filename
        self.verify_filename = verify_filename
        self.difficulty_filename = difficulty_filename
        self.verify_failed_filename = verify_failed_filename
        self.difficulty_failed_filename = difficulty_failed_filename

    def _ensure_workspace(self):
        PackageUpdater(
            target_root=self.target_root,
            order_filename=self.order_filename,
        )

    def _build_walker(self) -> OrderFolderWalker:
        self._ensure_workspace()
        return OrderFolderWalker(
            target_root=self.target_root,
            order_filename=self.order_filename,
        )

    def _build_store(self) -> BeatmapFolderStore:
        self._ensure_workspace()
        return BeatmapFolderStore(
            target_root=self.target_root,
            order_filename=self.order_filename,
        )

    def _build_verify_exporter(self) -> VerifyExporter:
        return VerifyExporter(
            walker=self._build_walker(),
            store=self._build_store(),
            verify_filename=self.verify_filename,
            failed_filename=self.verify_failed_filename,
        )

    def _build_difficulty_manager(self) -> DifficultyFileManager:
        return DifficultyFileManager(
            walker=self._build_walker(),
            store=self._build_store(),
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

    @classmethod
    def from_config(cls, config_path: Path | None = None) -> "CheckDataPipeline":
        config = load_check_data_config(config_path)
        return cls(**config)

    @classmethod
    def from_config_or_default(
        cls,
        config_path: Path | None = None,
    ) -> "CheckDataPipeline":
        try:
            return cls.from_config(config_path)
        except Exception as e:
            fallback_path = config_path or CONFIG_PATH
            print(
                f"\033[31m[error] {fallback_path} 读取失败，改用默认参数: {e} "
                f"config.json参数配置不合法\033[0m"
            )
            return cls()


def main():
    pipeline = CheckDataPipeline.from_config_or_default()
    pipeline.run(
        overwrite=False,
        run_get_files=True,
        run_verify_export=True,
        run_difficulty_export=True,
    )


if __name__ == "__main__":
    main()
