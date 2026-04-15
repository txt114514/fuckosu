from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from Traning.Lib.get_training_data.config_loader import (
    CONFIG_PATH,
    CheckDataConfigError,
    load_check_data_config,
)
from Traning.Lib.traning_package_manager.order_walker import OrderFolderWalker
from Traning.Lib.traning_package_manager.process_status_manager import (
    PROCESS_STEPS,
    ProcessStatusManager,
)

DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_TARGET_ROOT = DEFAULT_REPO_ROOT / "training_package" / "match-completed_package"
VIDEO_SUFFIXES = {".mp4", ".webm", ".mkv", ".avi", ".mov"}


class VideoInitChecker:
    def __init__(
        self,
        target_root: str = str(DEFAULT_TARGET_ROOT),
        order_filename: str = "order.txt",
    ):
        self.target_root = Path(target_root)
        self.order_filename = order_filename
        self.status_manager = ProcessStatusManager(
            target_root=str(self.target_root),
            order_filename=self.order_filename,
        )

    def _build_walker(self) -> OrderFolderWalker:
        return OrderFolderWalker(
            target_root=str(self.target_root),
            order_filename=self.order_filename,
        )

    def check_order_file(self) -> list[str]:
        walker = self._build_walker()
        folder_names = walker.read_folder_names()
        if not folder_names:
            raise ValueError(f"{walker.order_file} 为空，无法初始化视频流程")
        return folder_names

    def check_folders_after_check_data(self) -> list[Path]:
        folder_names = self.check_order_file()
        missing_folders: list[Path] = []

        for folder_name in folder_names:
            folder_path = self.target_root / folder_name
            if not folder_path.exists() or not folder_path.is_dir():
                missing_folders.append(folder_path)

        if missing_folders:
            missing_text = "\n".join(str(path) for path in missing_folders)
            raise FileNotFoundError(
                "检测到 check_data_main.py 之后应存在的文件夹缺失:\n"
                f"{missing_text}"
            )

        return [self.target_root / folder_name for folder_name in folder_names]

    def _folder_has_video(self, folder_path: Path) -> bool:
        return any(
            child.is_file() and child.suffix.lower() in VIDEO_SUFFIXES
            for child in folder_path.iterdir()
        )

    def check_video_progress(self) -> tuple[list[Path], list[Path]]:
        folder_paths = self.check_folders_after_check_data()
        folders_with_video: list[Path] = []
        folders_without_video: list[Path] = []

        for folder_path in folder_paths:
            if self._folder_has_video(folder_path):
                folders_with_video.append(folder_path)
            else:
                folders_without_video.append(folder_path)

        return folders_with_video, folders_without_video

    def _sync_video_matched_status(self, folder_name: str, folder_path: Path):
        has_video = self._folder_has_video(folder_path)
        is_done = self.status_manager.is_step_done(folder_name, "video_matched")

        if has_video and not is_done:
            self.status_manager.mark_step_done(
                folder_name,
                "video_matched",
                detail={"folder": str(folder_path)},
            )
            return

        if not has_video and is_done:
            self.status_manager.mark_step_pending(
                folder_name,
                "video_matched",
                detail={"error": "状态显示已匹配视频，但文件夹中未找到视频文件"},
            )

    def check_process_status(self) -> dict[str, int]:
        folder_names = self.check_order_file()
        counts = {step: 0 for step in PROCESS_STEPS}

        for folder_name in folder_names:
            self.status_manager.ensure_status_file(folder_name)
            self._sync_video_matched_status(
                folder_name,
                self.target_root / folder_name,
            )
            summary = self.status_manager.get_steps_summary(folder_name)
            for step, done in summary.items():
                if done:
                    counts[step] += 1

        return counts

    def run(self):
        folder_names = self.check_order_file()
        folder_paths = self.check_folders_after_check_data()
        folders_with_video, folders_without_video = self.check_video_progress()
        status_counts = self.check_process_status()

        print(f"[完成] order.txt 检查通过，共 {len(folder_names)} 项")
        print(f"[完成] 文件夹检查通过，共 {len(folder_paths)} 个")
        print(f"[完成] 已有视频文件夹 {len(folders_with_video)} 个")
        print(f"[完成] 待处理无视频文件夹 {len(folders_without_video)} 个")
        for step in PROCESS_STEPS:
            print(f"[完成] 状态 {step} 已完成 {status_counts[step]} 个")

    @classmethod
    def from_config(cls, config_path: Path | None = None) -> "VideoInitChecker":
        config = load_check_data_config(config_path)
        return cls(target_root=config["target_root"])

    @classmethod
    def from_config_or_default(
        cls,
        config_path: Path | None = None,
    ) -> "VideoInitChecker":
        try:
            return cls.from_config(config_path)
        except CheckDataConfigError as e:
            fallback_path = config_path or CONFIG_PATH
            print(
                f"\033[31m[error] {fallback_path} 读取失败，改用默认参数: {e} "
                f"config.json参数配置不合法\033[0m"
            )
            return cls()


def main():
    checker = VideoInitChecker.from_config_or_default()
    checker.run()


if __name__ == "__main__":
    main()
