from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from Traning.Lib.get_training_data.config_loader import (
    build_from_video_shared_config_or_default,
)
from Traning.Lib.get_training_data.process_status_manager import ProcessStatusManager
from Traning.Lib.traning_package_manager.order_walker import OrderFolderWalker


DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_TARGET_ROOT = DEFAULT_REPO_ROOT / "training_package" / "match-completed_package"
DEFAULT_ORDER_FILENAME = "order.txt"
DEFAULT_VIDEO_SUFFIXES = (".mp4", ".webm", ".mkv", ".avi", ".mov")
DEFAULT_OUTPUT_FILENAME = "video_processed.mp4"
DEFAULT_STATUS_STEP = "av_corresponded"

# 默认值保留在当前文件；config.json 里的合法参数只用于覆盖这些默认值。

class VideoInitChecker:
    def __init__(
        self,
        target_root: str = str(DEFAULT_TARGET_ROOT),
        order_filename: str = DEFAULT_ORDER_FILENAME,
        video_suffixes: Iterable[str] = DEFAULT_VIDEO_SUFFIXES,
        output_filename: str = DEFAULT_OUTPUT_FILENAME,
        status_step: str = DEFAULT_STATUS_STEP,
    ):
        self.target_root = Path(target_root)
        self.order_filename = order_filename
        self.video_suffixes = {suffix.lower() for suffix in video_suffixes}
        self.output_filename = output_filename
        self.status_step = status_step
        self.status_manager = ProcessStatusManager(
            target_root=str(self.target_root),
            order_filename=self.order_filename,
        )

    def _folder_has_video(self, folder_path: Path) -> bool:
        return any(
            child.is_file() and child.suffix.lower() in self.video_suffixes
            for child in folder_path.iterdir()
        )

    def _folder_items(self) -> list[tuple[str, Path]]:
        walker = OrderFolderWalker(
            target_root=str(self.target_root),
            order_filename=self.order_filename,
        )
        folder_names = walker.read_folder_names()
        if not folder_names:
            raise ValueError(f"{walker.order_file} 为空，无法初始化视频流程")

        items = [(folder_name, self.target_root / folder_name) for folder_name in folder_names]
        missing_paths = [folder_path for _, folder_path in items if not folder_path.is_dir()]
        if missing_paths:
            missing_text = "\n".join(str(path) for path in missing_paths)
            raise FileNotFoundError(
                "检测到 check_data_main.py 之后应存在的文件夹缺失:\n"
                f"{missing_text}"
            )

        return items

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

    def _sync_av_corresponded_status(self, folder_name: str, folder_path: Path):
        if self.status_step not in self.status_manager.process_steps:
            return

        output_path = folder_path / self.output_filename
        output_exists = output_path.is_file()
        is_done = self.status_manager.is_step_done(folder_name, self.status_step)

        if output_exists and not is_done:
            self.status_manager.mark_step_done(
                folder_name,
                self.status_step,
                detail={
                    "stage": "auto_synced",
                    "output_video_path": str(output_path),
                },
            )
            return

        if not output_exists and is_done:
            self.status_manager.mark_step_pending(
                folder_name,
                self.status_step,
                detail={"error": "状态显示已完成 AV 对齐，但未找到输出视频"},
            )

    def run(self):
        folder_items = self._folder_items()
        folders_with_video = 0
        folders_without_video = 0
        status_counts = {step: 0 for step in self.status_manager.process_steps}

        for folder_name, folder_path in folder_items:
            self.status_manager.ensure_status_file(folder_name)
            self._sync_video_matched_status(folder_name, folder_path)
            self._sync_av_corresponded_status(folder_name, folder_path)

            if self._folder_has_video(folder_path):
                folders_with_video += 1
            else:
                folders_without_video += 1

            for step, done in self.status_manager.get_steps_summary(folder_name).items():
                if done:
                    status_counts[step] += 1

        print(f"[完成] order.txt 检查通过，共 {len(folder_items)} 项")
        print(f"[完成] 文件夹检查通过，共 {len(folder_items)} 个")
        print(f"[完成] 已有视频文件夹 {folders_with_video} 个")
        print(f"[完成] 待处理无视频文件夹 {folders_without_video} 个")
        for step in self.status_manager.process_steps:
            print(f"[完成] 状态 {step} 已完成 {status_counts[step]} 个")


def main():
    checker = build_from_video_shared_config_or_default(
        VideoInitChecker,
        default_builder=VideoInitChecker,
    )
    checker.run()


if __name__ == "__main__":
    main()
