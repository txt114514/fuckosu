from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from Traning.Lib.get_training_data.config_loader import (
    build_from_check_data_config_or_default,
)
from Traning.Lib.traning_package_manager.order_walker import OrderFolderWalker
from Traning.Lib.traning_package_manager.process_status_manager import (
    ProcessStatusManager,
)

DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_VIDEO_ROOT = DEFAULT_REPO_ROOT / "training_package" / "video_package"
DEFAULT_TARGET_ROOT = DEFAULT_REPO_ROOT / "training_package" / "match-completed_package"

VIDEO_TIME_PATTERN = re.compile(
    r"(?P<year>\d{4})年(?P<month>\d{2})月(?P<day>\d{2})日\s+"
    r"(?P<hour>\d{2})时(?P<minute>\d{2})分(?P<second>\d{2})秒"
)
VIDEO_SUFFIXES = {".mp4", ".webm", ".mkv", ".avi", ".mov"}


class VideoPackageRenamer:
    def __init__(
        self,
        video_root: str = str(DEFAULT_VIDEO_ROOT),
        target_root: str = str(DEFAULT_TARGET_ROOT),
        order_filename: str = "order.txt",
        status_manager: ProcessStatusManager | None = None,
    ):
        self.video_root = Path(video_root)
        self.target_root = Path(target_root)
        self.order_filename = order_filename
        self.walker = OrderFolderWalker(
            target_root=str(self.target_root),
            order_filename=self.order_filename,
        )
        self.status_manager = status_manager or ProcessStatusManager(
            target_root=str(self.target_root),
            order_filename=self.order_filename,
        )

    def _folder_has_video(self, folder_path: Path) -> bool:
        return any(
            child.is_file() and child.suffix.lower() in VIDEO_SUFFIXES
            for child in folder_path.iterdir()
        )

    def _parse_video_time(self, path: Path) -> datetime:
        match = VIDEO_TIME_PATTERN.search(path.name)
        if match is None:
            raise ValueError(f"视频文件名中缺少可识别时间: {path.name}")

        return datetime(
            year=int(match.group("year")),
            month=int(match.group("month")),
            day=int(match.group("day")),
            hour=int(match.group("hour")),
            minute=int(match.group("minute")),
            second=int(match.group("second")),
        )

    def _list_videos_in_time_order(self) -> list[Path]:
        if not self.video_root.exists():
            raise FileNotFoundError(f"视频目录不存在: {self.video_root}")

        video_files = [
            path
            for path in self.video_root.iterdir()
            if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
        ]
        if not video_files:
            raise ValueError(f"{self.video_root} 中没有可重命名的视频文件")

        return sorted(
            video_files,
            key=lambda path: (self._parse_video_time(path), path.name.lower()),
        )

    def _pending_folder_names(self) -> list[str]:
        pending_names: list[str] = []

        for folder_name in self.walker.read_folder_names():
            destination_dir = self.target_root / folder_name
            if not destination_dir.exists():
                raise FileNotFoundError(
                    f"目标文件夹不存在，请先执行 check_data_main.py: {destination_dir}"
                )
            self.status_manager.ensure_status_file(folder_name)
            if self._folder_has_video(destination_dir):
                self.status_manager.mark_step_done(
                    folder_name,
                    "video_matched",
                    detail={"folder": str(destination_dir)},
                )
                continue
            if self.status_manager.is_step_done(folder_name, "video_matched"):
                self.status_manager.mark_step_pending(
                    folder_name,
                    "video_matched",
                    detail={"error": "状态显示已匹配视频，但文件夹中未找到视频文件"},
                )
            pending_names.append(folder_name)

        if not pending_names:
            raise ValueError("order.txt 对应文件夹都已经存在视频文件，无需继续处理")

        return pending_names

    def _build_rename_plan(self) -> list[tuple[str, Path, Path]]:
        ordered_videos = self._list_videos_in_time_order()
        pending_folder_names = self._pending_folder_names()

        if len(ordered_videos) != len(pending_folder_names):
            raise ValueError(
                "待处理视频数量与无视频文件夹数量不一致: "
                f"video={len(ordered_videos)}, pending_folder={len(pending_folder_names)}"
            )

        plan: list[tuple[str, Path, Path]] = []
        for source_path, folder_name in zip(ordered_videos, pending_folder_names):
            destination_dir = self.target_root / folder_name
            destination_path = destination_dir / f"{folder_name}{source_path.suffix}"
            plan.append((folder_name, source_path, destination_path))

        destination_paths = [destination_path for _, _, destination_path in plan]
        if len(set(destination_paths)) != len(destination_paths):
            raise ValueError("重命名目标中出现重复文件名")

        source_paths = {source_path for _, source_path, _ in plan}
        for _, _, destination_path in plan:
            if destination_path.exists() and destination_path not in source_paths:
                raise FileExistsError(f"目标文件已存在，无法覆盖: {destination_path}")

        return plan

    def run(self):
        plan = self._build_rename_plan()
        temp_plan: list[tuple[Path, Path]] = []
        completed_plan: list[tuple[str, Path, Path]] = []

        for index, (_folder_name, source_path, _destination_path) in enumerate(plan):
            temp_path = source_path.with_name(f".__rename_tmp__{index}{source_path.suffix}")
            if temp_path.exists():
                raise FileExistsError(f"临时文件已存在，请先清理: {temp_path}")
            source_path.rename(temp_path)
            temp_plan.append((temp_path, source_path))

        try:
            for (temp_path, original_path), (folder_name, _source_path, destination_path) in zip(
                temp_plan,
                plan,
            ):
                temp_path.rename(destination_path)
                completed_plan.append((folder_name, destination_path, original_path))
                self.status_manager.mark_step_done(
                    folder_name,
                    "video_matched",
                    detail={"video_path": str(destination_path)},
                )
                print(f"[完成] {original_path.name} -> {destination_path}")
        except Exception:
            for folder_name, destination_path, original_path in reversed(completed_plan):
                if destination_path.exists():
                    destination_path.rename(original_path)
                    self.status_manager.mark_step_pending(
                        folder_name,
                        "video_matched",
                        detail={"error": "视频移动过程中发生异常，已回滚"},
                    )
            for temp_path, original_path in temp_plan:
                if temp_path.exists():
                    temp_path.rename(original_path)
            raise

        print()
        print(f"处理完成：共重命名 {len(plan)} 个视频")

def main():
    renamer = build_from_check_data_config_or_default(
        lambda **config: VideoPackageRenamer(target_root=config["target_root"]),
        default_builder=VideoPackageRenamer,
    )
    renamer.run()


if __name__ == "__main__":
    main()
