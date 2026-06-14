from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from before_traning.conf import Settings
from before_traning.conf.legacy_config import assign_group, settings_namespace
from before_traning.Lib.beatmap.manifest import ManifestFolderWalker
from before_traning.Lib.common.failures import exception_detail, failure_detail
from before_traning.Lib.common.pathspec import filter_files, matches_name, suffix_spec
from before_traning.conf.defaults import DEFAULT_SETTINGS as DEFAULTS
from before_traning.state.process_status import ProcessStatusManager


VIDEO_TIME_PATTERN = re.compile(
    r"^osu_(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})_"
    r"(?P<hour>\d{2})-(?P<minute>\d{2})-(?P<second>\d{2})$"
)


class VideoPackageRenamer:
    def __init__(
        self,
        settings: Settings = DEFAULTS,
        status_manager: ProcessStatusManager | None = None,
        **overrides: object,
    ):
        if not isinstance(settings, Settings):
            overrides = {"video_root": settings, **overrides}
            settings = DEFAULTS

        config = settings_namespace(settings, processor="video_package", overrides=overrides)
        assign_group(self, config, "video_package")
        self.video_root = Path(self.video_root)
        self.target_root = Path(self.target_root)
        self.video_suffixes = {suffix.lower() for suffix in config.video_suffixes}
        self.video_file_spec = suffix_spec(self.video_suffixes)
        self.walker = ManifestFolderWalker(
            target_root=str(self.target_root),
            manifest_filename=self.manifest_filename,
        )
        self.status_manager = status_manager or ProcessStatusManager(
            target_root=str(self.target_root),
            manifest_filename=self.manifest_filename,
        )

    def _folder_has_video(self, folder_path: Path) -> bool:
        return any(
            child.is_file() and matches_name(self.video_file_spec, child)
            for child in folder_path.iterdir()
        )

    def _parse_video_time(self, path: Path) -> datetime:
        match = VIDEO_TIME_PATTERN.fullmatch(path.stem)
        if match is None:
            raise ValueError(
                "视频文件名格式非法，必须严格使用: osu_YYYY-MM-DD_HH-MM-SS，例如 "
                f"osu_2026-04-16_19-54-54{path.suffix or '.mkv'}；当前文件: {path.name}"
            )

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

        video_files = filter_files(self.video_root.iterdir(), self.video_file_spec)
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
                    detail=failure_detail(
                        "状态显示已匹配视频，但文件夹中未找到视频文件",
                        self._pending_folder_names,
                    ),
                )
            pending_names.append(folder_name)

        if not pending_names:
            raise ValueError("manifest 中的文件夹都已经存在视频文件，无需继续处理")

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
        except Exception as error:
            for folder_name, destination_path, original_path in reversed(completed_plan):
                if destination_path.exists():
                    destination_path.rename(original_path)
                    self.status_manager.mark_step_pending(
                        folder_name,
                        "video_matched",
                        detail=exception_detail(
                            error,
                            recovery="视频移动过程中发生异常，已回滚",
                        ),
                    )
            for temp_path, original_path in temp_plan:
                if temp_path.exists():
                    temp_path.rename(original_path)
            raise

        print()
        print(f"处理完成：共重命名 {len(plan)} 个视频")


class VideoMatchRenamer(VideoPackageRenamer):
    """Task-aligned name for sequence-based video matching."""
