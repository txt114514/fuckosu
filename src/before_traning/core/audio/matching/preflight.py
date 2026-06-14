from __future__ import annotations

from pathlib import Path

from before_traning.Lib.common.failures import failure_detail
from before_traning.Lib.common.pathspec import filter_files, matches_name


class AudioMatchPreflightMixin:
    def _folder_has_video(self, folder_name: str) -> bool:
        folder_path = self.store.get_folder_path(folder_name)
        return any(
            child.is_file() and matches_name(self.video_file_spec, child)
            for child in folder_path.iterdir()
        )

    def _sync_video_matched_status(self, folder_name: str):
        self.status_manager.ensure_status_file(folder_name)
        has_video = self._folder_has_video(folder_name)
        is_done = self.status_manager.is_step_done(
            folder_name,
            self.match_status_step,
        )
        folder_path = self.store.get_folder_path(folder_name)

        if has_video and not is_done:
            self.status_manager.mark_step_done(
                folder_name,
                self.match_status_step,
                detail={
                    "folder": str(folder_path),
                    "match_strategy": "existing_file",
                },
            )
            return

        if not has_video and is_done:
            self.status_manager.mark_step_pending(
                folder_name,
                self.match_status_step,
                detail=failure_detail(
                    "状态显示已匹配视频，但文件夹中未找到视频文件",
                    self._sync_video_matched_status,
                ),
            )

    def _pending_folder_names(self) -> list[str]:
        pending_names: list[str] = []
        for folder_name in self.walker.read_folder_names():
            if not self.store.folder_exists(folder_name):
                raise FileNotFoundError(f"目标文件夹不存在: {self.store.get_folder_path(folder_name)}")

            self._sync_video_matched_status(folder_name)
            if self._folder_has_video(folder_name):
                continue
            if not self.store.file_exists(folder_name, self.audio_filename):
                raise FileNotFoundError(
                    f"{folder_name} 中缺少实验匹配所需音频文件: {self.audio_filename}"
                )
            pending_names.append(folder_name)

        return pending_names

    def _candidate_folder_names(self, *, include_existing_video: bool) -> list[str]:
        pending_names = self._pending_folder_names()
        if pending_names or not include_existing_video:
            return pending_names

        return [
            folder_name
            for folder_name in self.walker.read_folder_names()
            if self.store.folder_exists(folder_name)
            and self.store.file_exists(folder_name, self.audio_filename)
        ]

    def _candidate_videos(self, *, allow_fallback: bool) -> list[Path]:
        if not self.video_root.exists():
            raise FileNotFoundError(f"视频目录不存在: {self.video_root}")

        videos = filter_files(self.video_root.iterdir(), self.video_file_spec)
        if videos:
            return sorted(videos, key=lambda path: path.name.lower())

        if not allow_fallback:
            raise ValueError(f"{self.video_root} 中没有可供实验匹配的视频文件")

        # 如果 video_root 里没有待匹配视频，则退回到已导入到谱面目录中的源视频，方便做本地实验。
        fallback_videos: list[Path] = []
        for folder_name in self.walker.read_folder_names():
            for suffix in sorted(self.video_suffixes):
                candidate = self.target_root / folder_name / f"{folder_name}{suffix}"
                if candidate.is_file():
                    fallback_videos.append(candidate)
                    break
        if not fallback_videos:
            raise ValueError("未找到可用于实验的视频文件")
        return fallback_videos
