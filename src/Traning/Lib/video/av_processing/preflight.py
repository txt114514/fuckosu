from __future__ import annotations

from pathlib import Path

import numpy as np

from Traning.Lib.common.failures import failure_detail
from Traning.Lib.common.pathspec import filter_files


class AVPreflightMixin:
    def _validate_config(self, config):
        if config.sample_rate <= 0:
            raise ValueError("sample_rate 必须大于 0")
        if config.envelope_hz <= 0:
            raise ValueError("envelope_hz 必须大于 0")
        if config.refine_hz <= 0:
            raise ValueError("refine_hz 必须大于 0")
        if config.refine_search_seconds <= 0:
            raise ValueError("refine_search_seconds 必须大于 0")
        if config.music_lowpass_hz <= 0:
            raise ValueError("music_lowpass_hz 必须大于 0")
        if config.verify_correction_window_ms <= 0:
            raise ValueError("verify_correction_window_ms 必须大于 0")
        if not np.isfinite(float(config.global_offset_ms)):
            raise ValueError("global_offset_ms 必须是有限数字")
        if not config.status_step.strip():
            raise ValueError("status_step 不能为空")

    def _ensure_status_steps_registered(self):
        registered_steps = set(self.status_manager.process_steps)
        required_registered_steps = set(self.required_steps)
        required_registered_steps.add(self.status_step)
        missing_steps = [step for step in required_registered_steps if step not in registered_steps]
        if missing_steps:
            raise ValueError(
                "配置中的 process_steps 缺少 AV 对齐所需步骤: "
                f"{', '.join(missing_steps)}"
            )

    def _resolve_source_video_path(self, folder_name: str) -> Path:
        folder_path = self.store.get_folder_path(folder_name)
        candidates = sorted(
            [
                path
                for path in filter_files(folder_path.iterdir(), self.video_file_spec)
                if path.stem == folder_name
            ],
            key=lambda path: path.name.lower(),
        )

        if not candidates:
            raise FileNotFoundError(
                f"{folder_name} 中未找到源视频，要求文件名为 {folder_name} + 视频后缀"
            )
        if len(candidates) > 1:
            names = ", ".join(path.name for path in candidates)
            raise ValueError(f"{folder_name} 中检测到多个源视频，无法确定使用哪一个: {names}")
        return candidates[0]

    def _resolve_song_audio_path(self, folder_name: str) -> Path:
        audio_path = self.store.get_file_path(folder_name, self.audio_filename)
        if not audio_path.exists():
            raise FileNotFoundError(f"{folder_name} 中缺少音频文件: {audio_path.name}")
        return audio_path

    def _resolve_verify_path(self, folder_name: str) -> Path:
        return self.store.get_file_path(folder_name, self.verify_filename)

    def _sync_output_status(self, folder_name: str) -> tuple[bool, bool]:
        output_exists = self.store.file_exists(folder_name, self.output_filename)
        step_done = self.status_manager.is_step_done(folder_name, self.status_step)

        if output_exists and not step_done:
            output_video_path = self.store.get_file_path(folder_name, self.output_filename)
            self.status_manager.mark_step_done(
                folder_name,
                self.status_step,
                detail={
                    "stage": "auto_synced",
                    "output_video_path": str(output_video_path),
                },
            )
            step_done = True

        if not output_exists and step_done:
            self.status_manager.mark_step_pending(
                folder_name,
                self.status_step,
                detail=failure_detail(
                    "状态显示已完成 AV 对齐，但输出文件不存在",
                    self._sync_output_status,
                ),
            )
            step_done = False

        return output_exists, step_done

    def _ensure_required_steps_done(self, folder_name: str):
        pending_steps = [
            step
            for step in self.required_steps
            if not self.status_manager.is_step_done(folder_name, step)
        ]
        if pending_steps:
            raise ValueError(
                "AV 对齐前置步骤未完成: "
                f"{', '.join(pending_steps)}"
            )
