from __future__ import annotations

from pathlib import Path

from Traning.Lib.common.failures import exception_detail


class ClipWrapUpMixin:
    def progress_message(self, index: int, total: int, folder_name: str) -> str | None:
        return f"[进度] {index}/{total} {folder_name}"

    def _reference_detail(self) -> dict[str, int]:
        return {
            "reference_width": self.crop_reference_width,
            "reference_height": self.crop_reference_height,
            "reference_crop_left": self.crop_left,
            "reference_crop_top": self.crop_top,
            "reference_crop_right": self.crop_right,
            "reference_crop_bottom": self.crop_bottom,
        }

    def _mark_cropping(
        self,
        folder_name: str,
        video_path: Path,
        crop_info: dict[str, int],
    ):
        self.status_manager.mark_step_pending(
            folder_name,
            self.status_step,
            detail={
                "stage": "cropping",
                "output_video_path": str(video_path),
                **self._reference_detail(),
                **crop_info,
            },
        )

    def _mark_done(
        self,
        folder_name: str,
        video_path: Path,
        video_width: int,
        video_height: int,
        crop_info: dict[str, int],
    ):
        self.status_manager.mark_step_done(
            folder_name,
            self.status_step,
            detail={
                "stage": "done",
                "output_video_path": str(video_path),
                "video_width": video_width,
                "video_height": video_height,
                **self._reference_detail(),
                **crop_info,
            },
        )

    def handle_failure(self, folder_name: str, error: Exception):
        if self.store.folder_exists(folder_name):
            self.status_manager.ensure_status_file(folder_name)
            self.status_manager.mark_step_pending(
                folder_name,
                self.status_step,
                detail=exception_detail(
                    error,
                    stage="failed",
                    **self._reference_detail(),
                ),
            )
