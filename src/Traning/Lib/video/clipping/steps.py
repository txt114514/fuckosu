from __future__ import annotations

from pathlib import Path

from Traning.Lib.common.batch import BatchProcessResult
from Traning.Lib.tools.ffmpeg import build_crop_video_args, run_ffmpeg


class ClipStepsMixin:
    def _crop_video_in_place(self, video_path: Path, crop_info: dict[str, int]):
        temp_output_path = video_path.with_name(f".__clip_tmp__{video_path.name}")
        if temp_output_path.exists():
            raise FileExistsError(f"临时裁剪文件已存在，请先清理: {temp_output_path}")

        try:
            run_ffmpeg(
                build_crop_video_args(
                    video_path,
                    temp_output_path,
                    crop_left=crop_info["crop_left"],
                    crop_top=crop_info["crop_top"],
                    crop_width=crop_info["crop_width"],
                    crop_height=crop_info["crop_height"],
                )
            )
            temp_output_path.replace(video_path)
        finally:
            if temp_output_path.exists():
                temp_output_path.unlink()

    def process_one(
        self,
        folder_name: str,
        overwrite: bool = False,
    ) -> BatchProcessResult:
        if not self._ensure_folder_ready(folder_name, overwrite):
            return "skip"

        video_path = self.store.get_file_path(folder_name, self.output_filename)
        if not video_path.is_file():
            raise FileNotFoundError(f"待裁剪视频不存在: {video_path}")

        video_width, video_height, crop_info = self._validate_crop_bounds(video_path)
        self._mark_cropping(folder_name, video_path, crop_info)
        self._crop_video_in_place(video_path, crop_info)
        self._mark_done(folder_name, video_path, video_width, video_height, crop_info)
        return "success"
