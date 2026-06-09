from __future__ import annotations

from pathlib import Path

from Traning.Lib.tools.ffmpeg import get_video_size as probe_video_size


class ClipGeometryMixin:
    def get_video_size(self, video_path: Path) -> tuple[int, int]:
        return probe_video_size(video_path)

    def _scale_crop_coordinate(
        self,
        value: int,
        reference_size: int,
        video_size: int,
    ) -> int:
        return int(value * video_size / reference_size + 0.5)

    def _resolve_scaled_crop(
        self,
        video_width: int,
        video_height: int,
    ) -> dict[str, int]:
        crop_left = self._scale_crop_coordinate(
            self.crop_left,
            self.crop_reference_width,
            video_width,
        )
        crop_top = self._scale_crop_coordinate(
            self.crop_top,
            self.crop_reference_height,
            video_height,
        )
        crop_right = self._scale_crop_coordinate(
            self.crop_right,
            self.crop_reference_width,
            video_width,
        )
        crop_bottom = self._scale_crop_coordinate(
            self.crop_bottom,
            self.crop_reference_height,
            video_height,
        )

        if crop_right <= crop_left or crop_bottom <= crop_top:
            raise ValueError(
                "按比例换算后的裁剪矩形非法: "
                f"video={video_width}x{video_height}, "
                f"crop=({crop_left}, {crop_top}, {crop_right}, {crop_bottom})"
            )

        crop_width = crop_right - crop_left
        crop_height = crop_bottom - crop_top

        # yuv420p 编码要求偶数尺寸，提前对齐，避免 ffmpeg 隐式调整后状态记录不一致。
        if crop_width % 2 != 0:
            crop_width -= 1
        if crop_height % 2 != 0:
            crop_height -= 1
        if crop_width <= 0 or crop_height <= 0:
            raise ValueError(
                "按比例换算后的裁剪尺寸非法: "
                f"video={video_width}x{video_height}, "
                f"crop_width={crop_width}, crop_height={crop_height}"
            )

        return {
            "crop_left": crop_left,
            "crop_top": crop_top,
            "crop_right": crop_left + crop_width,
            "crop_bottom": crop_top + crop_height,
            "crop_width": crop_width,
            "crop_height": crop_height,
        }

    def _validate_crop_bounds(self, video_path: Path) -> tuple[int, int, dict[str, int]]:
        video_width, video_height = self.get_video_size(video_path)
        crop_info = self._resolve_scaled_crop(video_width, video_height)

        if crop_info["crop_right"] > video_width or crop_info["crop_bottom"] > video_height:
            raise ValueError(
                "裁剪区域超出视频边界: "
                f"video={video_width}x{video_height}, "
                f"scaled_crop=({crop_info['crop_left']}, {crop_info['crop_top']}, "
                f"{crop_info['crop_right']}, {crop_info['crop_bottom']}), "
                f"reference={self.crop_reference_width}x{self.crop_reference_height}, "
                f"reference_crop=({self.crop_left}, {self.crop_top}, "
                f"{self.crop_right}, {self.crop_bottom})"
            )

        return video_width, video_height, crop_info

    def describe_crop_for_video(self, video_path: Path) -> tuple[int, int, dict[str, int]]:
        return self._validate_crop_bounds(video_path)
