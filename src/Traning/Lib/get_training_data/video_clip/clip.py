from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from Traning.Lib.function_tools.functions_process_tool import (
    BatchProcessResult,
    FolderBatchProcessor,
    read_config_values,
)
from Traning.Lib.function_tools.video_process_tool import (
    get_video_size as probe_video_size,
    run_ffmpeg,
)
from Traning.Lib.get_training_data.config_loader import (
    CLIP_PROCESSOR_CONFIG_SPECS,
    ConfigReader,
    build_from_config_or_default,
)
from Traning.Lib.get_training_data.process_status_manager import ProcessStatusManager
from Traning.Lib.traning_package_manager.files_manager import BeatmapFolderStore

DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_TARGET_ROOT = DEFAULT_REPO_ROOT / "training_package" / "match-completed_package"
DEFAULT_ORDER_FILENAME = "order.txt"
DEFAULT_OUTPUT_FILENAME = "video_processed.mp4"
DEFAULT_FAILED_FILENAME = "clip_failed.txt"
DEFAULT_STATUS_STEP = "video_processed"
DEFAULT_REQUIRED_STEPS = ("av_corresponded",)
DEFAULT_CROP_REFERENCE_WIDTH = 2048
DEFAULT_CROP_REFERENCE_HEIGHT = 1152
DEFAULT_CROP_LEFT = 186
DEFAULT_CROP_TOP = 178
DEFAULT_CROP_RIGHT = 1768
DEFAULT_CROP_BOTTOM = 1080

# 默认值保留在当前文件；config.json 里的合法参数只用于覆盖这些默认值。
# clip.py 当前使用基于 2048x1152 参考图的裁剪框：(186, 178) -> (1768, 1080)。


def _load_fixed_region_video_crop_processor_config(
    config: ConfigReader,
) -> dict[str, object]:
    # clip 阶段读取输出视频名、状态名、前置步骤和参考裁剪框。
    return read_config_values(config, CLIP_PROCESSOR_CONFIG_SPECS)

def build_fixed_region_video_crop_processor_from_config_or_default(
    config_path: Path | None = None,
) -> "FixedRegionVideoCropProcessor":
    return build_from_config_or_default(
        FixedRegionVideoCropProcessor,
        [_load_fixed_region_video_crop_processor_config],
        config_path=config_path,
        default_builder=FixedRegionVideoCropProcessor,
    )


class FixedRegionVideoCropProcessor(FolderBatchProcessor):
    def __init__(
        self,
        target_root: str = str(DEFAULT_TARGET_ROOT),
        order_filename: str = DEFAULT_ORDER_FILENAME,
        output_filename: str = DEFAULT_OUTPUT_FILENAME,
        failed_filename: str = DEFAULT_FAILED_FILENAME,
        status_step: str = DEFAULT_STATUS_STEP,
        required_steps: Iterable[str] = DEFAULT_REQUIRED_STEPS,
        crop_reference_width: int = DEFAULT_CROP_REFERENCE_WIDTH,
        crop_reference_height: int = DEFAULT_CROP_REFERENCE_HEIGHT,
        crop_left: int = DEFAULT_CROP_LEFT,
        crop_top: int = DEFAULT_CROP_TOP,
        crop_right: int = DEFAULT_CROP_RIGHT,
        crop_bottom: int = DEFAULT_CROP_BOTTOM,
        status_manager: ProcessStatusManager | None = None,
    ):
        if crop_reference_width <= 0 or crop_reference_height <= 0:
            raise ValueError("crop_reference_width 和 crop_reference_height 必须为正数")
        if crop_left < 0 or crop_top < 0:
            raise ValueError("crop_left 和 crop_top 不能为负数")
        if crop_right <= crop_left or crop_bottom <= crop_top:
            raise ValueError("裁剪矩形非法，必须满足 right > left 且 bottom > top")
        if crop_right > crop_reference_width or crop_bottom > crop_reference_height:
            raise ValueError(
                "参考裁剪区域超出参考分辨率边界: "
                f"reference={crop_reference_width}x{crop_reference_height}, "
                f"crop=({crop_left}, {crop_top}, {crop_right}, {crop_bottom})"
            )
        if not status_step.strip():
            raise ValueError("status_step 不能为空")

        self.target_root = Path(target_root)
        self.order_filename = order_filename
        self.output_filename = output_filename
        super().__init__(failed_filename)
        self.status_step = status_step.strip()
        self.required_steps = tuple(step.strip() for step in required_steps if step.strip())
        self.crop_reference_width = crop_reference_width
        self.crop_reference_height = crop_reference_height
        self.crop_left = crop_left
        self.crop_top = crop_top
        self.crop_right = crop_right
        self.crop_bottom = crop_bottom
        self.reference_crop_width = crop_right - crop_left
        self.reference_crop_height = crop_bottom - crop_top
        self.store = BeatmapFolderStore(
            target_root=str(self.target_root),
            order_filename=self.order_filename,
        )
        self.walker = self.store.walker
        self.status_manager = status_manager or ProcessStatusManager(
            target_root=str(self.target_root),
            order_filename=self.order_filename,
        )

        self._ensure_status_steps_registered()

    def _ensure_status_steps_registered(self):
        registered_steps = set(self.status_manager.process_steps)
        required_registered_steps = set(self.required_steps)
        required_registered_steps.add(self.status_step)
        missing_steps = [step for step in required_registered_steps if step not in registered_steps]
        if missing_steps:
            raise ValueError(
                "配置中的 process_steps 缺少 clip 阶段所需步骤: "
                f"{', '.join(missing_steps)}"
            )

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

    def _crop_video_in_place(self, video_path: Path, crop_info: dict[str, int]):
        temp_output_path = video_path.with_name(f".__clip_tmp__{video_path.name}")
        if temp_output_path.exists():
            raise FileExistsError(f"临时裁剪文件已存在，请先清理: {temp_output_path}")

        try:
            run_ffmpeg(
                [
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(video_path),
                    "-vf",
                    (
                        f"crop={crop_info['crop_width']}:{crop_info['crop_height']}:"
                        f"{crop_info['crop_left']}:{crop_info['crop_top']}"
                    ),
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "18",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    "-movflags",
                    "+faststart",
                    str(temp_output_path),
                ]
            )
            temp_output_path.replace(video_path)
        finally:
            if temp_output_path.exists():
                temp_output_path.unlink()

    def progress_message(self, index: int, total: int, folder_name: str) -> str | None:
        return f"[进度] {index}/{total} {folder_name}"

    def process_one(
        self,
        folder_name: str,
        overwrite: bool = False,
    ) -> BatchProcessResult:
        if not self.store.folder_exists(folder_name):
            return "skip"

        self.status_manager.ensure_status_file(folder_name)

        if not overwrite and self.status_manager.is_step_done(folder_name, self.status_step):
            return "skip"

        missing_steps = [
            step for step in self.required_steps
            if not self.status_manager.is_step_done(folder_name, step)
        ]
        if missing_steps:
            raise ValueError(
                f"缺少前置步骤: {', '.join(missing_steps)}"
            )

        video_path = self.store.get_file_path(folder_name, self.output_filename)
        if not video_path.is_file():
            raise FileNotFoundError(f"待裁剪视频不存在: {video_path}")

        video_width, video_height, crop_info = self._validate_crop_bounds(video_path)
        self.status_manager.mark_step_pending(
            folder_name,
            self.status_step,
            detail={
                "stage": "cropping",
                "output_video_path": str(video_path),
                "reference_width": self.crop_reference_width,
                "reference_height": self.crop_reference_height,
                "reference_crop_left": self.crop_left,
                "reference_crop_top": self.crop_top,
                "reference_crop_right": self.crop_right,
                "reference_crop_bottom": self.crop_bottom,
                **crop_info,
            },
        )
        self._crop_video_in_place(video_path, crop_info)
        self.status_manager.mark_step_done(
            folder_name,
            self.status_step,
            detail={
                "stage": "done",
                "output_video_path": str(video_path),
                "video_width": video_width,
                "video_height": video_height,
                "reference_width": self.crop_reference_width,
                "reference_height": self.crop_reference_height,
                "reference_crop_left": self.crop_left,
                "reference_crop_top": self.crop_top,
                "reference_crop_right": self.crop_right,
                "reference_crop_bottom": self.crop_bottom,
                **crop_info,
            },
        )
        return "success"

    def handle_failure(self, folder_name: str, error: Exception):
        if self.store.folder_exists(folder_name):
            self.status_manager.ensure_status_file(folder_name)
            self.status_manager.mark_step_pending(
                folder_name,
                self.status_step,
                detail={
                    "stage": "failed",
                    "error": str(error),
                    "reference_width": self.crop_reference_width,
                    "reference_height": self.crop_reference_height,
                    "reference_crop_left": self.crop_left,
                    "reference_crop_top": self.crop_top,
                    "reference_crop_right": self.crop_right,
                    "reference_crop_bottom": self.crop_bottom,
                },
            )


def main():
    processor = build_fixed_region_video_crop_processor_from_config_or_default()
    processor.run(overwrite=False)


if __name__ == "__main__":
    main()
