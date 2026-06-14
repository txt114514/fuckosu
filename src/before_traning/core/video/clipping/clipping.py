from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from loguru import logger

from before_traning.Lib.beatmap.folder_store import BeatmapFolderStore
from before_traning.Lib.common.batch import FolderBatchProcessor
from before_traning.Lib.common.failures import format_exception
from before_traning.conf.defaults import DEFAULT_SETTINGS as DEFAULTS
from before_traning.Lib.video.clipping.geometry import ClipGeometryMixin
from before_traning.core.video.clipping.preflight import ClipPreflightMixin
from before_traning.core.video.clipping.steps import ClipStepsMixin
from before_traning.core.video.clipping.wrapup import ClipWrapUpMixin
from before_traning.conf import Settings, load_settings
from before_traning.conf.legacy_config import assign_group, group_values, settings_namespace
from before_traning.state.process_status import ProcessStatusManager


def build_fixed_region_video_crop_processor_from_config_or_default(
    config_path: Path | None = None,
) -> "FixedRegionVideoCropProcessor":
    try:
        settings = load_settings(config_path)
    except Exception as e:
        logger.error(
            "{} 读取失败，改用默认参数: {}",
            config_path or "默认配置",
            format_exception(e),
        )
        settings = DEFAULTS
    return FixedRegionVideoCropProcessor(settings)


class FixedRegionVideoCropProcessor(
    ClipWrapUpMixin,
    ClipStepsMixin,
    ClipGeometryMixin,
    ClipPreflightMixin,
    FolderBatchProcessor,
):
    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        status_manager: ProcessStatusManager | None = None,
    ) -> "FixedRegionVideoCropProcessor":
        return cls(settings, status_manager=status_manager)

    def __init__(
        self,
        settings: Settings = DEFAULTS,
        status_manager: ProcessStatusManager | None = None,
        **overrides: object,
    ):
        if not isinstance(settings, Settings):
            overrides = {"target_root": settings, **overrides}
            settings = DEFAULTS

        config = settings_namespace(settings, processor="clip", overrides=overrides)
        (
            crop_reference_width,
            crop_reference_height,
            crop_left,
            crop_top,
            crop_right,
            crop_bottom,
        ) = group_values(config, "clip_crop")

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
        if not config.status_step.strip():
            raise ValueError("status_step 不能为空")

        self.target_root = Path(config.target_root)
        self.manifest_filename = config.manifest_filename
        self.output_filename = config.output_filename
        super().__init__()
        self.status_step = config.status_step.strip()
        self.required_steps = tuple(
            step.strip()
            for step in config.required_steps
            if step.strip()
        )
        assign_group(self, config, "clip_crop")
        self.reference_crop_width = self.crop_right - self.crop_left
        self.reference_crop_height = self.crop_bottom - self.crop_top
        self.store = BeatmapFolderStore(
            target_root=str(self.target_root),
            manifest_filename=self.manifest_filename,
        )
        self.walker = self.store.walker
        self.status_manager = status_manager or ProcessStatusManager(
            target_root=str(self.target_root),
            manifest_filename=self.manifest_filename,
        )

        self._ensure_status_steps_registered()


class VideoClipProcessor(FixedRegionVideoCropProcessor):
    """Task-aligned name for the fixed-region video clip processor."""


def main():
    processor = build_fixed_region_video_crop_processor_from_config_or_default()
    processor.run(overwrite=False)


if __name__ == "__main__":
    main()
