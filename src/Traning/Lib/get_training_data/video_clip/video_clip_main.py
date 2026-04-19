from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from Traning.Lib.function_tools.functions_process_tool import (
    read_config_values,
)
from Traning.Lib.get_training_data.config_loader import (
    ConfigReader,
    VIDEO_CLIP_PIPELINE_CONFIG_SPECS,
    build_from_config_or_default,
)
from Traning.Lib.get_training_data.video_clip.AV_correspondence import (
    AVCorrespondenceProcessor,
)
from Traning.Lib.get_training_data.video_clip.clip import (
    DEFAULT_CROP_BOTTOM as DEFAULT_CLIP_CROP_BOTTOM,
    DEFAULT_CROP_LEFT as DEFAULT_CLIP_CROP_LEFT,
    DEFAULT_CROP_REFERENCE_HEIGHT as DEFAULT_CLIP_CROP_REFERENCE_HEIGHT,
    DEFAULT_CROP_REFERENCE_WIDTH as DEFAULT_CLIP_CROP_REFERENCE_WIDTH,
    DEFAULT_CROP_RIGHT as DEFAULT_CLIP_CROP_RIGHT,
    DEFAULT_CROP_TOP as DEFAULT_CLIP_CROP_TOP,
    DEFAULT_FAILED_FILENAME as DEFAULT_CLIP_FAILED_FILENAME,
    DEFAULT_REQUIRED_STEPS as DEFAULT_CLIP_REQUIRED_STEPS,
    DEFAULT_STATUS_STEP as DEFAULT_CLIP_STATUS_STEP,
    FixedRegionVideoCropProcessor,
)
from Traning.Lib.get_training_data.video_clip.read_video import VideoPackageRenamer
from Traning.Lib.get_training_data.video_clip.video_init import VideoInitChecker


DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_TARGET_ROOT = DEFAULT_REPO_ROOT / "training_package" / "match-completed_package"
DEFAULT_ORDER_FILENAME = "order.txt"
DEFAULT_VIDEO_ROOT = DEFAULT_REPO_ROOT / "training_package" / "video_package"
DEFAULT_VIDEO_SUFFIXES = (".mp4", ".webm", ".mkv", ".avi", ".mov")
DEFAULT_AUDIO_FILENAME = "audio.mp3"
DEFAULT_OUTPUT_FILENAME = "video_processed.mp4"
DEFAULT_STATUS_STEP = "av_corresponded"
DEFAULT_FAILED_FILENAME = "av_correspondence_failed.txt"
DEFAULT_REQUIRED_STEPS = ("audio_imported", "video_matched")
DEFAULT_SAMPLE_RATE = 8000
DEFAULT_ENVELOPE_HZ = 100
DEFAULT_REFINE_HZ = 1000
DEFAULT_REFINE_SEARCH_SECONDS = 1.5
DEFAULT_RUN_CLIP_STAGE = True

# 默认值保留在当前文件；config.json 里的合法参数只用于覆盖这些默认值。


def _load_video_clip_pipeline_config(config: ConfigReader) -> dict[str, object]:
    # video_clip 总流程会统一读取视频目录、AV 对齐参数，以及 clip 阶段裁剪参数。
    return read_config_values(config, VIDEO_CLIP_PIPELINE_CONFIG_SPECS)


def build_video_clip_pipeline_from_config_or_default(
    config_path: Path | None = None,
) -> "VideoClipPipeline":
    return build_from_config_or_default(
        VideoClipPipeline,
        [_load_video_clip_pipeline_config],
        config_path=config_path,
        default_builder=VideoClipPipeline,
    )


class VideoClipPipeline:
    def __init__(
        self,
        target_root: str = str(DEFAULT_TARGET_ROOT),
        order_filename: str = DEFAULT_ORDER_FILENAME,
        video_root: str = str(DEFAULT_VIDEO_ROOT),
        video_suffixes: Iterable[str] = DEFAULT_VIDEO_SUFFIXES,
        audio_filename: str = DEFAULT_AUDIO_FILENAME,
        output_filename: str = DEFAULT_OUTPUT_FILENAME,
        status_step: str = DEFAULT_STATUS_STEP,
        failed_filename: str = DEFAULT_FAILED_FILENAME,
        required_steps: Iterable[str] = DEFAULT_REQUIRED_STEPS,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        envelope_hz: int = DEFAULT_ENVELOPE_HZ,
        refine_hz: int = DEFAULT_REFINE_HZ,
        refine_search_seconds: float = DEFAULT_REFINE_SEARCH_SECONDS,
        clip_failed_filename: str = DEFAULT_CLIP_FAILED_FILENAME,
        clip_status_step: str = DEFAULT_CLIP_STATUS_STEP,
        clip_required_steps: Iterable[str] = DEFAULT_CLIP_REQUIRED_STEPS,
        clip_crop_reference_width: int = DEFAULT_CLIP_CROP_REFERENCE_WIDTH,
        clip_crop_reference_height: int = DEFAULT_CLIP_CROP_REFERENCE_HEIGHT,
        clip_crop_left: int = DEFAULT_CLIP_CROP_LEFT,
        clip_crop_top: int = DEFAULT_CLIP_CROP_TOP,
        clip_crop_right: int = DEFAULT_CLIP_CROP_RIGHT,
        clip_crop_bottom: int = DEFAULT_CLIP_CROP_BOTTOM,
    ):
        self.target_root = target_root
        self.order_filename = order_filename
        self.video_root = video_root
        self.video_suffixes = tuple(video_suffixes)
        self.audio_filename = audio_filename
        self.output_filename = output_filename
        self.status_step = status_step
        self.failed_filename = failed_filename
        self.required_steps = tuple(required_steps)
        self.sample_rate = sample_rate
        self.envelope_hz = envelope_hz
        self.refine_hz = refine_hz
        self.refine_search_seconds = refine_search_seconds
        self.clip_failed_filename = clip_failed_filename
        self.clip_status_step = clip_status_step
        self.clip_required_steps = tuple(clip_required_steps)
        self.clip_crop_reference_width = clip_crop_reference_width
        self.clip_crop_reference_height = clip_crop_reference_height
        self.clip_crop_left = clip_crop_left
        self.clip_crop_top = clip_crop_top
        self.clip_crop_right = clip_crop_right
        self.clip_crop_bottom = clip_crop_bottom

    def _build_video_init_checker(self) -> VideoInitChecker:
        return VideoInitChecker(
            target_root=self.target_root,
            order_filename=self.order_filename,
            video_suffixes=self.video_suffixes,
            output_filename=self.output_filename,
            status_step=self.status_step,
            clip_status_step=self.clip_status_step,
            clip_failed_filename=self.clip_failed_filename,
            clip_required_steps=self.clip_required_steps,
            clip_crop_reference_width=self.clip_crop_reference_width,
            clip_crop_reference_height=self.clip_crop_reference_height,
            clip_crop_left=self.clip_crop_left,
            clip_crop_top=self.clip_crop_top,
            clip_crop_right=self.clip_crop_right,
            clip_crop_bottom=self.clip_crop_bottom,
        )

    def _build_video_package_renamer(self) -> VideoPackageRenamer:
        return VideoPackageRenamer(
            video_root=self.video_root,
            target_root=self.target_root,
            order_filename=self.order_filename,
            video_suffixes=self.video_suffixes,
        )

    def _build_av_correspondence_processor(self) -> AVCorrespondenceProcessor:
        return AVCorrespondenceProcessor(
            target_root=self.target_root,
            order_filename=self.order_filename,
            audio_filename=self.audio_filename,
            output_filename=self.output_filename,
            failed_filename=self.failed_filename,
            status_step=self.status_step,
            required_steps=self.required_steps,
            sample_rate=self.sample_rate,
            envelope_hz=self.envelope_hz,
            refine_hz=self.refine_hz,
            refine_search_seconds=self.refine_search_seconds,
            video_suffixes=self.video_suffixes,
        )

    def _build_clip_processor(self) -> FixedRegionVideoCropProcessor:
        return FixedRegionVideoCropProcessor(
            target_root=self.target_root,
            order_filename=self.order_filename,
            output_filename=self.output_filename,
            failed_filename=self.clip_failed_filename,
            status_step=self.clip_status_step,
            required_steps=self.clip_required_steps,
            crop_reference_width=self.clip_crop_reference_width,
            crop_reference_height=self.clip_crop_reference_height,
            crop_left=self.clip_crop_left,
            crop_top=self.clip_crop_top,
            crop_right=self.clip_crop_right,
            crop_bottom=self.clip_crop_bottom,
        )

    def _run_clip_stage(self, overwrite: bool):
        self._build_clip_processor().run(overwrite=overwrite)

    def run(
        self,
        overwrite: bool = False,
        run_init_check: bool = True,
        run_video_match: bool = True,
        run_av_correspondence: bool = True,
        run_clip_stage: bool = DEFAULT_RUN_CLIP_STAGE,
    ):
        if run_init_check:
            print("[阶段] 初始化视频流程状态")
            self._build_video_init_checker().run()

        if run_video_match:
            print()
            print("[阶段] 匹配视频到谱面目录")
            self._build_video_package_renamer().run()

        if run_av_correspondence:
            print()
            print("[阶段] 执行 AV 对齐并生成处理后视频")
            self._build_av_correspondence_processor().run(overwrite=overwrite)

        if run_clip_stage:
            print()
            print("[阶段] 执行固定区域裁剪")
            self._run_clip_stage(overwrite=overwrite)


def main():
    pipeline = build_video_clip_pipeline_from_config_or_default()
    pipeline.run(
        overwrite=False,
        run_init_check=True,
        run_video_match=True,
        run_av_correspondence=True,
        run_clip_stage=DEFAULT_RUN_CLIP_STAGE,
    )


if __name__ == "__main__":
    main()
