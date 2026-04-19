from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Iterable, List

import numpy as np
from scipy import signal
from scipy.io import wavfile

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from Traning.Lib.function_tools.functions_process_tool import (
    BatchProcessResult,
    FolderBatchProcessor,
    read_config_values,
)
from Traning.Lib.function_tools.video_process_tool import (
    get_audio_stream_start_time,
    run_ffmpeg,
)
from Traning.Lib.get_training_data.config_loader import (
    AV_CORRESPONDENCE_PROCESSOR_CONFIG_SPECS,
    ConfigReader,
    build_from_config_or_default,
)
from Traning.Lib.get_training_data.process_status_manager import ProcessStatusManager
from Traning.Lib.traning_package_manager.files_manager import BeatmapFolderStore

DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_TARGET_ROOT = DEFAULT_REPO_ROOT / "training_package" / "match-completed_package"
DEFAULT_ORDER_FILENAME = "order.txt"
DEFAULT_AUDIO_FILENAME = "audio.mp3"
DEFAULT_OUTPUT_FILENAME = "video_processed.mp4"
DEFAULT_FAILED_FILENAME = "av_correspondence_failed.txt"
DEFAULT_STATUS_STEP = "av_corresponded"
DEFAULT_REQUIRED_STEPS = ("audio_imported", "video_matched")
DEFAULT_SAMPLE_RATE = 8000
DEFAULT_ENVELOPE_HZ = 100
DEFAULT_REFINE_HZ = 1000
DEFAULT_REFINE_SEARCH_SECONDS = 1.5
DEFAULT_VIDEO_SUFFIXES = (".mp4", ".webm", ".mkv", ".avi", ".mov")

# 默认值保留在当前文件；config.json 里的合法参数只用于覆盖这些默认值。

# 共享参数优先从 config.json 的 video_shared 读取；
# AV 对齐专属参数优先从 config.json 的 av_correspondence 读取。


def _load_av_correspondence_processor_config(config: ConfigReader) -> dict[str, object]:
    return read_config_values(config, AV_CORRESPONDENCE_PROCESSOR_CONFIG_SPECS)


def build_av_correspondence_processor_from_config_or_default(
    config_path: Path | None = None,
) -> "AVCorrespondenceProcessor":
    return build_from_config_or_default(
        AVCorrespondenceProcessor,
        [_load_av_correspondence_processor_config],
        config_path=config_path,
        default_builder=AVCorrespondenceProcessor,
    )


class AVCorrespondenceProcessor(FolderBatchProcessor):
    def __init__(
        self,
        target_root: str = str(DEFAULT_TARGET_ROOT),
        order_filename: str = DEFAULT_ORDER_FILENAME,
        audio_filename: str = DEFAULT_AUDIO_FILENAME,
        output_filename: str = DEFAULT_OUTPUT_FILENAME,
        failed_filename: str = DEFAULT_FAILED_FILENAME,
        status_step: str = DEFAULT_STATUS_STEP,
        required_steps: Iterable[str] = DEFAULT_REQUIRED_STEPS,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        envelope_hz: int = DEFAULT_ENVELOPE_HZ,
        refine_hz: int = DEFAULT_REFINE_HZ,
        refine_search_seconds: float = DEFAULT_REFINE_SEARCH_SECONDS,
        video_suffixes: Iterable[str] = DEFAULT_VIDEO_SUFFIXES,
        status_manager: ProcessStatusManager | None = None,
    ):
        if sample_rate <= 0:
            raise ValueError("sample_rate 必须大于 0")
        if envelope_hz <= 0:
            raise ValueError("envelope_hz 必须大于 0")
        if refine_hz <= 0:
            raise ValueError("refine_hz 必须大于 0")
        if refine_search_seconds <= 0:
            raise ValueError("refine_search_seconds 必须大于 0")
        if not status_step.strip():
            raise ValueError("status_step 不能为空")

        self.target_root = Path(target_root)
        self.order_filename = order_filename
        self.audio_filename = audio_filename
        self.output_filename = output_filename
        super().__init__(failed_filename)
        self.status_step = status_step.strip()
        self.required_steps = tuple(step.strip() for step in required_steps if step.strip())
        self.sample_rate = sample_rate
        self.envelope_hz = envelope_hz
        self.refine_hz = min(sample_rate, refine_hz)
        self.refine_search_seconds = refine_search_seconds
        self.video_suffixes = {suffix.lower() for suffix in video_suffixes}
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
                "配置中的 process_steps 缺少 AV 对齐所需步骤: "
                f"{', '.join(missing_steps)}"
            )

    def _extract_audio_to_wav(self, source_path: Path, output_path: Path, from_video: bool):
        args = [
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source_path),
        ]
        if from_video:
            args.extend(["-vn"])
        args.extend(
            [
                "-ac",
                "1",
                "-ar",
                str(self.sample_rate),
                "-f",
                "wav",
                "-acodec",
                "pcm_s16le",
                str(output_path),
            ]
        )
        run_ffmpeg(args)

    def _load_wav_samples(self, wav_path: Path) -> np.ndarray:
        sample_rate, samples = wavfile.read(wav_path)
        if sample_rate != self.sample_rate:
            raise ValueError(
                f"采样率不匹配，期望 {self.sample_rate}，实际 {sample_rate}: {wav_path}"
            )

        if samples.ndim == 2:
            samples = samples.mean(axis=1)

        if np.issubdtype(samples.dtype, np.integer):
            info = np.iinfo(samples.dtype)
            scale = float(max(abs(info.min), info.max))
            normalized = samples.astype(np.float32) / scale
        else:
            normalized = samples.astype(np.float32)

        if normalized.size == 0:
            raise ValueError(f"音频为空: {wav_path}")

        return normalized

    def _normalize_series(self, values: np.ndarray) -> np.ndarray:
        normalized = values.astype(np.float32)
        normalized -= float(normalized.mean())
        std = float(normalized.std())
        if std > 1e-8:
            normalized /= std
        return normalized

    def _build_feature_series(
        self,
        samples: np.ndarray,
        target_hz: int,
        mode: str = "energy",
    ) -> np.ndarray:
        hop_size = max(1, int(round(self.sample_rate / target_hz)))
        if mode == "energy":
            source = np.abs(samples)
        elif mode == "transient":
            source = np.abs(np.diff(samples, prepend=samples[:1]))
        else:
            raise ValueError(f"未知特征模式: {mode}")

        usable_length = (source.size // hop_size) * hop_size
        if usable_length == 0:
            feature = np.array([float(source.mean())], dtype=np.float32)
        else:
            feature = source[:usable_length].reshape(-1, hop_size).mean(axis=1)
            if usable_length < source.size:
                tail_mean = float(source[usable_length:].mean())
                feature = np.concatenate(
                    [feature.astype(np.float32), np.array([tail_mean], dtype=np.float32)]
                )

        return self._normalize_series(feature)

    def _estimate_best_start_frame(
        self,
        long_series: np.ndarray,
        short_series: np.ndarray,
    ) -> tuple[float, float]:
        if long_series.size < short_series.size:
            raise ValueError("长序列长度不能小于短序列长度")

        correlation = signal.correlate(
            long_series,
            short_series,
            mode="valid",
            method="fft",
        )
        best_index = int(np.argmax(correlation))
        fractional_offset = 0.0

        if 0 < best_index < correlation.size - 1:
            left = float(correlation[best_index - 1])
            center = float(correlation[best_index])
            right = float(correlation[best_index + 1])
            denominator = left - 2.0 * center + right
            if abs(denominator) > 1e-8:
                fractional_offset = 0.5 * (left - right) / denominator
                fractional_offset = float(np.clip(fractional_offset, -1.0, 1.0))

        best_score = float(correlation[best_index] / max(short_series.size, 1))
        return best_index + fractional_offset, best_score

    def _estimate_offset_seconds(
        self,
        video_audio_samples: np.ndarray,
        song_audio_samples: np.ndarray,
    ) -> tuple[float, float, float]:
        coarse_video = self._build_feature_series(
            video_audio_samples,
            self.envelope_hz,
            mode="energy",
        )
        coarse_song = self._build_feature_series(
            song_audio_samples,
            self.envelope_hz,
            mode="energy",
        )

        coarse_start_frame, coarse_score = self._estimate_best_start_frame(
            coarse_video,
            coarse_song,
        )
        coarse_offset_seconds = coarse_start_frame / float(self.envelope_hz)

        fine_video = self._build_feature_series(
            video_audio_samples,
            self.refine_hz,
            mode="transient",
        )
        fine_song = self._build_feature_series(
            song_audio_samples,
            self.refine_hz,
            mode="transient",
        )
        if fine_video.size < fine_song.size:
            raise ValueError(
                "视频音频时长短于歌曲音频，无法在视频中找到完整歌曲片段"
            )

        search_margin_frames = max(1, int(round(self.refine_search_seconds * self.refine_hz)))
        coarse_start_frame_fine = int(round(coarse_offset_seconds * self.refine_hz))
        search_start = max(0, coarse_start_frame_fine - search_margin_frames)
        search_end = min(
            fine_video.size,
            coarse_start_frame_fine + fine_song.size + search_margin_frames,
        )
        if search_end - search_start < fine_song.size:
            search_start = max(0, fine_video.size - fine_song.size)
            search_end = fine_video.size

        fine_search_region = fine_video[search_start:search_end]
        fine_start_frame, fine_score = self._estimate_best_start_frame(
            fine_search_region,
            fine_song,
        )
        refined_offset_seconds = (search_start + fine_start_frame) / float(self.refine_hz)
        return refined_offset_seconds, fine_score, coarse_score

    def _resolve_source_video_path(self, folder_name: str) -> Path:
        candidates = [
            self.store.get_file_path(folder_name, f"{folder_name}{suffix}")
            for suffix in sorted(self.video_suffixes)
            if self.store.file_exists(folder_name, f"{folder_name}{suffix}")
        ]

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

    def _validate_trim_window(
        self,
        offset_seconds: float,
        song_duration_seconds: float,
        video_duration_seconds: float,
    ) -> float:
        tolerance = min(0.02, 2.0 / max(self.refine_hz, self.envelope_hz))

        if offset_seconds < -tolerance:
            raise ValueError(
                "视频音频起点晚于歌曲音频起点，无法仅通过裁剪得到完整对齐视频: "
                f"offset={offset_seconds:.3f}s"
            )

        trim_start_seconds = max(0.0, offset_seconds)
        required_end = trim_start_seconds + song_duration_seconds
        if required_end > video_duration_seconds + tolerance:
            raise ValueError(
                "视频时长不足，无法裁出与歌曲音频等长且对齐的视频: "
                f"需要结束时间 {required_end:.3f}s，视频只有 {video_duration_seconds:.3f}s"
            )

        return trim_start_seconds

    def _trim_video(
        self,
        source_video_path: Path,
        output_video_path: Path,
        trim_start_seconds: float,
        trim_duration_seconds: float,
    ):
        run_ffmpeg(
            [
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(source_video_path),
                "-ss",
                f"{trim_start_seconds:.6f}",
                "-t",
                f"{trim_duration_seconds:.6f}",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
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
                str(output_video_path),
            ]
        )

    def _update_progress(self, folder_name: str, stage: str, detail: dict | None = None):
        payload = {"stage": stage}
        if detail:
            payload.update(detail)
        self.status_manager.mark_step_pending(folder_name, self.status_step, detail=payload)

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
                detail={"error": "状态显示已完成 AV 对齐，但输出文件不存在"},
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
        output_exists, step_done = self._sync_output_status(folder_name)
        if not overwrite and output_exists and step_done:
            return "skip"

        self._ensure_required_steps_done(folder_name)

        source_video_path = self._resolve_source_video_path(folder_name)
        song_audio_path = self._resolve_song_audio_path(folder_name)
        output_video_path = self.store.get_file_path(folder_name, self.output_filename)
        video_audio_start_time = get_audio_stream_start_time(source_video_path)
        song_audio_start_time = get_audio_stream_start_time(song_audio_path)

        self._update_progress(
            folder_name,
            "extracting_audio",
            detail={
                "source_video_path": str(source_video_path),
                "audio_path": str(song_audio_path),
                "output_video_path": str(output_video_path),
            },
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            video_audio_wav = tmpdir_path / "video_audio.wav"
            song_audio_wav = tmpdir_path / "song_audio.wav"

            self._extract_audio_to_wav(source_video_path, video_audio_wav, from_video=True)
            self._extract_audio_to_wav(song_audio_path, song_audio_wav, from_video=False)

            video_audio_samples = self._load_wav_samples(video_audio_wav)
            song_audio_samples = self._load_wav_samples(song_audio_wav)

            self._update_progress(folder_name, "aligning_audio")
            raw_offset_seconds, score, coarse_score = self._estimate_offset_seconds(
                video_audio_samples,
                song_audio_samples,
            )
            metadata_adjustment_seconds = video_audio_start_time - song_audio_start_time
            offset_seconds = raw_offset_seconds + metadata_adjustment_seconds
            song_duration_seconds = song_audio_samples.size / float(self.sample_rate)
            video_duration_seconds = video_audio_samples.size / float(self.sample_rate)
            trim_start_seconds = self._validate_trim_window(
                offset_seconds,
                song_duration_seconds,
                video_duration_seconds,
            )

        self._update_progress(
            folder_name,
            "trimming_video",
            detail={
                "raw_offset_seconds": round(raw_offset_seconds, 6),
                "offset_seconds": round(offset_seconds, 6),
                "trim_start_seconds": round(trim_start_seconds, 6),
                "trim_duration_seconds": round(song_duration_seconds, 6),
            },
        )
        self._trim_video(
            source_video_path=source_video_path,
            output_video_path=output_video_path,
            trim_start_seconds=trim_start_seconds,
            trim_duration_seconds=song_duration_seconds,
        )

        self.status_manager.mark_step_done(
            folder_name,
            self.status_step,
            detail={
                "stage": "done",
                "source_video_path": str(source_video_path),
                "output_video_path": str(output_video_path),
                "audio_path": str(song_audio_path),
                "raw_offset_seconds": round(raw_offset_seconds, 6),
                "metadata_adjustment_seconds": round(metadata_adjustment_seconds, 6),
                "offset_seconds": round(offset_seconds, 6),
                "video_audio_start_time": round(video_audio_start_time, 6),
                "song_audio_start_time": round(song_audio_start_time, 6),
                "trim_start_seconds": round(trim_start_seconds, 6),
                "trim_duration_seconds": round(song_duration_seconds, 6),
                "match_score": round(score, 6),
                "coarse_match_score": round(coarse_score, 6),
                "refine_hz": self.refine_hz,
            },
        )
        return "success"

    def handle_failure(self, folder_name: str, error: Exception):
        if self.store.folder_exists(folder_name):
            self.status_manager.ensure_status_file(folder_name)
            self.status_manager.mark_step_pending(
                folder_name,
                self.status_step,
                detail={"stage": "failed", "error": str(error)},
            )


def main():
    processor = build_av_correspondence_processor_from_config_or_default()
    processor.run(overwrite=False)


if __name__ == "__main__":
    main()
