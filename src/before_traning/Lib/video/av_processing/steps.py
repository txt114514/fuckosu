from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import wavfile

from before_traning.Lib.tools.ffmpeg import extract_wav, trim_video


class AVCoreStepsMixin:
    def _extract_audio_to_wav(self, source_path: Path, output_path: Path, from_video: bool):
        extract_wav(
            source_path,
            output_path,
            sample_rate=self.sample_rate,
            from_video=from_video,
        )

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

    def _lowpass_samples(self, samples: np.ndarray) -> np.ndarray:
        nyquist_hz = self.sample_rate / 2.0
        normalized_cutoff = self.music_lowpass_hz / nyquist_hz
        if normalized_cutoff >= 1.0:
            return samples.astype(np.float32, copy=False)

        b, a = signal.butter(4, normalized_cutoff, btype="low")
        pad_length = 3 * (max(len(a), len(b)) - 1)
        if samples.size <= pad_length:
            return samples.astype(np.float32, copy=False)

        filtered = signal.filtfilt(b, a, samples)
        return filtered.astype(np.float32)

    def _build_music_refine_series(self, samples: np.ndarray) -> np.ndarray:
        filtered_samples = self._lowpass_samples(samples)
        return self._build_feature_series(
            filtered_samples,
            self.refine_hz,
            mode="energy",
        )

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

        fine_video = self._build_music_refine_series(video_audio_samples)
        fine_song = self._build_music_refine_series(song_audio_samples)
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

    def _parse_verify_hit_times_ms(self, verify_path: Path) -> list[int]:
        if not verify_path.is_file():
            return []

        hit_times_ms: list[int] = []
        for raw_line in verify_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            name, rest = line.split("(", 1)
            args = ast.literal_eval(f"({rest}")
            if name in {"Circle", "Slider", "Spinner"}:
                hit_times_ms.append(int(args[0]))
        return hit_times_ms

    def _build_verify_click_train(self, hit_times_ms: list[int], length_frames: int) -> np.ndarray:
        click_train = np.zeros(length_frames, dtype=np.float32)
        radius = int(round(self.refine_hz * 0.03))
        sigma = max(radius / 2, 1)
        for time_ms in hit_times_ms:
            center = int(round((time_ms / 1000.0) * self.refine_hz))
            left = max(0, center - radius)
            right = min(click_train.size, center + radius + 1)
            if left >= right:
                continue
            xs = np.arange(left, right) - center
            pulse = np.exp(-(xs.astype(np.float32) ** 2) / (2.0 * sigma**2))
            click_train[left:right] += pulse.astype(np.float32)
        return self._normalize_series(click_train)

    def _estimate_verify_adjustment_seconds(
        self,
        transient_series: np.ndarray,
        verify_path: Path,
        base_offset_seconds: float,
    ) -> tuple[float, dict[str, float]] | None:
        hit_times_ms = self._parse_verify_hit_times_ms(verify_path)
        if not hit_times_ms:
            return None

        last_hit_seconds = max(hit_times_ms) / 1000.0
        click_train = self._build_verify_click_train(
            hit_times_ms,
            int(np.ceil(last_hit_seconds * self.refine_hz)) + 1,
        )
        base_frame = int(round(base_offset_seconds * self.refine_hz))
        window_frames = int(
            round(self.verify_correction_window_ms / 1000.0 * self.refine_hz)
        )

        best_delta_frames: int | None = None
        best_score: float | None = None
        for delta_frames in range(-window_frames, window_frames + 1):
            start = base_frame + delta_frames
            end = start + click_train.size
            if start < 0 or end > transient_series.size:
                continue

            score = float(
                np.dot(
                    self._normalize_series(transient_series[start:end]),
                    click_train,
                )
                / max(click_train.size, 1)
            )
            if best_score is None or score > best_score:
                best_delta_frames = delta_frames
                best_score = score

        if best_delta_frames is None or best_score is None or best_score <= 0.0:
            return None

        verify_adjustment_seconds = best_delta_frames / float(self.refine_hz)
        verify_adjustment_ms = verify_adjustment_seconds * 1000.0
        return verify_adjustment_seconds, {
            "verify_adjustment_seconds": verify_adjustment_seconds,
            "verify_adjustment_ms": verify_adjustment_ms,
            "verify_score": best_score,
            "verify_window_ms": self.verify_correction_window_ms,
        }

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
        trim_video(
            source_video_path,
            output_video_path,
            start_seconds=trim_start_seconds,
            duration_seconds=trim_duration_seconds,
        )
