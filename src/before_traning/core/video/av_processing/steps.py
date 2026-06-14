from __future__ import annotations

import tempfile
from pathlib import Path

from before_traning.Lib.common.batch import BatchProcessResult


class AVProcessStepsMixin:
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
        verify_path = self._resolve_verify_path(folder_name)
        output_video_path = self.store.get_file_path(
            folder_name,
            self.output_filename,
        )
        self._update_progress(
            folder_name,
            "extracting_audio",
            detail={
                "source_video_path": str(source_video_path),
                "audio_path": str(song_audio_path),
                "verify_path": str(verify_path),
                "output_video_path": str(output_video_path),
            },
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            video_audio_wav = tmpdir_path / "video_audio.wav"
            song_audio_wav = tmpdir_path / "song_audio.wav"
            self._extract_audio_to_wav(
                source_video_path,
                video_audio_wav,
                from_video=True,
            )
            self._extract_audio_to_wav(
                song_audio_path,
                song_audio_wav,
                from_video=False,
            )
            video_audio_samples = self._load_wav_samples(video_audio_wav)
            song_audio_samples = self._load_wav_samples(song_audio_wav)
            self._update_progress(folder_name, "aligning_audio")
            raw_offset_seconds, score, coarse_score = (
                self._estimate_offset_seconds(
                    video_audio_samples,
                    song_audio_samples,
                )
            )
            verify_adjustment_seconds = 0.0
            verify_detail = None
            verify_adjustment = self._estimate_verify_adjustment_seconds(
                self._build_feature_series(
                    video_audio_samples,
                    self.refine_hz,
                    mode="transient",
                ),
                verify_path,
                raw_offset_seconds,
            )
            if verify_adjustment is not None:
                verify_adjustment_seconds, verify_detail = verify_adjustment
            global_offset_seconds = self.global_offset_ms / 1000.0
            offset_seconds = (
                raw_offset_seconds
                + verify_adjustment_seconds
                + global_offset_seconds
            )
            song_duration_seconds = (
                song_audio_samples.size / float(self.sample_rate)
            )
            video_duration_seconds = (
                video_audio_samples.size / float(self.sample_rate)
            )
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
                "verify_adjustment_seconds": round(
                    verify_adjustment_seconds,
                    6,
                ),
                "global_offset_seconds": round(global_offset_seconds, 6),
                "offset_seconds": round(offset_seconds, 6),
                "trim_start_seconds": round(trim_start_seconds, 6),
                "trim_duration_seconds": round(
                    song_duration_seconds,
                    6,
                ),
            },
        )
        self._trim_video(
            source_video_path=source_video_path,
            output_video_path=output_video_path,
            trim_start_seconds=trim_start_seconds,
            trim_duration_seconds=song_duration_seconds,
        )
        self._mark_done(
            folder_name=folder_name,
            source_video_path=source_video_path,
            output_video_path=output_video_path,
            song_audio_path=song_audio_path,
            verify_path=verify_path,
            raw_offset_seconds=raw_offset_seconds,
            verify_adjustment_seconds=verify_adjustment_seconds,
            global_offset_seconds=global_offset_seconds,
            offset_seconds=offset_seconds,
            trim_start_seconds=trim_start_seconds,
            song_duration_seconds=song_duration_seconds,
            score=score,
            coarse_score=coarse_score,
            verify_detail=verify_detail,
        )
        return "success"


__all__ = ["AVProcessStepsMixin"]
