from __future__ import annotations


class AVWrapUpMixin:
    def _update_progress(self, folder_name: str, stage: str, detail: dict | None = None):
        payload = {"stage": stage}
        if detail:
            payload.update(detail)
        self.status_manager.mark_step_pending(folder_name, self.status_step, detail=payload)

    def progress_message(self, index: int, total: int, folder_name: str) -> str | None:
        return f"[进度] {index}/{total} {folder_name}"

    def _build_done_detail(
        self,
        *,
        source_video_path,
        output_video_path,
        song_audio_path,
        verify_path,
        raw_offset_seconds: float,
        verify_adjustment_seconds: float,
        global_offset_seconds: float,
        offset_seconds: float,
        trim_start_seconds: float,
        song_duration_seconds: float,
        score: float,
        coarse_score: float,
        verify_detail: dict[str, float] | None,
    ) -> dict:
        detail = {
            "stage": "done",
            "source_video_path": str(source_video_path),
            "output_video_path": str(output_video_path),
            "audio_path": str(song_audio_path),
            "verify_path": str(verify_path),
            "raw_offset_seconds": round(raw_offset_seconds, 6),
            "verify_adjustment_seconds": round(verify_adjustment_seconds, 6),
            "global_offset_seconds": round(global_offset_seconds, 6),
            "global_offset_ms": round(self.global_offset_ms, 6),
            "offset_seconds": round(offset_seconds, 6),
            "trim_start_seconds": round(trim_start_seconds, 6),
            "trim_duration_seconds": round(song_duration_seconds, 6),
            "match_score": round(score, 6),
            "coarse_match_score": round(coarse_score, 6),
            "refine_hz": self.refine_hz,
            "music_lowpass_hz": self.music_lowpass_hz,
        }
        if verify_detail is not None:
            detail.update(
                {
                    key: round(value, 6)
                    for key, value in verify_detail.items()
                }
            )
        return detail

    def _mark_done(self, folder_name: str, **detail_kwargs):
        self.status_manager.mark_step_done(
            folder_name,
            self.status_step,
            detail=self._build_done_detail(**detail_kwargs),
        )

    def handle_failure(self, folder_name: str, error: Exception):
        if self.store.folder_exists(folder_name):
            self.status_manager.ensure_status_file(folder_name)
            self.status_manager.mark_step_pending(
                folder_name,
                self.status_step,
                detail={"stage": "failed", "error": str(error)},
            )
