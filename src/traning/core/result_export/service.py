from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from traning.lib.data import SegmentFrameDataset
from traning.lib.visualization import (
    GalleryResult,
    OutputIdentity,
    VisualizationResult,
    allocate_output_identity,
    launch_image_window,
    render_annotated_frame,
    save_annotated_frame,
)
from traning.conf import VisualizationSettings
from traning.state import BatchGalleryRequest
from visualization.lib.gallery_api import export_best_trial_gallery


class OptionalTrainingVisualizer:
    """Best-effort visualization that never raises into training code."""

    def __init__(self, settings: VisualizationSettings):
        self.settings = settings
        self._render_disabled = False
        self._display_disabled = False
        self._warning_returned = False
        self._display_process: subprocess.Popen[bytes] | None = None

    def _warning_once(self, message: str) -> str | None:
        if self._warning_returned:
            return None
        self._warning_returned = True
        return message

    def visualize(
        self,
        sample: dict[str, Any],
        *,
        target_source_index: int | None = None,
        output_path: Path | None = None,
        force: bool = False,
        show_window: bool | None = None,
    ) -> VisualizationResult:
        if not force and not self.settings.enabled:
            return VisualizationResult(status="disabled")
        if self._render_disabled:
            return VisualizationResult(status="skipped")

        selected_show = (
            self.settings.show_window
            if show_window is None
            else show_window
        )
        try:
            output_identity = allocate_output_identity(
                self.settings.output_dir
            )
            selected_output = output_path or self._default_output_path(
                sample,
                output_identity,
            )
            image = render_annotated_frame(
                sample,
                target_source_index=target_source_index,
                metadata_lines=(
                    f"output={output_identity.sequence:06d}",
                    f"output_time={output_identity.created_at_utc}",
                ),
            )
            saved_path = save_annotated_frame(image, selected_output)
        except Exception as error:
            self._render_disabled = True
            return VisualizationResult(
                status="failed",
                warning=self._warning_once(
                    "training visualization disabled after render failure: "
                    f"{type(error).__name__}: {error}"
                ),
            )

        if not selected_show or self._display_disabled:
            return VisualizationResult(status="saved", output_path=saved_path)

        try:
            self._display_process = launch_image_window(
                saved_path,
                title=self.settings.window_title,
                ffplay_binary=self.settings.ffplay_binary,
                previous_process=self._display_process,
            )
            return VisualizationResult(
                status="displayed",
                output_path=saved_path,
            )
        except Exception as error:
            self._display_disabled = True
            return VisualizationResult(
                status="saved",
                output_path=saved_path,
                warning=self._warning_once(
                    "training visualization window disabled after display "
                    f"failure: {type(error).__name__}: {error}"
                ),
            )

    def maybe_visualize_step(
        self,
        sample: dict[str, Any],
        *,
        global_step: int,
        target_source_index: int | None = None,
    ) -> VisualizationResult:
        if global_step < 0:
            return VisualizationResult(
                status="failed",
                warning=self._warning_once(
                    "training visualization received a negative global_step"
                ),
            )
        if global_step % self.settings.every_n_steps != 0:
            return VisualizationResult(status="skipped")
        return self.visualize(
            sample,
            target_source_index=target_source_index,
        )

    def save_gallery(
        self,
        dataset: SegmentFrameDataset,
        request: BatchGalleryRequest,
        *,
        output_root: Path | None = None,
        samples_per_group: int | None = None,
    ) -> GalleryResult:
        if self._render_disabled:
            return GalleryResult(status="skipped")
        selected_count = (
            self.settings.gallery_samples_per_group
            if samples_per_group is None
            else samples_per_group
        )
        try:
            output_dir, saved_count, issues = export_best_trial_gallery(
                dataset,
                request,
                output_root=output_root or self.settings.output_dir,
                samples_per_group=selected_count,
            )
        except Exception as error:
            self._render_disabled = True
            return GalleryResult(
                status="failed",
                selected_trial_id=request.best_trial.trial_id,
                warning=self._warning_once(
                    "training annotation gallery disabled after save failure: "
                    f"{type(error).__name__}: {error}"
                ),
            )

        warning = None
        if issues:
            warning = self._warning_once(
                f"training annotation gallery skipped {len(issues)} "
                f"invalid frame record(s); see manifest.json"
            )
        return GalleryResult(
            status="saved",
            output_dir=output_dir,
            selected_trial_id=request.best_trial.trial_id,
            saved_frame_count=saved_count,
            warning=warning,
        )

    def _default_output_path(
        self,
        sample: dict[str, Any],
        output_identity: OutputIdentity,
    ) -> Path:
        safe_key = str(sample["sample_key"]).replace("/", "__")
        timestamp = round(float(sample["timestamp_ms"]))
        return self.settings.output_dir / (
            f"{output_identity.prefix}__{safe_key}"
            f"__{timestamp:09d}ms.png"
        )


__all__ = ["OptionalTrainingVisualizer"]
