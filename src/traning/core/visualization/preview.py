from __future__ import annotations

from pathlib import Path

from traning.Lib.visualization import (
    GalleryResult,
    VisualizationResult,
    select_click_frame,
)
from traning.conf import Settings
from traning.core.dataset_import import build_dataset
from traning.core.visualization.service import OptionalTrainingVisualizer
from traning.state import BatchGalleryRequest


def visualize_click_label(
    settings: Settings,
    *,
    segment_index: int = 0,
    object_index: int = 0,
    output_path: Path | None = None,
    show_window: bool | None = None,
) -> VisualizationResult:
    dataset = build_dataset(settings)
    selected = select_click_frame(
        dataset,
        segment_index=segment_index,
        object_index=object_index,
    )
    sample = dataset[selected.dataset_index]
    visualizer = OptionalTrainingVisualizer(settings.visualization)
    return visualizer.visualize(
        sample,
        target_source_index=selected.target_source_index,
        output_path=output_path,
        force=True,
        show_window=show_window,
    )


def save_annotation_gallery(
    settings: Settings,
    request: BatchGalleryRequest,
    *,
    output_root: Path | None = None,
    samples_per_group: int | None = None,
) -> GalleryResult:
    dataset = build_dataset(settings)
    visualizer = OptionalTrainingVisualizer(settings.visualization)
    return visualizer.save_gallery(
        dataset,
        request,
        output_root=output_root,
        samples_per_group=samples_per_group,
    )


__all__ = ["save_annotation_gallery", "visualize_click_label"]
