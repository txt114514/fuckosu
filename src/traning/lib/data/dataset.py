from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset

from traning.lib.data.annotation import visible_hit_objects
from traning.lib.data.models import FrameReference, SegmentRecord
from traning.lib.data.sampling import build_frame_references
from traning.lib.data.video_reader import VideoReader


class SegmentFrameDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        records: tuple[SegmentRecord, ...],
        *,
        sample_fps: float,
        frame_step: int = 1,
        max_frames_per_segment: int | None = None,
        visibility_post_ms: float = 100.0,
        normalize_images: bool = True,
        coordinate_transform: dict[str, Any] | None = None,
    ):
        if not records:
            raise ValueError("records must not be empty")
        self.records = records
        self.references: tuple[FrameReference, ...] = build_frame_references(
            records,
            sample_fps=sample_fps,
            frame_step=frame_step,
            max_frames_per_segment=max_frames_per_segment,
        )
        self.visibility_post_ms = visibility_post_ms
        self.normalize_images = normalize_images
        self.coordinate_transform = coordinate_transform
        self._reader: VideoReader | None = None

    def __len__(self) -> int:
        return len(self.references)

    def _video_reader(self) -> VideoReader:
        if self._reader is None:
            self._reader = VideoReader()
        return self._reader

    def __getitem__(self, index: int) -> dict[str, Any]:
        reference = self.references[index]
        record = self.records[reference.record_index]
        frame = self._video_reader().read_frame_at(
            record.video_path,
            reference.timestamp_ms,
        )
        image = torch.from_numpy(frame.copy()).permute(2, 0, 1)
        if self.normalize_images:
            image = image.to(dtype=torch.float32).div_(255.0)

        visible = visible_hit_objects(
            record.annotation,
            reference.timestamp_ms,
            visibility_post_ms=self.visibility_post_ms,
        )
        return {
            "image": image,
            "sample_key": record.key,
            "item_name": record.item_name,
            "segment_id": record.annotation.segment_id,
            "dataset_dimension": record.dataset_dimension,
            "category": record.category,
            "frame_index": reference.frame_index,
            "timestamp_ms": reference.timestamp_ms,
            "hit_objects": tuple(
                item.model_dump(mode="python")
                for item in record.annotation.hit_objects
            ),
            "visible_hit_objects": tuple(
                item.model_dump(mode="python") for item in visible
            ),
            "circle_radius_osu_pixels": (
                record.annotation.difficulty.circle_radius_osu_pixels
            ),
            "approach_preempt_ms": record.annotation.difficulty.approach_preempt_ms,
            "coordinate_transform": self.coordinate_transform,
        }

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_reader"] = None
        return state


__all__ = ["SegmentFrameDataset"]
