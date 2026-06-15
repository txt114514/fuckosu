from __future__ import annotations

from math import ceil

from traning.Lib.data.models import FrameReference, SegmentRecord


def build_frame_references(
    records: tuple[SegmentRecord, ...],
    *,
    sample_fps: float,
    frame_step: int,
    max_frames_per_segment: int | None,
) -> tuple[FrameReference, ...]:
    references: list[FrameReference] = []
    frame_interval_ms = 1000.0 / sample_fps

    for record_index, record in enumerate(records):
        frame_count = max(1, ceil(record.annotation.duration_ms / frame_interval_ms))
        frame_indexes = range(0, frame_count, frame_step)
        if max_frames_per_segment is not None:
            frame_indexes = tuple(frame_indexes)[:max_frames_per_segment]
        references.extend(
            FrameReference(
                record_index=record_index,
                frame_index=frame_index,
                timestamp_ms=frame_index * frame_interval_ms,
            )
            for frame_index in frame_indexes
        )
    return tuple(references)


__all__ = ["build_frame_references"]
