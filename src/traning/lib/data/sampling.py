"""把每个 segment 的时长确定性展开为相对时间帧引用。"""

from __future__ import annotations

from math import ceil

from traning.lib.data.models import FrameReference, SegmentRecord


def build_frame_references(
    records: tuple[SegmentRecord, ...],
    *,
    sample_fps: float,
    frame_step: int,
    max_frames_per_segment: int | None,
) -> tuple[FrameReference, ...]:
    """按固定采样频率生成 segment 相对时间引用，不读取视频内容。"""

    references: list[FrameReference] = []
    frame_interval_ms = 1000.0 / sample_fps

    for record_index, record in enumerate(records):
        # ceil 保留末尾不足一个间隔的 segment，max(1) 保证每段至少有一帧。
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
