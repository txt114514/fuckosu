"""为指定 segment 物件选择时间上最接近其打击时刻的抽样帧。"""

from __future__ import annotations

from traning.lib.data.dataset import SegmentFrameDataset
from traning.lib.visualization.models import SelectedFrame


def select_click_frame(
    dataset: SegmentFrameDataset,
    *,
    segment_index: int,
    object_index: int = 0,
) -> SelectedFrame:
    """选择指定物件开始时间附近、实际存在于 Dataset 的最近抽样帧。"""

    if not 0 <= segment_index < len(dataset.records):
        raise IndexError(
            f"segment_index {segment_index} 超出范围 "
            f"[0, {len(dataset.records)})"
        )
    record = dataset.records[segment_index]
    hit_objects = record.annotation.hit_objects
    if not 0 <= object_index < len(hit_objects):
        raise IndexError(
            f"object_index {object_index} 超出范围 [0, {len(hit_objects)})"
        )

    target = hit_objects[object_index]
    candidates = (
        (dataset_index, reference)
        for dataset_index, reference in enumerate(dataset.references)
        if reference.record_index == segment_index
    )
    # start_ms 通常不落在离散采样点上，因此选择绝对时间差最小的帧。
    try:
        dataset_index, reference = min(
            candidates,
            key=lambda item: abs(item[1].timestamp_ms - target.start_ms),
        )
    except ValueError as error:
        raise ValueError(
            f"片段 {record.key} 没有抽样帧引用"
        ) from error

    return SelectedFrame(
        dataset_index=dataset_index,
        segment_index=segment_index,
        object_index=object_index,
        timestamp_ms=reference.timestamp_ms,
        target_source_index=target.source_index,
    )


__all__ = ["select_click_frame"]
