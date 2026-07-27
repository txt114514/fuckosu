"""定义 osu! 原始 timing point 的结构化数据模型。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OsuOriginalTimingPoint:
    time: int
    beat_length: float
    meter: int
    sample_set: int
    sample_index: int
    volume: int
    uninherited: bool
    effects: int
