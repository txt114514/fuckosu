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