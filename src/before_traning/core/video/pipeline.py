"""依次编排录像匹配、AV 对齐、裁剪与谱面驱动分段阶段。"""

from __future__ import annotations

from before_traning.conf import Settings
from before_traning.core.beatmap.pipeline import (
    TRAINING_PIPELINE,
    VIDEO_TASK_KEYS,
)


def prepare_videos(settings: Settings) -> dict[str, bool]:
    return TRAINING_PIPELINE.run_direct(
        settings,
        only=VIDEO_TASK_KEYS,
    )


__all__ = [
    "prepare_videos",
]
