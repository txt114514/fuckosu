from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np


class VideoReader:
    def __init__(self, max_open_videos: int = 4):
        if max_open_videos <= 0:
            raise ValueError("max_open_videos must be positive")
        self.max_open_videos = max_open_videos
        self._captures: OrderedDict[Path, cv2.VideoCapture] = OrderedDict()

    def _capture(self, path: Path) -> cv2.VideoCapture:
        capture = self._captures.pop(path, None)
        if capture is None:
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                capture.release()
                raise ValueError(f"failed to open video: {path}")
        self._captures[path] = capture
        while len(self._captures) > self.max_open_videos:
            _, old_capture = self._captures.popitem(last=False)
            old_capture.release()
        return capture

    def read_frame(self, path: Path, frame_index: int) -> np.ndarray:
        capture = self._capture(path)
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = capture.read()
        if not success:
            raise IndexError(f"failed to decode frame {frame_index}: {path}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def read_frame_at(self, path: Path, timestamp_ms: float) -> np.ndarray:
        if timestamp_ms < 0:
            raise ValueError("timestamp_ms must be nonnegative")
        capture = self._capture(path)
        capture.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms)
        success, frame = capture.read()
        if not success:
            raise IndexError(
                f"failed to decode frame at {timestamp_ms:.3f} ms: {path}"
            )
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        for capture in self._captures.values():
            capture.release()
        self._captures.clear()

    def __del__(self) -> None:
        self.close()


__all__ = ["VideoReader"]
