"""游戏画面坐标校准的公开入口。"""

from traning.core.calibration.playfield import (
    CalibrationResult,
    calibrate_playfield_transform,
)

__all__ = ["CalibrationResult", "calibrate_playfield_transform"]
