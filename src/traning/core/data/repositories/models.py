"""数据 repository 对外暴露的稳定领域对象。"""

from __future__ import annotations

import math
from dataclasses import dataclass

from traning.state.common import require_identifier


def _require_positive_integer(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} 必须是整数")
    if value < 1:
        raise ValueError(f"{field_name} 必须为正整数")


def _require_nonnegative_integer(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} 必须是整数")
    if value < 0:
        raise ValueError(f"{field_name} 不得为负数")


def _require_optional_text(value: str | None, field_name: str) -> None:
    if value is not None:
        require_identifier(value, field_name)


@dataclass(frozen=True, slots=True)
class PreprocessingMetadata:
    """单个数据项完成视频预处理后形成的坐标链元数据。"""

    item_name: str
    source_width: int
    source_height: int
    crop_left: int
    crop_top: int
    crop_width: int
    crop_height: int
    updated_at: str

    def __post_init__(self) -> None:
        require_identifier(self.item_name, "item_name")
        _require_positive_integer(self.source_width, "source_width")
        _require_positive_integer(self.source_height, "source_height")
        _require_nonnegative_integer(self.crop_left, "crop_left")
        _require_nonnegative_integer(self.crop_top, "crop_top")
        _require_positive_integer(self.crop_width, "crop_width")
        _require_positive_integer(self.crop_height, "crop_height")
        require_identifier(self.updated_at, "updated_at")
        if self.crop_left + self.crop_width > self.source_width:
            raise ValueError("裁剪矩形的右边界超出源视频")
        if self.crop_top + self.crop_height > self.source_height:
            raise ValueError("裁剪矩形的下边界超出源视频")


@dataclass(frozen=True, slots=True)
class DatasetCatalogEntry:
    """训练数据目录中的一个可寻址数据项。"""

    folder_name: str
    source_name: str
    sequence: int
    osu_filename: str
    source_osz_name: str | None = None
    source_mtime_ns: int | None = None
    difficulty_value: float | None = None
    active: bool = True

    def __post_init__(self) -> None:
        require_identifier(self.folder_name, "folder_name")
        require_identifier(self.source_name, "source_name")
        _require_nonnegative_integer(self.sequence, "sequence")
        require_identifier(self.osu_filename, "osu_filename")
        _require_optional_text(self.source_osz_name, "source_osz_name")
        if self.source_mtime_ns is not None:
            _require_nonnegative_integer(self.source_mtime_ns, "source_mtime_ns")
        if self.difficulty_value is not None:
            if isinstance(self.difficulty_value, bool) or not isinstance(
                self.difficulty_value, (int, float)
            ):
                raise TypeError("difficulty_value 必须是数值")
            if not math.isfinite(float(self.difficulty_value)):
                raise ValueError("difficulty_value 必须是有限数值")
            if not 0.0 <= float(self.difficulty_value):
                raise ValueError("difficulty_value 不得为负数")
        if not isinstance(self.active, bool):
            raise TypeError("active 必须是布尔值")


class RepositoryError(RuntimeError):
    """repository 无法完成持久化操作时抛出的显式错误。"""


__all__ = ("DatasetCatalogEntry", "PreprocessingMetadata", "RepositoryError")
