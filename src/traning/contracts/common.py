"""V2 领域契约共享的类型与校验工具。"""

from __future__ import annotations

import math
import re
from typing import TypeAlias


JSONScalar: TypeAlias = None | bool | int | float | str
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]
JSONObject: TypeAlias = dict[str, JSONValue]


def require_identifier(value: str, field_name: str) -> None:
    """校验稳定标识符，拒绝空白和首尾空格。"""

    if not isinstance(value, str):
        raise TypeError(f"{field_name} 必须是字符串")
    if not value or value != value.strip():
        raise ValueError(f"{field_name} 必须是非空且无首尾空格的标识符")


def require_finite(value: float, field_name: str) -> None:
    """拒绝 NaN 和无穷值。"""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} 必须是数值")
    if not math.isfinite(value):
        raise ValueError(f"{field_name} 必须是有限数值")


def require_nonnegative(value: float, field_name: str) -> None:
    """校验有限非负数。"""

    require_finite(value, field_name)
    if value < 0.0:
        raise ValueError(f"{field_name} 不得为负数")


def require_probability(value: float, field_name: str) -> None:
    """校验闭区间 [0, 1] 内的有限概率。"""

    require_finite(value, field_name)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name} 必须位于 [0, 1]")


def require_probability_sum(values: tuple[float, ...], field_name: str) -> None:
    """校验归一化概率分布。"""

    for index, value in enumerate(values):
        require_probability(value, f"{field_name}[{index}]")
    if not math.isclose(sum(values), 1.0, rel_tol=1e-7, abs_tol=1e-7):
        raise ValueError(f"{field_name} 的概率和必须为 1")


def require_sha256(value: str) -> None:
    """校验小写或大写 SHA-256 十六进制摘要。"""

    if not isinstance(value, str):
        raise TypeError("sha256 必须是字符串")
    if re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise ValueError("sha256 必须是 64 位十六进制摘要")


def require_transform_fingerprint(
    value: str,
    field_name: str = "transform_fingerprint",
) -> None:
    """校验共享坐标 API 生成的稳定变换指纹。

    当坐标变换矩阵、原帧尺寸或标定身份变化时，该值会变化，
    因此它必须作为训练样本和候选缓存的一级身份。
    """

    if not isinstance(value, str):
        raise TypeError(f"{field_name} 必须是字符串")
    if re.fullmatch(r"transform-[0-9a-f]{16}", value) is None:
        raise ValueError(f"{field_name} 必须是 transform- 前缀加 16 位小写十六进制指纹")
