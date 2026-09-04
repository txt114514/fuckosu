"""训练边界使用的基础值校验原语。

本模块只负责把外部输入收敛为可信的 Python 值。业务层应消费这些函数的返回值，
而不是在每个内部函数中重复相同的 ``isinstance``、范围和有限性检查。
"""

from __future__ import annotations

from enum import Enum
from math import isfinite
from os import PathLike, fspath
from pathlib import Path
from typing import TypeVar


_EnumT = TypeVar("_EnumT", bound=Enum)


def require_int(
    value: object,
    name: str = "value",
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """返回严格整数，并拒绝 Python 中作为 ``int`` 子类的布尔值。"""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be greater than or equal to {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be less than or equal to {maximum}")
    return value


def require_real(
    value: object,
    name: str = "value",
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """返回有限浮点数，可选地执行闭区间范围检查。"""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} 必须是有限数值 (must be finite)")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be greater than or equal to {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be less than or equal to {maximum}")
    return result


def require_bool(value: object, name: str = "value") -> bool:
    """返回严格布尔值，不接受 ``0``、``1`` 等隐式替代。"""

    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value


def require_path(
    value: object,
    name: str = "value",
    *,
    allow_string: bool = True,
    must_exist: bool = False,
    directory: bool | None = None,
) -> Path:
    """返回未解析的 :class:`Path`，并可检查存在性和文件种类。

    不调用 ``resolve`` 或 ``expanduser``，因此校验不会偷偷改变调用方提供的路径语义。
    ``directory=True`` 要求目录，``False`` 要求普通文件，``None`` 不限制种类。
    """

    if not isinstance(allow_string, bool):
        raise TypeError("allow_string must be a boolean")
    allowed_type = (str, PathLike) if allow_string else (Path,)
    if isinstance(value, bool) or not isinstance(value, allowed_type):
        raise TypeError(f"{name} must be a path")
    raw_path = fspath(value)
    if not isinstance(raw_path, str):
        raise TypeError(f"{name} must be a text path")
    if not raw_path.strip():
        raise ValueError(f"{name} must not be empty")
    if "\x00" in raw_path:
        raise ValueError(f"{name} must not contain a NUL byte")
    result = Path(raw_path)
    if must_exist and not result.exists():
        raise ValueError(f"{name} does not exist: {result}")
    if directory is True and (not result.exists() or not result.is_dir()):
        raise ValueError(f"{name} must be an existing directory: {result}")
    if directory is False and (not result.exists() or not result.is_file()):
        raise ValueError(f"{name} must be an existing file: {result}")
    return result


def require_non_empty_str(value: object, name: str = "value") -> str:
    """返回非空字符串；只含空白的字符串视为无效。"""

    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value


def require_enum(
    value: object,
    enum_type: type[_EnumT],
    name: str = "value",
    *,
    coerce: bool = True,
) -> _EnumT:
    """返回指定枚举成员，并可在外部边界把原始枚举值显式转换一次。"""

    if isinstance(value, enum_type):
        return value
    if not coerce:
        raise TypeError(f"{name} must be an instance of {enum_type.__name__}")
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        choices = ", ".join(repr(member.value) for member in enum_type)
        raise ValueError(
            f"{name} must be one of the {enum_type.__name__} values: {choices}"
        ) from exc


def require_finite(value: object, name: str = "value") -> float:
    """返回有限实数；这是无范围约束的 ``require_real`` 快捷入口。"""

    return require_real(value, name)


__all__ = [
    "require_bool",
    "require_enum",
    "require_finite",
    "require_int",
    "require_non_empty_str",
    "require_path",
    "require_real",
]
