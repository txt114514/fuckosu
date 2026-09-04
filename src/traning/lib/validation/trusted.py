"""已验证值标记与可选的内部诊断开关。"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Generic, TypeVar


_T = TypeVar("_T")

STRICT_INTERNAL_CHECKS_ENV = "TRANING_STRICT_INTERNAL_CHECKS"


@dataclass(frozen=True, slots=True)
class Trusted(Generic[_T]):
    """携带边界身份的不可变已验证值。

    包装器使内部 API 能在类型层表达“已经由哪个边界验证”，无需通过全局缓存或修改
    Tensor。调用 ``unwrap`` 只取回原对象，不复制内容。
    """

    value: _T
    boundary: str

    def __post_init__(self) -> None:
        if not self.boundary.strip():
            raise ValueError("Trusted boundary must not be empty")

    def unwrap(self) -> _T:
        """返回边界已经验证的原对象。"""

        return self.value


def strict_internal_checks_enabled(
    environ: Mapping[str, str] | None = None,
) -> bool:
    """仅在环境变量精确等于 ``1`` 时启用昂贵的内部诊断。"""

    source = os.environ if environ is None else environ
    return source.get(STRICT_INTERNAL_CHECKS_ENV) == "1"


def validate_internal(
    value: _T,
    validator: Callable[[_T], object],
) -> _T:
    """按诊断开关执行内部断言；默认热路径直接返回同一对象。"""

    if strict_internal_checks_enabled():
        validator(value)
    return value


def unwrap_trusted(value: Trusted[_T], *, boundary: str | None = None) -> _T:
    """取出已验证值，并可廉价核对其来源边界。"""

    if not isinstance(value, Trusted):
        raise TypeError("value must be Trusted")
    if boundary is not None and value.boundary != boundary:
        raise ValueError(
            f"trusted value came from {value.boundary!r}, expected {boundary!r}"
        )
    return value.value


__all__ = [
    "STRICT_INTERNAL_CHECKS_ENV",
    "Trusted",
    "strict_internal_checks_enabled",
    "unwrap_trusted",
    "validate_internal",
]
