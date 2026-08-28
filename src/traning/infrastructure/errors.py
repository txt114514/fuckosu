"""基础设施边界使用的显式错误类型。"""

from __future__ import annotations


class InfrastructureError(RuntimeError):
    """OSU V2 基础设施错误的共同基类。"""


class SchemaMismatchError(InfrastructureError):
    """持久化数据的结构与调用方要求的 schema 不一致。"""


class IntegrityError(InfrastructureError):
    """持久化数据不完整、损坏或不满足严格格式要求。"""


class AtomicWriteError(InfrastructureError):
    """原子写入未能完整发布并持久化。"""


__all__ = (
    "AtomicWriteError",
    "InfrastructureError",
    "IntegrityError",
    "SchemaMismatchError",
)
