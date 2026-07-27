"""提供所有共享 dataclass 契约的递归序列化与反序列化基类。"""

from __future__ import annotations

from dataclasses import asdict, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar


ContractT = TypeVar("ContractT", bound="ContractMixin")


def contract_to_dict(value: Any) -> Any:
    """递归把契约 dataclass、枚举和路径转换为 JSON 友好容器。"""
    if is_dataclass(value):
        return {
            key: contract_to_dict(item)
            for key, item in asdict(value).items()
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, tuple):
        return [contract_to_dict(item) for item in value]
    if isinstance(value, list):
        return [contract_to_dict(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): contract_to_dict(item)
            for key, item in value.items()
        }
    return value


class ContractMixin:
    """为稳定 dataclass 契约提供容错字段构造与统一序列化。"""

    @classmethod
    def from_mapping(cls: type[ContractT], data: dict[str, Any]) -> ContractT:
        if not isinstance(data, dict):
            raise TypeError("contract data must be a mapping")
        # 忽略未来版本增加的未知字段，使旧消费者能读取向前扩展的 manifest。
        valid_names = {
            field.name
            for field in fields(cls)
            if field.init
        }
        return cls(
            **{
                name: data[name]
                for name in valid_names
                if name in data
            }
        )

    def as_dict(self) -> dict[str, Any]:
        rendered = contract_to_dict(self)
        if not isinstance(rendered, dict):
            raise TypeError("contract did not render as a mapping")
        return rendered


__all__ = ["ContractMixin", "contract_to_dict"]
