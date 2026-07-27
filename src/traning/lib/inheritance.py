"""训练继承包 API 的兼容转发入口。"""

from traning.core.training_inheritance import (
    InheritanceLoadResult,
    InheritancePackage,
    create_inheritance_package,
    load_inheritance_package,
)

__all__ = [
    "InheritanceLoadResult",
    "InheritancePackage",
    "create_inheritance_package",
    "load_inheritance_package",
]
