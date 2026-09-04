"""已弃用的环境导入路径；实现位于 :mod:`traning.lib.environment`。"""

from traning.lib.environment import (
    OPTIONAL_PACKAGES,
    REQUIRED_PACKAGES,
    EnvironmentReport,
    PackageCheck,
    PackageSpec,
    TorchCheck,
    check_package,
    collect_environment_report,
    collect_torch_check,
)

__deprecated__ = True

__all__ = [
    "EnvironmentReport",
    "OPTIONAL_PACKAGES",
    "PackageCheck",
    "PackageSpec",
    "REQUIRED_PACKAGES",
    "TorchCheck",
    "check_package",
    "collect_environment_report",
    "collect_torch_check",
]
