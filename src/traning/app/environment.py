"""已弃用；环境实现已迁至 :mod:`traning.lib.environment.training`。"""

from traning.lib.environment.training import (
    ConfiguredEnvironmentReport,
    EnvironmentCheckResult,
    EnvironmentCheckStatus,
    EnvironmentNotReadyError,
    EnvironmentReport,
    check_v2_environment,
    require_v2_environment,
)

__deprecated__ = True

__all__ = (
    "ConfiguredEnvironmentReport",
    "EnvironmentCheckResult",
    "EnvironmentCheckStatus",
    "EnvironmentNotReadyError",
    "EnvironmentReport",
    "check_v2_environment",
    "require_v2_environment",
)
