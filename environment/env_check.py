"""已弃用的兼容包装；请改用 :mod:`traning.lib.environment.env_check`。"""

from traning.lib.environment.env_check import (
    OPTIONAL_PACKAGES,
    REQUIRED_PACKAGES,
    EnvironmentReport,
    PackageCheck,
    PackageSpec,
    TorchCheck,
    build_parser,
    check_package,
    collect_environment_report,
    collect_torch_check,
    main,
    render_environment_report,
)

__deprecated__ = True

if __name__ == "__main__":  # pragma: no cover - 兼容旧命令。
    raise SystemExit(main())

__all__ = (
    "EnvironmentReport",
    "OPTIONAL_PACKAGES",
    "PackageCheck",
    "PackageSpec",
    "REQUIRED_PACKAGES",
    "TorchCheck",
    "build_parser",
    "check_package",
    "collect_environment_report",
    "collect_torch_check",
    "main",
    "render_environment_report",
)
