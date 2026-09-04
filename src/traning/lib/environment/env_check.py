"""环境报告的公开查询入口和无副作用命令行格式化。"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from traning.state.environment import (
    EnvironmentReport,
    PackageCheck,
    PackageSpec,
    TorchCheck,
)

from .report import (
    OPTIONAL_PACKAGES,
    REQUIRED_PACKAGES,
    check_package,
    collect_environment_report,
    collect_torch_check,
)


def render_environment_report(report: EnvironmentReport) -> str:
    """将报告渲染为确定性、机器可读的 JSON。"""

    return json.dumps(report.as_dict(), ensure_ascii=False, indent=2, sort_keys=True)


def build_parser() -> argparse.ArgumentParser:
    """构造独立环境检查命令的参数解析器。"""

    parser = argparse.ArgumentParser(
        prog="python -m traning.lib.environment.env_check",
        description="Inspect Python packages, FFmpeg, PyTorch and CUDA.",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="consider the report not ready when CUDA is unavailable",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="return a non-zero exit status when the environment is not ready",
    )
    parser.add_argument(
        "--disallow-cpu",
        action="store_true",
        help="declare CPU-only execution unacceptable for this invocation",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """打印报告；仅 strict 模式把未就绪状态转换为失败退出码。"""

    args = build_parser().parse_args(argv)
    report = collect_environment_report(cpu_mode_allowed=not args.disallow_cpu)
    print(render_environment_report(report))
    require_cuda = bool(args.require_cuda or args.disallow_cpu)
    return int(args.strict and not report.ready(require_cuda=require_cuda))


if __name__ == "__main__":  # pragma: no cover - 由命令行 smoke 覆盖。
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
