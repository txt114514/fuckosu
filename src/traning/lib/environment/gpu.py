"""CUDA/GPU 专项探测及其独立命令行入口。"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from traning.state.common import JSONObject
from traning.state.environment import TorchCheck

from .report import collect_torch_check


def gpu_check_as_dict(check: TorchCheck) -> JSONObject:
    """将 GPU 探测结果转换为稳定 JSON 数据。"""

    return {
        "torch_available": check.available,
        "torch_version": check.version,
        "torchvision_version": check.torchvision_version,
        "torch_cuda": check.torch_cuda,
        "cuda_available": check.cuda_available,
        "cuda_unavailable_reason": check.cuda_unavailable_reason,
        "gpu_name": check.gpu_name,
        "compute_capability": check.compute_capability,
        "cudnn_version": check.cudnn_version,
        "bf16_supported": check.bf16_supported,
        "total_vram_gib": check.total_vram_gib,
        "free_vram_gib": check.free_vram_gib,
        "error": check.error,
    }


def build_parser() -> argparse.ArgumentParser:
    """构造 GPU 专项检查命令参数。"""

    parser = argparse.ArgumentParser(
        prog="python -m traning.lib.environment.gpu",
        description="Inspect the CUDA device visible to PyTorch.",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=0,
        help="CUDA device index to inspect (default: 0)",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="return a non-zero exit status when CUDA is unavailable",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """输出 GPU 报告，并按 require-cuda 给出真实退出状态。"""

    args = build_parser().parse_args(argv)
    if args.device_index < 0:
        raise SystemExit("--device-index must be non-negative")
    check = collect_torch_check(device_index=args.device_index)
    print(json.dumps(gpu_check_as_dict(check), ensure_ascii=False, indent=2))
    return int(args.require_cuda and not check.cuda_available)


if __name__ == "__main__":  # pragma: no cover - 由 shell smoke 覆盖。
    raise SystemExit(main())


__all__ = ("build_parser", "gpu_check_as_dict", "main")
