"""只读收集 Python、依赖、系统工具和 PyTorch/CUDA 环境信息。"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import platform
import shutil
import sys
from collections.abc import Iterable

from traning.state.environment import (
    EnvironmentReport,
    PackageCheck,
    PackageSpec,
    TorchCheck,
)


# 规格表是环境报告的唯一包清单；展示、严格判断和测试均消费同一份数据。
REQUIRED_PACKAGES: tuple[PackageSpec, ...] = (
    PackageSpec("torch", "torch", ("torch",)),
    PackageSpec("torchvision", "torchvision", ("torchvision",)),
    PackageSpec("opencv", "cv2", ("opencv-python-headless", "opencv-python")),
    PackageSpec("pyav", "av", ("av",)),
    PackageSpec("numpy", "numpy", ("numpy",)),
    PackageSpec("scipy", "scipy", ("scipy",)),
    PackageSpec("pillow", "PIL", ("Pillow",)),
    PackageSpec("typer", "typer", ("typer",)),
    PackageSpec(
        "pydantic-settings",
        "pydantic_settings",
        ("pydantic-settings",),
    ),
    PackageSpec("pyyaml", "yaml", ("PyYAML",)),
    PackageSpec("prefect", "prefect", ("prefect",)),
    PackageSpec("einops", "einops", ("einops",)),
    PackageSpec("safetensors", "safetensors", ("safetensors",)),
    PackageSpec("psutil", "psutil", ("psutil",)),
)

OPTIONAL_PACKAGES: tuple[PackageSpec, ...] = (
    PackageSpec("optuna", "optuna", ("optuna",), required=False),
    PackageSpec("timm", "timm", ("timm",), required=False),
    PackageSpec(
        "huggingface-hub",
        "huggingface_hub",
        ("huggingface-hub",),
        required=False,
    ),
    PackageSpec("tensorboard", "tensorboard", ("tensorboard",), required=False),
    PackageSpec("pytest", "pytest", ("pytest",), required=False),
    PackageSpec("pytest-cov", "pytest_cov", ("pytest-cov",), required=False),
    PackageSpec("ruff", "ruff", ("ruff",), required=False),
    PackageSpec("mypy", "mypy", ("mypy",), required=False),
    PackageSpec("types-PyYAML", None, ("types-PyYAML",), required=False),
)


def _metadata_version(distributions: Iterable[str]) -> str | None:
    """返回第一个已安装发行包的版本，不导入其运行时代码。"""

    for distribution in distributions:
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def check_package(spec: PackageSpec) -> PackageCheck:
    """只通过模块规格和发行元数据检查一个依赖。"""

    version = _metadata_version(spec.distributions)
    try:
        available = (
            version is not None
            if spec.import_name is None
            else importlib.util.find_spec(spec.import_name) is not None
        )
    except (ImportError, ModuleNotFoundError, ValueError) as exc:
        return PackageCheck(
            spec=spec,
            available=False,
            version=version,
            error=f"{type(exc).__name__}: {exc}",
        )
    return PackageCheck(spec=spec, available=available, version=version)


def _torch_unavailable() -> TorchCheck:
    """构造 torch 完全不可导入时的显式失败结果。"""

    reason = "torch is not importable"
    return TorchCheck(
        available=False,
        version=None,
        torchvision_version=_metadata_version(("torchvision",)),
        torch_cuda=None,
        cuda_available=False,
        gpu_name=None,
        compute_capability=None,
        cudnn_version=None,
        bf16_supported=None,
        total_vram_gib=None,
        free_vram_gib=None,
        error=reason,
        cuda_unavailable_reason=reason,
    )


def _cuda_unavailable_reason(torch_module: object) -> str:
    """依据 PyTorch 构建信息解释 CUDA 不可用，而不是静默写 False。"""

    version = getattr(torch_module, "version", None)
    compiled_cuda = getattr(version, "cuda", None)
    if compiled_cuda is None:
        return "installed PyTorch build does not include CUDA support"
    return (
        "torch.cuda.is_available() returned False; no usable CUDA driver/device "
        f"is visible to this process (PyTorch CUDA build {compiled_cuda})"
    )


def collect_torch_check(*, device_index: int = 0) -> TorchCheck:
    """探测 PyTorch 构建与一个 CUDA 设备；探测异常作为报告返回。"""

    if importlib.util.find_spec("torch") is None:
        return _torch_unavailable()

    # 延迟导入保证环境检查在 torch 安装损坏时仍能给出结构化报告。
    try:
        import torch
    except Exception as exc:  # pragma: no cover - 取决于本机二进制环境。
        reason = f"torch import failed: {type(exc).__name__}: {exc}"
        return TorchCheck(
            available=False,
            version=None,
            torchvision_version=_metadata_version(("torchvision",)),
            torch_cuda=None,
            cuda_available=False,
            gpu_name=None,
            compute_capability=None,
            cudnn_version=None,
            bf16_supported=None,
            total_vram_gib=None,
            free_vram_gib=None,
            error=reason,
            cuda_unavailable_reason=reason,
        )

    cuda_available = False
    gpu_name: str | None = None
    compute_capability: str | None = None
    bf16_supported: bool | None = None
    total_vram_gib: float | None = None
    free_vram_gib: float | None = None
    probe_error: str | None = None
    unavailable_reason: str | None = None

    try:
        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(device_index)
            capability = torch.cuda.get_device_capability(device_index)
            compute_capability = f"{capability[0]}.{capability[1]}"
            bf16_supported = bool(torch.cuda.is_bf16_supported())
            free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
            free_vram_gib = free_bytes / 1024**3
            total_vram_gib = total_bytes / 1024**3
        else:
            unavailable_reason = _cuda_unavailable_reason(torch)
    except Exception as exc:  # pragma: no cover - 取决于驱动和设备状态。
        cuda_available = False
        probe_error = f"CUDA probe failed: {type(exc).__name__}: {exc}"
        unavailable_reason = probe_error

    cudnn_version: str | None = None
    try:
        cudnn = torch.backends.cudnn.version()
        cudnn_version = str(cudnn) if cudnn is not None else None
    except Exception as exc:  # pragma: no cover - 取决于 torch backend 状态。
        backend_error = f"cuDNN probe failed: {type(exc).__name__}: {exc}"
        probe_error = (
            f"{probe_error}; {backend_error}" if probe_error else backend_error
        )

    return TorchCheck(
        available=True,
        version=str(torch.__version__),
        torchvision_version=_metadata_version(("torchvision",)),
        torch_cuda=torch.version.cuda,
        cuda_available=cuda_available,
        gpu_name=gpu_name,
        compute_capability=compute_capability,
        cudnn_version=cudnn_version,
        bf16_supported=bf16_supported,
        total_vram_gib=total_vram_gib,
        free_vram_gib=free_vram_gib,
        error=probe_error,
        cuda_unavailable_reason=unavailable_reason,
    )


def collect_environment_report(
    *,
    cpu_mode_allowed: bool = True,
) -> EnvironmentReport:
    """收集完整环境报告，不安装包、不修改 torch 或 CUDA 全局状态。"""

    packages = tuple(
        check_package(spec) for spec in (*REQUIRED_PACKAGES, *OPTIONAL_PACKAGES)
    )
    torch_check = collect_torch_check()
    return EnvironmentReport(
        python_version=sys.version.replace("\n", " "),
        python_executable=sys.executable,
        platform=platform.platform(),
        ffmpeg_path=shutil.which("ffmpeg"),
        nvidia_smi_path=shutil.which("nvidia-smi"),
        torch=torch_check,
        packages=packages,
        cpu_mode_allowed=cpu_mode_allowed,
        cuda_unavailable_reason=torch_check.cuda_unavailable_reason,
    )


__all__ = (
    "OPTIONAL_PACKAGES",
    "REQUIRED_PACKAGES",
    "check_package",
    "collect_environment_report",
    "collect_torch_check",
)
