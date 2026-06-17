from __future__ import annotations

import importlib.metadata
import importlib.util
import platform
import shutil
import sys
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class PackageSpec:
    label: str
    import_name: str | None
    distributions: tuple[str, ...]
    required: bool = True


@dataclass(frozen=True)
class PackageCheck:
    spec: PackageSpec
    available: bool
    version: str | None


@dataclass(frozen=True)
class TorchCheck:
    available: bool
    version: str | None
    torchvision_version: str | None
    torch_cuda: str | None
    cuda_available: bool
    gpu_name: str | None
    compute_capability: str | None
    cudnn_version: str | None
    bf16_supported: bool | None
    total_vram_gib: float | None
    free_vram_gib: float | None
    error: str | None = None


@dataclass(frozen=True)
class EnvironmentReport:
    python_version: str
    python_executable: str
    platform: str
    ffmpeg_path: str | None
    nvidia_smi_path: str | None
    torch: TorchCheck
    packages: tuple[PackageCheck, ...]

    @property
    def missing_required_packages(self) -> tuple[str, ...]:
        return tuple(
            check.spec.label
            for check in self.packages
            if check.spec.required and not check.available
        )

    def ready(self, *, require_cuda: bool = False) -> bool:
        if self.missing_required_packages:
            return False
        if self.ffmpeg_path is None:
            return False
        if require_cuda and not self.torch.cuda_available:
            return False
        return self.torch.available


REQUIRED_PACKAGES: tuple[PackageSpec, ...] = (
    PackageSpec("torch", "torch", ("torch",)),
    PackageSpec("torchvision", "torchvision", ("torchvision",)),
    PackageSpec("opencv", "cv2", ("opencv-python-headless", "opencv-python")),
    PackageSpec("pyav", "av", ("av",)),
    PackageSpec("numpy", "numpy", ("numpy",)),
    PackageSpec("scipy", "scipy", ("scipy",)),
    PackageSpec("pillow", "PIL", ("Pillow",)),
    PackageSpec("typer", "typer", ("typer",)),
    PackageSpec("pydantic-settings", "pydantic_settings", ("pydantic-settings",)),
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
    for distribution in distributions:
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def check_package(spec: PackageSpec) -> PackageCheck:
    version = _metadata_version(spec.distributions)
    if spec.import_name is None:
        available = version is not None
    else:
        available = importlib.util.find_spec(spec.import_name) is not None
    return PackageCheck(
        spec=spec,
        available=available,
        version=version,
    )


def collect_torch_check() -> TorchCheck:
    if importlib.util.find_spec("torch") is None:
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
            error="torch is not importable",
        )

    try:
        import torch
    except Exception as exc:  # pragma: no cover - import failure is env-specific.
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
            error=f"{type(exc).__name__}: {exc}",
        )

    cuda_available = False
    gpu_name = None
    compute_capability = None
    bf16_supported = None
    total_vram_gib = None
    free_vram_gib = None
    error = None

    try:
        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            capability = torch.cuda.get_device_capability(0)
            compute_capability = f"{capability[0]}.{capability[1]}"
            bf16_supported = bool(torch.cuda.is_bf16_supported())
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            free_vram_gib = free_bytes / 1024**3
            total_vram_gib = total_bytes / 1024**3
    except Exception as exc:  # pragma: no cover - GPU runtime is env-specific.
        error = f"{type(exc).__name__}: {exc}"

    cudnn_version = None
    try:
        cudnn = torch.backends.cudnn.version()
        cudnn_version = str(cudnn) if cudnn is not None else None
    except Exception as exc:  # pragma: no cover - backend state is env-specific.
        error = f"{type(exc).__name__}: {exc}"

    return TorchCheck(
        available=True,
        version=torch.__version__,
        torchvision_version=_metadata_version(("torchvision",)),
        torch_cuda=torch.version.cuda,
        cuda_available=cuda_available,
        gpu_name=gpu_name,
        compute_capability=compute_capability,
        cudnn_version=cudnn_version,
        bf16_supported=bf16_supported,
        total_vram_gib=total_vram_gib,
        free_vram_gib=free_vram_gib,
        error=error,
    )


def collect_environment_report() -> EnvironmentReport:
    packages = tuple(
        check_package(spec) for spec in (*REQUIRED_PACKAGES, *OPTIONAL_PACKAGES)
    )
    return EnvironmentReport(
        python_version=sys.version.replace("\n", " "),
        python_executable=sys.executable,
        platform=platform.platform(),
        ffmpeg_path=shutil.which("ffmpeg"),
        nvidia_smi_path=shutil.which("nvidia-smi"),
        torch=collect_torch_check(),
        packages=packages,
    )
