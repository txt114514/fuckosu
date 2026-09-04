"""环境探测与训练启动检查共享的不可变数据契约。

本模块只声明数据，不执行导入探测、CUDA 初始化或配置检查。这样环境实现可以
依赖这些类型，而不会让 :mod:`traning.state` 反向依赖运行时工具。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .common import JSONObject


@dataclass(frozen=True, slots=True)
class PackageSpec:
    """描述一个发行包、可导入模块及其是否为训练硬依赖。"""

    label: str
    import_name: str | None
    distributions: tuple[str, ...]
    required: bool = True

    def __post_init__(self) -> None:
        """在外部规格表边界拒绝含糊或不可查询的包描述。"""

        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("package label 必须是非空字符串")
        if self.label != self.label.strip():
            raise ValueError("package label 不得包含首尾空格")
        if self.import_name is not None and (
            not isinstance(self.import_name, str) or not self.import_name.strip()
        ):
            raise ValueError("import_name 必须为 None 或非空字符串")
        if not isinstance(self.distributions, tuple) or not self.distributions:
            raise TypeError("distributions 必须是非空字符串元组")
        if any(
            not isinstance(item, str) or not item.strip() for item in self.distributions
        ):
            raise ValueError("distributions 只能包含非空字符串")
        if not isinstance(self.required, bool):
            raise TypeError("required 必须是 bool")


@dataclass(frozen=True, slots=True)
class PackageCheck:
    """单个依赖的可发现状态、发行版本和可选失败原因。"""

    spec: PackageSpec
    available: bool
    version: str | None
    error: str | None = None

    def __post_init__(self) -> None:
        """验证环境探测器输出的最小结构约束。"""

        if not isinstance(self.spec, PackageSpec):
            raise TypeError("spec 必须是 PackageSpec")
        if not isinstance(self.available, bool):
            raise TypeError("available 必须是 bool")


@dataclass(frozen=True, slots=True)
class TorchCheck:
    """PyTorch 构建信息和零号 CUDA 设备能力。"""

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
    cuda_unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        """确保布尔状态不会被字符串或整数冒充。"""

        if not isinstance(self.available, bool):
            raise TypeError("available 必须是 bool")
        if not isinstance(self.cuda_available, bool):
            raise TypeError("cuda_available 必须是 bool")
        if self.cuda_available and self.cuda_unavailable_reason is not None:
            raise ValueError("CUDA 可用时不得同时声明不可用原因")


@dataclass(frozen=True, slots=True)
class EnvironmentReport:
    """聚合依赖、工具链和设备状态的宿主环境报告。"""

    python_version: str
    python_executable: str
    platform: str
    ffmpeg_path: str | None
    nvidia_smi_path: str | None
    torch: TorchCheck
    packages: tuple[PackageCheck, ...]
    cpu_mode_allowed: bool = True
    cuda_unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        """验证报告组合，并统一顶层 CUDA 不可用原因。"""

        if not isinstance(self.torch, TorchCheck):
            raise TypeError("torch 必须是 TorchCheck")
        if not isinstance(self.packages, tuple) or any(
            not isinstance(item, PackageCheck) for item in self.packages
        ):
            raise TypeError("packages 必须是 PackageCheck 元组")
        if not isinstance(self.cpu_mode_allowed, bool):
            raise TypeError("cpu_mode_allowed 必须是 bool")
        if self.torch.cuda_available and self.cuda_unavailable_reason is not None:
            raise ValueError("CUDA 可用时不得同时声明不可用原因")
        if not self.torch.cuda_available and self.cuda_unavailable_reason is None:
            object.__setattr__(
                self,
                "cuda_unavailable_reason",
                self.torch.cuda_unavailable_reason or "CUDA is unavailable",
            )

    @property
    def missing_required_packages(self) -> tuple[str, ...]:
        """返回未发现的硬依赖标签，顺序与规格表一致。"""

        return tuple(
            check.spec.label
            for check in self.packages
            if check.spec.required and not check.available
        )

    def ready(self, *, require_cuda: bool = False) -> bool:
        """按调用方设备要求判断报告是否满足启动条件。"""

        if self.missing_required_packages or self.ffmpeg_path is None:
            return False
        if not self.torch.available:
            return False
        if require_cuda and not self.torch.cuda_available:
            return False
        return self.cpu_mode_allowed or self.torch.cuda_available

    def as_dict(self) -> JSONObject:
        """生成可直接 JSON 序列化且字段稳定的环境报告。"""

        return {
            "python": {
                "version": self.python_version,
                "executable": self.python_executable,
                "platform": self.platform,
            },
            "tools": {
                "ffmpeg": self.ffmpeg_path,
                "nvidia_smi": self.nvidia_smi_path,
            },
            "torch": {
                "available": self.torch.available,
                "version": self.torch.version,
                "torchvision_version": self.torch.torchvision_version,
                "torch_cuda": self.torch.torch_cuda,
                "cuda_available": self.torch.cuda_available,
                "cuda_unavailable_reason": self.torch.cuda_unavailable_reason,
                "gpu_name": self.torch.gpu_name,
                "compute_capability": self.torch.compute_capability,
                "cudnn_version": self.torch.cudnn_version,
                "bf16_supported": self.torch.bf16_supported,
                "total_vram_gib": self.torch.total_vram_gib,
                "free_vram_gib": self.torch.free_vram_gib,
                "error": self.torch.error,
            },
            "packages": {
                check.spec.label: {
                    "available": check.available,
                    "version": check.version,
                    "required": check.spec.required,
                    "error": check.error,
                }
                for check in self.packages
            },
            "cpu_mode_allowed": self.cpu_mode_allowed,
            "cuda_unavailable_reason": self.cuda_unavailable_reason,
            "missing_required_packages": list(self.missing_required_packages),
        }


class EnvironmentCheckStatus(str, Enum):
    """一项配置化训练环境检查的非歧义状态。"""

    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class EnvironmentCheckResult:
    """配置化训练启动检查的单项结果。"""

    name: str
    status: EnvironmentCheckStatus
    message: str

    def __post_init__(self) -> None:
        """在检查结果产生边界验证展示字段。"""

        if (
            not isinstance(self.name, str)
            or not self.name
            or self.name != self.name.strip()
        ):
            raise ValueError("check name 必须非空且无首尾空格")
        if not isinstance(self.status, EnvironmentCheckStatus):
            raise TypeError("status 必须是 EnvironmentCheckStatus")
        if (
            not isinstance(self.message, str)
            or not self.message
            or self.message != self.message.strip()
        ):
            raise ValueError("message 必须非空且无首尾空格")


@dataclass(frozen=True, slots=True)
class ConfiguredEnvironmentReport:
    """已结合训练配置、设备和坐标证据的检查报告。"""

    results: tuple[EnvironmentCheckResult, ...]

    def __post_init__(self) -> None:
        """在报告边界验证类型、非空性和检查名唯一性。"""

        if not isinstance(self.results, tuple) or not self.results:
            raise TypeError("results 必须是非空 EnvironmentCheckResult 元组")
        if any(not isinstance(item, EnvironmentCheckResult) for item in self.results):
            raise TypeError("results 只能包含 EnvironmentCheckResult")
        names = tuple(item.name for item in self.results)
        if len(names) != len(set(names)):
            raise ValueError("environment check name 不得重复")

    @property
    def ok(self) -> bool:
        """没有阻断失败时为真；warning 保持可见但不阻断。"""

        return all(
            item.status is not EnvironmentCheckStatus.FAILED for item in self.results
        )

    def as_dict(self) -> JSONObject:
        """生成训练 CLI 使用的稳定 JSON 对象。"""

        return {
            "ok": self.ok,
            "results": [
                {
                    "name": item.name,
                    "status": item.status.value,
                    "message": item.message,
                }
                for item in self.results
            ],
        }


__all__ = (
    "ConfiguredEnvironmentReport",
    "EnvironmentCheckResult",
    "EnvironmentCheckStatus",
    "EnvironmentReport",
    "PackageCheck",
    "PackageSpec",
    "TorchCheck",
)
