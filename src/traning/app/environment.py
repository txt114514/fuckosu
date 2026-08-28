"""V2 配置与 PyTorch 设备的只读启动前检查。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch
from package import AffineOsuVideoTransform

from traning.config import RuntimeDevice, V2Config
from traning.data import (
    FrameCoordinateTransform,
    audit_affine_calibration,
    load_affine_calibration_evidence,
)
from traning.infrastructure import IntegrityError, SchemaMismatchError


class EnvironmentCheckStatus(str, Enum):
    """单项启动检查的非歧义状态。"""

    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class EnvironmentCheckResult:
    """一项无副作用环境检查的 typed 结果。"""

    name: str
    status: EnvironmentCheckStatus
    message: str

    def __post_init__(self) -> None:
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
class EnvironmentReport:
    """V2 启动边界消费的完整不可变检查报告。"""

    results: tuple[EnvironmentCheckResult, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.results, tuple) or not self.results:
            raise TypeError("results 必须是非空 EnvironmentCheckResult 元组")
        if any(not isinstance(item, EnvironmentCheckResult) for item in self.results):
            raise TypeError("results 只能包含 EnvironmentCheckResult")
        names = tuple(item.name for item in self.results)
        if len(names) != len(set(names)):
            raise ValueError("environment check name 不得重复")

    @property
    def ok(self) -> bool:
        """没有阻断失败时返回真；可见 warning 不会伪装成完整证据。"""

        return all(
            item.status is not EnvironmentCheckStatus.FAILED for item in self.results
        )


class EnvironmentNotReadyError(RuntimeError):
    """配置要求的运行设备不可用。"""

    def __init__(self, report: EnvironmentReport) -> None:
        if report.ok:
            raise ValueError("EnvironmentNotReadyError 只能包装失败报告")
        self.report = report
        failed = "; ".join(
            item.message
            for item in report.results
            if item.status is EnvironmentCheckStatus.FAILED
        )
        super().__init__(failed)


def check_v2_environment(config: V2Config) -> EnvironmentReport:
    """检查配置一致性与实际 CUDA 可见性，不修改全局 torch 状态。"""

    if not isinstance(config, V2Config):
        raise TypeError("config 必须是 V2Config")
    results = [
        EnvironmentCheckResult(
            name="config",
            status=EnvironmentCheckStatus.PASSED,
            message=f"V2 config schema {config.schema_version} 已验证",
        )
    ]
    cuda_available = torch.cuda.is_available()
    cuda_required = (
        config.runtime.device is RuntimeDevice.CUDA or config.runtime.require_cuda
    )
    if cuda_required and not cuda_available:
        results.append(
            EnvironmentCheckResult(
                name="device",
                status=EnvironmentCheckStatus.FAILED,
                message="配置要求 CUDA，但当前 PyTorch namespace 不可见 CUDA",
            )
        )
    else:
        selected = "cuda" if config.runtime.device is RuntimeDevice.CUDA else "cpu"
        results.append(
            EnvironmentCheckResult(
                name="device",
                status=EnvironmentCheckStatus.PASSED,
                message=f"运行设备 {selected} 可用",
            )
        )
    coordinate_error: str | None = None
    coordinate_transform: FrameCoordinateTransform | None = None
    if config.coordinates.affine_matrix is None:
        coordinate_error = "缺少离线训练/评分 affine 坐标标定"
    else:
        try:
            coordinate_transform = FrameCoordinateTransform(
                source_frame_width=config.coordinates.source_width,
                source_frame_height=config.coordinates.source_height,
                transform_identity=config.coordinates.transform_identity,
                transform=AffineOsuVideoTransform(config.coordinates.affine_matrix),
            )
        except (TypeError, ValueError) as exc:
            coordinate_error = f"affine 坐标标定无效：{exc}"
    results.append(
        EnvironmentCheckResult(
            name="coordinates",
            status=(
                EnvironmentCheckStatus.PASSED
                if coordinate_error is None
                else EnvironmentCheckStatus.FAILED
            ),
            message=(
                "离线训练/评分 affine 坐标标定已配置"
                if coordinate_error is None
                else coordinate_error
            ),
        )
    )
    evidence_path = config.coordinates.calibration_evidence_path
    if coordinate_transform is not None and evidence_path is not None:
        try:
            loaded_evidence = load_affine_calibration_evidence(evidence_path)
            audit = audit_affine_calibration(
                coordinate_transform,
                loaded_evidence,
            )
        except (IntegrityError, SchemaMismatchError, TypeError, ValueError) as exc:
            results.append(
                EnvironmentCheckResult(
                    name="coordinate_evidence",
                    status=EnvironmentCheckStatus.FAILED,
                    message=f"坐标证据无法验证：{exc}",
                )
            )
        else:
            if not audit.ok:
                evidence_status = EnvironmentCheckStatus.FAILED
                evidence_message = (
                    "坐标证据与正式变换不一致，或独立控制点超过 "
                    f"{audit.max_allowed_error_px:.3f}px 门限"
                )
            elif audit.fit_reproducible:
                evidence_status = EnvironmentCheckStatus.PASSED
                evidence_message = (
                    f"坐标拟合与 {audit.control_count} 个独立控制点均可复现；"
                    f"最大残差 {audit.max_error_px:.3f}px"
                )
            else:
                evidence_status = EnvironmentCheckStatus.WARNING
                evidence_message = (
                    f"{audit.control_count} 个独立控制点验证通过，最大残差 "
                    f"{audit.max_error_px:.3f}px；原始拟合集未入库，当前方程只能"
                    "标记为 legacy control-validated"
                )
            results.append(
                EnvironmentCheckResult(
                    name="coordinate_evidence",
                    status=evidence_status,
                    message=evidence_message,
                )
            )
    return EnvironmentReport(tuple(results))


def require_v2_environment(config: V2Config) -> EnvironmentReport:
    """返回通过报告；任一失败则抛出携带完整报告的 typed error。"""

    report = check_v2_environment(config)
    if not report.ok:
        raise EnvironmentNotReadyError(report)
    return report


__all__ = (
    "EnvironmentCheckResult",
    "EnvironmentCheckStatus",
    "EnvironmentNotReadyError",
    "EnvironmentReport",
    "check_v2_environment",
    "require_v2_environment",
)
