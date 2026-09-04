"""结合严格训练配置、CUDA 可见性和坐标证据的启动门禁。"""

from __future__ import annotations

import torch
from package import AffineOsuVideoTransform

from traning.conf import RuntimeDevice, V2Config
from traning.core.data import (
    FrameCoordinateTransform,
    audit_affine_calibration,
    load_affine_calibration_evidence,
)
from traning.lib.infrastructure import IntegrityError, SchemaMismatchError
from traning.state.environment import (
    ConfiguredEnvironmentReport,
    EnvironmentCheckResult,
    EnvironmentCheckStatus,
)


class EnvironmentNotReadyError(RuntimeError):
    """配置要求的运行环境不可用。"""

    def __init__(self, report: ConfiguredEnvironmentReport) -> None:
        """保留完整 typed 报告并汇总阻断原因。"""

        if report.ok:
            raise ValueError("EnvironmentNotReadyError 只能包装失败报告")
        self.report = report
        failed = "; ".join(
            item.message
            for item in report.results
            if item.status is EnvironmentCheckStatus.FAILED
        )
        super().__init__(failed)


def check_v2_environment(config: V2Config) -> ConfiguredEnvironmentReport:
    """检查配置、实际 CUDA 可见性和正式坐标证据，不修改 torch 状态。"""

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
                message=(
                    "配置要求 CUDA，但 torch.cuda.is_available() 为 False；"
                    "未静默回退 CPU"
                ),
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
            audit = audit_affine_calibration(coordinate_transform, loaded_evidence)
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
    return ConfiguredEnvironmentReport(tuple(results))


def require_v2_environment(config: V2Config) -> ConfiguredEnvironmentReport:
    """返回通过的配置化报告；阻断项存在时抛出携带报告的异常。"""

    report = check_v2_environment(config)
    if not report.ok:
        raise EnvironmentNotReadyError(report)
    return report


# 旧 ``traning.app.environment.EnvironmentReport`` 名称仅是同一类型对象的别名。
EnvironmentReport = ConfiguredEnvironmentReport


__all__ = (
    "ConfiguredEnvironmentReport",
    "EnvironmentCheckResult",
    "EnvironmentCheckStatus",
    "EnvironmentNotReadyError",
    "EnvironmentReport",
    "check_v2_environment",
    "require_v2_environment",
)
