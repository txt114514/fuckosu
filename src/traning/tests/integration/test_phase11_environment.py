"""V2 启动环境检查的集成验收。"""

from __future__ import annotations

import pytest
from pathlib import Path

from traning.app.environment import (
    EnvironmentCheckStatus,
    EnvironmentNotReadyError,
    check_v2_environment,
    require_v2_environment,
)
from traning.config import CoordinateConfig, RuntimeConfig, RuntimeDevice, V2Config
from traning.config import load_v2_config


_IDENTITY_MATRIX = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))


def test_cpu_environment_with_explicit_coordinates_passes(monkeypatch) -> None:
    """CPU smoke 不得因 sandbox 看不到 CUDA 而失败。"""

    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    config = V2Config(
        runtime=RuntimeConfig(
            device=RuntimeDevice.CPU,
            require_cuda=False,
            amp=False,
        ),
        coordinates=CoordinateConfig(
            513,
            385,
            "test-identity-v1",
            _IDENTITY_MATRIX,
        ),
    )

    report = require_v2_environment(config)
    assert report.ok
    assert all(item.status is EnvironmentCheckStatus.PASSED for item in report.results)


def test_required_cuda_unavailable_is_a_typed_failure(monkeypatch) -> None:
    """设备不可见必须报告真实失败，不能静默退回 CPU。"""

    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    config = V2Config(
        coordinates=CoordinateConfig(
            513,
            385,
            "test-identity-v1",
            _IDENTITY_MATRIX,
        ),
    )

    report = check_v2_environment(config)
    assert not report.ok
    with pytest.raises(EnvironmentNotReadyError) as captured:
        require_v2_environment(config)
    assert captured.value.report == report


def test_missing_coordinate_calibration_blocks_formal_environment(monkeypatch) -> None:
    """正式训练/评分不得使用 centered transform 静默兜底。"""

    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    report = check_v2_environment(V2Config())

    assert not report.ok
    coordinate = next(item for item in report.results if item.name == "coordinates")
    assert coordinate.status is EnvironmentCheckStatus.FAILED


def test_formal_config_exposes_validation_only_coordinate_warning(monkeypatch) -> None:
    """控制点可通过启动门，但缺失的原拟合集必须保持可见 warning。"""

    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    config = load_v2_config(Path("configs/traning.yaml"))

    report = check_v2_environment(config)
    evidence = next(
        item for item in report.results if item.name == "coordinate_evidence"
    )

    assert report.ok
    assert evidence.status is EnvironmentCheckStatus.WARNING
    assert "原始拟合集未入库" in evidence.message
