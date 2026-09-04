"""坐标方程拟合、证据来源与独立控制点审计验收。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from traning.core.app import build_frame_coordinate_transform
from traning.conf import load_v2_config
from traning.core.data import (
    CalibrationObservation,
    audit_affine_calibration,
    fit_affine_least_squares,
    load_affine_calibration_evidence,
)
from traning.lib.infrastructure import SchemaMismatchError


_WORKSPACE = Path(__file__).resolve().parents[4]
_CONFIG_PATH = _WORKSPACE / "configs/traning.yaml"
_EVIDENCE_PATH = _WORKSPACE / "configs/traning_coordinate_evidence.json"


def _observation(
    identifier: str,
    osu_x: float,
    osu_y: float,
    matrix: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> CalibrationObservation:
    """按指定精确方程生成无噪声测试观测。"""

    return CalibrationObservation(
        control_id=identifier,
        osu_x=osu_x,
        osu_y=osu_y,
        video_x=matrix[0][0] * osu_x + matrix[0][1] * osu_y + matrix[0][2],
        video_y=matrix[1][0] * osu_x + matrix[1][1] * osu_y + matrix[1][2],
        source_reference=f"synthetic:{identifier}",
    )


def test_checked_in_coordinate_evidence_validates_but_does_not_claim_refit() -> None:
    """五个控制点通过不等于原始 passed 拟合集可重放。"""

    config = load_v2_config(_CONFIG_PATH)
    loaded = load_affine_calibration_evidence(_EVIDENCE_PATH)
    report = audit_affine_calibration(
        build_frame_coordinate_transform(config),
        loaded,
    )

    assert report.ok
    assert report.control_count == 5
    assert report.fit_reproducible is False
    assert report.mean_error_px == pytest.approx(0.4486107834, abs=1e-9)
    assert report.rmse_error_px == pytest.approx(0.6701845065, abs=1e-9)
    assert report.max_error_px == pytest.approx(1.3936472394, abs=1e-9)
    assert report.worst_control_id == "legacy-roi-control-005"
    assert "pass-sample" not in loaded.evidence.transform_identity
    assert "ransac" not in loaded.evidence.transform_identity


def test_collective_least_squares_fit_is_order_independent() -> None:
    """未来完整观测集可由统一方程循环拟合，输入顺序不改变矩阵或摘要。"""

    matrix = ((2.0, 0.25, 10.0), (-0.1, 1.5, 20.0))
    observations = (
        _observation("p0", 0.0, 0.0, matrix),
        _observation("p1", 100.0, 0.0, matrix),
        _observation("p2", 0.0, 100.0, matrix),
        _observation("p3", 150.0, 75.0, matrix),
    )

    forward = fit_affine_least_squares(observations)
    reverse = fit_affine_least_squares(tuple(reversed(observations)))

    assert forward.observation_count == 4
    assert forward.observation_sha256 == reverse.observation_sha256
    for expected_row, actual_row in zip(matrix, forward.affine_matrix, strict=True):
        assert actual_row == pytest.approx(expected_row, abs=1e-12)
    assert reverse.affine_matrix == forward.affine_matrix
    assert forward.max_error_px < 1e-10


def test_coordinate_evidence_rejects_unknown_or_drifting_schema(tmp_path: Path) -> None:
    """证据未知字段与方程漂移不能被宽松加载或当成已验证配置。"""

    payload = json.loads(_EVIDENCE_PATH.read_text(encoding="utf-8"))
    payload["unknown"] = True
    bad_path = tmp_path / "bad-evidence.json"
    bad_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SchemaMismatchError, match="根字段集合"):
        load_affine_calibration_evidence(bad_path)

    del payload["unknown"]
    payload["affine_matrix"][0][2] += 1.0
    bad_path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_affine_calibration_evidence(bad_path)
    config = load_v2_config(_CONFIG_PATH)
    report = audit_affine_calibration(
        build_frame_coordinate_transform(config),
        loaded,
    )
    assert not report.ok
    assert report.matrix_matches is False


def test_affine_fit_rejects_collinear_observations() -> None:
    """共同样本没有二维覆盖时不得发布不可逆或欠定方程。"""

    matrix = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    observations = tuple(
        _observation(f"line-{index}", float(index), float(index), matrix)
        for index in range(3)
    )

    with pytest.raises(ValueError, match="共线"):
        fit_affine_least_squares(observations)
