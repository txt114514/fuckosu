"""加载、拟合并审计 V2 osu→原帧仿射坐标证据。

生产坐标方程与验证控制点是两个不同事实：前者决定训练、评分和图库的
共同映射，后者只能证明该方程在已观测位置上的残差。模块因此显式记录
原始拟合集是否仍可获得，绝不把少量验证点伪装成“大量通过样本”。
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from package import AffineOsuVideoTransform, OSU_PLAYFIELD_HEIGHT, OSU_PLAYFIELD_WIDTH

from traning.conf import AffineMatrix, CALIBRATION_EVIDENCE_SCHEMA_VERSION
from traning.lib.infrastructure import (
    SchemaMismatchError,
    read_json_object,
    sha256_file,
)

from .coordinates import FrameCoordinateTransform


CALIBRATION_EVIDENCE_ARTIFACT_TYPE = "traning_affine_calibration_evidence"
"""坐标证据 JSON 的固定制品类型。"""

_ROOT_KEYS = frozenset(
    {
        "artifact_type",
        "schema_version",
        "source_frame_width",
        "source_frame_height",
        "transform_identity",
        "affine_matrix",
        "max_control_error_px",
        "fit_provenance",
        "validation_controls",
    }
)
_FIT_KEYS = frozenset(
    {
        "method",
        "observations_available",
        "observation_count",
        "observation_sha256",
        "note",
    }
)
_CONTROL_KEYS = frozenset(
    {
        "control_id",
        "osu_x",
        "osu_y",
        "video_x",
        "video_y",
        "source_reference",
    }
)


def _require_finite(name: str, value: float) -> None:
    """拒绝布尔、非数值和非有限坐标。"""

    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} 必须是数值")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} 必须是有限数值")


def _require_text(name: str, value: str) -> None:
    """要求标识和说明为非空、无首尾空格的字符串。"""

    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} 必须是非空且无首尾空格的字符串")


def _require_sha256(name: str, value: str) -> None:
    """校验 canonical 小写 SHA-256 文本。"""

    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} 必须是小写 SHA-256")


@dataclass(frozen=True, slots=True)
class CalibrationObservation:
    """一个独立 osu 坐标与原视频像素中心的成对观测。"""

    control_id: str
    osu_x: float
    osu_y: float
    video_x: float
    video_y: float
    source_reference: str

    def __post_init__(self) -> None:
        """验证标识、坐标范围及可追溯来源。"""

        _require_text("control_id", self.control_id)
        _require_text("source_reference", self.source_reference)
        for name, value in (
            ("osu_x", self.osu_x),
            ("osu_y", self.osu_y),
            ("video_x", self.video_x),
            ("video_y", self.video_y),
        ):
            _require_finite(name, value)
        if not 0.0 <= self.osu_x <= OSU_PLAYFIELD_WIDTH:
            raise ValueError("osu_x 超出 osu!standard playfield")
        if not 0.0 <= self.osu_y <= OSU_PLAYFIELD_HEIGHT:
            raise ValueError("osu_y 超出 osu!standard playfield")


@dataclass(frozen=True, slots=True)
class CalibrationFitProvenance:
    """原始拟合方法、样本数量与样本集合摘要的可用性声明。"""

    method: str
    observations_available: bool
    observation_count: int | None
    observation_sha256: str | None
    note: str

    def __post_init__(self) -> None:
        """可复现声明必须同时具备数量和摘要，缺失时必须同时为空。"""

        _require_text("fit_provenance.method", self.method)
        _require_text("fit_provenance.note", self.note)
        if not isinstance(self.observations_available, bool):
            raise TypeError("observations_available 必须是 bool")
        if self.observations_available:
            if (
                isinstance(self.observation_count, bool)
                or not isinstance(self.observation_count, int)
                or self.observation_count < 3
            ):
                raise ValueError("可复现拟合集必须至少包含三个观测")
            if self.observation_sha256 is None:
                raise ValueError("可复现拟合集必须记录 observation_sha256")
            _require_sha256("observation_sha256", self.observation_sha256)
        elif self.observation_count is not None or self.observation_sha256 is not None:
            raise ValueError("拟合集不可用时 count 和 sha256 必须同时为 null")

    @property
    def reproducible(self) -> bool:
        """仅当完整拟合集数量与摘要均可校验时返回真。"""

        return self.observations_available


@dataclass(frozen=True, slots=True)
class AffineCalibrationEvidence:
    """绑定矩阵、原帧、拟合来源及独立控制点的严格证据。"""

    schema_version: int
    artifact_type: str
    source_frame_width: int
    source_frame_height: int
    transform_identity: str
    affine_matrix: AffineMatrix
    max_control_error_px: float
    fit_provenance: CalibrationFitProvenance
    validation_controls: tuple[CalibrationObservation, ...]

    def __post_init__(self) -> None:
        """验证制品版本、方程、尺寸和控制点集合的不变量。"""

        if self.schema_version != CALIBRATION_EVIDENCE_SCHEMA_VERSION:
            raise ValueError(
                "calibration evidence schema_version 必须是 "
                f"{CALIBRATION_EVIDENCE_SCHEMA_VERSION}"
            )
        if self.artifact_type != CALIBRATION_EVIDENCE_ARTIFACT_TYPE:
            raise ValueError(
                f"artifact_type 必须是 {CALIBRATION_EVIDENCE_ARTIFACT_TYPE!r}"
            )
        for name, value in (
            ("source_frame_width", self.source_frame_width),
            ("source_frame_height", self.source_frame_height),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} 必须是正整数")
        _require_text("transform_identity", self.transform_identity)
        AffineOsuVideoTransform(self.affine_matrix)
        _require_finite("max_control_error_px", self.max_control_error_px)
        if self.max_control_error_px <= 0.0:
            raise ValueError("max_control_error_px 必须大于 0")
        if not isinstance(self.fit_provenance, CalibrationFitProvenance):
            raise TypeError("fit_provenance 必须是 CalibrationFitProvenance")
        if (
            not isinstance(self.validation_controls, tuple)
            or len(self.validation_controls) < 3
            or any(
                not isinstance(item, CalibrationObservation)
                for item in self.validation_controls
            )
        ):
            raise TypeError("validation_controls 必须至少含三个 CalibrationObservation")
        identifiers = tuple(item.control_id for item in self.validation_controls)
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("validation control_id 不得重复")
        for control in self.validation_controls:
            if not 0.0 <= control.video_x < self.source_frame_width:
                raise ValueError(f"{control.control_id} 的 video_x 超出原帧")
            if not 0.0 <= control.video_y < self.source_frame_height:
                raise ValueError(f"{control.control_id} 的 video_y 超出原帧")


@dataclass(frozen=True, slots=True)
class LoadedCalibrationEvidence:
    """保留证据内容与原始 JSON 字节摘要的加载结果。"""

    evidence: AffineCalibrationEvidence
    artifact_sha256: str

    def __post_init__(self) -> None:
        """阻止加载边界丢失制品完整性身份。"""

        if not isinstance(self.evidence, AffineCalibrationEvidence):
            raise TypeError("evidence 必须是 AffineCalibrationEvidence")
        _require_sha256("artifact_sha256", self.artifact_sha256)


@dataclass(frozen=True, slots=True)
class AffineFitResult:
    """由完整观测集合最小二乘拟合出的方程及残差摘要。"""

    affine_matrix: AffineMatrix
    observation_count: int
    observation_sha256: str
    mean_error_px: float
    rmse_error_px: float
    max_error_px: float

    def __post_init__(self) -> None:
        """确保拟合摘要自身有限且能作为后续 manifest 证据。"""

        AffineOsuVideoTransform(self.affine_matrix)
        if self.observation_count < 3:
            raise ValueError("observation_count 必须至少为 3")
        _require_sha256("observation_sha256", self.observation_sha256)
        for name, value in (
            ("mean_error_px", self.mean_error_px),
            ("rmse_error_px", self.rmse_error_px),
            ("max_error_px", self.max_error_px),
        ):
            _require_finite(name, value)
            if value < 0.0:
                raise ValueError(f"{name} 不得为负数")


@dataclass(frozen=True, slots=True)
class CalibrationAuditReport:
    """配置坐标变换对证据矩阵和独立控制点的完整审计结果。"""

    transform_fingerprint: str
    evidence_artifact_sha256: str
    control_set_sha256: str
    control_count: int
    identity_matches: bool
    frame_size_matches: bool
    matrix_matches: bool
    fit_reproducible: bool
    mean_error_px: float
    rmse_error_px: float
    max_error_px: float
    max_allowed_error_px: float
    worst_control_id: str

    def __post_init__(self) -> None:
        """校验报告字段，使 CLI 和环境门禁无需重新推导语义。"""

        _require_text("transform_fingerprint", self.transform_fingerprint)
        _require_sha256("evidence_artifact_sha256", self.evidence_artifact_sha256)
        _require_sha256("control_set_sha256", self.control_set_sha256)
        if self.control_count < 3:
            raise ValueError("control_count 必须至少为 3")
        for name in (
            "identity_matches",
            "frame_size_matches",
            "matrix_matches",
            "fit_reproducible",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} 必须是 bool")
        for name, value in (
            ("mean_error_px", self.mean_error_px),
            ("rmse_error_px", self.rmse_error_px),
            ("max_error_px", self.max_error_px),
            ("max_allowed_error_px", self.max_allowed_error_px),
        ):
            _require_finite(name, value)
            if value < 0.0:
                raise ValueError(f"{name} 不得为负数")
        _require_text("worst_control_id", self.worst_control_id)

    @property
    def ok(self) -> bool:
        """身份、尺寸、方程和全部控制点同时通过时才返回真。"""

        return (
            self.identity_matches
            and self.frame_size_matches
            and self.matrix_matches
            and self.max_error_px <= self.max_allowed_error_px
        )


def load_affine_calibration_evidence(path: Path) -> LoadedCalibrationEvidence:
    """严格加载版本化证据 JSON，并保留原始文件 SHA-256。"""

    if not isinstance(path, Path):
        raise TypeError("path 必须是 pathlib.Path")
    payload = read_json_object(path)
    if set(payload) != _ROOT_KEYS:
        raise SchemaMismatchError("calibration evidence 根字段集合不匹配")
    try:
        fit_payload = _object(payload["fit_provenance"], "fit_provenance")
        if set(fit_payload) != _FIT_KEYS:
            raise SchemaMismatchError("fit_provenance 字段集合不匹配")
        controls_payload = _array(payload["validation_controls"], "validation_controls")
        controls = tuple(
            _parse_control(item, index) for index, item in enumerate(controls_payload)
        )
        evidence = AffineCalibrationEvidence(
            schema_version=_integer(payload["schema_version"], "schema_version"),
            artifact_type=_string(payload["artifact_type"], "artifact_type"),
            source_frame_width=_integer(
                payload["source_frame_width"], "source_frame_width"
            ),
            source_frame_height=_integer(
                payload["source_frame_height"], "source_frame_height"
            ),
            transform_identity=_string(
                payload["transform_identity"], "transform_identity"
            ),
            affine_matrix=_matrix(payload["affine_matrix"]),
            max_control_error_px=_number(
                payload["max_control_error_px"], "max_control_error_px"
            ),
            fit_provenance=CalibrationFitProvenance(
                method=_string(fit_payload["method"], "fit_provenance.method"),
                observations_available=_boolean(
                    fit_payload["observations_available"],
                    "fit_provenance.observations_available",
                ),
                observation_count=_optional_integer(
                    fit_payload["observation_count"],
                    "fit_provenance.observation_count",
                ),
                observation_sha256=_optional_string(
                    fit_payload["observation_sha256"],
                    "fit_provenance.observation_sha256",
                ),
                note=_string(fit_payload["note"], "fit_provenance.note"),
            ),
            validation_controls=controls,
        )
    except (TypeError, ValueError) as exc:
        if isinstance(exc, SchemaMismatchError):
            raise
        raise SchemaMismatchError("calibration evidence typed schema 不匹配") from exc
    return LoadedCalibrationEvidence(evidence, sha256_file(path))


def fit_affine_least_squares(
    observations: tuple[CalibrationObservation, ...],
) -> AffineFitResult:
    """用全部给定观测确定性拟合 2×3 方程，不执行随机抽样或静默剔除。"""

    ordered = _validated_observations(observations)
    design = np.asarray(
        [(item.osu_x, item.osu_y, 1.0) for item in ordered],
        dtype=np.float64,
    )
    target = np.asarray(
        [(item.video_x, item.video_y) for item in ordered],
        dtype=np.float64,
    )
    solution, _residuals, rank, _singular_values = np.linalg.lstsq(
        design,
        target,
        rcond=None,
    )
    if int(rank) != 3:
        raise ValueError("校准观测在 osu 平面上共线，无法拟合 affine 方程")
    matrix = _matrix_from_array(solution.T)
    predicted = design @ solution
    errors = np.linalg.norm(predicted - target, axis=1)
    return AffineFitResult(
        affine_matrix=matrix,
        observation_count=len(ordered),
        observation_sha256=_observation_sha256(ordered),
        mean_error_px=float(np.mean(errors)),
        rmse_error_px=float(np.sqrt(np.mean(np.square(errors)))),
        max_error_px=float(np.max(errors)),
    )


def audit_affine_calibration(
    transform: FrameCoordinateTransform,
    loaded: LoadedCalibrationEvidence,
) -> CalibrationAuditReport:
    """以正式共享变换复算每个控制点，返回唯一可消费的门禁报告。"""

    if not isinstance(transform, FrameCoordinateTransform):
        raise TypeError("transform 必须是 FrameCoordinateTransform")
    if not isinstance(loaded, LoadedCalibrationEvidence):
        raise TypeError("loaded 必须是 LoadedCalibrationEvidence")
    evidence = loaded.evidence
    controls = tuple(
        sorted(evidence.validation_controls, key=lambda item: item.control_id)
    )
    residuals: list[tuple[str, float]] = []
    for item in controls:
        predicted_x, predicted_y = transform.transform.osu_to_video(
            item.osu_x,
            item.osu_y,
        )
        residuals.append(
            (
                item.control_id,
                math.hypot(predicted_x - item.video_x, predicted_y - item.video_y),
            )
        )
    worst_control_id, max_error = max(residuals, key=lambda item: (item[1], item[0]))
    errors = tuple(error for _identifier, error in residuals)
    return CalibrationAuditReport(
        transform_fingerprint=transform.transform_fingerprint,
        evidence_artifact_sha256=loaded.artifact_sha256,
        control_set_sha256=_observation_sha256(controls),
        control_count=len(controls),
        identity_matches=transform.transform_identity == evidence.transform_identity,
        frame_size_matches=(
            transform.source_frame_width == evidence.source_frame_width
            and transform.source_frame_height == evidence.source_frame_height
        ),
        matrix_matches=transform.transform.matrix == evidence.affine_matrix,
        fit_reproducible=evidence.fit_provenance.reproducible,
        mean_error_px=sum(errors) / len(errors),
        rmse_error_px=math.sqrt(sum(error * error for error in errors) / len(errors)),
        max_error_px=max_error,
        max_allowed_error_px=evidence.max_control_error_px,
        worst_control_id=worst_control_id,
    )


def _validated_observations(
    observations: tuple[CalibrationObservation, ...],
) -> tuple[CalibrationObservation, ...]:
    if (
        not isinstance(observations, tuple)
        or len(observations) < 3
        or any(not isinstance(item, CalibrationObservation) for item in observations)
    ):
        raise TypeError("observations 必须至少含三个 CalibrationObservation")
    identifiers = tuple(item.control_id for item in observations)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("observation control_id 不得重复")
    return tuple(sorted(observations, key=lambda item: item.control_id))


def _observation_sha256(observations: tuple[CalibrationObservation, ...]) -> str:
    payload = [
        {
            "control_id": item.control_id,
            "osu_x": item.osu_x,
            "osu_y": item.osu_y,
            "source_reference": item.source_reference,
            "video_x": item.video_x,
            "video_y": item.video_y,
        }
        for item in observations
    ]
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _parse_control(value: object, index: int) -> CalibrationObservation:
    payload = _object(value, f"validation_controls[{index}]")
    if set(payload) != _CONTROL_KEYS:
        raise SchemaMismatchError(f"validation_controls[{index}] 字段集合不匹配")
    prefix = f"validation_controls[{index}]"
    return CalibrationObservation(
        control_id=_string(payload["control_id"], f"{prefix}.control_id"),
        osu_x=_number(payload["osu_x"], f"{prefix}.osu_x"),
        osu_y=_number(payload["osu_y"], f"{prefix}.osu_y"),
        video_x=_number(payload["video_x"], f"{prefix}.video_x"),
        video_y=_number(payload["video_y"], f"{prefix}.video_y"),
        source_reference=_string(
            payload["source_reference"], f"{prefix}.source_reference"
        ),
    )


def _matrix(value: object) -> AffineMatrix:
    rows = _array(value, "affine_matrix")
    if len(rows) != 2:
        raise SchemaMismatchError("affine_matrix 必须有两行")
    parsed_rows: list[tuple[float, float, float]] = []
    for row_index, row in enumerate(rows):
        values = _array(row, f"affine_matrix[{row_index}]")
        if len(values) != 3:
            raise SchemaMismatchError("affine_matrix 每行必须有三个数值")
        parsed_rows.append(
            tuple(
                _number(item, f"affine_matrix[{row_index}][{column_index}]")
                for column_index, item in enumerate(values)
            )
        )
    return (parsed_rows[0], parsed_rows[1])


def _matrix_from_array(value: np.ndarray) -> AffineMatrix:
    return (
        (float(value[0, 0]), float(value[0, 1]), float(value[0, 2])),
        (float(value[1, 0]), float(value[1, 1]), float(value[1, 2])),
    )


def _object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise SchemaMismatchError(f"{name} 必须是 JSON object")
    return value


def _array(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise SchemaMismatchError(f"{name} 必须是 JSON array")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise SchemaMismatchError(f"{name} 必须是字符串")
    return value


def _optional_string(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _string(value, name)


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SchemaMismatchError(f"{name} 必须是整数")
    return value


def _optional_integer(value: object, name: str) -> int | None:
    if value is None:
        return None
    return _integer(value, name)


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise SchemaMismatchError(f"{name} 必须是数值")
    converted = float(value)
    if not math.isfinite(converted):
        raise SchemaMismatchError(f"{name} 必须是有限数值")
    return converted


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise SchemaMismatchError(f"{name} 必须是 bool")
    return value


__all__ = (
    "CALIBRATION_EVIDENCE_ARTIFACT_TYPE",
    "AffineCalibrationEvidence",
    "AffineFitResult",
    "CalibrationAuditReport",
    "CalibrationFitProvenance",
    "CalibrationObservation",
    "LoadedCalibrationEvidence",
    "audit_affine_calibration",
    "fit_affine_least_squares",
    "load_affine_calibration_evidence",
)
