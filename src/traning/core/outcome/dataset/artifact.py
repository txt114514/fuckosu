"""反事实 Outcome 数据集的不可变 generation 发布与严格加载。"""

from __future__ import annotations

import json
import math
import time
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from traning.state import (
    ArtifactManifest,
    BeliefState,
    DataSplit,
    DecisionAction,
    JSONObject,
    ObjectTypeDistribution,
    OutcomeCategory,
    OutcomeTrainingSample,
    Point2D,
    require_transform_fingerprint,
)
from traning.lib.infrastructure import (
    IntegrityError,
    SchemaMismatchError,
    atomic_write_json,
    atomic_write_jsonl,
    read_json_object,
    sha256_file,
)
from traning.core.evaluation import SCORE_VERSION
from traning.core.outcome.oracle import OUTCOME_ORACLE_VERSION

from .builder import CounterfactualOutcomeDataset


OUTCOME_DATASET_SCHEMA_VERSION = 2
OUTCOME_DATASET_ARTIFACT_TYPE = "counterfactual_outcome_dataset"
MANIFEST_FILENAME = "manifest.json"
RECORDS_FILENAME = "records.<generation>.jsonl"

_MANIFEST_FIELDS = frozenset(
    {
        "artifact_id",
        "artifact_type",
        "schema_version",
        "dataset_id",
        "split",
        "producer_id",
        "row_count",
        "sha256",
        "created_at_ms",
        "metadata",
    }
)
_METADATA_FIELDS = frozenset(
    {
        "records_filename",
        "oracle_version",
        "scoring_version",
        "transform_fingerprint",
    }
)
_RECORD_FIELDS = frozenset(
    {
        "sample_id",
        "split",
        "source_sample_id",
        "oracle_state_id",
        "belief",
        "action",
        "action_track_id",
        "horizon_ms",
        "target_category",
        "target_score",
        "valid",
        "expires",
        "target_object_id",
    }
)
_BELIEF_FIELDS = frozenset(
    {
        "track_id",
        "timestamp_ms",
        "belief_embedding",
        "position_mean",
        "position_uncertainty",
        "visibility_probability",
        "object_type_distribution",
        "age",
        "time_since_seen_ms",
        "uncertainty",
    }
)
_POINT_FIELDS = frozenset({"x", "y"})
_TYPE_FIELDS = frozenset({"p_ring", "p_slider", "p_spinner", "p_unknown"})


@dataclass(frozen=True, slots=True)
class OutcomeDatasetManifest:
    """复用 canonical ArtifactManifest 的 Outcome 数据集清单。"""

    artifact: ArtifactManifest

    def __post_init__(self) -> None:
        if not isinstance(self.artifact, ArtifactManifest):
            raise TypeError("artifact 必须是 ArtifactManifest")
        if self.artifact.artifact_type != OUTCOME_DATASET_ARTIFACT_TYPE:
            raise ValueError(f"artifact_type 必须为 {OUTCOME_DATASET_ARTIFACT_TYPE}")
        if self.artifact.schema_version != OUTCOME_DATASET_SCHEMA_VERSION:
            raise ValueError(f"schema_version 必须为 {OUTCOME_DATASET_SCHEMA_VERSION}")
        if not isinstance(self.artifact.split, DataSplit):
            raise TypeError("artifact.split 必须是 DataSplit")
        if self.artifact.split is DataSplit.ALL:
            raise ValueError("split 必须是具体 DataSplit")
        metadata = dict(self.artifact.metadata)
        if set(metadata) != _METADATA_FIELDS:
            raise ValueError("Outcome dataset metadata 字段不匹配")
        if (
            not isinstance(metadata["records_filename"], str)
            or Path(metadata["records_filename"]).name != metadata["records_filename"]
            or not metadata["records_filename"].startswith("records.")
            or not metadata["records_filename"].endswith(".jsonl")
        ):
            raise ValueError("records_filename 必须是安全的 generation 文件名")
        _identifier(metadata["oracle_version"], "oracle_version")
        _identifier(metadata["scoring_version"], "scoring_version")
        transform_fingerprint = metadata["transform_fingerprint"]
        if not isinstance(transform_fingerprint, str):
            raise TypeError("transform_fingerprint 必须是字符串")
        require_transform_fingerprint(transform_fingerprint)

    @property
    def records_filename(self) -> str:
        """返回 manifest 已提交的不可变 records generation 文件名。"""

        value = dict(self.artifact.metadata)["records_filename"]
        if not isinstance(value, str):  # pragma: no cover - 构造期已校验
            raise TypeError("records_filename 必须是字符串")
        return value

    @property
    def schema_version(self) -> int:
        """返回 Outcome 数据集制品的 schema 版本。"""

        return self.artifact.schema_version

    @property
    def dataset_id(self) -> str:
        """返回反事实样本所归属的数据集稳定标识。"""

        return self.artifact.dataset_id

    @property
    def split(self) -> DataSplit:
        """返回该制品唯一且具体的数据切分。"""

        return self.artifact.split

    @property
    def producer_id(self) -> str:
        """返回生成反事实数据集的生产者标识。"""

        return self.artifact.producer_id

    @property
    def row_count(self) -> int:
        """返回 manifest 承诺的 Outcome 样本行数。"""

        return self.artifact.row_count

    @property
    def sha256(self) -> str:
        """返回不可变 records generation 的 SHA-256。"""

        return self.artifact.sha256

    @property
    def oracle_version(self) -> str:
        """返回生成反事实标签时使用的 OutcomeOracle 版本。"""

        value = dict(self.artifact.metadata)["oracle_version"]
        if not isinstance(value, str):  # pragma: no cover - 构造期已校验
            raise TypeError("oracle_version 必须是字符串")
        return value

    @property
    def scoring_version(self) -> str:
        """返回生成标签时使用的 canonical scorer 版本。"""

        value = dict(self.artifact.metadata)["scoring_version"]
        if not isinstance(value, str):  # pragma: no cover - 构造期已校验
            raise TypeError("scoring_version 必须是字符串")
        return value

    @property
    def transform_fingerprint(self) -> str:
        """返回从 runtime 原帧坐标生成 oracle label 时使用的变换指纹。"""

        value = dict(self.artifact.metadata)["transform_fingerprint"]
        if not isinstance(value, str):  # pragma: no cover - 构造期已校验
            raise TypeError("transform_fingerprint 必须是字符串")
        return value


class OutcomeDatasetArtifactStore:
    """以 manifest-last 协议发布并严格恢复 typed Outcome 样本。"""

    def __init__(self, directory: Path) -> None:
        if not isinstance(directory, Path):
            raise TypeError("directory 必须是 pathlib.Path")
        self._directory = directory

    def publish(
        self,
        dataset: CounterfactualOutcomeDataset,
        *,
        dataset_id: str,
        producer_id: str,
        created_at_ms: float | None = None,
    ) -> OutcomeDatasetManifest:
        """完整发布新 generation 后，才原子替换唯一提交点 manifest。"""

        if not isinstance(dataset, CounterfactualOutcomeDataset):
            raise TypeError("dataset 必须是 CounterfactualOutcomeDataset")
        timestamp = time.time() * 1000.0 if created_at_ms is None else created_at_ms
        # 在触碰文件前先校验全部 manifest 身份字段。
        self._manifest(
            dataset_id=dataset_id,
            split=dataset.split,
            producer_id=producer_id,
            row_count=0,
            digest="0" * 64,
            records_filename="records.preflight.jsonl",
            oracle_version=OUTCOME_ORACLE_VERSION,
            scoring_version=SCORE_VERSION,
            transform_fingerprint=dataset.transform_fingerprint,
            created_at_ms=timestamp,
        )
        row_count = 0
        sample_ids: set[str] = set()

        def encoded_records() -> Iterable[JSONObject]:
            """稳定遍历 typed 样本，并在编码前拒绝重复 sample_id。"""

            nonlocal row_count
            for record in dataset.records:
                if record.sample_id in sample_ids:
                    raise ValueError(
                        f"Outcome dataset sample_id 重复：{record.sample_id}"
                    )
                sample_ids.add(record.sample_id)
                row_count += 1
                yield _record_to_json(record)

        filename = f"records.{uuid.uuid4().hex}.jsonl"
        records_path = self._directory / filename
        atomic_write_jsonl(records_path, encoded_records())
        manifest = self._manifest(
            dataset_id=dataset_id,
            split=dataset.split,
            producer_id=producer_id,
            row_count=row_count,
            digest=sha256_file(records_path),
            records_filename=filename,
            oracle_version=OUTCOME_ORACLE_VERSION,
            scoring_version=SCORE_VERSION,
            transform_fingerprint=dataset.transform_fingerprint,
            created_at_ms=timestamp,
        )
        atomic_write_json(
            self._directory / MANIFEST_FILENAME, _manifest_to_json(manifest)
        )
        return manifest

    def load(
        self,
        *,
        expected_dataset_id: str,
        expected_split: DataSplit,
        expected_producer_id: str,
        expected_transform_fingerprint: str,
        expected_schema_version: int = OUTCOME_DATASET_SCHEMA_VERSION,
    ) -> CounterfactualOutcomeDataset:
        """验证 schema、身份、版本、摘要和行数后恢复 typed samples。"""

        manifest = _manifest_from_json(
            read_json_object(self._directory / MANIFEST_FILENAME)
        )
        artifact = manifest.artifact
        require_transform_fingerprint(
            expected_transform_fingerprint,
            "expected_transform_fingerprint",
        )
        expected = (
            (
                "artifact_id",
                artifact.artifact_id,
                f"{expected_dataset_id}-counterfactual-outcomes",
            ),
            ("schema_version", artifact.schema_version, expected_schema_version),
            ("dataset_id", artifact.dataset_id, expected_dataset_id),
            ("split", artifact.split, expected_split),
            ("producer_id", artifact.producer_id, expected_producer_id),
            ("oracle_version", manifest.oracle_version, OUTCOME_ORACLE_VERSION),
            ("scoring_version", manifest.scoring_version, SCORE_VERSION),
            (
                "transform_fingerprint",
                manifest.transform_fingerprint,
                expected_transform_fingerprint,
            ),
        )
        for name, actual, wanted in expected:
            if actual != wanted:
                raise SchemaMismatchError(
                    f"Outcome dataset {name} 不匹配：实际 {actual!r}，预期 {wanted!r}"
                )
        records_path = self._directory / manifest.records_filename
        actual_digest = sha256_file(records_path)
        if actual_digest != artifact.sha256:
            raise IntegrityError("Outcome dataset SHA-256 与 manifest 不一致")

        decoded: list[OutcomeTrainingSample] = []
        try:
            with records_path.open("rb") as handle:
                for line_number, raw_line in enumerate(handle, start=1):
                    if not raw_line.endswith(b"\n"):
                        raise IntegrityError(
                            f"Outcome dataset 第 {line_number} 行被截断"
                        )
                    if raw_line == b"\n":
                        raise SchemaMismatchError(
                            f"Outcome dataset 第 {line_number} 行不得为空"
                        )
                    decoded.append(_decode_record_line(raw_line, line_number))
        except (OSError, UnicodeError) as exc:
            raise IntegrityError(
                f"无法完整读取 Outcome dataset：{records_path}"
            ) from exc
        if len(decoded) != artifact.row_count:
            raise IntegrityError(
                f"Outcome dataset 行数不匹配：实际 {len(decoded)}，清单 {artifact.row_count}"
            )
        sample_ids = tuple(record.sample_id for record in decoded)
        if len(sample_ids) != len(set(sample_ids)):
            raise SchemaMismatchError("Outcome dataset 含重复 sample_id")
        try:
            return CounterfactualOutcomeDataset(
                split=artifact.split,
                records=tuple(decoded),
                transform_fingerprint=manifest.transform_fingerprint,
            )
        except (TypeError, ValueError) as exc:
            raise SchemaMismatchError(
                "Outcome dataset records 与 manifest split 不一致"
            ) from exc

    def _manifest(
        self,
        *,
        dataset_id: str,
        split: DataSplit,
        producer_id: str,
        row_count: int,
        digest: str,
        records_filename: str,
        oracle_version: str,
        scoring_version: str,
        transform_fingerprint: str,
        created_at_ms: float,
    ) -> OutcomeDatasetManifest:
        return OutcomeDatasetManifest(
            ArtifactManifest(
                artifact_id=f"{dataset_id}-counterfactual-outcomes",
                artifact_type=OUTCOME_DATASET_ARTIFACT_TYPE,
                schema_version=OUTCOME_DATASET_SCHEMA_VERSION,
                dataset_id=dataset_id,
                split=split,
                producer_id=producer_id,
                row_count=row_count,
                sha256=digest,
                created_at_ms=created_at_ms,
                metadata=(
                    ("records_filename", records_filename),
                    ("oracle_version", oracle_version),
                    ("scoring_version", scoring_version),
                    ("transform_fingerprint", transform_fingerprint),
                ),
            )
        )


def _record_to_json(record: OutcomeTrainingSample) -> JSONObject:
    belief = record.belief
    return {
        "sample_id": record.sample_id,
        "split": record.split.value,
        "source_sample_id": record.source_sample_id,
        "oracle_state_id": record.oracle_state_id,
        "belief": {
            "track_id": belief.track_id,
            "timestamp_ms": belief.timestamp_ms,
            "belief_embedding": list(belief.belief_embedding),
            "position_mean": {"x": belief.position_mean.x, "y": belief.position_mean.y},
            "position_uncertainty": {
                "x": belief.position_uncertainty.x,
                "y": belief.position_uncertainty.y,
            },
            "visibility_probability": belief.visibility_probability,
            "object_type_distribution": {
                "p_ring": belief.object_type_distribution.p_ring,
                "p_slider": belief.object_type_distribution.p_slider,
                "p_spinner": belief.object_type_distribution.p_spinner,
                "p_unknown": belief.object_type_distribution.p_unknown,
            },
            "age": belief.age,
            "time_since_seen_ms": belief.time_since_seen_ms,
            "uncertainty": belief.uncertainty,
        },
        "action": record.action.value,
        "action_track_id": record.action_track_id,
        "horizon_ms": record.horizon_ms,
        "target_category": int(record.target_category),
        "target_score": record.target_score,
        "valid": record.valid,
        "expires": record.expires,
        "target_object_id": record.target_object_id,
    }


def _manifest_to_json(manifest: OutcomeDatasetManifest) -> JSONObject:
    artifact = manifest.artifact
    return {
        "artifact_id": artifact.artifact_id,
        "artifact_type": artifact.artifact_type,
        "schema_version": artifact.schema_version,
        "dataset_id": artifact.dataset_id,
        "split": artifact.split.value,
        "producer_id": artifact.producer_id,
        "row_count": artifact.row_count,
        "sha256": artifact.sha256,
        "created_at_ms": artifact.created_at_ms,
        "metadata": {
            "records_filename": manifest.records_filename,
            "oracle_version": manifest.oracle_version,
            "scoring_version": manifest.scoring_version,
            "transform_fingerprint": manifest.transform_fingerprint,
        },
    }


def _manifest_from_json(payload: JSONObject) -> OutcomeDatasetManifest:
    _require_fields(payload, _MANIFEST_FIELDS, "manifest")
    try:
        split = DataSplit(_string(payload["split"], "manifest.split"))
    except ValueError as exc:
        raise SchemaMismatchError("manifest.split 不是 canonical DataSplit") from exc
    try:
        metadata = _object(payload["metadata"], "manifest.metadata")
        _require_fields(metadata, _METADATA_FIELDS, "manifest.metadata")
        return OutcomeDatasetManifest(
            ArtifactManifest(
                artifact_id=_string(payload["artifact_id"], "manifest.artifact_id"),
                artifact_type=_string(
                    payload["artifact_type"], "manifest.artifact_type"
                ),
                schema_version=_integer(
                    payload["schema_version"], "manifest.schema_version"
                ),
                dataset_id=_string(payload["dataset_id"], "manifest.dataset_id"),
                split=split,
                producer_id=_string(payload["producer_id"], "manifest.producer_id"),
                row_count=_integer(payload["row_count"], "manifest.row_count"),
                sha256=_string(payload["sha256"], "manifest.sha256"),
                created_at_ms=_real(payload["created_at_ms"], "manifest.created_at_ms"),
                metadata=(
                    (
                        "records_filename",
                        _string(
                            metadata["records_filename"],
                            "manifest.metadata.records_filename",
                        ),
                    ),
                    (
                        "oracle_version",
                        _string(
                            metadata["oracle_version"],
                            "manifest.metadata.oracle_version",
                        ),
                    ),
                    (
                        "scoring_version",
                        _string(
                            metadata["scoring_version"],
                            "manifest.metadata.scoring_version",
                        ),
                    ),
                    (
                        "transform_fingerprint",
                        _string(
                            metadata["transform_fingerprint"],
                            "manifest.metadata.transform_fingerprint",
                        ),
                    ),
                ),
            )
        )
    except (TypeError, ValueError) as exc:
        raise SchemaMismatchError("manifest 不满足 Outcome dataset 契约") from exc


def _decode_record_line(raw_line: bytes, line_number: int) -> OutcomeTrainingSample:
    try:
        decoded: object = json.loads(
            raw_line.decode("utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_non_finite_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IntegrityError(
            f"Outcome dataset 第 {line_number} 行不是完整 JSON"
        ) from exc
    if not isinstance(decoded, dict):
        raise SchemaMismatchError(f"Outcome dataset 第 {line_number} 行必须是 object")
    return _record_from_json(cast(JSONObject, decoded), line_number)


def _record_from_json(payload: JSONObject, line_number: int) -> OutcomeTrainingSample:
    context = f"records[{line_number}]"
    _require_fields(payload, _RECORD_FIELDS, context)
    belief_payload = _object(payload["belief"], f"{context}.belief")
    _require_fields(belief_payload, _BELIEF_FIELDS, f"{context}.belief")
    try:
        belief = BeliefState(
            track_id=_string(belief_payload["track_id"], f"{context}.belief.track_id"),
            timestamp_ms=_real(
                belief_payload["timestamp_ms"], f"{context}.belief.timestamp_ms"
            ),
            belief_embedding=_float_tuple(
                belief_payload["belief_embedding"], f"{context}.belief.belief_embedding"
            ),
            position_mean=_point(
                belief_payload["position_mean"], f"{context}.belief.position_mean"
            ),
            position_uncertainty=_point(
                belief_payload["position_uncertainty"],
                f"{context}.belief.position_uncertainty",
            ),
            visibility_probability=_real(
                belief_payload["visibility_probability"],
                f"{context}.belief.visibility_probability",
            ),
            object_type_distribution=_type_distribution(
                belief_payload["object_type_distribution"],
                f"{context}.belief.object_type_distribution",
            ),
            age=_integer(belief_payload["age"], f"{context}.belief.age"),
            time_since_seen_ms=_real(
                belief_payload["time_since_seen_ms"],
                f"{context}.belief.time_since_seen_ms",
            ),
            uncertainty=_real(
                belief_payload["uncertainty"], f"{context}.belief.uncertainty"
            ),
        )
        action = DecisionAction(_string(payload["action"], f"{context}.action"))
        action_track_value = payload["action_track_id"]
        action_track_id = (
            None
            if action_track_value is None
            else _string(action_track_value, f"{context}.action_track_id")
        )
        try:
            split = DataSplit(_string(payload["split"], f"{context}.split"))
        except ValueError as exc:
            raise SchemaMismatchError(
                f"{context}.split 不是 canonical DataSplit"
            ) from exc
        target_object_value = payload["target_object_id"]
        target_object_id = (
            None
            if target_object_value is None
            else _string(target_object_value, f"{context}.target_object_id")
        )
        return OutcomeTrainingSample(
            sample_id=_string(payload["sample_id"], f"{context}.sample_id"),
            split=split,
            source_sample_id=_string(
                payload["source_sample_id"], f"{context}.source_sample_id"
            ),
            oracle_state_id=_string(
                payload["oracle_state_id"], f"{context}.oracle_state_id"
            ),
            belief=belief,
            action=action,
            action_track_id=action_track_id,
            horizon_ms=_real(payload["horizon_ms"], f"{context}.horizon_ms"),
            target_category=OutcomeCategory(
                _integer(payload["target_category"], f"{context}.target_category")
            ),
            target_score=_real(payload["target_score"], f"{context}.target_score"),
            valid=_boolean(payload["valid"], f"{context}.valid"),
            expires=_boolean(payload["expires"], f"{context}.expires"),
            target_object_id=target_object_id,
        )
    except (TypeError, ValueError) as exc:
        raise SchemaMismatchError(
            f"{context} 不满足 OutcomeTrainingSample 契约"
        ) from exc


def _point(value: object, context: str) -> Point2D:
    payload = _object(value, context)
    _require_fields(payload, _POINT_FIELDS, context)
    return Point2D(
        x=_real(payload["x"], f"{context}.x"), y=_real(payload["y"], f"{context}.y")
    )


def _type_distribution(value: object, context: str) -> ObjectTypeDistribution:
    payload = _object(value, context)
    _require_fields(payload, _TYPE_FIELDS, context)
    return ObjectTypeDistribution(
        p_ring=_real(payload["p_ring"], f"{context}.p_ring"),
        p_slider=_real(payload["p_slider"], f"{context}.p_slider"),
        p_spinner=_real(payload["p_spinner"], f"{context}.p_spinner"),
        p_unknown=_real(payload["p_unknown"], f"{context}.p_unknown"),
    )


def _require_fields(
    payload: JSONObject, expected: frozenset[str], context: str
) -> None:
    actual = set(payload)
    if actual != expected:
        raise SchemaMismatchError(
            f"{context} 字段不匹配：缺少 {sorted(expected - actual)}，未知 {sorted(actual - expected)}"
        )


def _object(value: object, context: str) -> JSONObject:
    if not isinstance(value, dict):
        raise SchemaMismatchError(f"{context} 必须是 object")
    return cast(JSONObject, value)


def _string(value: object, context: str) -> str:
    if not isinstance(value, str):
        raise SchemaMismatchError(f"{context} 必须是字符串")
    return value


def _integer(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SchemaMismatchError(f"{context} 必须是整数")
    return value


def _real(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaMismatchError(f"{context} 必须是数值")
    result = float(value)
    if not math.isfinite(result):
        raise SchemaMismatchError(f"{context} 必须是有限数值")
    return result


def _boolean(value: object, context: str) -> bool:
    if not isinstance(value, bool):
        raise SchemaMismatchError(f"{context} 必须是布尔值")
    return value


def _float_tuple(value: object, context: str) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise SchemaMismatchError(f"{context} 必须是数组")
    return tuple(_real(item, f"{context}[{index}]") for index, item in enumerate(value))


def _identifier(value: object, name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} 必须是字符串")
    if not value or value != value.strip():
        raise ValueError(f"{name} 必须非空且无首尾空格")


def _object_without_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise IntegrityError(f"JSON object 含重复 key：{key}")
        result[key] = value
    return result


def _reject_non_finite_constant(value: str) -> object:
    raise IntegrityError(f"JSON 含非标准数值常量：{value}")


__all__ = (
    "MANIFEST_FILENAME",
    "OUTCOME_DATASET_ARTIFACT_TYPE",
    "OUTCOME_DATASET_SCHEMA_VERSION",
    "OutcomeDatasetArtifactStore",
    "OutcomeDatasetManifest",
    "RECORDS_FILENAME",
)
