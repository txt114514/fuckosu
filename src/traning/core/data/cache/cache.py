"""推理候选缓存的原子发布与严格加载。"""

from __future__ import annotations

import json
import math
import time
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from traning.conf.versions import CANDIDATE_CACHE_SCHEMA_VERSION
from traning.state import (
    ArtifactManifest,
    CandidateObservation,
    DataSplit,
    InferenceCandidateRecord,
    JSONObject,
    JSONValue,
    ObjectTypeDistribution,
    Point2D,
    RingAttributes,
    SliderAttributes,
    SpinnerAttributes,
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


CANDIDATE_CACHE_ARTIFACT_TYPE = "inference_candidate_cache"
# Manifest 指向不可变 generation 文件；它是唯一提交点。发布中断时，旧
# manifest 仍能读取旧 generation，不会被一个半发布的新 records 覆盖。
RECORDS_FILENAME = "records.<generation>.jsonl"
MANIFEST_FILENAME = "manifest.json"

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


@dataclass(frozen=True, slots=True)
class CandidateCacheManifest:
    """候选缓存清单；缓存身份必须包含坐标变换指纹。"""

    artifact: ArtifactManifest

    def __post_init__(self) -> None:
        if self.artifact.artifact_type != CANDIDATE_CACHE_ARTIFACT_TYPE:
            raise ValueError(f"artifact_type 必须为 {CANDIDATE_CACHE_ARTIFACT_TYPE!r}")
        metadata = dict(self.artifact.metadata)
        if set(metadata) != {"records_filename", "transform_fingerprint"}:
            raise ValueError(
                "candidate cache metadata 必须且只能包含 "
                "records_filename 和 transform_fingerprint"
            )
        records_filename = metadata["records_filename"]
        if not isinstance(records_filename, str):
            raise TypeError("records_filename 必须是字符串")
        if (
            Path(records_filename).name != records_filename
            or not records_filename.startswith("records.")
            or not records_filename.endswith(".jsonl")
        ):
            raise ValueError("records_filename 必须是安全的 generation 文件名")
        transform_fingerprint = metadata["transform_fingerprint"]
        if not isinstance(transform_fingerprint, str):
            raise TypeError("transform_fingerprint 必须是字符串")
        require_transform_fingerprint(
            transform_fingerprint,
            "candidate cache transform_fingerprint",
        )

    @property
    def schema_version(self) -> int:
        """返回通用制品清单中的缓存 schema 版本。"""

        return self.artifact.schema_version

    @property
    def dataset_id(self) -> str:
        """返回生成本缓存的数据集稳定标识。"""

        return self.artifact.dataset_id

    @property
    def producer_id(self) -> str:
        """返回生成候选记录的模型或生产者标识。"""

        return self.artifact.producer_id

    @property
    def row_count(self) -> int:
        """返回清单承诺的候选记录总行数。"""

        return self.artifact.row_count

    @property
    def sha256(self) -> str:
        """返回不可变 records generation 的 SHA-256。"""

        return self.artifact.sha256

    @property
    def records_filename(self) -> str:
        """返回 manifest 已提交的不可变 records generation。"""

        value = dict(self.artifact.metadata)["records_filename"]
        if not isinstance(value, str):  # pragma: no cover - 构造期已校验
            raise TypeError("records_filename 必须是字符串")
        return value

    @property
    def transform_fingerprint(self) -> str:
        """返回生成本缓存时的完整坐标变换指纹。"""

        value = dict(self.artifact.metadata)["transform_fingerprint"]
        if not isinstance(value, str):  # pragma: no cover - 构造期已校验
            raise TypeError("transform_fingerprint 必须是字符串")
        return value


@dataclass(frozen=True, slots=True)
class CandidateCacheDataset:
    """将加载出的候选记录与其完整 manifest 来源永久绑定。"""

    manifest: CandidateCacheManifest
    records: tuple[InferenceCandidateRecord, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, CandidateCacheManifest):
            raise TypeError("manifest 必须是 CandidateCacheManifest")
        if not isinstance(self.records, tuple) or any(
            not isinstance(record, InferenceCandidateRecord) for record in self.records
        ):
            raise TypeError("records 必须是 InferenceCandidateRecord 元组")
        if len(self.records) != self.manifest.row_count:
            raise ValueError("records 数量必须与 manifest.row_count 一致")

    @property
    def transform_fingerprint(self) -> str:
        """返回候选像素坐标所绑定的变换指纹。"""

        return self.manifest.transform_fingerprint


class CandidateCacheWriter:
    """先完整发布记录文件，再以摘要和行数生成并发布清单。"""

    def __init__(
        self,
        cache_dir: Path,
        *,
        artifact_id: str,
        dataset_id: str,
        producer_id: str,
        transform_fingerprint: str,
        split: DataSplit = DataSplit.ALL,
        schema_version: int = CANDIDATE_CACHE_SCHEMA_VERSION,
    ) -> None:
        self._cache_dir = Path(cache_dir)
        self._artifact_id = artifact_id
        self._dataset_id = dataset_id
        self._producer_id = producer_id
        self._transform_fingerprint = transform_fingerprint
        self._split = split
        self._schema_version = schema_version

    def write(
        self,
        records: Iterable[InferenceCandidateRecord],
        *,
        created_at_ms: float | None = None,
    ) -> CandidateCacheManifest:
        """原子写入全量 JSONL，校验落盘内容后发布对应清单。"""

        row_count = 0
        timestamp = time.time() * 1000.0 if created_at_ms is None else created_at_ms
        # 在触碰现有缓存前校验全部清单身份字段与时间戳。
        self._build_manifest(
            row_count=0,
            digest="0" * 64,
            created_at_ms=timestamp,
            records_filename="records.preflight.jsonl",
        )

        def encoded_records() -> Iterable[JSONObject]:
            """逐条校验推理记录并生成严格 JSON object。"""

            nonlocal row_count
            for record in records:
                if not isinstance(record, InferenceCandidateRecord):
                    raise TypeError("records 只能包含 InferenceCandidateRecord")
                row_count += 1
                yield _record_to_json(record)

        records_filename = f"records.{uuid.uuid4().hex}.jsonl"
        records_path = self._cache_dir / records_filename
        atomic_write_jsonl(records_path, encoded_records())
        digest = sha256_file(records_path)
        manifest = self._build_manifest(
            row_count=row_count,
            digest=digest,
            created_at_ms=timestamp,
            records_filename=records_filename,
        )
        atomic_write_json(
            self._cache_dir / MANIFEST_FILENAME, _manifest_to_json(manifest)
        )
        return manifest

    def _build_manifest(
        self,
        *,
        row_count: int,
        digest: str,
        created_at_ms: float,
        records_filename: str,
    ) -> CandidateCacheManifest:
        return CandidateCacheManifest(
            ArtifactManifest(
                artifact_id=self._artifact_id,
                artifact_type=CANDIDATE_CACHE_ARTIFACT_TYPE,
                schema_version=self._schema_version,
                dataset_id=self._dataset_id,
                split=self._split,
                producer_id=self._producer_id,
                row_count=row_count,
                sha256=digest,
                created_at_ms=created_at_ms,
                metadata=(
                    ("records_filename", records_filename),
                    ("transform_fingerprint", self._transform_fingerprint),
                ),
            )
        )


class CandidateCacheReader:
    """仅加载身份、版本、行数和摘要全部符合预期的候选缓存。"""

    def __init__(
        self,
        cache_dir: Path,
        *,
        expected_artifact_id: str,
        expected_dataset_id: str,
        expected_producer_id: str,
        expected_transform_fingerprint: str,
        expected_schema_version: int = CANDIDATE_CACHE_SCHEMA_VERSION,
    ) -> None:
        # 在读磁盘前校验期望身份，避免非法输入变成“缓存未命中”。
        require_transform_fingerprint(
            expected_transform_fingerprint,
            "expected_transform_fingerprint",
        )
        self._cache_dir = Path(cache_dir)
        self._expected_artifact_id = expected_artifact_id
        self._expected_dataset_id = expected_dataset_id
        self._expected_producer_id = expected_producer_id
        self._expected_transform_fingerprint = expected_transform_fingerprint
        self._expected_schema_version = expected_schema_version

    def read_manifest(self) -> CandidateCacheManifest:
        """读取清单并硬校验调用方要求的缓存身份。"""

        payload = read_json_object(self._cache_dir / MANIFEST_FILENAME)
        manifest = _manifest_from_json(payload)
        artifact = manifest.artifact
        mismatches = (
            ("artifact_id", artifact.artifact_id, self._expected_artifact_id),
            ("schema_version", artifact.schema_version, self._expected_schema_version),
            ("dataset_id", artifact.dataset_id, self._expected_dataset_id),
            ("producer_id", artifact.producer_id, self._expected_producer_id),
            (
                "transform_fingerprint",
                manifest.transform_fingerprint,
                self._expected_transform_fingerprint,
            ),
        )
        for field_name, actual, expected in mismatches:
            if actual != expected:
                raise SchemaMismatchError(
                    f"候选缓存 {field_name} 不匹配：实际 {actual!r}，预期 {expected!r}"
                )
        return manifest

    def read(self) -> CandidateCacheDataset:
        """验证完整性后返回不会丢失 manifest 来源的 typed dataset。"""

        manifest = self.read_manifest()
        records_path = self._cache_dir / manifest.records_filename
        actual_digest = sha256_file(records_path)
        if actual_digest != manifest.sha256:
            raise IntegrityError(
                f"候选缓存 SHA-256 不匹配：实际 {actual_digest}，清单 {manifest.sha256}"
            )

        decoded: list[InferenceCandidateRecord] = []
        try:
            with records_path.open("rb") as handle:
                for line_number, raw_line in enumerate(handle, start=1):
                    if not raw_line.endswith(b"\n"):
                        raise IntegrityError(
                            f"候选缓存第 {line_number} 行未完整换行，疑似截断"
                        )
                    if raw_line == b"\n":
                        raise SchemaMismatchError(
                            f"候选缓存第 {line_number} 行不得为空"
                        )
                    decoded.append(_decode_record_line(raw_line, line_number))
        except (OSError, UnicodeError) as exc:
            raise IntegrityError(f"无法完整读取候选缓存：{records_path}") from exc

        if len(decoded) != manifest.row_count:
            raise IntegrityError(
                f"候选缓存行数不匹配：实际 {len(decoded)}，清单 {manifest.row_count}"
            )
        return CandidateCacheDataset(manifest=manifest, records=tuple(decoded))

    load = read


def publish_candidate_cache(
    directory: Path,
    records: Iterable[InferenceCandidateRecord],
    *,
    dataset_id: str,
    producer_id: str,
    transform_fingerprint: str,
    created_at_ms: float,
) -> ArtifactManifest:
    """发布候选缓存；坐标指纹是清单身份的必填部分。"""

    manifest = CandidateCacheWriter(
        directory,
        artifact_id=f"{dataset_id}.inference-candidates",
        dataset_id=dataset_id,
        producer_id=producer_id,
        transform_fingerprint=transform_fingerprint,
    ).write(records, created_at_ms=created_at_ms)
    return manifest.artifact


def load_candidate_cache(
    directory: Path,
    *,
    expected_artifact_id: str,
    expected_dataset_id: str,
    expected_producer_id: str,
    expected_transform_fingerprint: str,
    expected_schema_version: int = CANDIDATE_CACHE_SCHEMA_VERSION,
) -> CandidateCacheDataset:
    """加载经全部身份与完整性校验、且保留 manifest 的缓存。"""

    return CandidateCacheReader(
        directory,
        expected_artifact_id=expected_artifact_id,
        expected_dataset_id=expected_dataset_id,
        expected_producer_id=expected_producer_id,
        expected_transform_fingerprint=expected_transform_fingerprint,
        expected_schema_version=expected_schema_version,
    ).read()


def _record_to_json(record: InferenceCandidateRecord) -> JSONObject:
    observation = record.observation
    payload: JSONObject = {
        "frame_id": record.frame_id,
        "observation": {
            "frame_id": observation.frame_id,
            "frame_index": observation.frame_index,
            "timestamp_ms": observation.timestamp_ms,
            "candidate_id": observation.candidate_id,
            "x": observation.x,
            "y": observation.y,
            "confidence": observation.confidence,
            "visibility_probability": observation.visibility_probability,
            "object_type_distribution": {
                "p_ring": observation.object_type_distribution.p_ring,
                "p_slider": observation.object_type_distribution.p_slider,
                "p_spinner": observation.object_type_distribution.p_spinner,
                "p_unknown": observation.object_type_distribution.p_unknown,
            },
            "appearance_embedding": list(observation.appearance_embedding),
            "ring": _ring_to_json(observation.ring),
            "slider": _slider_to_json(observation.slider),
            "spinner": _spinner_to_json(observation.spinner),
        },
    }
    return payload


def _ring_to_json(value: RingAttributes | None) -> JSONValue:
    return (
        None
        if value is None
        else {"probability": value.probability, "radius_px": value.radius_px}
    )


def _slider_to_json(value: SliderAttributes | None) -> JSONValue:
    if value is None:
        return None
    return {
        "probability": value.probability,
        "direction": {"x": value.direction.x, "y": value.direction.y},
        "path": [{"x": point.x, "y": point.y} for point in value.path],
    }


def _spinner_to_json(value: SpinnerAttributes | None) -> JSONValue:
    return None if value is None else {"probability": value.probability}


def _decode_record_line(raw_line: bytes, line_number: int) -> InferenceCandidateRecord:
    try:
        text = raw_line.decode("utf-8")
        decoded: object = json.loads(
            text,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_non_finite_constant,
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise IntegrityError(f"候选缓存第 {line_number} 行不是完整 UTF-8 JSON") from exc
    if not isinstance(decoded, dict):
        raise SchemaMismatchError(f"候选缓存第 {line_number} 行必须是 object")
    return _record_from_json(cast(JSONObject, decoded), line_number)


def _record_from_json(
    payload: JSONObject, line_number: int
) -> InferenceCandidateRecord:
    context = f"records[{line_number}]"
    _require_fields(payload, {"frame_id", "observation"}, context)
    observation_payload = _object(payload["observation"], f"{context}.observation")
    observation = _observation_from_json(observation_payload, f"{context}.observation")
    try:
        return InferenceCandidateRecord(
            frame_id=_string(payload["frame_id"], f"{context}.frame_id"),
            observation=observation,
        )
    except (TypeError, ValueError) as exc:
        raise SchemaMismatchError(f"{context} 不满足推理候选契约") from exc


def _observation_from_json(payload: JSONObject, context: str) -> CandidateObservation:
    fields = {
        "frame_id",
        "frame_index",
        "timestamp_ms",
        "candidate_id",
        "x",
        "y",
        "confidence",
        "visibility_probability",
        "object_type_distribution",
        "appearance_embedding",
        "ring",
        "slider",
        "spinner",
    }
    _require_fields(payload, fields, context)
    distribution_payload = _object(
        payload["object_type_distribution"], f"{context}.object_type_distribution"
    )
    _require_fields(
        distribution_payload,
        {"p_ring", "p_slider", "p_spinner", "p_unknown"},
        f"{context}.object_type_distribution",
    )
    embedding = _array(
        payload["appearance_embedding"], f"{context}.appearance_embedding"
    )
    try:
        return CandidateObservation(
            frame_id=_string(payload["frame_id"], f"{context}.frame_id"),
            frame_index=_integer(payload["frame_index"], f"{context}.frame_index"),
            timestamp_ms=_number(payload["timestamp_ms"], f"{context}.timestamp_ms"),
            candidate_id=_string(payload["candidate_id"], f"{context}.candidate_id"),
            x=_number(payload["x"], f"{context}.x"),
            y=_number(payload["y"], f"{context}.y"),
            confidence=_number(payload["confidence"], f"{context}.confidence"),
            visibility_probability=_number(
                payload["visibility_probability"], f"{context}.visibility_probability"
            ),
            object_type_distribution=ObjectTypeDistribution(
                p_ring=_number(
                    distribution_payload["p_ring"],
                    f"{context}.object_type_distribution.p_ring",
                ),
                p_slider=_number(
                    distribution_payload["p_slider"],
                    f"{context}.object_type_distribution.p_slider",
                ),
                p_spinner=_number(
                    distribution_payload["p_spinner"],
                    f"{context}.object_type_distribution.p_spinner",
                ),
                p_unknown=_number(
                    distribution_payload["p_unknown"],
                    f"{context}.object_type_distribution.p_unknown",
                ),
            ),
            appearance_embedding=tuple(
                _number(value, f"{context}.appearance_embedding[{index}]")
                for index, value in enumerate(embedding)
            ),
            ring=_ring_from_json(payload["ring"], f"{context}.ring"),
            slider=_slider_from_json(payload["slider"], f"{context}.slider"),
            spinner=_spinner_from_json(payload["spinner"], f"{context}.spinner"),
        )
    except (TypeError, ValueError) as exc:
        if isinstance(exc, SchemaMismatchError):
            raise
        raise SchemaMismatchError(
            f"{context} 不满足 CandidateObservation 契约"
        ) from exc


def _ring_from_json(value: JSONValue, context: str) -> RingAttributes | None:
    if value is None:
        return None
    payload = _object(value, context)
    _require_fields(payload, {"probability", "radius_px"}, context)
    return RingAttributes(
        probability=_number(payload["probability"], f"{context}.probability"),
        radius_px=_number(payload["radius_px"], f"{context}.radius_px"),
    )


def _slider_from_json(value: JSONValue, context: str) -> SliderAttributes | None:
    if value is None:
        return None
    payload = _object(value, context)
    _require_fields(payload, {"probability", "direction", "path"}, context)
    direction_payload = _object(payload["direction"], f"{context}.direction")
    _require_fields(direction_payload, {"x", "y"}, f"{context}.direction")
    path_payload = _array(payload["path"], f"{context}.path")
    points: list[Point2D] = []
    for index, point_value in enumerate(path_payload):
        point_context = f"{context}.path[{index}]"
        point = _object(point_value, point_context)
        _require_fields(point, {"x", "y"}, point_context)
        points.append(
            Point2D(
                _number(point["x"], f"{point_context}.x"),
                _number(point["y"], f"{point_context}.y"),
            )
        )
    return SliderAttributes(
        probability=_number(payload["probability"], f"{context}.probability"),
        direction=Point2D(
            _number(direction_payload["x"], f"{context}.direction.x"),
            _number(direction_payload["y"], f"{context}.direction.y"),
        ),
        path=tuple(points),
    )


def _spinner_from_json(value: JSONValue, context: str) -> SpinnerAttributes | None:
    if value is None:
        return None
    payload = _object(value, context)
    _require_fields(payload, {"probability"}, context)
    return SpinnerAttributes(
        probability=_number(payload["probability"], f"{context}.probability")
    )


def _manifest_to_json(manifest: CandidateCacheManifest) -> JSONObject:
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
        "metadata": [[key, value] for key, value in artifact.metadata],
    }


def _manifest_from_json(payload: JSONObject) -> CandidateCacheManifest:
    _require_fields(payload, set(_MANIFEST_FIELDS), "manifest")
    metadata_payload = _array(payload["metadata"], "manifest.metadata")
    metadata: list[tuple[str, JSONValue]] = []
    for index, item in enumerate(metadata_payload):
        pair = _array(item, f"manifest.metadata[{index}]")
        if len(pair) != 2:
            raise SchemaMismatchError(f"manifest.metadata[{index}] 必须是二元组")
        metadata.append((_string(pair[0], f"manifest.metadata[{index}][0]"), pair[1]))
    try:
        artifact = ArtifactManifest(
            artifact_id=_string(payload["artifact_id"], "manifest.artifact_id"),
            artifact_type=_string(payload["artifact_type"], "manifest.artifact_type"),
            schema_version=_integer(
                payload["schema_version"], "manifest.schema_version"
            ),
            dataset_id=_string(payload["dataset_id"], "manifest.dataset_id"),
            split=DataSplit(_string(payload["split"], "manifest.split")),
            producer_id=_string(payload["producer_id"], "manifest.producer_id"),
            row_count=_integer(payload["row_count"], "manifest.row_count"),
            sha256=_string(payload["sha256"], "manifest.sha256"),
            created_at_ms=_number(payload["created_at_ms"], "manifest.created_at_ms"),
            metadata=tuple(metadata),
        )
        return CandidateCacheManifest(artifact)
    except (TypeError, ValueError) as exc:
        if isinstance(exc, SchemaMismatchError):
            raise
        raise SchemaMismatchError(
            "manifest 不满足 CandidateCacheManifest 契约"
        ) from exc


def _require_fields(payload: JSONObject, expected: set[str], context: str) -> None:
    actual = set(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise SchemaMismatchError(f"{context} 字段不匹配：缺少 {missing}，多出 {extra}")


def _object(value: JSONValue, context: str) -> JSONObject:
    if not isinstance(value, dict):
        raise SchemaMismatchError(f"{context} 必须是 object")
    return value


def _array(value: JSONValue, context: str) -> list[JSONValue]:
    if not isinstance(value, list):
        raise SchemaMismatchError(f"{context} 必须是 array")
    return value


def _string(value: JSONValue, context: str) -> str:
    if not isinstance(value, str):
        raise SchemaMismatchError(f"{context} 必须是 string")
    return value


def _integer(value: JSONValue, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SchemaMismatchError(f"{context} 必须是 integer")
    return value


def _number(value: JSONValue, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaMismatchError(f"{context} 必须是 number")
    if not math.isfinite(value):
        raise SchemaMismatchError(f"{context} 必须是有限数值")
    return float(value)


def _object_without_duplicate_keys(pairs: list[tuple[str, JSONValue]]) -> JSONObject:
    result: JSONObject = {}
    for key, value in pairs:
        if key in result:
            raise IntegrityError(f"候选缓存 JSON object 含有重复键：{key}")
        result[key] = value
    return result


def _reject_non_finite_constant(value: str) -> JSONValue:
    raise IntegrityError(f"候选缓存含有非标准数值常量：{value}")


__all__ = (
    "CANDIDATE_CACHE_ARTIFACT_TYPE",
    "CANDIDATE_CACHE_SCHEMA_VERSION",
    "CandidateCacheManifest",
    "CandidateCacheDataset",
    "CandidateCacheReader",
    "CandidateCacheWriter",
    "MANIFEST_FILENAME",
    "RECORDS_FILENAME",
    "load_candidate_cache",
    "publish_candidate_cache",
)
