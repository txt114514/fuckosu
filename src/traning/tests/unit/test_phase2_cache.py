"""验证候选缓存的事务提交、完整性和无 GT schema。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import traning.core.data.cache.cache as cache_module
from traning.state import (
    CandidateObservation,
    InferenceCandidateRecord,
    ObjectTypeDistribution,
)
from traning.core.data.cache import (
    MANIFEST_FILENAME,
    load_candidate_cache,
    publish_candidate_cache,
)
from traning.lib.infrastructure import (
    AtomicWriteError,
    IntegrityError,
    SchemaMismatchError,
    atomic_write_json,
    atomic_write_jsonl,
    read_json_object,
    sha256_file,
)


TRANSFORM_FINGERPRINT = "transform-0123456789abcdef"


def _record(candidate_id: str, *, x: float = 32.0) -> InferenceCandidateRecord:
    observation = CandidateObservation(
        frame_id="frame-0001",
        frame_index=1,
        timestamp_ms=16.0,
        candidate_id=candidate_id,
        x=x,
        y=48.0,
        confidence=0.9,
        visibility_probability=0.8,
        object_type_distribution=ObjectTypeDistribution(
            p_ring=0.7,
            p_slider=0.1,
            p_spinner=0.1,
            p_unknown=0.1,
        ),
        appearance_embedding=(1.0, 0.0, 0.0),
    )
    return InferenceCandidateRecord(frame_id="frame-0001", observation=observation)


def _records_path(cache_dir: Path) -> Path:
    manifest = read_json_object(cache_dir / MANIFEST_FILENAME)
    metadata = manifest["metadata"]
    assert isinstance(metadata, list) and len(metadata) == 2
    metadata_by_name = {
        pair[0]: pair[1]
        for pair in metadata
        if isinstance(pair, list) and len(pair) == 2 and isinstance(pair[0], str)
    }
    records_filename = metadata_by_name["records_filename"]
    assert isinstance(records_filename, str)
    return cache_dir / records_filename


def test_candidate_cache_round_trip_and_runtime_schema(tmp_path) -> None:
    """发布后恢复相同 typed record，持久化文本也不含 GT-only 字段。"""

    cache_dir = tmp_path / "cache"
    records = (_record("candidate-1"), _record("candidate-2", x=96.0))
    manifest = publish_candidate_cache(
        cache_dir,
        records,
        dataset_id="dataset-a",
        producer_id="perception-a",
        transform_fingerprint=TRANSFORM_FINGERPRINT,
        created_at_ms=1000.0,
    )
    assert manifest.row_count == 2
    loaded = load_candidate_cache(
        cache_dir,
        expected_artifact_id="dataset-a.inference-candidates",
        expected_dataset_id="dataset-a",
        expected_producer_id="perception-a",
        expected_transform_fingerprint=TRANSFORM_FINGERPRINT,
    )
    assert loaded.records == records
    assert loaded.manifest.artifact == manifest
    assert loaded.transform_fingerprint == TRANSFORM_FINGERPRINT
    persisted = _records_path(cache_dir).read_text(encoding="utf-8")
    for forbidden in (
        "temporal_target",
        "selected_candidate_id",
        "hit_objects",
        "oracle_label",
    ):
        assert forbidden not in persisted


def test_candidate_cache_rejects_checksum_and_row_count_mismatch(tmp_path) -> None:
    """内容篡改和 manifest 行数谎报分别被拒绝。"""

    cache_dir = tmp_path / "cache"
    publish_candidate_cache(
        cache_dir,
        (_record("candidate-1"),),
        dataset_id="dataset-a",
        producer_id="perception-a",
        transform_fingerprint=TRANSFORM_FINGERPRINT,
        created_at_ms=1000.0,
    )
    records_path = _records_path(cache_dir)
    records_path.write_bytes(records_path.read_bytes() + b"\n")
    with pytest.raises(IntegrityError, match="SHA-256"):
        load_candidate_cache(
            cache_dir,
            expected_artifact_id="dataset-a.inference-candidates",
            expected_dataset_id="dataset-a",
            expected_producer_id="perception-a",
            expected_transform_fingerprint=TRANSFORM_FINGERPRINT,
        )

    # 重新发布恢复合法 records，再单独篡改 row_count，确保走精确计数门禁。
    publish_candidate_cache(
        cache_dir,
        (_record("candidate-1"),),
        dataset_id="dataset-a",
        producer_id="perception-a",
        transform_fingerprint=TRANSFORM_FINGERPRINT,
        created_at_ms=1001.0,
    )
    manifest_path = cache_dir / MANIFEST_FILENAME
    manifest = read_json_object(manifest_path)
    manifest["row_count"] = 2
    atomic_write_json(manifest_path, manifest)
    with pytest.raises(IntegrityError, match="行数"):
        load_candidate_cache(
            cache_dir,
            expected_artifact_id="dataset-a.inference-candidates",
            expected_dataset_id="dataset-a",
            expected_producer_id="perception-a",
            expected_transform_fingerprint=TRANSFORM_FINGERPRINT,
        )


def test_candidate_cache_rejects_identity_and_gt_field_injection(tmp_path) -> None:
    """调用方身份不匹配和持久化 GT 注入都硬失败。"""

    cache_dir = tmp_path / "cache"
    publish_candidate_cache(
        cache_dir,
        (_record("candidate-1"),),
        dataset_id="dataset-a",
        producer_id="perception-a",
        transform_fingerprint=TRANSFORM_FINGERPRINT,
        created_at_ms=1000.0,
    )
    with pytest.raises(SchemaMismatchError, match="dataset_id"):
        load_candidate_cache(
            cache_dir,
            expected_artifact_id="dataset-a.inference-candidates",
            expected_dataset_id="dataset-b",
            expected_producer_id="perception-a",
            expected_transform_fingerprint=TRANSFORM_FINGERPRINT,
        )
    with pytest.raises(SchemaMismatchError, match="artifact_id"):
        load_candidate_cache(
            cache_dir,
            expected_artifact_id="another-artifact",
            expected_dataset_id="dataset-a",
            expected_producer_id="perception-a",
            expected_transform_fingerprint=TRANSFORM_FINGERPRINT,
        )

    records_path = _records_path(cache_dir)
    row = json.loads(records_path.read_text(encoding="utf-8"))
    row["temporal_target"] = {"action": "click"}
    atomic_write_jsonl(records_path, (row,))
    manifest_path = cache_dir / MANIFEST_FILENAME
    manifest = read_json_object(manifest_path)
    manifest["sha256"] = sha256_file(records_path)
    atomic_write_json(manifest_path, manifest)
    with pytest.raises(SchemaMismatchError, match="字段不匹配"):
        load_candidate_cache(
            cache_dir,
            expected_artifact_id="dataset-a.inference-candidates",
            expected_dataset_id="dataset-a",
            expected_producer_id="perception-a",
            expected_transform_fingerprint=TRANSFORM_FINGERPRINT,
        )


def test_candidate_cache_rejects_stale_or_legacy_coordinate_identity(
    tmp_path,
) -> None:
    """新坐标标定不得复用旧指纹或 schema v1 缓存。"""

    cache_dir = tmp_path / "cache"
    manifest = publish_candidate_cache(
        cache_dir,
        (_record("candidate-1"),),
        dataset_id="dataset-a",
        producer_id="perception-a",
        transform_fingerprint=TRANSFORM_FINGERPRINT,
        created_at_ms=1000.0,
    )
    assert dict(manifest.metadata)["transform_fingerprint"] == TRANSFORM_FINGERPRINT

    with pytest.raises(SchemaMismatchError, match="transform_fingerprint"):
        load_candidate_cache(
            cache_dir,
            expected_artifact_id="dataset-a.inference-candidates",
            expected_dataset_id="dataset-a",
            expected_producer_id="perception-a",
            expected_transform_fingerprint="transform-fedcba9876543210",
        )

    # 模拟升级前清单：schema v1 没有坐标指纹，即使 records 完整也拒绝。
    manifest_path = cache_dir / MANIFEST_FILENAME
    legacy_manifest = read_json_object(manifest_path)
    legacy_manifest["schema_version"] = 1
    metadata = legacy_manifest["metadata"]
    assert isinstance(metadata, list)
    legacy_manifest["metadata"] = [
        item
        for item in metadata
        if isinstance(item, list) and item[0] != "transform_fingerprint"
    ]
    atomic_write_json(manifest_path, legacy_manifest)
    with pytest.raises(SchemaMismatchError, match="CandidateCacheManifest"):
        load_candidate_cache(
            cache_dir,
            expected_artifact_id="dataset-a.inference-candidates",
            expected_dataset_id="dataset-a",
            expected_producer_id="perception-a",
            expected_transform_fingerprint=TRANSFORM_FINGERPRINT,
        )


def test_candidate_cache_rejects_malformed_transform_before_publication(
    tmp_path,
) -> None:
    """非共享坐标 API 指纹在落盘前就必须失败。"""

    cache_dir = tmp_path / "cache"
    with pytest.raises(ValueError, match="transform_fingerprint"):
        publish_candidate_cache(
            cache_dir,
            (_record("candidate-1"),),
            dataset_id="dataset-a",
            producer_id="perception-a",
            transform_fingerprint="centered-fallback",
            created_at_ms=1000.0,
        )
    assert not (cache_dir / MANIFEST_FILENAME).exists()


def test_failed_manifest_commit_keeps_previous_generation_readable(
    tmp_path,
    monkeypatch,
) -> None:
    """manifest 提交失败时，旧事务仍完整可读。"""

    cache_dir = tmp_path / "cache"
    old_records = (_record("candidate-old"),)
    publish_candidate_cache(
        cache_dir,
        old_records,
        dataset_id="dataset-a",
        producer_id="perception-a",
        transform_fingerprint=TRANSFORM_FINGERPRINT,
        created_at_ms=1000.0,
    )

    def fail_manifest(*args: object, **kwargs: object) -> None:
        """模拟 manifest 提交失败以检查 generation 原子性。"""

        del args, kwargs
        raise AtomicWriteError("injected manifest failure")

    monkeypatch.setattr(cache_module, "atomic_write_json", fail_manifest)
    with pytest.raises(AtomicWriteError):
        publish_candidate_cache(
            cache_dir,
            (_record("candidate-new"),),
            dataset_id="dataset-a",
            producer_id="perception-a",
            transform_fingerprint=TRANSFORM_FINGERPRINT,
            created_at_ms=1001.0,
        )
    loaded = load_candidate_cache(
        cache_dir,
        expected_artifact_id="dataset-a.inference-candidates",
        expected_dataset_id="dataset-a",
        expected_producer_id="perception-a",
        expected_transform_fingerprint=TRANSFORM_FINGERPRINT,
    )
    assert loaded.records == old_records
