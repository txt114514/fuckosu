"""验证反事实 Outcome 数据集的确定性构造与严格制品边界。"""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import fields, replace
from pathlib import Path

import pytest

from package import AffineOsuVideoTransform
from traning.contracts import (
    ArtifactManifest,
    BeliefState,
    CandidateObservation,
    DataSplit,
    InferenceCandidateRecord,
    ObjectType,
    ObjectTypeDistribution,
    Point2D,
)
from traning.data import FrameCoordinateTransform
from traning.evaluation import SCORE_VERSION
from traning.infrastructure import AtomicWriteError, IntegrityError, SchemaMismatchError
from traning.outcome.dataset import (
    MANIFEST_FILENAME,
    OUTCOME_DATASET_ARTIFACT_TYPE,
    OUTCOME_DATASET_SCHEMA_VERSION,
    CounterfactualFrame,
    CounterfactualOutcomeDataset,
    CounterfactualOutcomeDatasetBuilder,
    OutcomeDatasetArtifactStore,
)
from traning.outcome.dataset import artifact as artifact_module
from traning.outcome.oracle import (
    OUTCOME_ORACLE_VERSION,
    OracleState,
    OracleTarget,
    OutcomeCategory,
    OutcomeOracle,
)


_SOURCE_WIDTH = 513
_SOURCE_HEIGHT = 385
_COORDINATE_TRANSFORM = FrameCoordinateTransform(
    source_frame_width=_SOURCE_WIDTH,
    source_frame_height=_SOURCE_HEIGHT,
    transform_identity="outcome-dataset-identity-v1",
    transform=AffineOsuVideoTransform(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))),
)


def _belief(track_id: str, timestamp_ms: float, *, x: float) -> BeliefState:
    return BeliefState(
        track_id=track_id,
        timestamp_ms=timestamp_ms,
        belief_embedding=(0.1, 0.2, 0.3),
        position_mean=Point2D(x, 50.0),
        position_uncertainty=Point2D(1.0, 1.5),
        visibility_probability=0.9,
        object_type_distribution=ObjectTypeDistribution(1.0, 0.0, 0.0, 0.0),
        age=3,
        time_since_seen_ms=0.0,
        uncertainty=0.2,
    )


def _frame(sample_id: str, timestamp_ms: float) -> CounterfactualFrame:
    beliefs = (
        _belief("track-b", timestamp_ms, x=80.0),
        _belief("track-a", timestamp_ms, x=20.0),
    )
    targets = (
        OracleTarget(
            track_id="track-b",
            object_id=f"{sample_id}-object-b",
            object_type=ObjectType.RING,
            position=Point2D(80.0, 50.0),
            start_time_ms=timestamp_ms,
            end_time_ms=timestamp_ms + 100.0,
        ),
        OracleTarget(
            track_id="track-a",
            object_id=f"{sample_id}-object-a",
            object_type=ObjectType.RING,
            position=Point2D(20.0, 50.0),
            start_time_ms=timestamp_ms,
            end_time_ms=timestamp_ms + 100.0,
        ),
    )
    return CounterfactualFrame(
        sample_id=sample_id,
        split=DataSplit.TRAIN,
        source_frame_width=_SOURCE_WIDTH,
        source_frame_height=_SOURCE_HEIGHT,
        transform_fingerprint=_COORDINATE_TRANSFORM.transform_fingerprint,
        beliefs=beliefs,
        oracle_state=OracleState(f"{sample_id}-state", timestamp_ms, targets),
    )


def _builder(
    horizons: tuple[float, ...] = (0.0, 125.0, 300.0),
) -> CounterfactualOutcomeDatasetBuilder:
    return CounterfactualOutcomeDatasetBuilder(
        OutcomeOracle(circle_radius=20.0),
        horizons,
        _COORDINATE_TRANSFORM,
    )


def _publish(
    directory: Path,
    dataset: CounterfactualOutcomeDataset,
) -> OutcomeDatasetArtifactStore:
    store = OutcomeDatasetArtifactStore(directory)
    store.publish(
        dataset,
        dataset_id="dataset-1",
        producer_id="producer-1",
        created_at_ms=123.0,
    )
    return store


def _load(store: OutcomeDatasetArtifactStore) -> CounterfactualOutcomeDataset:
    return store.load(
        expected_dataset_id="dataset-1",
        expected_split=DataSplit.TRAIN,
        expected_producer_id="producer-1",
        expected_transform_fingerprint=(_COORDINATE_TRANSFORM.transform_fingerprint),
    )


def _manifest_payload(directory: Path) -> dict[str, object]:
    decoded = json.loads((directory / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert isinstance(decoded, dict)
    return decoded


def _records_path(directory: Path, manifest: dict[str, object]) -> Path:
    metadata = manifest["metadata"]
    assert isinstance(metadata, dict)
    filename = metadata["records_filename"]
    assert isinstance(filename, str)
    return directory / filename


def _write_manifest(directory: Path, payload: dict[str, object]) -> None:
    (directory / MANIFEST_FILENAME).write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _rehash_records(directory: Path, manifest: dict[str, object]) -> None:
    records_path = _records_path(directory, manifest)
    manifest["sha256"] = hashlib.sha256(records_path.read_bytes()).hexdigest()
    _write_manifest(directory, manifest)


def test_builder_order_labels_and_serialized_bytes_are_deterministic(
    tmp_path: Path,
) -> None:
    """frames/beliefs 反序仍产生相同 records 次序和相同 JSONL 字节。"""

    frame_a = _frame("sample-a", 0.0)
    frame_b = _frame("sample-b", 1000.0)
    forward_dataset = _builder().build((frame_b, frame_a))
    reversed_dataset = _builder().build(
        (
            replace(frame_a, beliefs=tuple(reversed(frame_a.beliefs))),
            replace(frame_b, beliefs=tuple(reversed(frame_b.beliefs))),
        )
    )
    assert forward_dataset == reversed_dataset
    forward = forward_dataset.records
    assert tuple((item.belief.track_id, item.horizon_ms) for item in forward[:6]) == (
        ("track-a", 0.0),
        ("track-a", 125.0),
        ("track-a", 300.0),
        ("track-b", 0.0),
        ("track-b", 125.0),
        ("track-b", 300.0),
    )
    first_track = forward[:3]
    assert tuple(item.target_category for item in first_track) == (
        OutcomeCategory.HIGH,
        OutcomeCategory.MEDIUM,
        OutcomeCategory.INVALID,
    )
    assert first_track[0].target_score > first_track[1].target_score > 0.0
    assert tuple((item.valid, item.expires) for item in first_track) == (
        (True, False),
        (True, False),
        (False, True),
    )
    assert all(item.split is DataSplit.TRAIN for item in forward)
    assert all(item.source_sample_id in {"sample-a", "sample-b"} for item in forward)
    assert all(item.oracle_state_id.endswith("-state") for item in forward)
    assert first_track[0].target_object_id == "sample-a-object-a"

    directories = (tmp_path / "forward", tmp_path / "reverse")
    first_store = _publish(directories[0], forward_dataset)
    second_store = _publish(directories[1], reversed_dataset)
    first_manifest = _manifest_payload(directories[0])
    second_manifest = _manifest_payload(directories[1])
    assert (
        _records_path(directories[0], first_manifest).read_bytes()
        == _records_path(directories[1], second_manifest).read_bytes()
    )
    assert _load(first_store) == _load(second_store) == forward_dataset


def test_sample_ids_are_unique_and_encode_changed_horizon() -> None:
    """相同 index 上 horizon 值变化必须改变 ID，不能只依赖数组位置。"""

    frame = _frame("sample-a", 0.0)
    at_16 = _builder((16.0,)).build((frame,)).records
    at_32 = _builder((32.0,)).build((frame,)).records
    assert len({record.sample_id for record in at_16}) == len(at_16)
    assert len({record.sample_id for record in at_32}) == len(at_32)
    assert {record.sample_id for record in at_16}.isdisjoint(
        record.sample_id for record in at_32
    )


def test_length_prefixed_sample_id_prevents_component_boundary_collision() -> None:
    """``a:b + c`` 与 ``a + b:c`` 不得生成旧分隔符方案下的同一 sample ID。"""

    def one_frame(sample_id: str, track_id: str, object_id: str) -> CounterfactualFrame:
        """构造单轨单目标帧以验证长度前缀 ID 编码。"""

        belief = _belief(track_id, 0.0, x=20.0)
        target = OracleTarget(
            track_id=track_id,
            object_id=object_id,
            object_type=ObjectType.RING,
            position=belief.position_mean,
            start_time_ms=0.0,
            end_time_ms=100.0,
        )
        return CounterfactualFrame(
            sample_id=sample_id,
            split=DataSplit.TRAIN,
            source_frame_width=_SOURCE_WIDTH,
            source_frame_height=_SOURCE_HEIGHT,
            transform_fingerprint=_COORDINATE_TRANSFORM.transform_fingerprint,
            beliefs=(belief,),
            oracle_state=OracleState(f"state-{object_id}", 0.0, (target,)),
        )

    dataset = _builder((0.0,)).build(
        (
            one_frame("a:b", "c", "object-left"),
            one_frame("a", "b:c", "object-right"),
        )
    )
    sample_ids = tuple(record.sample_id for record in dataset.records)
    assert len(sample_ids) == len(set(sample_ids)) == 2


def test_dataset_wrapper_rejects_empty_or_mixed_split_records() -> None:
    """Typed dataset 必须非空，且 manifest split 不能与 record lineage 分裂。"""

    with pytest.raises(ValueError, match="records 不得为空"):
        CounterfactualOutcomeDataset(
            DataSplit.TRAIN,
            (),
            _COORDINATE_TRANSFORM.transform_fingerprint,
        )
    with pytest.raises(ValueError, match="frames 不得为空"):
        _builder().build(())

    dataset = _builder((0.0,)).build((_frame("sample-a", 0.0),))
    wrong_split = replace(dataset.records[0], split=DataSplit.TEST)
    with pytest.raises(ValueError, match="dataset.split"):
        CounterfactualOutcomeDataset(
            DataSplit.TRAIN,
            (wrong_split,),
            _COORDINATE_TRANSFORM.transform_fingerprint,
        )


def test_artifact_round_trip_uses_canonical_manifest(tmp_path: Path) -> None:
    """manifest 必须包装 canonical ArtifactManifest 并记录精确版本和摘要。"""

    dataset = _builder().build((_frame("sample-a", 0.0),))
    store = OutcomeDatasetArtifactStore(tmp_path)
    manifest = store.publish(
        dataset,
        dataset_id="dataset-1",
        producer_id="producer-1",
        created_at_ms=123.0,
    )
    assert isinstance(manifest.artifact, ArtifactManifest)
    assert manifest.artifact.artifact_id == "dataset-1-counterfactual-outcomes"
    assert manifest.artifact.artifact_type == OUTCOME_DATASET_ARTIFACT_TYPE
    assert manifest.schema_version == OUTCOME_DATASET_SCHEMA_VERSION == 2
    assert manifest.dataset_id == "dataset-1"
    assert manifest.split is DataSplit.TRAIN
    assert manifest.producer_id == "producer-1"
    assert manifest.row_count == len(dataset.records)
    assert dict(manifest.artifact.metadata) == {
        "records_filename": manifest.records_filename,
        "oracle_version": OUTCOME_ORACLE_VERSION,
        "scoring_version": SCORE_VERSION,
        "transform_fingerprint": _COORDINATE_TRANSFORM.transform_fingerprint,
    }
    records_path = tmp_path / manifest.records_filename
    assert hashlib.sha256(records_path.read_bytes()).hexdigest() == manifest.sha256
    assert _load(store) == dataset


@pytest.mark.parametrize(
    ("case", "expected_error"),
    (
        ("checksum", IntegrityError),
        ("row_count", IntegrityError),
        ("schema", SchemaMismatchError),
        ("dataset", SchemaMismatchError),
        ("split", SchemaMismatchError),
        ("producer", SchemaMismatchError),
        ("oracle", SchemaMismatchError),
        ("scoring", SchemaMismatchError),
        ("transform", SchemaMismatchError),
    ),
)
def test_manifest_identity_and_integrity_tampering_is_rejected(
    tmp_path: Path,
    case: str,
    expected_error: type[Exception],
) -> None:
    """摘要、行数、schema、身份及 canonical 版本任一篡改都必须硬失败。"""

    directory = tmp_path / case
    dataset = _builder().build((_frame("sample-a", 0.0),))
    store = _publish(directory, dataset)
    manifest = _manifest_payload(directory)
    metadata = manifest["metadata"]
    assert isinstance(metadata, dict)
    mutations: dict[str, tuple[dict[str, object], str, object]] = {
        "checksum": (manifest, "sha256", "0" * 64),
        "row_count": (manifest, "row_count", len(dataset.records) + 1),
        "schema": (manifest, "schema_version", 1),
        "dataset": (manifest, "dataset_id", "other-dataset"),
        "split": (manifest, "split", DataSplit.TEST.value),
        "producer": (manifest, "producer_id", "other-producer"),
        "oracle": (metadata, "oracle_version", "other-oracle"),
        "scoring": (metadata, "scoring_version", "other-scoring"),
        "transform": (
            metadata,
            "transform_fingerprint",
            "transform-0000000000000000",
        ),
    }
    target, key, value = mutations[case]
    target[key] = value
    _write_manifest(directory, manifest)
    with pytest.raises(expected_error):
        _load(store)


@pytest.mark.parametrize("injected_field", ("ground_truth", "unknown_field"))
def test_rehashed_gt_or_unknown_record_field_is_rejected(
    tmp_path: Path,
    injected_field: str,
) -> None:
    """即使攻击者重算摘要，GT 或未知字段也不能越过 exact-schema 边界。"""

    directory = tmp_path / injected_field
    dataset = _builder().build((_frame("sample-a", 0.0),))
    store = _publish(directory, dataset)
    manifest = _manifest_payload(directory)
    records_path = _records_path(directory, manifest)
    lines = records_path.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    assert isinstance(first, dict)
    first[injected_field] = {"hit_objects": []}
    lines[0] = json.dumps(first, separators=(",", ":"), sort_keys=True)
    records_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _rehash_records(directory, manifest)
    with pytest.raises(SchemaMismatchError):
        _load(store)


def test_rehashed_duplicate_sample_id_is_rejected(tmp_path: Path) -> None:
    """重复 sample_id 即使行数和摘要一致，也不是合法 typed dataset。"""

    dataset = _builder().build((_frame("sample-a", 0.0),))
    store = _publish(tmp_path, dataset)
    manifest = _manifest_payload(tmp_path)
    records_path = _records_path(tmp_path, manifest)
    lines = records_path.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    second = json.loads(lines[1])
    second["sample_id"] = first["sample_id"]
    lines[1] = json.dumps(second, separators=(",", ":"), sort_keys=True)
    records_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _rehash_records(tmp_path, manifest)
    with pytest.raises(SchemaMismatchError, match="重复 sample_id"):
        _load(store)


def test_rehashed_record_split_mismatch_is_rejected(tmp_path: Path) -> None:
    """record 自带 split 即使合法，也必须与 manifest split 完全一致。"""

    dataset = _builder().build((_frame("sample-a", 0.0),))
    store = _publish(tmp_path, dataset)
    manifest = _manifest_payload(tmp_path)
    records_path = _records_path(tmp_path, manifest)
    lines = records_path.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    first["split"] = DataSplit.TEST.value
    lines[0] = json.dumps(first, separators=(",", ":"), sort_keys=True)
    records_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _rehash_records(tmp_path, manifest)
    with pytest.raises(SchemaMismatchError, match="manifest split"):
        _load(store)


def test_manifest_commit_failure_preserves_old_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """新 generation 已写完但 manifest 提交失败时，旧提交仍须完整可读。"""

    old_dataset = _builder((0.0,)).build((_frame("sample-a", 0.0),))
    store = _publish(tmp_path, old_dataset)
    old_manifest = (tmp_path / MANIFEST_FILENAME).read_bytes()
    new_dataset = _builder((16.0,)).build((_frame("sample-b", 1000.0),))

    def fail_manifest_commit(path: Path, payload: object) -> None:
        """模拟最终 manifest 原子提交失败。"""

        raise AtomicWriteError(f"injected manifest failure: {path.name}")

    monkeypatch.setattr(artifact_module, "atomic_write_json", fail_manifest_commit)
    with pytest.raises(AtomicWriteError, match="injected manifest failure"):
        store.publish(
            new_dataset,
            dataset_id="dataset-1",
            producer_id="producer-1",
            created_at_ms=456.0,
        )
    assert (tmp_path / MANIFEST_FILENAME).read_bytes() == old_manifest
    assert _load(store) == old_dataset


def test_dataset_source_has_no_legacy_or_any_and_inference_has_no_oracle_fields() -> (
    None
):
    """Dataset 不得依赖 legacy/Any，推理契约在类型上不得暴露 oracle 标签。"""

    dataset_root = Path(__file__).parents[2] / "outcome" / "dataset"
    for source_path in dataset_root.rglob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(
                    alias.name != "osu_v2" and not alias.name.startswith("osu_v2.")
                    for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert module != "osu_v2" and not module.startswith("osu_v2.")
                assert all(alias.name != "Any" for alias in node.names)

    inference_fields = {field.name for field in fields(InferenceCandidateRecord)} | {
        field.name for field in fields(CandidateObservation)
    }
    assert inference_fields.isdisjoint(
        {
            "oracle_state",
            "oracle_outcome",
            "target_category",
            "target_score",
            "valid",
            "expires",
            "ground_truth",
            "hit_objects",
        }
    )
