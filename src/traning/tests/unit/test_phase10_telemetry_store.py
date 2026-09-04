"""Phase 10 typed telemetry event 与四通道状态存储验收。"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from traning.state import TelemetryEvent
from traning.core.evaluation import (
    EvaluationTag,
    PrimaryError,
    SequenceScore,
    build_sequence_evaluation_events,
)
from traning.lib.infrastructure.errors import IntegrityError, SchemaMismatchError
from traning.lib.telemetry.events import (
    TELEMETRY_SCHEMA_VERSION,
    EvaluationEvent,
    MetricsEvent,
    ResourceEvent,
)
from traning.lib.telemetry.store import StateStore


_FILENAMES = (
    "metrics.jsonl",
    "resources.jsonl",
    "evaluation.jsonl",
    "events.jsonl",
)


def _metrics(step: int = 1) -> MetricsEvent:
    return MetricsEvent(
        schema_version=TELEMETRY_SCHEMA_VERSION,
        timestamp_ms=float(step),
        run_id="run-telemetry",
        step=step,
        loss=0.5,
        perception_recall=0.9,
        tracking_id_switches=2,
        outcome_nll=0.4,
        outcome_brier=0.2,
        outcome_ece=0.1,
        expected_score_error=0.03,
        decision_utility=-0.2,
        wait_click_ratio=0.25,
        score=0.8,
    )


def _resources(step: int = 1) -> ResourceEvent:
    return ResourceEvent(
        schema_version=TELEMETRY_SCHEMA_VERSION,
        timestamp_ms=float(step),
        run_id="run-telemetry",
        step=step,
        gpu_utilization=0.75,
        vram_used_mb=4096.0,
        vram_total_mb=8192.0,
        throughput=123.5,
    )


def _unresolved_evaluation() -> EvaluationEvent:
    score = SequenceScore(
        clicks=(),
        resolved_targets=(),
        unresolved_target_ids=("target-105",),
    )
    canonical = build_sequence_evaluation_events("sample-105", 105, score)[0]
    return EvaluationEvent(
        schema_version=TELEMETRY_SCHEMA_VERSION,
        timestamp_ms=105.0,
        run_id="run-telemetry",
        event=canonical,
    )


def _lifecycle(payload: object | None = None) -> TelemetryEvent:
    event_payload = () if payload is None else (("detail", payload),)
    return TelemetryEvent(
        schema_version=TELEMETRY_SCHEMA_VERSION,
        event_type="training.stage",
        timestamp_ms=1.0,
        run_id="run-telemetry",
        metrics=(("epoch", 1.0),),
        payload=event_payload,
    )


def _read_lines(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_four_channels_persist_versioned_strict_json_and_recover(
    tmp_path: Path,
) -> None:
    """四类通道必须用版本化严格 JSON 持久化并可恢复。"""

    directory = tmp_path / "telemetry"
    store = StateStore(directory)
    metrics = _metrics()
    resources = _resources()
    evaluation = _unresolved_evaluation()
    lifecycle = _lifecycle()

    for event in (metrics, resources, evaluation, lifecycle):
        store.publish(event)

    assert tuple(sorted(path.name for path in directory.iterdir())) == tuple(
        sorted(_FILENAMES)
    )
    for filename in _FILENAMES:
        rows = _read_lines(directory / filename)
        assert len(rows) == 1
        assert rows[0]["schema_version"] == TELEMETRY_SCHEMA_VERSION
        assert (directory / filename).read_bytes().endswith(b"\n")

    recovered = StateStore(directory)
    history = recovered.history()
    assert history.metrics == (metrics,)
    assert history.resources == (resources,)
    assert history.evaluations == (evaluation,)
    assert history.events == (lifecycle,)


def test_snapshot_is_frozen_copy_and_keeps_canonical_event_identity(
    tmp_path: Path,
) -> None:
    """快照必须不可变，并保留进程内 canonical 事件对象身份。"""

    mutable_payload: dict[str, object] = {"items": [1]}
    lifecycle = _lifecycle(mutable_payload)
    evaluation = _unresolved_evaluation()
    canonical = evaluation.event
    store = StateStore(tmp_path / "telemetry")
    store.publish(evaluation)
    store.publish(lifecycle)

    snapshot = store.snapshot()
    assert snapshot.evaluation is evaluation
    assert snapshot.evaluation.event is canonical
    assert snapshot.evaluation.event.primary_error is PrimaryError.DECISION
    assert snapshot.evaluation.event.error_tags == (EvaluationTag.UNRESOLVED_TARGET,)
    with pytest.raises(FrozenInstanceError):
        snapshot.metrics_count = 100  # type: ignore[misc]

    mutable_items = mutable_payload["items"]
    assert isinstance(mutable_items, list)
    mutable_items.append(2)
    copied_payload = dict(store.snapshot().event.payload)  # type: ignore[union-attr]
    assert copied_payload == {"detail": {"items": [1]}}


def test_evaluation_disk_roundtrip_preserves_pass_and_error_semantics(
    tmp_path: Path,
) -> None:
    """评估事件磁盘往返不得改变通过状态或错误归因。"""

    envelope = _unresolved_evaluation()
    store = StateStore(tmp_path / "telemetry")
    store.publish(envelope)

    recovered = StateStore(tmp_path / "telemetry").snapshot().evaluation
    assert recovered is not None
    assert recovered.event == envelope.event
    assert recovered.event.event_id == envelope.event.event_id
    assert recovered.event.passed is False
    assert recovered.event.primary_error is PrimaryError.DECISION
    assert recovered.event.primary_error is not PrimaryError.SPATIAL


def test_concurrent_publish_keeps_every_complete_record(tmp_path: Path) -> None:
    """并发发布必须保留每条完整记录且不能发生交错损坏。"""

    store = StateStore(tmp_path / "telemetry")
    event_count = 64

    with ThreadPoolExecutor(max_workers=8) as executor:
        tuple(
            executor.map(lambda step: store.publish(_metrics(step)), range(event_count))
        )

    history = store.history()
    assert len(history.metrics) == event_count
    assert {event.step for event in history.metrics} == set(range(event_count))
    rows = _read_lines(tmp_path / "telemetry" / "metrics.jsonl")
    assert len(rows) == event_count
    assert {row["step"] for row in rows} == set(range(event_count))
    assert StateStore(tmp_path / "telemetry").history().metrics == history.metrics


def test_publish_calls_fsync_and_rejects_unknown_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """发布必须执行 fsync，并拒绝未注册事件类型。"""

    import traning.lib.telemetry.store as store_module

    store = StateStore(tmp_path / "telemetry")
    real_fsync = store_module.os.fsync
    descriptors: list[int] = []

    def observe_fsync(descriptor: int) -> None:
        """记录 fsync 调用后委托给真实实现。"""

        descriptors.append(descriptor)
        real_fsync(descriptor)

    monkeypatch.setattr(store_module.os, "fsync", observe_fsync)
    store.publish(_metrics())
    assert descriptors
    with pytest.raises(TypeError, match="typed telemetry event"):
        store.publish(object())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("perception_recall", 1.1, r"\[0, 1\]"),
        ("tracking_id_switches", -1, "不得为负数"),
        ("outcome_nll", float("nan"), "有限数值"),
        ("wait_click_ratio", float("inf"), "有限数值"),
    ),
)
def test_metrics_reject_bad_values(field: str, value: object, message: str) -> None:
    """指标契约必须拒绝非法数值与错误类型。"""

    values: dict[str, object] = {
        "schema_version": TELEMETRY_SCHEMA_VERSION,
        "timestamp_ms": 1.0,
        "run_id": "run-telemetry",
        "step": 1,
        "loss": 0.5,
        "perception_recall": 0.9,
        "tracking_id_switches": 0,
        "outcome_nll": 0.4,
        "outcome_brier": 0.2,
        "outcome_ece": 0.1,
        "expected_score_error": 0.03,
        "decision_utility": 0.2,
        "wait_click_ratio": 0.25,
    }
    values[field] = value
    with pytest.raises((TypeError, ValueError), match=message):
        MetricsEvent(**values)  # type: ignore[arg-type]


def test_store_rejects_partial_channel_set_and_corrupt_json(tmp_path: Path) -> None:
    """恢复时必须拒绝残缺通道集合与损坏 JSON。"""

    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "metrics.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(IntegrityError, match="四通道文件集合不完整"):
        StateStore(partial)

    corrupt = tmp_path / "corrupt"
    StateStore(corrupt)
    (corrupt / "events.jsonl").write_text(
        '{"schema_version":1,"schema_version":1}\n', encoding="utf-8"
    )
    with pytest.raises(IntegrityError, match="重复键"):
        StateStore(corrupt)


def test_store_rejects_wrong_channel_schema_and_truncated_tail(tmp_path: Path) -> None:
    """恢复时必须拒绝错误通道 schema 与截断尾记录。"""

    wrong = tmp_path / "wrong"
    StateStore(wrong)
    row = {
        "schema_version": 1,
        "record_type": "resources",
    }
    (wrong / "metrics.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(SchemaMismatchError, match="record_type"):
        StateStore(wrong)

    truncated = tmp_path / "truncated"
    StateStore(truncated)
    (truncated / "metrics.jsonl").write_text('{"schema_version":1', encoding="utf-8")
    with pytest.raises(IntegrityError, match="尾行不完整"):
        StateStore(truncated)


def test_store_rejects_path_and_schema_mismatch(tmp_path: Path) -> None:
    """事件通道路径和声明 schema 不一致时必须硬失败。"""

    with pytest.raises(TypeError, match="pathlib.Path"):
        StateStore(str(tmp_path))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="schema_version"):
        StateStore(
            tmp_path / "bad-version",
            schema_version=TELEMETRY_SCHEMA_VERSION + 1,
        )

    store = StateStore(tmp_path / "telemetry")
    mismatched = TelemetryEvent(
        schema_version=TELEMETRY_SCHEMA_VERSION + 1,
        event_type="training.stage",
        timestamp_ms=1.0,
        run_id="run-telemetry",
    )
    with pytest.raises(ValueError, match="StateStore 不一致"):
        store.publish(mismatched)


def test_store_binds_first_run_and_recovery_rejects_mixed_runs(tmp_path: Path) -> None:
    """store 必须绑定首个 run，并拒绝混入其他 run 的历史。"""

    directory = tmp_path / "telemetry"
    store = StateStore(directory)
    store.publish(_metrics())
    with pytest.raises(ValueError, match="已绑定 run_id"):
        store.publish(
            MetricsEvent(
                schema_version=TELEMETRY_SCHEMA_VERSION,
                timestamp_ms=2.0,
                run_id="another-run",
                step=2,
                loss=0.4,
                perception_recall=0.9,
                tracking_id_switches=0,
                outcome_nll=0.3,
                outcome_brier=0.2,
                outcome_ece=0.1,
                expected_score_error=0.02,
                decision_utility=0.1,
                wait_click_ratio=0.5,
            )
        )
    assert len(store.history().metrics) == 1

    store.publish(_resources())
    resources_path = directory / "resources.jsonl"
    resources_row = _read_lines(resources_path)[0]
    resources_row["run_id"] = "another-run"
    resources_path.write_text(
        json.dumps(resources_row, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(IntegrityError, match="多个 run_id"):
        StateStore(directory)
