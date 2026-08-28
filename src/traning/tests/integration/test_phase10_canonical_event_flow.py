"""验证归因事件在优化、遥测与 gallery 投影间保持唯一身份。"""

from __future__ import annotations

from traning.contracts import DataSplit
from traning.evaluation import (
    PrimaryError,
    SequenceScore,
    build_sequence_evaluation_events,
)
from traning.telemetry import (
    TELEMETRY_SCHEMA_VERSION,
    EvaluationEvent,
    StateStore,
    TelemetryReporter,
)
from traning.training import (
    EvaluationSplitEvent,
    HardExampleConsumer,
    HardExampleDestination,
    build_hard_example_plan,
)
from traning.visualization import RichDashboardRenderer


def test_frame_105_event_identity_is_shared_by_all_consumers(tmp_path) -> None:
    """未点击但画面很准的目标必须始终归入 Decision。"""

    score = SequenceScore(
        clicks=(),
        resolved_targets=(),
        unresolved_target_ids=("target-105",),
    )
    event = build_sequence_evaluation_events(
        "long_sequence_000008",
        105,
        score,
    )[0]
    hard_examples = build_hard_example_plan(
        (EvaluationSplitEvent(event, DataSplit.TRAIN),)
    )
    reporter = TelemetryReporter("run-frame-105", StateStore(tmp_path / "telemetry"))
    reporter.publish(
        EvaluationEvent(
            TELEMETRY_SCHEMA_VERSION,
            105.0,
            "run-frame-105",
            event,
        )
    )
    gallery = RichDashboardRenderer.render(reporter.snapshot())

    consumer_events = tuple(
        hard_examples.events_for(consumer)[0] for consumer in HardExampleConsumer
    )
    gallery_event = gallery.evaluations[0].event
    assert all(item is event for item in (*consumer_events, gallery_event))
    assert hard_examples.routes[0].destination is HardExampleDestination.DECISION
    assert gallery_event.primary_error is PrimaryError.DECISION
    assert gallery_event.primary_error is not PrimaryError.SPATIAL
    assert gallery_event.passed is False
