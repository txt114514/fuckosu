"""Phase 6 canonical scoring 与离线 Outcome oracle 验收。"""

from __future__ import annotations

import ast
import importlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from traning.state import (
    BeliefState,
    DataSplit,
    DecisionAction,
    ObjectType,
    ObjectTypeDistribution,
    OutcomeCategory as ContractOutcomeCategory,
    OutcomeTrainingSample,
    Point2D,
)
from traning.core.evaluation import (
    CombinedScore,
    PointScore,
    PredictedClick,
    SequenceScoreSpec,
    TargetObject,
    score_click_sequence,
    score_point,
    score_slider,
)
from traning.core.outcome.oracle import (
    HypotheticalClick,
    OracleOutcome,
    OracleState,
    OracleTarget,
    OutcomeCategory,
    OutcomeOracle,
)


_TRANING_ROOT = Path(__file__).resolve().parents[2]
_GOLDEN_PATH = _TRANING_ROOT / "tests/regression/fixtures/legacy_golden_v1.json"


def _golden() -> dict[str, object]:
    payload = json.loads(_GOLDEN_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _ring_target(
    *,
    track_id: str = "ring-track",
    object_id: str = "ring-object",
    start_time_ms: float = 1000.0,
    end_time_ms: float = 2000.0,
) -> OracleTarget:
    return OracleTarget(
        track_id=track_id,
        object_id=object_id,
        object_type=ObjectType.RING,
        position=Point2D(100.0, 80.0),
        start_time_ms=start_time_ms,
        end_time_ms=end_time_ms,
    )


def _state(
    target: OracleTarget,
    *,
    timestamp_ms: float = 1000.0,
    resolved: bool = False,
) -> OracleState:
    return OracleState(
        state_id="state-1",
        timestamp_ms=timestamp_ms,
        targets=(target,),
        resolved_track_ids=(target.track_id,) if resolved else (),
    )


def _click(
    target: OracleTarget,
    *,
    horizon_ms: float = 0.0,
    position: Point2D | None = None,
    path: tuple[Point2D, ...] = (),
) -> HypotheticalClick:
    return HypotheticalClick(
        track_id=target.track_id,
        horizon_ms=horizon_ms,
        position=position or target.position,
        path=path,
    )


def test_v2_point_slider_and_sequence_match_existing_golden() -> None:
    """V2 canonical scorer 必须逐字段保持冻结 golden。"""

    fixture = _golden()
    oracle_fixture = fixture["oracle"]
    sequence_fixture = fixture["sequence"]
    assert isinstance(oracle_fixture, dict)
    assert isinstance(sequence_fixture, dict)
    expected_point = oracle_fixture["point"]
    expected_slider = oracle_fixture["slider"]
    assert isinstance(expected_point, dict)
    assert isinstance(expected_slider, dict)

    point = score_point(
        (100.0, 80.0),
        (106.0, 88.0),
        circle_radius=20.0,
        reference_time_ms=1000.0,
        predicted_time_ms=1075.0,
    )
    assert point.distance == pytest.approx(expected_point["distance"])
    assert point.distance_ratio == pytest.approx(expected_point["distance_ratio"])
    assert point.time_error_ms == pytest.approx(expected_point["time_error_ms"])
    assert point.score.spatial == pytest.approx(expected_point["spatial"])
    assert point.score.temporal == pytest.approx(expected_point["temporal"])
    assert point.score.raw == pytest.approx(expected_point["raw"])
    assert point.score.normalized == pytest.approx(expected_point["normalized"])
    assert point.passed is expected_point["passed"]

    slider = score_slider(
        (20.0, 20.0),
        (21.0, 22.0),
        ((20.0, 20.0), (50.0, 20.0), (80.0, 40.0)),
        ((21.0, 22.0), (50.0, 22.0), (79.0, 41.0)),
        circle_radius=16.0,
        reference_start_ms=2000.0,
        predicted_start_ms=2040.0,
    )
    assert slider.head.distance == pytest.approx(expected_slider["head_distance"])
    assert slider.path.reference_coverage == pytest.approx(
        expected_slider["path_reference_coverage"]
    )
    assert slider.path.prediction_precision == pytest.approx(
        expected_slider["path_prediction_precision"]
    )
    assert slider.score.raw == pytest.approx(expected_slider["raw"])
    assert slider.score.normalized == pytest.approx(expected_slider["normalized"])
    assert slider.passed is expected_slider["passed"]

    targets = (
        TargetObject("a", "circle", 1000.0, 1000.0, x=100.0, y=80.0, source_index=0),
        TargetObject("b", "circle", 1200.0, 1200.0, x=160.0, y=80.0, source_index=1),
    )
    clicks = (
        PredictedClick(1005.0, 102.0, 82.0),
        PredictedClick(1020.0, 101.0, 81.0),
        PredictedClick(1205.0, 161.0, 79.0),
        PredictedClick(1500.0, 0.0, 0.0),
    )
    sequence = score_click_sequence(targets, clicks, circle_radius=20.0)
    assert sequence.hit_count == sequence_fixture["hit_count"]
    assert sequence.miss_count == sequence_fixture["miss_count"]
    assert (
        sequence.frequency_limited_count == sequence_fixture["frequency_limited_count"]
    )
    assert (
        list(sequence.unresolved_target_ids)
        == sequence_fixture["unresolved_target_ids"]
    )
    assert [item.status for item in sequence.clicks] == sequence_fixture["statuses"]
    assert [item.target_id for item in sequence.clicks] == sequence_fixture[
        "target_ids"
    ]
    assert [list(item.error_tags) for item in sequence.clicks] == sequence_fixture[
        "error_tags"
    ]


def test_ring_oracle_is_sensitive_to_space_horizon_and_category() -> None:
    """ring 标签必须由 canonical 空间/时间评分共同决定。"""

    target = _ring_target()
    oracle = OutcomeOracle(circle_radius=20.0)
    state = _state(target)

    current = oracle.evaluate(state, _click(target))
    medium = oracle.evaluate(state, _click(target, horizon_ms=150.0))
    temporal_miss = oracle.evaluate(state, _click(target, horizon_ms=151.0))
    spatial_miss = oracle.evaluate(
        state,
        _click(target, position=Point2D(200.0, 180.0)),
    )

    assert current.category is OutcomeCategory.HIGH
    assert current.score == pytest.approx(1.0)
    assert current.valid and current.passed and not current.expires
    assert medium.category is OutcomeCategory.MEDIUM
    assert 0.5 <= medium.score < 0.8
    assert medium.score < current.score
    assert temporal_miss.category is OutcomeCategory.MISS
    assert temporal_miss.valid and not temporal_miss.passed
    assert spatial_miss.category is OutcomeCategory.MISS
    assert spatial_miss.spatial_error is not None
    assert spatial_miss.spatial_error > 20.0


def test_oracle_invalid_states_cover_expired_unknown_resolved_and_spinner() -> None:
    """不可评分状态必须明确映射为 INVALID，只有过期分支设置 expires。"""

    oracle = OutcomeOracle(circle_radius=20.0)
    expiring = _ring_target(end_time_ms=1000.0)
    expired = oracle.evaluate(_state(expiring), _click(expiring, horizon_ms=151.0))
    unknown_track = oracle.evaluate(
        _state(expiring),
        HypotheticalClick("missing-track", 0.0, Point2D(0.0, 0.0)),
    )
    resolved = oracle.evaluate(_state(expiring, resolved=True), _click(expiring))
    spinner = OracleTarget(
        "spinner-track",
        "spinner-object",
        ObjectType.SPINNER,
        Point2D(100.0, 80.0),
        1000.0,
        2000.0,
    )
    spinner_outcome = oracle.evaluate(_state(spinner), _click(spinner))

    assert expired.category is OutcomeCategory.INVALID
    assert expired.expires and not expired.valid
    for outcome in (unknown_track, resolved, spinner_outcome):
        assert outcome.category is OutcomeCategory.INVALID
        assert not outcome.valid and not outcome.expires and not outcome.passed
        assert outcome.score == 0.0


def test_slider_head_only_and_full_path_use_canonical_slider_score() -> None:
    """head-only 合法；完整路径必须与共享 score_slider 数值完全一致。"""

    reference_path = (
        Point2D(20.0, 20.0),
        Point2D(50.0, 20.0),
        Point2D(80.0, 40.0),
    )
    predicted_path = (
        Point2D(21.0, 22.0),
        Point2D(50.0, 22.0),
        Point2D(79.0, 41.0),
    )
    target = OracleTarget(
        "slider-track",
        "slider-object",
        ObjectType.SLIDER,
        reference_path[0],
        2000.0,
        3000.0,
        reference_path,
    )
    state = _state(target, timestamp_ms=2000.0)
    oracle = OutcomeOracle(circle_radius=16.0)
    head_only = oracle.evaluate(
        state,
        _click(target, horizon_ms=40.0, position=Point2D(21.0, 22.0)),
    )
    full_path = oracle.evaluate(
        state,
        _click(
            target,
            horizon_ms=40.0,
            position=Point2D(21.0, 22.0),
            path=predicted_path,
        ),
    )
    canonical = score_slider(
        (20.0, 20.0),
        (21.0, 22.0),
        tuple((point.x, point.y) for point in reference_path),
        tuple((point.x, point.y) for point in predicted_path),
        circle_radius=16.0,
        reference_start_ms=2000.0,
        predicted_start_ms=2040.0,
    )

    assert head_only.valid and head_only.passed
    assert full_path.score == pytest.approx(canonical.score.normalized)
    assert full_path.passed is canonical.passed
    assert full_path.spatial_error == pytest.approx(canonical.head.distance)
    assert full_path.time_error_ms == pytest.approx(canonical.head.time_error_ms)


def test_oracle_category_is_contract_identity_and_slider_head_matches_path_start() -> (
    None
):
    """Oracle 只重导出 canonical enum，并拒绝相互矛盾的 slider 头与路径。"""

    assert OutcomeCategory is ContractOutcomeCategory
    with pytest.raises(ValueError, match="path 起点"):
        OracleTarget(
            "slider-track",
            "slider-object",
            ObjectType.SLIDER,
            Point2D(10.0, 10.0),
            100.0,
            200.0,
            (Point2D(11.0, 10.0), Point2D(20.0, 10.0)),
        )


@pytest.mark.parametrize(
    ("normalized", "expected_category"),
    (
        (0.499999, OutcomeCategory.LOW),
        (0.5, OutcomeCategory.MEDIUM),
        (0.799999, OutcomeCategory.MEDIUM),
        (0.8, OutcomeCategory.HIGH),
    ),
)
def test_oracle_thresholds_produce_valid_training_sample_categories(
    monkeypatch: pytest.MonkeyPatch,
    normalized: float,
    expected_category: OutcomeCategory,
) -> None:
    """Oracle 阈值边界必须直接满足 OutcomeTrainingSample 的同源语义。"""

    oracle_module = importlib.import_module("traning.core.outcome.oracle.oracle")
    point_score = PointScore(
        distance=0.0,
        distance_ratio=0.0,
        time_error_ms=0.0,
        score=CombinedScore(0.0, 0.0, 0.0, normalized),
        passed=True,
    )
    monkeypatch.setattr(
        oracle_module, "score_point", lambda *args, **kwargs: point_score
    )
    target = _ring_target()
    outcome = OutcomeOracle(circle_radius=20.0).evaluate(_state(target), _click(target))
    assert outcome.category is expected_category

    belief = BeliefState(
        track_id=target.track_id,
        timestamp_ms=target.start_time_ms,
        belief_embedding=(1.0,),
        position_mean=target.position,
        position_uncertainty=Point2D(1.0, 1.0),
        visibility_probability=1.0,
        object_type_distribution=ObjectTypeDistribution(1.0, 0.0, 0.0, 0.0),
        age=1,
        time_since_seen_ms=0.0,
        uncertainty=0.0,
    )
    sample = OutcomeTrainingSample(
        sample_id=f"sample-{normalized}",
        split=DataSplit.TRAIN,
        source_sample_id="source-1",
        oracle_state_id="state-1",
        belief=belief,
        action=DecisionAction.CLICK,
        action_track_id=target.track_id,
        horizon_ms=0.0,
        target_category=outcome.category,
        target_score=outcome.score,
        valid=outcome.valid,
        expires=outcome.expires,
        target_object_id=outcome.target_object_id,
    )
    assert sample.target_category is expected_category
    contradictory_category = {
        OutcomeCategory.LOW: OutcomeCategory.MEDIUM,
        OutcomeCategory.MEDIUM: OutcomeCategory.LOW,
        OutcomeCategory.HIGH: OutcomeCategory.MEDIUM,
    }[expected_category]
    with pytest.raises(ValueError):
        replace(sample, target_category=contradictory_category)


def test_oracle_sequence_is_exact_canonical_delegation() -> None:
    """Oracle sequence 入口不得改变 canonical 结果或错误归因。"""

    targets = (
        TargetObject("a", "circle", 1000.0, 1000.0, x=100.0, y=80.0),
        TargetObject("b", "circle", 1200.0, 1200.0, x=160.0, y=80.0),
    )
    clicks = (
        PredictedClick(1005.0, 102.0, 82.0),
        PredictedClick(1020.0, 101.0, 81.0),
        PredictedClick(1205.0, 161.0, 79.0),
    )
    oracle = OutcomeOracle(circle_radius=20.0)
    direct = score_click_sequence(
        targets,
        clicks,
        circle_radius=20.0,
        spec=SequenceScoreSpec(
            min_click_interval_ms=50.0,
            object_score_spec=oracle.spec,
        ),
    )
    assert oracle.evaluate_sequence(targets, clicks) == direct


@pytest.mark.parametrize(
    "kwargs",
    (
        {"category": OutcomeCategory.INVALID, "valid": True},
        {
            "category": OutcomeCategory.HIGH,
            "valid": True,
            "expires": True,
            "passed": True,
        },
        {"category": OutcomeCategory.MISS, "valid": True, "passed": True},
        {
            "category": OutcomeCategory.HIGH,
            "valid": True,
            "passed": True,
            "target_object_id": None,
        },
    ),
)
def test_oracle_outcome_rejects_category_validity_contradictions(
    kwargs: dict[str, object],
) -> None:
    """OracleOutcome 不允许类别、valid、expires 和 passed 相互矛盾。"""

    values: dict[str, object] = {
        "track_id": "track-1",
        "horizon_ms": 0.0,
        "category": OutcomeCategory.INVALID,
        "score": 0.0,
        "valid": False,
        "expires": False,
        "passed": False,
        "target_object_id": "object-1",
        "spatial_error": None,
        "time_error_ms": None,
    }
    values.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        OracleOutcome(**values)  # type: ignore[arg-type]


def test_oracle_outcome_rejects_invalid_category_type_and_negative_error() -> None:
    """类别枚举与误差范围必须是硬契约。"""

    with pytest.raises(TypeError):
        OracleOutcome(
            "track-1",
            0.0,
            4,
            1.0,
            True,
            False,
            True,
            "object-1",
            0.0,
            0.0,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError):
        OracleOutcome(
            "track-1",
            0.0,
            OutcomeCategory.HIGH,
            1.0,
            True,
            False,
            True,
            "object-1",
            -1.0,
            0.0,
        )


def test_evaluation_and_oracle_ast_use_one_shared_scoring_implementation() -> None:
    """禁止 legacy/Any 渗透，Oracle 必须调用共享 scorer 而非复制公式。"""

    source_paths = tuple(
        sorted((_TRANING_ROOT / "core/evaluation").rglob("*.py"))
    ) + tuple(
        sorted((_TRANING_ROOT / "core/outcome/oracle").rglob("*.py"))
    )
    forbidden_definitions = {
        "combine_coefficients",
        "score_click_sequence",
        "score_point",
        "score_slider",
        "score_slider_path",
        "spatial_coefficient",
        "temporal_coefficient",
    }
    oracle_tree = ast.parse(
        (_TRANING_ROOT / "core/outcome/oracle/oracle.py").read_text(encoding="utf-8")
    )
    for path in source_paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(not alias.name.startswith("osu_v2") for alias in node.names)
            if isinstance(node, ast.ImportFrom):
                assert node.module is None or not node.module.startswith("osu_v2")
            if isinstance(node, ast.Name):
                assert node.id != "Any"
        if "outcome/oracle" in path.as_posix():
            definitions = {
                node.name
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            assert definitions.isdisjoint(forbidden_definitions)

    imported_scorers = {
        alias.name
        for node in ast.walk(oracle_tree)
        if isinstance(node, ast.ImportFrom) and node.module == "traning.core.evaluation"
        for alias in node.names
    }
    called_functions = {
        node.func.id
        for node in ast.walk(oracle_tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert {"score_point", "score_slider", "score_click_sequence"} <= imported_scorers
    assert {"score_point", "score_slider", "score_click_sequence"} <= called_functions
