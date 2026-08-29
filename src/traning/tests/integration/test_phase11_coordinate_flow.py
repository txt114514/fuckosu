"""验证标定坐标在训练、评分与 gallery 之间的单一全链路。"""

from __future__ import annotations

from math import hypot
from pathlib import Path

import pytest
import torch
from PIL import Image

from package import AffineOsuVideoTransform, OsuVideoTransform
from traning.config import PerceptionConfig
from traning.contracts import (
    BeliefState,
    DataSplit,
    GroundTruthObject,
    ObjectType,
    ObjectTypeDistribution,
    Point2D,
    RuntimeFrame,
    TrainingSample,
)
from traning.data.coordinates import (
    FrameCoordinateTransform,
    FramePixelPoint,
    OsuPoint,
)
from traning.evaluation import (
    FramePredictedClick,
    PrimaryError,
    TargetObject,
    build_sequence_evaluation_events,
    score_frame_click_sequence,
)
from traning.perception import (
    DensePerceptionOutput,
    build_coordinate_training_targets,
    decode_candidates,
    rasterize_perception_targets,
)
from traning.outcome.dataset import (
    CounterfactualFrame,
    CounterfactualOutcomeDatasetBuilder,
)
from traning.outcome.oracle import (
    OracleState,
    OracleTarget,
    OutcomeCategory,
    OutcomeOracle,
)
from traning.visualization import (
    build_gallery_frame_overlay,
    project_gallery_target_overlays,
    render_gallery_png,
)


# 该矩阵由 legacy 大量 pass 样本联合拟合；生产代码不复制其系数。
PASS_SAMPLE_MATRIX = (
    (2.115860914627143, 0.0011971920855575358, 242.59057485632047),
    (0.0003418231662923798, 2.1166805757239477, 16.12108357719331),
)
SOURCE_WIDTH = 1484
SOURCE_HEIGHT = 846

# 独立控制点未用于生产方程，只用于防止标定回归。
CONTROL_POINTS = (
    (OsuPoint(508.0, 237.0), (1317.5, 517.5)),
    (OsuPoint(80.0, 101.0), (412.0, 230.0)),
    (OsuPoint(395.0, 215.0), (1078.5, 471.5)),
    (OsuPoint(213.0, 179.0), (693.5, 395.0)),
    (OsuPoint(256.0, 183.0), (785.5, 404.5)),
)


@pytest.fixture
def shared_transform() -> AffineOsuVideoTransform:
    """使用 ``package`` 公开类构造唯一共享的生产变换对象。"""

    return AffineOsuVideoTransform(matrix=PASS_SAMPLE_MATRIX)


@pytest.fixture
def adapter(shared_transform: AffineOsuVideoTransform) -> FrameCoordinateTransform:
    """将共享变换显式绑定到标定原帧及其身份。"""

    return FrameCoordinateTransform(
        source_frame_width=SOURCE_WIDTH,
        source_frame_height=SOURCE_HEIGHT,
        transform_identity="legacy-control-validated-v1",
        transform=shared_transform,
    )


def test_three_consumers_share_one_transform_and_inverse(
    adapter: FrameCoordinateTransform,
    shared_transform: AffineOsuVideoTransform,
) -> None:
    """真实训练 target、sequence scoring 和 gallery API 必须共用变换。"""

    target = OsuPoint(79.89, 101.22)
    sample = TrainingSample(
        sample_id="frame-36",
        split=DataSplit.TRAIN,
        frame_index=36,
        timestamp_ms=576.0,
        width=SOURCE_WIDTH,
        height=SOURCE_HEIGHT,
        image_bytes=b"frame-bytes",
        transform_fingerprint=adapter.transform_fingerprint,
        candidates=(),
        ground_truth_objects=(
            GroundTruthObject(
                object_id="target-36",
                object_type=ObjectType.RING,
                position=Point2D(target.x, target.y),
                start_time_ms=576.0,
                end_time_ms=576.0,
                score=1.0,
                radius_osu=20.0,
            ),
        ),
        selected_candidate_id=None,
    )
    training_target = build_coordinate_training_targets(sample, adapter)[0].position
    gallery_target = project_gallery_target_overlays(
        (
            TargetObject(
                "target-36",
                "circle",
                576.0,
                576.0,
                x=target.x,
                y=target.y,
            ),
        ),
        adapter,
    )[0].head
    scored_sequence = score_frame_click_sequence(
        (
            TargetObject(
                "target-36",
                "circle",
                576.0,
                576.0,
                x=target.x,
                y=target.y,
            ),
        ),
        (FramePredictedClick(576.0, training_target),),
        coordinate_transform=adapter,
        circle_radius=20.0,
    )
    scored_target = adapter.prediction_to_canonical_scoring(training_target)

    assert adapter.transform is shared_transform
    assert adapter.transform_fingerprint.startswith("transform-")
    assert training_target == gallery_target
    assert training_target.x == pytest.approx(411.75, abs=0.01)
    assert training_target.y == pytest.approx(230.40, abs=0.01)
    assert scored_target.x == pytest.approx(target.x, abs=1e-9)
    assert scored_target.y == pytest.approx(target.y, abs=1e-9)
    assert scored_sequence.hit_count == 1
    assert scored_sequence.unresolved_target_ids == ()


def test_slider_direction_preserves_legal_out_of_playfield_control_point(
    adapter: FrameCoordinateTransform,
) -> None:
    """slider 控制点越界时必须变换向量，不能裁剪或拒绝真实标注。"""

    sample = TrainingSample(
        sample_id="slider-outside-control",
        split=DataSplit.TRAIN,
        frame_index=0,
        timestamp_ms=0.0,
        width=SOURCE_WIDTH,
        height=SOURCE_HEIGHT,
        image_bytes=b"frame-bytes",
        transform_fingerprint=adapter.transform_fingerprint,
        candidates=(),
        ground_truth_objects=(
            GroundTruthObject(
                object_id="slider-1",
                object_type=ObjectType.SLIDER,
                position=Point2D(500.0, 200.0),
                start_time_ms=0.0,
                end_time_ms=100.0,
                score=1.0,
                path=(Point2D(500.0, 200.0), Point2D(540.0, 210.0)),
            ),
        ),
        selected_candidate_id=None,
    )

    target = build_coordinate_training_targets(sample, adapter)[0]
    assert target.slider_direction is not None
    raw_delta = (
        PASS_SAMPLE_MATRIX[0][0] * 40.0 + PASS_SAMPLE_MATRIX[0][1] * 10.0,
        PASS_SAMPLE_MATRIX[1][0] * 40.0 + PASS_SAMPLE_MATRIX[1][1] * 10.0,
    )
    norm = hypot(*raw_delta)
    assert target.slider_direction == pytest.approx(
        (raw_delta[0] / norm, raw_delta[1] / norm),
        abs=1e-12,
    )


def test_slider_outside_control_point_is_shared_by_training_scoring_and_gallery(
    adapter: FrameCoordinateTransform,
) -> None:
    """合法越界控制点必须在训练、评分与 gallery 中使用同一未裁剪投影。"""

    start = Point2D(500.0, 200.0)
    outside = Point2D(620.0, 210.0)
    sample = TrainingSample(
        sample_id="slider-shared-outside-control",
        split=DataSplit.TRAIN,
        frame_index=12,
        timestamp_ms=200.0,
        width=SOURCE_WIDTH,
        height=SOURCE_HEIGHT,
        image_bytes=b"frame-bytes",
        transform_fingerprint=adapter.transform_fingerprint,
        candidates=(),
        ground_truth_objects=(
            GroundTruthObject(
                object_id="slider-shared",
                object_type=ObjectType.SLIDER,
                position=start,
                start_time_ms=200.0,
                end_time_ms=300.0,
                score=1.0,
                path=(start, outside),
            ),
        ),
        selected_candidate_id=None,
    )
    training_target = build_coordinate_training_targets(sample, adapter)[0]
    target = TargetObject(
        "slider-shared",
        "slider",
        200.0,
        300.0,
        path=((start.x, start.y), (outside.x, outside.y)),
    )
    frame_start = adapter.ground_truth_to_training_target(
        OsuPoint(start.x, start.y),
        source_frame_width=SOURCE_WIDTH,
        source_frame_height=SOURCE_HEIGHT,
    )
    projected_outside = adapter.ground_truth_geometry_to_frame(
        outside,
        source_frame_width=SOURCE_WIDTH,
        source_frame_height=SOURCE_HEIGHT,
    )
    score = score_frame_click_sequence(
        (target,),
        (
            FramePredictedClick(
                200.0,
                frame_start,
                path=(
                    frame_start,
                    FramePixelPoint(
                        min(projected_outside.x, SOURCE_WIDTH - 1.0),
                        min(projected_outside.y, SOURCE_HEIGHT - 1.0),
                        SOURCE_WIDTH,
                        SOURCE_HEIGHT,
                        adapter.transform_fingerprint,
                    ),
                ),
            ),
        ),
        coordinate_transform=adapter,
        circle_radius=20.0,
    )
    events = build_sequence_evaluation_events(
        sample.sample_id, sample.frame_index, score
    )
    overlay = build_gallery_frame_overlay(
        (target,), score, events, adapter, frame_index=sample.frame_index
    )

    assert training_target.slider_direction is not None
    assert overlay.targets[0].path[1] == projected_outside
    assert overlay.targets[0].path[1].x > SOURCE_WIDTH
    # PIL 会在最终画布边界裁切线段；领域投影本身不得把控制点夹到边缘。
    assert overlay.targets[0].path[1].x != SOURCE_WIDTH - 1.0


def test_pass_sample_is_rasterized_with_decoder_inverse_equation(
    adapter: FrameCoordinateTransform,
) -> None:
    """frame 36 必须通过正式 target 栅格化与 decoder 精确回到同一原帧点。"""

    target = OsuPoint(79.89, 101.22)
    sample = TrainingSample(
        sample_id="frame-36-grid",
        split=DataSplit.TRAIN,
        frame_index=36,
        timestamp_ms=576.0,
        width=SOURCE_WIDTH,
        height=SOURCE_HEIGHT,
        image_bytes=b"frame-bytes",
        transform_fingerprint=adapter.transform_fingerprint,
        candidates=(),
        ground_truth_objects=(
            GroundTruthObject(
                object_id="target-36",
                object_type=ObjectType.RING,
                position=Point2D(target.x, target.y),
                start_time_ms=576.0,
                end_time_ms=576.0,
                score=1.0,
                radius_osu=20.0,
            ),
        ),
        selected_candidate_id=None,
    )
    scalar = torch.zeros((1, 1, 36, 64))
    template = DensePerceptionOutput(
        center_logits=scalar.clone(),
        visibility_logits=scalar.clone(),
        type_logits=torch.zeros((1, 4, 36, 64)),
        xy_offsets=torch.zeros((1, 2, 36, 64)),
        ring_logits=scalar.clone(),
        ring_radius=scalar.clone(),
        slider_logits=scalar.clone(),
        slider_direction=torch.zeros((1, 2, 36, 64)),
        spinner_logits=scalar.clone(),
        identity_embedding=torch.ones((1, 2, 36, 64)),
    )
    targets = rasterize_perception_targets((sample,), template, adapter)
    positive = torch.nonzero(targets.center_heatmap[0, 0], as_tuple=False)
    assert positive.tolist() == [[9, 17]]
    row, column = positive[0].tolist()
    expected = adapter.ground_truth_to_training_target(
        target,
        source_frame_width=SOURCE_WIDTH,
        source_frame_height=SOURCE_HEIGHT,
    )
    assert targets.xy_offsets[0, :, row, column].tolist() == pytest.approx(
        (
            expected.x / (SOURCE_WIDTH / 64) - (column + 0.5),
            expected.y / (SOURCE_HEIGHT / 36) - (row + 0.5),
        ),
        abs=1e-5,
    )

    # 用栅格化监督构造理想预测，并通过正式 decoder 走回原帧像素。
    ideal = DensePerceptionOutput(
        center_logits=targets.center_heatmap * 40.0 - 20.0,
        visibility_logits=targets.visibility * 40.0 - 20.0,
        type_logits=torch.where(
            targets.type_indices[:, None].eq(torch.arange(4).view(1, 4, 1, 1)),
            torch.tensor(20.0),
            torch.tensor(-20.0),
        ),
        xy_offsets=targets.xy_offsets,
        ring_logits=targets.ring * 40.0 - 20.0,
        ring_radius=targets.ring_radius,
        slider_logits=targets.slider * 40.0 - 20.0,
        slider_direction=torch.where(
            targets.slider_direction.ne(0.0),
            targets.slider_direction,
            torch.tensor(1.0),
        ),
        spinner_logits=targets.spinner * 40.0 - 20.0,
        identity_embedding=torch.nn.functional.normalize(
            torch.ones((1, 2, 36, 64)), dim=1
        ),
    )
    candidate = decode_candidates(
        ideal,
        frame_id="frame-36-grid",
        frame_index=36,
        timestamp_ms=576.0,
        frame_width=SOURCE_WIDTH,
        frame_height=SOURCE_HEIGHT,
        config=PerceptionConfig(score_threshold=0.1, nms_radius_px=0.0),
    )[0]
    assert candidate.x == pytest.approx(expected.x, abs=1e-4)
    assert candidate.y == pytest.approx(expected.y, abs=1e-4)
    assert candidate.ring is not None
    assert candidate.ring.radius_px == pytest.approx(
        adapter.ground_truth_radius_to_training_target(
            20.0,
            source_frame_width=SOURCE_WIDTH,
            source_frame_height=SOURCE_HEIGHT,
        ),
        abs=1e-4,
    )


def test_pass_sample_control_residuals_stay_within_four_pixels(
    adapter: FrameCoordinateTransform,
) -> None:
    """大量 pass 样本拟合后的五个独立控制点均不得偏移。"""

    residuals: list[float] = []
    for osu_point, expected_frame_point in CONTROL_POINTS:
        actual = adapter.ground_truth_to_training_target(
            osu_point,
            source_frame_width=SOURCE_WIDTH,
            source_frame_height=SOURCE_HEIGHT,
        )
        residuals.append(
            hypot(
                actual.x - expected_frame_point[0],
                actual.y - expected_frame_point[1],
            )
        )
    assert max(residuals) <= 4.0


def test_frame_score_event_and_real_png_keep_one_transform_identity(
    adapter: FrameCoordinateTransform,
    tmp_path: Path,
) -> None:
    """评分、归因、gallery 点位和真实 PNG 必须保留同一标定指纹。"""

    target = TargetObject(
        "target-36",
        "circle",
        576.0,
        576.0,
        x=79.89,
        y=101.22,
    )
    position = adapter.ground_truth_to_training_target(
        OsuPoint(79.89, 101.22),
        source_frame_width=SOURCE_WIDTH,
        source_frame_height=SOURCE_HEIGHT,
    )
    score = score_frame_click_sequence(
        (target,),
        (FramePredictedClick(576.0, position),),
        coordinate_transform=adapter,
        circle_radius=20.0,
    )
    events = build_sequence_evaluation_events("frame-36", 36, score)
    overlay = build_gallery_frame_overlay(
        (target,), score, events, adapter, frame_index=36
    )

    assert events[0].coordinate_transform_fingerprint == adapter.transform_fingerprint
    assert events[0].click_x == pytest.approx(position.x)
    assert events[0].click_y == pytest.approx(position.y)
    assert overlay.events[0] is events[0]
    assert overlay.predictions[0].event is events[0]
    assert overlay.predictions[0].position == overlay.targets[0].head

    frame = RuntimeFrame(
        frame_id="frame-36",
        frame_index=36,
        timestamp_ms=576.0,
        width=SOURCE_WIDTH,
        height=SOURCE_HEIGHT,
        image_bytes=bytes(SOURCE_WIDTH * SOURCE_HEIGHT * 3),
    )
    output_path = tmp_path / "frame-36.png"
    render_gallery_png(frame, overlay, output_path)
    with Image.open(output_path) as image:
        assert image.format == "PNG"
        assert image.size == (SOURCE_WIDTH, SOURCE_HEIGHT)
        assert image.getpixel((round(position.x), round(position.y))) != (0, 0, 0)


def test_frame_105_unresolved_stays_decision_with_coordinate_provenance(
    adapter: FrameCoordinateTransform,
) -> None:
    """图上存在准确 GT 但无实际 click 时，frame 105 只能归入 Decision。"""

    target = TargetObject(
        "target-105",
        "circle",
        1680.0,
        1680.0,
        x=80.0,
        y=101.0,
    )
    score = score_frame_click_sequence(
        (target,),
        (),
        coordinate_transform=adapter,
        circle_radius=20.0,
    )
    events = build_sequence_evaluation_events("frame-105", 105, score)
    overlay = build_gallery_frame_overlay(
        (target,), score, events, adapter, frame_index=105
    )

    assert len(events) == 1
    assert events[0].primary_error is PrimaryError.DECISION
    assert events[0].coordinate_transform_fingerprint == adapter.transform_fingerprint
    assert overlay.predictions == ()
    assert overlay.events[0] is events[0]


def test_multiframe_events_keep_click_and_unresolved_source_frames(
    adapter: FrameCoordinateTransform,
) -> None:
    """同一长序列的事件必须保留各自帧号，不能全部归到最后一帧。"""

    first_position = adapter.ground_truth_to_training_target(
        OsuPoint(80.0, 101.0),
        source_frame_width=SOURCE_WIDTH,
        source_frame_height=SOURCE_HEIGHT,
    )
    targets = (
        TargetObject(
            "target-105",
            "circle",
            1680.0,
            1680.0,
            x=80.0,
            y=101.0,
            frame_index=105,
        ),
        TargetObject(
            "target-107",
            "circle",
            1712.0,
            1712.0,
            x=120.0,
            y=110.0,
            frame_index=107,
        ),
    )
    score = score_frame_click_sequence(
        targets,
        (FramePredictedClick(1680.0, first_position, frame_index=105),),
        coordinate_transform=adapter,
        circle_radius=20.0,
    )
    events = build_sequence_evaluation_events(
        "long_sequence_000008",
        999,
        score,
    )

    assert tuple(event.frame_index for event in events) == (105, 107)
    assert events[0].passed is True
    assert events[1].primary_error is PrimaryError.DECISION
    assert events[1].target_id == "target-107"


def test_frame_margin_prediction_is_scored_as_spatial_miss(
    adapter: FrameCoordinateTransform,
) -> None:
    """映射到 playfield 外的合法原帧点击应计 miss，而不是中止整段评估。"""

    result = score_frame_click_sequence(
        (
            TargetObject(
                "target-margin",
                "circle",
                100.0,
                100.0,
                x=80.0,
                y=101.0,
            ),
        ),
        (
            FramePredictedClick(
                100.0,
                FramePixelPoint(
                    0.0,
                    0.0,
                    SOURCE_WIDTH,
                    SOURCE_HEIGHT,
                    adapter.transform_fingerprint,
                ),
            ),
        ),
        coordinate_transform=adapter,
        circle_radius=20.0,
    )

    assert result.hit_count == 0
    assert result.miss_count == 1
    assert result.clicks[0].primary_error == "spatial"


def test_counterfactual_labels_inverse_frame_belief_before_oracle(
    adapter: FrameCoordinateTransform,
) -> None:
    """Outcome dataset 不得把原帧 belief 像素直接与 osu! oracle target 相减。"""

    target = OsuPoint(79.89, 101.22)
    frame_position = adapter.ground_truth_to_training_target(
        target,
        source_frame_width=SOURCE_WIDTH,
        source_frame_height=SOURCE_HEIGHT,
    )
    belief = BeliefState(
        track_id="track-36",
        timestamp_ms=576.0,
        belief_embedding=(0.1, 0.2),
        position_mean=Point2D(frame_position.x, frame_position.y),
        position_uncertainty=Point2D(1.0, 1.0),
        visibility_probability=1.0,
        object_type_distribution=ObjectTypeDistribution(1.0, 0.0, 0.0, 0.0),
        age=2,
        time_since_seen_ms=0.0,
        uncertainty=0.1,
    )
    oracle_state = OracleState(
        state_id="state-36",
        timestamp_ms=576.0,
        targets=(
            OracleTarget(
                track_id="track-36",
                object_id="target-36",
                object_type=ObjectType.RING,
                position=Point2D(target.x, target.y),
                start_time_ms=576.0,
                end_time_ms=676.0,
            ),
        ),
    )
    dataset = CounterfactualOutcomeDatasetBuilder(
        OutcomeOracle(circle_radius=20.0),
        (0.0,),
        adapter,
    ).build(
        (
            CounterfactualFrame(
                sample_id="frame-36",
                split=DataSplit.TRAIN,
                source_frame_width=SOURCE_WIDTH,
                source_frame_height=SOURCE_HEIGHT,
                transform_fingerprint=adapter.transform_fingerprint,
                beliefs=(belief,),
                oracle_state=oracle_state,
            ),
        )
    )

    assert dataset.transform_fingerprint == adapter.transform_fingerprint
    assert dataset.records[0].target_category is OutcomeCategory.HIGH
    assert dataset.records[0].target_score == pytest.approx(1.0)


def test_fingerprint_binds_matrix_identity_and_source_frame(
    adapter: FrameCoordinateTransform,
    shared_transform: AffineOsuVideoTransform,
) -> None:
    """指纹可比较，且尺寸或标定身份变化时不得复用。"""

    same_binding = FrameCoordinateTransform(
        SOURCE_WIDTH,
        SOURCE_HEIGHT,
        "legacy-control-validated-v1",
        shared_transform,
    )
    different_identity = FrameCoordinateTransform(
        SOURCE_WIDTH,
        SOURCE_HEIGHT,
        "pass-sample-ransac-v2",
        shared_transform,
    )
    different_size = FrameCoordinateTransform(
        SOURCE_WIDTH + 1,
        SOURCE_HEIGHT,
        "legacy-control-validated-v1",
        shared_transform,
    )

    assert adapter == same_binding
    assert adapter.transform_fingerprint == same_binding.transform_fingerprint
    assert adapter.transform_fingerprint != different_identity.transform_fingerprint
    assert adapter.transform_fingerprint != different_size.transform_fingerprint


def test_mismatched_frame_size_is_rejected_by_every_consumer(
    adapter: FrameCoordinateTransform,
) -> None:
    """三个消费者都必须对错误原帧尺寸硬失败。"""

    target = OsuPoint(80.0, 101.0)
    with pytest.raises(ValueError, match="标定尺寸不一致"):
        adapter.ground_truth_to_training_target(
            target,
            source_frame_width=SOURCE_WIDTH - 1,
            source_frame_height=SOURCE_HEIGHT,
        )
    with pytest.raises(ValueError, match="标定尺寸不一致"):
        adapter.target_to_gallery_overlay(
            target,
            source_frame_width=SOURCE_WIDTH,
            source_frame_height=SOURCE_HEIGHT - 1,
        )
    wrong_frame_prediction = FramePixelPoint(
        412.0,
        230.0,
        SOURCE_WIDTH - 1,
        SOURCE_HEIGHT,
        adapter.transform_fingerprint,
    )
    with pytest.raises(ValueError, match="标定尺寸不一致"):
        adapter.prediction_to_canonical_scoring(wrong_frame_prediction)

    wrong_fingerprint_prediction = FramePixelPoint(
        412.0,
        230.0,
        SOURCE_WIDTH,
        SOURCE_HEIGHT,
        "transform-0123456789abcdef",
    )
    with pytest.raises(ValueError, match="指纹不一致"):
        adapter.prediction_to_canonical_scoring(wrong_fingerprint_prediction)


def test_nonfinite_out_of_bounds_and_centered_fallback_are_rejected(
    shared_transform: AffineOsuVideoTransform,
) -> None:
    """非法点和未标定居中变换不得被 clamp 或静默接受。"""

    with pytest.raises(ValueError, match="有限"):
        OsuPoint(float("nan"), 1.0)
    with pytest.raises(ValueError, match="playfield"):
        OsuPoint(513.0, 1.0)
    with pytest.raises(ValueError, match="像素边界"):
        FramePixelPoint(
            SOURCE_WIDTH,
            1.0,
            SOURCE_WIDTH,
            SOURCE_HEIGHT,
            "transform-0123456789abcdef",
        )

    centered = OsuVideoTransform.fit_centered(SOURCE_WIDTH, SOURCE_HEIGHT)
    with pytest.raises(TypeError, match="AffineOsuVideoTransform"):
        FrameCoordinateTransform(
            SOURCE_WIDTH,
            SOURCE_HEIGHT,
            "implicit-centered-fallback",
            centered,
        )

    # 即使仿射类型正确，标定四角超出原帧也必须在构造时失败。
    outside_transform = AffineOsuVideoTransform(
        matrix=((2.0, 0.0, 1000.0), (0.0, 2.0, 100.0))
    )
    with pytest.raises(ValueError, match="像素边界"):
        FrameCoordinateTransform(
            SOURCE_WIDTH,
            SOURCE_HEIGHT,
            "outside-frame",
            outside_transform,
        )

    # 保留 fixture 引用，确认拒绝分支未替换共享生产对象。
    assert isinstance(shared_transform, AffineOsuVideoTransform)
