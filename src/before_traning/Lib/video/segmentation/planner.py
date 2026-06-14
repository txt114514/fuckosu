from __future__ import annotations

from dataclasses import dataclass
from math import acos, ceil, hypot, isfinite, pi, sqrt
from typing import Literal

from slider import Position
from slider.curve import Curve

from before_traning.Lib.beatmap.hit_objects import Circle, HitObject, Slider, Spinner
from before_traning.Lib.beatmap.standard import (
    ParsedStandardBeatmap,
    approach_preempt_ms,
    parse_standard_beatmap,
    parse_standard_hit_objects,
)


SegmentCategory = Literal[
    "single_point",
    "multi_point",
    "slider",
    "point_slider",
    "spinner",
]
SegmentDimension = Literal["atomic", "long_sequence"]
SEGMENT_CATEGORIES: tuple[SegmentCategory, ...] = (
    "single_point",
    "multi_point",
    "slider",
    "point_slider",
    "spinner",
)
OSU_OBJECT_RADIUS = 64.0
OSU_LEGACY_RADIUS_FUDGE = 1.00041
SLIDER_PATH_SAMPLE_STEP_PIXELS = 4.0


@dataclass(frozen=True)
class SegmentPlan:
    dimension: SegmentDimension
    category: SegmentCategory
    source_plan_count: int
    hit_start_ms: int
    hit_end_ms: int
    clip_start_seconds: float
    clip_end_seconds: float
    object_types: tuple[str, ...]
    object_start_times_ms: tuple[int, ...]
    object_end_times_ms: tuple[int, ...]
    object_indexes: tuple[int, ...]
    hit_objects: tuple[HitObject, ...]
    circle_size: float
    circle_radius: float

    @property
    def duration_seconds(self) -> float:
        return self.clip_end_seconds - self.clip_start_seconds

    @property
    def pre_context_seconds(self) -> float:
        return self.hit_start_ms / 1000.0 - self.clip_start_seconds

    @property
    def post_context_seconds(self) -> float:
        return self.clip_end_seconds - self.hit_end_ms / 1000.0

    @property
    def clip_start_ms(self) -> int:
        return int(round(self.clip_start_seconds * 1000.0))

    @property
    def clip_end_ms(self) -> int:
        return int(round(self.clip_end_seconds * 1000.0))


def circle_radius_from_size(circle_size: float) -> float:
    if not isfinite(circle_size):
        raise ValueError("CircleSize 必须是有限数值")
    difficulty_range = (circle_size - 5.0) / 5.0
    scale = (
        (1.0 - 0.7 * difficulty_range)
        / 2.0
        * OSU_LEGACY_RADIUS_FUDGE
    )
    radius = OSU_OBJECT_RADIUS * scale
    if radius <= 0:
        raise ValueError(f"CircleSize={circle_size} 产生了非法圆半径")
    return radius


def circle_overlap_ratio(distance: float, radius: float) -> float:
    if distance <= 0:
        return 1.0
    if distance >= 2.0 * radius:
        return 0.0

    normalized_distance = distance / (2.0 * radius)
    overlap_area = 2.0 * radius * radius * (
        acos(normalized_distance)
        - normalized_distance * sqrt(1.0 - normalized_distance**2)
    )
    return overlap_area / (pi * radius * radius)


def _slider_polyline(slider: Slider) -> tuple[tuple[float, float], ...]:
    if len(slider.path) < 2 or not slider.pixel_length:
        return tuple(slider.path)

    curve = Curve.from_kind_and_points(
        slider.curve_type,
        [Position(x, y) for x, y in slider.path],
        slider.pixel_length,
    )
    sample_count = max(
        2,
        ceil(slider.pixel_length / SLIDER_PATH_SAMPLE_STEP_PIXELS) + 1,
    )
    return tuple(
        (float(position.x), float(position.y))
        for position in (
            curve(index / (sample_count - 1))
            for index in range(sample_count)
        )
    )


def _object_polyline(
    hit_object: HitObject,
) -> tuple[tuple[float, float], ...]:
    if isinstance(hit_object, Circle):
        return ((hit_object.x, hit_object.y),)
    if isinstance(hit_object, Slider):
        return _slider_polyline(hit_object)
    raise TypeError("Spinner 不参与坐标重叠判断")


def _point_to_segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    segment_x = end[0] - start[0]
    segment_y = end[1] - start[1]
    squared_length = segment_x * segment_x + segment_y * segment_y
    if squared_length == 0:
        return hypot(point[0] - start[0], point[1] - start[1])

    projection = (
        (point[0] - start[0]) * segment_x
        + (point[1] - start[1]) * segment_y
    ) / squared_length
    projection = min(1.0, max(0.0, projection))
    closest_x = start[0] + projection * segment_x
    closest_y = start[1] + projection * segment_y
    return hypot(point[0] - closest_x, point[1] - closest_y)


def _orientation(
    first: tuple[float, float],
    second: tuple[float, float],
    third: tuple[float, float],
) -> float:
    return (
        (second[0] - first[0]) * (third[1] - first[1])
        - (second[1] - first[1]) * (third[0] - first[0])
    )


def _segments_intersect(
    first_start: tuple[float, float],
    first_end: tuple[float, float],
    second_start: tuple[float, float],
    second_end: tuple[float, float],
) -> bool:
    first_side_start = _orientation(first_start, first_end, second_start)
    first_side_end = _orientation(first_start, first_end, second_end)
    second_side_start = _orientation(second_start, second_end, first_start)
    second_side_end = _orientation(second_start, second_end, first_end)
    epsilon = 1e-9

    if (
        first_side_start * first_side_end < -epsilon
        and second_side_start * second_side_end < -epsilon
    ):
        return True
    return min(
        _point_to_segment_distance(first_start, second_start, second_end),
        _point_to_segment_distance(first_end, second_start, second_end),
        _point_to_segment_distance(second_start, first_start, first_end),
        _point_to_segment_distance(second_end, first_start, first_end),
    ) <= epsilon


def _polyline_distance(
    first: tuple[tuple[float, float], ...],
    second: tuple[tuple[float, float], ...],
) -> float:
    if not first or not second:
        return float("inf")
    if len(first) == 1 and len(second) == 1:
        return hypot(first[0][0] - second[0][0], first[0][1] - second[0][1])
    if len(first) == 1:
        return min(
            _point_to_segment_distance(first[0], start, end)
            for start, end in zip(second, second[1:])
        )
    if len(second) == 1:
        return min(
            _point_to_segment_distance(second[0], start, end)
            for start, end in zip(first, first[1:])
        )

    minimum_distance = float("inf")
    for first_start, first_end in zip(first, first[1:]):
        for second_start, second_end in zip(second, second[1:]):
            if _segments_intersect(
                first_start,
                first_end,
                second_start,
                second_end,
            ):
                return 0.0
            minimum_distance = min(
                minimum_distance,
                _point_to_segment_distance(
                    first_start,
                    second_start,
                    second_end,
                ),
                _point_to_segment_distance(
                    first_end,
                    second_start,
                    second_end,
                ),
                _point_to_segment_distance(
                    second_start,
                    first_start,
                    first_end,
                ),
                _point_to_segment_distance(
                    second_end,
                    first_start,
                    first_end,
                ),
            )
    return minimum_distance


def hit_objects_overlap_ratio(
    first: HitObject,
    second: HitObject,
    *,
    circle_radius: float,
) -> float:
    distance = _polyline_distance(
        _object_polyline(first),
        _object_polyline(second),
    )
    return circle_overlap_ratio(distance, circle_radius)


def group_hit_objects(
    hit_objects: list[HitObject],
    overlap_merge_window_ms: int,
    *,
    circle_size: float = 5.0,
    min_circle_overlap_ratio: float = 0.5,
    priority_merge_window_ms: int = 0,
    use_priority_merge: bool = True,
) -> list[list[HitObject]]:
    if overlap_merge_window_ms < 0:
        raise ValueError("overlap_merge_window_ms 必须是非负整数")
    if priority_merge_window_ms < 0:
        raise ValueError("priority_merge_window_ms 必须是非负整数")
    if not 0.0 <= min_circle_overlap_ratio <= 1.0:
        raise ValueError("min_circle_overlap_ratio 必须在 0 到 1 之间")

    groups: list[list[HitObject]] = []
    current: list[HitObject] = []
    current_end_ms = 0
    circle_radius = circle_radius_from_size(circle_size)
    polylines = {
        id(hit_object): _object_polyline(hit_object)
        for hit_object in hit_objects
        if not isinstance(hit_object, Spinner)
    }

    for hit_object in hit_objects:
        if isinstance(hit_object, Spinner):
            if current:
                groups.append(current)
                current = []
            groups.append([hit_object])
            current_end_ms = hit_object.t_end
            continue

        time_matches = (
            current
            and hit_object.t_start - current_end_ms
            <= overlap_merge_window_ms
        )
        priority_merge_matches = (
            use_priority_merge
            and current
            and hit_object.t_start
            <= current_end_ms + priority_merge_window_ms
        )
        space_matches = time_matches and any(
            circle_overlap_ratio(
                _polyline_distance(
                    polylines[id(grouped_object)],
                    polylines[id(hit_object)],
                ),
                circle_radius,
            )
            >= min_circle_overlap_ratio
            for grouped_object in current
        )
        if priority_merge_matches or (time_matches and space_matches):
            current.append(hit_object)
            current_end_ms = max(current_end_ms, hit_object.t_end)
            continue

        if current:
            groups.append(current)
        current = [hit_object]
        current_end_ms = hit_object.t_end

    if current:
        groups.append(current)
    return groups


def classify_hit_group(hit_group: list[HitObject]) -> SegmentCategory:
    """Classify by contained object types; mixed groups may contain many sliders."""
    if not hit_group:
        raise ValueError("不能分类空 HitObject 组")
    if len(hit_group) == 1 and isinstance(hit_group[0], Spinner):
        return "spinner"

    has_circle = any(isinstance(item, Circle) for item in hit_group)
    has_slider = any(isinstance(item, Slider) for item in hit_group)
    has_spinner = any(isinstance(item, Spinner) for item in hit_group)
    if has_spinner:
        raise ValueError("Spinner 必须作为独立片段")
    if has_circle and has_slider:
        return "point_slider"
    if has_slider:
        return "slider"
    if has_circle and len(hit_group) == 1:
        return "single_point"
    if has_circle:
        return "multi_point"
    raise TypeError(f"无法分类 HitObject 组: {hit_group!r}")


def _build_plan(
    hit_group: list[HitObject],
    object_indexes: list[int],
    *,
    dimension: SegmentDimension,
    source_plan_count: int,
    circle_size: float,
    circle_radius: float,
    approach_context_seconds: float,
    post_context_seconds: float,
    video_duration_seconds: float,
) -> SegmentPlan:
    hit_start_ms = min(item.t_start for item in hit_group)
    hit_end_ms = max(item.t_end for item in hit_group)
    if hit_start_ms / 1000.0 >= video_duration_seconds:
        raise ValueError(
            f"谱面对象起点超出视频时长: hit={hit_start_ms}ms, "
            f"video={video_duration_seconds:.3f}s"
        )
    if hit_end_ms / 1000.0 > video_duration_seconds:
        raise ValueError(
            f"谱面对象终点超出视频时长: hit={hit_end_ms}ms, "
            f"video={video_duration_seconds:.3f}s"
        )

    clip_start_seconds = max(
        0.0,
        hit_start_ms / 1000.0 - approach_context_seconds,
    )
    clip_end_seconds = min(
        video_duration_seconds,
        hit_end_ms / 1000.0 + post_context_seconds,
    )
    if clip_end_seconds <= clip_start_seconds:
        raise ValueError(
            f"切分区间非法: {clip_start_seconds:.3f}s - "
            f"{clip_end_seconds:.3f}s"
        )
    return SegmentPlan(
        dimension=dimension,
        category=classify_hit_group(hit_group),
        source_plan_count=source_plan_count,
        hit_start_ms=hit_start_ms,
        hit_end_ms=hit_end_ms,
        clip_start_seconds=clip_start_seconds,
        clip_end_seconds=clip_end_seconds,
        object_types=tuple(item.type for item in hit_group),
        object_start_times_ms=tuple(item.t_start for item in hit_group),
        object_end_times_ms=tuple(item.t_end for item in hit_group),
        object_indexes=tuple(object_indexes),
        hit_objects=tuple(hit_group),
        circle_size=circle_size,
        circle_radius=circle_radius,
    )


def build_segment_plans(
    hit_objects: list[HitObject],
    *,
    approach_preempt_ratio: float,
    circle_size: float,
    min_circle_overlap_ratio: float,
    priority_merge_window_ms: int,
    use_priority_merge: bool,
    approach_preempt_seconds: float,
    post_context_seconds: float,
    video_duration_seconds: float,
) -> list[SegmentPlan]:
    if not isfinite(approach_preempt_seconds) or approach_preempt_seconds < 0:
        raise ValueError("approach_preempt_seconds 必须是非负有限数值")
    if not 0.0 <= approach_preempt_ratio <= 1.0:
        raise ValueError("approach_preempt_ratio 必须在 0 到 1 之间")
    if not isfinite(post_context_seconds) or post_context_seconds < 0:
        raise ValueError("post_context_seconds 必须是非负有限数值")

    approach_context_seconds = (
        approach_preempt_seconds * approach_preempt_ratio
    )
    overlap_merge_window_ms = round(approach_context_seconds * 1000.0)
    circle_radius = circle_radius_from_size(circle_size)
    hit_groups = group_hit_objects(
        hit_objects,
        overlap_merge_window_ms,
        circle_size=circle_size,
        min_circle_overlap_ratio=min_circle_overlap_ratio,
        priority_merge_window_ms=priority_merge_window_ms,
        use_priority_merge=use_priority_merge,
    )
    object_indexes = {id(item): index for index, item in enumerate(hit_objects)}
    plans: list[SegmentPlan] = []

    for hit_group in hit_groups:
        plans.append(
            _build_plan(
                hit_group,
                [
                    object_indexes[id(item)] for item in hit_group
                ],
                dimension="atomic",
                source_plan_count=1,
                circle_size=circle_size,
                circle_radius=circle_radius,
                approach_context_seconds=approach_context_seconds,
                post_context_seconds=post_context_seconds,
                video_duration_seconds=video_duration_seconds,
            )
        )

    assigned_indexes = [
        object_index
        for plan in plans
        for object_index in plan.object_indexes
    ]
    if sorted(assigned_indexes) != list(range(len(hit_objects))):
        raise RuntimeError("HitObject 分配不完整或存在重复归属")
    return plans


def build_long_sequence_plans(
    atomic_plans: list[SegmentPlan],
    *,
    approach_preempt_seconds: float,
    approach_preempt_ratio: float,
    post_context_seconds: float,
    video_duration_seconds: float,
    max_objects: int,
    max_duration_seconds: float,
) -> list[SegmentPlan]:
    if max_objects < 4:
        raise ValueError("long_sequence max_objects 不能小于 4")
    if not isfinite(max_duration_seconds) or max_duration_seconds <= 0:
        raise ValueError("long_sequence max_duration_seconds 必须是正有限数值")
    if not isfinite(approach_preempt_seconds) or approach_preempt_seconds < 0:
        raise ValueError("approach_preempt_seconds 必须是非负有限数值")
    if not 0.0 <= approach_preempt_ratio <= 1.0:
        raise ValueError("approach_preempt_ratio 必须在 0 到 1 之间")
    if not isfinite(post_context_seconds) or post_context_seconds < 0:
        raise ValueError("post_context_seconds 必须是非负有限数值")
    if any(plan.dimension != "atomic" for plan in atomic_plans):
        raise ValueError("长序列只能由 atomic SegmentPlan 构建")

    approach_context_seconds = (
        approach_preempt_seconds * approach_preempt_ratio
    )
    continuity_window_ms = round(approach_preempt_seconds * 1000.0)
    sequences: list[SegmentPlan] = []
    current: list[SegmentPlan] = []

    def combined_plan(plans: list[SegmentPlan]) -> SegmentPlan:
        hit_objects = [
            hit_object
            for plan in plans
            for hit_object in plan.hit_objects
        ]
        object_indexes = [
            object_index
            for plan in plans
            for object_index in plan.object_indexes
        ]
        return _build_plan(
            hit_objects,
            object_indexes,
            dimension="long_sequence",
            source_plan_count=len(plans),
            circle_size=plans[0].circle_size,
            circle_radius=plans[0].circle_radius,
            approach_context_seconds=approach_context_seconds,
            post_context_seconds=post_context_seconds,
            video_duration_seconds=video_duration_seconds,
        )

    def flush() -> None:
        nonlocal current
        if len(current) < 2:
            current = []
            return
        plan = combined_plan(current)
        circle_count = sum(
            isinstance(item, Circle) for item in plan.hit_objects
        )
        slider_count = sum(
            isinstance(item, Slider) for item in plan.hit_objects
        )
        if (
            circle_count >= 2
            and slider_count >= 2
            and len(plan.hit_objects) <= max_objects
            and plan.duration_seconds <= max_duration_seconds
        ):
            sequences.append(plan)
        current = []

    for plan in atomic_plans:
        if plan.category == "spinner":
            flush()
            continue
        if not current:
            current = [plan]
            continue

        candidate = [*current, plan]
        candidate_object_count = sum(
            len(item.hit_objects) for item in candidate
        )
        continuity_gap_ms = plan.hit_start_ms - current[-1].hit_end_ms
        candidate_plan = combined_plan(candidate)
        if (
            continuity_gap_ms > continuity_window_ms
            or candidate_object_count > max_objects
            or candidate_plan.duration_seconds > max_duration_seconds
        ):
            flush()
            current = [plan]
        else:
            current.append(plan)
    flush()

    assigned_indexes = [
        object_index
        for plan in sequences
        for object_index in plan.object_indexes
    ]
    if len(assigned_indexes) != len(set(assigned_indexes)):
        raise RuntimeError("long_sequence HitObject 存在重复归属")
    atomic_indexes = {
        object_index
        for plan in atomic_plans
        for object_index in plan.object_indexes
    }
    if not set(assigned_indexes) <= atomic_indexes:
        raise RuntimeError("long_sequence 包含 atomic 计划之外的 HitObject")
    return sequences


__all__ = [
    "SEGMENT_CATEGORIES",
    "ParsedStandardBeatmap",
    "SegmentCategory",
    "SegmentDimension",
    "SegmentPlan",
    "approach_preempt_ms",
    "build_long_sequence_plans",
    "build_segment_plans",
    "circle_overlap_ratio",
    "circle_radius_from_size",
    "classify_hit_group",
    "group_hit_objects",
    "hit_objects_overlap_ratio",
    "parse_standard_beatmap",
    "parse_standard_hit_objects",
]
