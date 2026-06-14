from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path

from before_traning.Lib.beatmap.folder_store import BeatmapFolderStore
from before_traning.Lib.beatmap.hit_objects import (
    Circle,
    HitObject,
    Slider,
    Spinner,
)
from before_traning.Lib.beatmap.osu_parser import VerifyOsuParser


BEATMAP_DATA_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ParsedStandardBeatmap:
    hp_drain_rate: float
    circle_size: float
    overall_difficulty: float
    approach_rate: float
    slider_multiplier: float
    slider_tick_rate: float
    stack_leniency: float
    hit_objects: tuple[HitObject, ...]

    @property
    def approach_preempt_ms(self) -> float:
        return approach_preempt_ms(self.approach_rate)


def approach_preempt_ms(approach_rate: float) -> float:
    if not isfinite(approach_rate):
        raise ValueError("ApproachRate 必须是有限数值")
    if approach_rate < 5.0:
        return 1200.0 + 120.0 * (5.0 - approach_rate)
    if approach_rate > 5.0:
        return 1200.0 - 150.0 * (approach_rate - 5.0)
    return 1200.0


def parse_standard_beatmap(
    osu_path: Path,
    parser: VerifyOsuParser | None = None,
) -> ParsedStandardBeatmap:
    parser = parser or VerifyOsuParser()
    _version, sections = parser.parse_sections(osu_path)
    general = parser.parse_key_value_section(sections.get("General", []))
    difficulty = parser.parse_key_value_section(sections.get("Difficulty", []))

    mode = int(general.get("Mode", "0"))
    if mode != 0:
        raise NotImplementedError(
            f"视频切分仅支持 osu!standard，检测到 Mode={mode}"
        )
    if "SliderMultiplier" not in difficulty:
        raise ValueError(f"{osu_path} 缺少 SliderMultiplier")
    if "CircleSize" not in difficulty:
        raise ValueError(f"{osu_path} 缺少 CircleSize")

    overall_difficulty = float(difficulty.get("OverallDifficulty", "5"))
    slider_multiplier = float(difficulty["SliderMultiplier"])
    timing_points = parser.parse_timing_points(
        sections.get("TimingPoints", [])
    )
    hit_objects = parser.parse_hitobjects(
        sections.get("HitObjects", []),
        timing_points,
        slider_multiplier,
    )
    if not hit_objects:
        raise ValueError(f"{osu_path} 没有可处理的 HitObjects")
    return ParsedStandardBeatmap(
        hp_drain_rate=float(difficulty.get("HPDrainRate", "5")),
        circle_size=float(difficulty["CircleSize"]),
        overall_difficulty=overall_difficulty,
        approach_rate=float(
            difficulty.get("ApproachRate", str(overall_difficulty))
        ),
        slider_multiplier=slider_multiplier,
        slider_tick_rate=float(difficulty.get("SliderTickRate", "1")),
        stack_leniency=float(general.get("StackLeniency", "0.7")),
        hit_objects=tuple(
            sorted(hit_objects, key=lambda item: (item.t_start, item.t_end))
        ),
    )


def parse_standard_hit_objects(
    osu_path: Path,
    parser: VerifyOsuParser | None = None,
) -> list[HitObject]:
    return list(parse_standard_beatmap(osu_path, parser).hit_objects)


def _beatmap_to_payload(
    beatmap: ParsedStandardBeatmap,
    parser: VerifyOsuParser,
) -> dict[str, object]:
    return {
        "hp_drain_rate": beatmap.hp_drain_rate,
        "circle_size": beatmap.circle_size,
        "overall_difficulty": beatmap.overall_difficulty,
        "approach_rate": beatmap.approach_rate,
        "slider_multiplier": beatmap.slider_multiplier,
        "slider_tick_rate": beatmap.slider_tick_rate,
        "stack_leniency": beatmap.stack_leniency,
        "hit_objects": [
            parser.hit_object_to_dict(hit_object)
            for hit_object in beatmap.hit_objects
        ],
    }


def _hit_object_from_payload(payload: dict[str, object]) -> HitObject:
    object_type = payload["type"]
    start_ms = int(payload["start_ms"])
    end_ms = int(payload["end_ms"])
    if object_type == "circle":
        return Circle(
            start_ms,
            end_ms,
            float(payload["x"]),
            float(payload["y"]),
        )
    if object_type == "slider":
        raw_path = payload["path"]
        if not isinstance(raw_path, list):
            raise ValueError("Slider path 缓存格式非法")
        return Slider(
            start_ms,
            end_ms,
            [
                (float(point[0]), float(point[1]))
                for point in raw_path
            ],
            int(payload["repeats"]),
            str(payload["curve_type"]),
            (
                float(payload["pixel_length"])
                if payload.get("pixel_length") is not None
                else None
            ),
        )
    if object_type == "spinner":
        return Spinner(start_ms, end_ms)
    raise ValueError(f"未知缓存 HitObject 类型: {object_type!r}")


def _beatmap_from_payload(
    payload: dict[str, object],
) -> ParsedStandardBeatmap:
    raw_objects = payload.get("hit_objects")
    if not isinstance(raw_objects, list):
        raise ValueError("谱面缓存缺少 hit_objects")
    return ParsedStandardBeatmap(
        hp_drain_rate=float(payload["hp_drain_rate"]),
        circle_size=float(payload["circle_size"]),
        overall_difficulty=float(payload["overall_difficulty"]),
        approach_rate=float(payload["approach_rate"]),
        slider_multiplier=float(payload["slider_multiplier"]),
        slider_tick_rate=float(payload["slider_tick_rate"]),
        stack_leniency=float(payload["stack_leniency"]),
        hit_objects=tuple(
            _hit_object_from_payload(item)
            for item in raw_objects
            if isinstance(item, dict)
        ),
    )


def load_standard_beatmap(
    store: BeatmapFolderStore,
    folder_name: str,
    *,
    refresh: bool = False,
    parser: VerifyOsuParser | None = None,
) -> tuple[Path, ParsedStandardBeatmap]:
    parser = parser or VerifyOsuParser()
    osu_files = store.find_osu_files(folder_name)
    if not osu_files:
        raise FileNotFoundError(f"{folder_name} 中没有 .osu 谱面文件")
    osu_path = osu_files[0]
    source_mtime_ns = osu_path.stat().st_mtime_ns
    cached = store.walker.manifest.beatmap_data_for(folder_name)
    if not refresh and cached is not None:
        filename, cached_mtime_ns, schema_version, payload = cached
        if (
            filename == osu_path.name
            and cached_mtime_ns == source_mtime_ns
            and schema_version == BEATMAP_DATA_SCHEMA_VERSION
        ):
            return osu_path, _beatmap_from_payload(payload)

    beatmap = parse_standard_beatmap(osu_path, parser)
    store.walker.manifest.save_beatmap_data(
        folder_name,
        osu_filename=osu_path.name,
        source_mtime_ns=source_mtime_ns,
        schema_version=BEATMAP_DATA_SCHEMA_VERSION,
        payload=_beatmap_to_payload(beatmap, parser),
    )
    return osu_path, beatmap


__all__ = [
    "BEATMAP_DATA_SCHEMA_VERSION",
    "ParsedStandardBeatmap",
    "approach_preempt_ms",
    "load_standard_beatmap",
    "parse_standard_beatmap",
    "parse_standard_hit_objects",
]
