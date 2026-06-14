from __future__ import annotations

from pathlib import Path
from typing import Any, List

from before_traning.Lib.beatmap.hit_objects import Circle, HitObject, Slider, Spinner
from before_traning.Lib.beatmap.timing_points import OsuOriginalTimingPoint


class VerifyOsuParser:
    def parse_sections(self, osu_path: Path) -> tuple[str | None, dict[str, list[str]]]:
        version = None
        sections: dict[str, list[str]] = {}
        current_section = None

        with osu_path.open("r", encoding="utf-8-sig") as f:
            for raw_line in f:
                line = raw_line.strip()

                if not line or line.startswith("//"):
                    continue

                if version is None and line.startswith("osu file format v"):
                    version = line
                    continue

                if line.startswith("[") and line.endswith("]"):
                    current_section = line[1:-1]
                    sections.setdefault(current_section, [])
                    continue

                if current_section is not None:
                    sections[current_section].append(line)

        return version, sections

    def parse_key_value_section(self, lines: List[str]) -> dict[str, str]:
        result = {}
        for line in lines:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            result[key.strip()] = value.strip()
        return result

    def parse_timing_points(self, lines: List[str]) -> List[OsuOriginalTimingPoint]:
        timing_points: List[OsuOriginalTimingPoint] = []

        for line in lines:
            parts = line.split(",")
            if len(parts) < 8:
                continue

            timing_points.append(
                OsuOriginalTimingPoint(
                    time=int(float(parts[0])),
                    beat_length=float(parts[1]),
                    meter=int(parts[2]),
                    sample_set=int(parts[3]),
                    sample_index=int(parts[4]),
                    volume=int(parts[5]),
                    uninherited=(int(parts[6]) == 1),
                    effects=int(parts[7]),
                )
            )

        timing_points.sort(key=lambda tp: tp.time)
        return timing_points

    def get_effective_timing(
        self,
        t: int,
        timing_points: List[OsuOriginalTimingPoint],
    ) -> tuple[OsuOriginalTimingPoint, float]:
        red_tp = None
        sv_multiplier = 1.0

        for tp in timing_points:
            if tp.time > t:
                break

            if tp.uninherited:
                red_tp = tp
            else:
                if tp.beat_length != 0:
                    sv_multiplier = -100.0 / tp.beat_length

        if red_tp is None:
            for tp in timing_points:
                if tp.uninherited:
                    red_tp = tp
                    break

        if red_tp is None:
            raise ValueError("找不到有效的 uninherited timing point")

        return red_tp, sv_multiplier

    def parse_hitobjects(
        self,
        hitobject_lines: List[str],
        timing_points: List[OsuOriginalTimingPoint],
        slider_multiplier: float,
    ) -> List[object]:
        objects = []

        for line in hitobject_lines:
            parts = line.split(",")
            if len(parts) < 5:
                continue

            x = float(parts[0])
            y = float(parts[1])
            t_start = int(parts[2])
            type_flag = int(parts[3])

            if type_flag & 128:
                raise NotImplementedError("当前不支持 mania hold note")

            if type_flag & 1:
                objects.append(Circle(t_start, t_start, x, y))
                continue

            if type_flag & 2:
                if len(parts) < 8:
                    raise ValueError(f"Slider 字段不足: {line}")

                slider_info = parts[5]
                slides = int(parts[6])
                length = float(parts[7])

                tokens = slider_info.split("|")
                path = [(x, y)]

                for token in tokens[1:]:
                    px, py = token.split(":")
                    path.append((float(px), float(py)))

                red_tp, sv_multiplier = self.get_effective_timing(t_start, timing_points)
                beat_length = red_tp.beat_length

                one_slide_duration = (
                    length / (slider_multiplier * 100.0 * sv_multiplier)
                ) * beat_length
                total_duration = one_slide_duration * slides
                t_end = int(round(t_start + total_duration))

                objects.append(
                    Slider(
                        t_start,
                        t_end,
                        path,
                        repeats=slides,
                        curve_type=tokens[0],
                        pixel_length=length,
                    )
                )
                continue

            if type_flag & 8:
                if len(parts) < 6:
                    raise ValueError(f"Spinner 字段不足: {line}")

                t_end = int(parts[5])
                objects.append(Spinner(t_start, t_end))
                continue

            raise NotImplementedError(f"未支持的 HitObject 类型: {line}")

        return objects

    def objects_to_lines(self, objects: List[object]) -> List[str]:
        lines: List[str] = []

        for obj in objects:
            if isinstance(obj, Circle):
                lines.append(f"Circle({obj.t_start}, {obj.t_end}, {obj.x}, {obj.y})")
            elif isinstance(obj, Slider):
                lines.append(
                    f"Slider({obj.t_start}, {obj.t_end}, {repr(obj.path)}, "
                    f"{obj.repeats}, {obj.curve_type!r}, {obj.pixel_length!r})"
                )
            elif isinstance(obj, Spinner):
                lines.append(f"Spinner({obj.t_start}, {obj.t_end})")
            else:
                raise TypeError(f"未知对象类型: {type(obj)}")

        return lines

    def hit_object_to_dict(
        self,
        hit_object: HitObject,
        *,
        time_offset_ms: int = 0,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": hit_object.type,
            "start_ms": hit_object.t_start - time_offset_ms,
            "end_ms": hit_object.t_end - time_offset_ms,
        }
        if isinstance(hit_object, Circle):
            payload.update(x=hit_object.x, y=hit_object.y)
        elif isinstance(hit_object, Slider):
            payload.update(
                path=[[x, y] for x, y in hit_object.path],
                repeats=hit_object.repeats,
                curve_type=hit_object.curve_type,
                pixel_length=hit_object.pixel_length,
            )
        elif not isinstance(hit_object, Spinner):
            raise TypeError(f"未知对象类型: {type(hit_object)}")
        return payload
