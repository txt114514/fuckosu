from __future__ import annotations

from pathlib import Path
import sys
from typing import List

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from Traning.Lib.function_tools.functions_process_tool import (
    BatchProcessResult,
    FolderBatchProcessor,
    read_config_values,
)
from Traning.Lib.get_training_data.config_loader import (
    build_from_config_or_default,
    ConfigReader,
    VERIFY_EXPORTER_CONFIG_SPECS,
)
from Traning.Lib.data_class_manager.data_type_group import Circle, Slider, Spinner
from Traning.Lib.data_class_manager.data_osu_original import OsuOriginalTimingPoint
from Traning.Lib.traning_package_manager.order_walker import OrderFolderWalker
from Traning.Lib.traning_package_manager.files_manager import BeatmapFolderStore
from Traning.Lib.get_training_data.process_status_manager import (
    ProcessStatusManager,
)


DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_TARGET_ROOT = DEFAULT_REPO_ROOT / "training_package" / "match-completed_package"
DEFAULT_ORDER_FILENAME = "order.txt"
DEFAULT_VERIFY_FILENAME = "verify.txt"
DEFAULT_VERIFY_FAILED_FILENAME = "verify_failed.txt"

# 默认值保留在当前文件；config.json 里的合法参数只用于覆盖这些默认值。


def _load_verify_exporter_config(config: ConfigReader) -> dict[str, object]:
    # verify 导出只读取目标目录、order 文件名和 verify 输出/失败文件名。
    return read_config_values(config, VERIFY_EXPORTER_CONFIG_SPECS)

class VerifyExporter(FolderBatchProcessor):
    def __init__(
        self,
        walker: OrderFolderWalker,
        store: BeatmapFolderStore,
        verify_filename: str = DEFAULT_VERIFY_FILENAME,
        failed_filename: str = DEFAULT_VERIFY_FAILED_FILENAME,
        status_manager: ProcessStatusManager | None = None,
    ):
        self.walker = walker
        self.store = store
        self.verify_filename = verify_filename
        super().__init__(failed_filename)
        self.status_manager = status_manager or ProcessStatusManager(
            target_root=str(store.target_root),
            order_filename=store.order_filename,
        )

    def _parse_sections(self, osu_path: Path) -> tuple[str | None, dict[str, list[str]]]:
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

    def _parse_key_value_section(self, lines: List[str]) -> dict[str, str]:
        result = {}
        for line in lines:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            result[key.strip()] = value.strip()
        return result

    def _parse_timing_points(self, lines: List[str]) -> List[OsuOriginalTimingPoint]:
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

    def _get_effective_timing(
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

    def _parse_hitobjects(
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

                red_tp, sv_multiplier = self._get_effective_timing(t_start, timing_points)
                beat_length = red_tp.beat_length

                one_slide_duration = (
                    length / (slider_multiplier * 100.0 * sv_multiplier)
                ) * beat_length
                total_duration = one_slide_duration * slides
                t_end = int(round(t_start + total_duration))

                objects.append(Slider(t_start, t_end, path, slides))
                continue

            if type_flag & 8:
                if len(parts) < 6:
                    raise ValueError(f"Spinner 字段不足: {line}")

                t_end = int(parts[5])
                objects.append(Spinner(t_start, t_end))
                continue

            raise NotImplementedError(f"未支持的 HitObject 类型: {line}")

        return objects

    def _objects_to_lines(self, objects: List[object]) -> List[str]:
        lines: List[str] = []

        for obj in objects:
            if isinstance(obj, Circle):
                lines.append(f"Circle({obj.t_start}, {obj.t_end}, {obj.x}, {obj.y})")
            elif isinstance(obj, Slider):
                lines.append(
                    f"Slider({obj.t_start}, {obj.t_end}, {repr(obj.path)}, {obj.repeats})"
                )
            elif isinstance(obj, Spinner):
                lines.append(f"Spinner({obj.t_start}, {obj.t_end})")
            else:
                raise TypeError(f"未知对象类型: {type(obj)}")

        return lines

    def process_one(
        self,
        folder_name: str,
        overwrite: bool = False,
    ) -> BatchProcessResult:
        if not self.store.folder_exists(folder_name):
            return "skip"

        self.status_manager.ensure_status_file(folder_name)
        verify_exists = self.store.file_exists(folder_name, self.verify_filename)
        verify_done = self.status_manager.is_step_done(folder_name, "verify_exported")
        if not overwrite and verify_exists and verify_done:
            return "skip"

        osu_files = self.store.find_osu_files(folder_name)
        if not osu_files:
            return "skip"

        osu_path = osu_files[0]

        _, sections = self._parse_sections(osu_path)
        general = self._parse_key_value_section(sections.get("General", []))
        difficulty = self._parse_key_value_section(sections.get("Difficulty", []))
        hitobjects = sections.get("HitObjects", [])
        timing_lines = sections.get("TimingPoints", [])

        mode = int(general.get("Mode", "0"))
        if mode != 0:
            raise NotImplementedError(f"当前仅支持 osu!standard (Mode=0)，检测到 Mode={mode}")

        if "SliderMultiplier" not in difficulty:
            raise ValueError(f"{osu_path} 缺少 SliderMultiplier")
        if not timing_lines:
            raise ValueError(f"{osu_path} 缺少 TimingPoints")
        if not hitobjects:
            raise ValueError(f"{osu_path} 缺少 HitObjects")

        slider_multiplier = float(difficulty["SliderMultiplier"])
        timing_points = self._parse_timing_points(timing_lines)
        objects = self._parse_hitobjects(hitobjects, timing_points, slider_multiplier)
        verify_lines = self._objects_to_lines(objects)

        write_mode = "overwrite" if overwrite else "skip_if_exists"

        verify_result = self.store.write_lines(
            folder_name=folder_name,
            filename=self.verify_filename,
            lines=verify_lines,
            mode=write_mode,
        )

        if verify_result == "skipped":
            self.status_manager.mark_step_done(
                folder_name,
                "verify_exported",
                detail={"filename": self.verify_filename},
            )
            if not verify_done:
                return "success"
            return "skip"

        self.status_manager.mark_step_done(
            folder_name,
            "verify_exported",
            detail={"filename": self.verify_filename},
        )
        return "success"

    def handle_failure(self, folder_name: str, error: Exception):
        if self.store.folder_exists(folder_name):
            self.status_manager.ensure_status_file(folder_name)
            self.status_manager.mark_step_pending(
                folder_name,
                "verify_exported",
                detail={"error": str(error)},
            )


def _build_verify_exporter_from_config(
    target_root: str = str(DEFAULT_TARGET_ROOT),
    order_filename: str = DEFAULT_ORDER_FILENAME,
    verify_filename: str = DEFAULT_VERIFY_FILENAME,
    verify_failed_filename: str = DEFAULT_VERIFY_FAILED_FILENAME,
) -> VerifyExporter:
    walker = OrderFolderWalker(target_root=target_root, order_filename=order_filename)
    store = BeatmapFolderStore(target_root=target_root, order_filename=order_filename)
    return VerifyExporter(
        walker=walker,
        store=store,
        verify_filename=verify_filename,
        failed_filename=verify_failed_filename,
    )


def build_verify_exporter_from_config_or_default(
    config_path: Path | None = None,
) -> VerifyExporter:
    return build_from_config_or_default(
        _build_verify_exporter_from_config,
        [_load_verify_exporter_config],
        config_path=config_path,
        default_builder=_build_verify_exporter_from_config,
    )


def main():
    exporter = build_verify_exporter_from_config_or_default()
    exporter.run(overwrite=False)


if __name__ == "__main__":
    main()
