from __future__ import annotations

from Traning.Lib.common.batch import BatchProcessResult


class VerifyStepsMixin:
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

        _, sections = self.parser.parse_sections(osu_path)
        general = self.parser.parse_key_value_section(sections.get("General", []))
        difficulty = self.parser.parse_key_value_section(sections.get("Difficulty", []))
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
        timing_points = self.parser.parse_timing_points(timing_lines)
        objects = self.parser.parse_hitobjects(hitobjects, timing_points, slider_multiplier)
        verify_lines = self.parser.objects_to_lines(objects)

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
