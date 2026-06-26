from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from traning.core.full_flow import (
    FULL_FLOW_STAGES,
    FullFlowConfig,
    load_full_flow_status,
    run_full_flow,
    stage_ids,
)


class FullFlowTests(unittest.TestCase):
    def test_stage_ids_are_unique_and_ordered(self) -> None:
        ids = stage_ids()
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(ids[0], "SOURCE_CHANGE_CHECK")
        self.assertEqual(ids[-1], "REPORT_GENERATION")
        self.assertTrue(all(stage.display_name for stage in FULL_FLOW_STAGES))

    def test_plan_mode_writes_manifest_state_and_reports(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_full_flow(
                FullFlowConfig(
                    config_path=Path("configs/model_small_vram.yaml"),
                    mode="plan",
                    device="cpu",
                    output_root=Path(temp_dir),
                    run_id="plan-test",
                    run_full_checks=False,
                    progress_ui="off",
                )
            )
            self.assertEqual(result.status, "planned")
            self.assertTrue(result.manifest_path.is_file())
            self.assertTrue(result.state_path.is_file())
            self.assertTrue(result.report_json_path.is_file())
            self.assertTrue(result.report_markdown_path.is_file())
            self.assertTrue(
                all(stage.status == "READY" for stage in result.stages)
            )

            loaded = load_full_flow_status(Path(temp_dir), run_id="plan-test")
            self.assertEqual(loaded.status, "planned")
            self.assertEqual(loaded.run_id, "plan-test")

    def test_critical_stages_cannot_be_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ValueError):
                run_full_flow(
                    FullFlowConfig(
                        config_path=Path("configs/model_small_vram.yaml"),
                        mode="plan",
                        device="cpu",
                        output_root=Path(temp_dir),
                        run_id="bad-skip",
                        skip_stages=("DATA_QUALITY_CHECK",),
                    )
                )

    def test_force_stage_is_reported_in_plan_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_full_flow(
                FullFlowConfig(
                    config_path=Path("configs/model_small_vram.yaml"),
                    mode="plan",
                    device="cpu",
                    output_root=Path(temp_dir),
                    run_id="force-plan",
                    run_full_checks=False,
                    progress_ui="off",
                    force_stages=("RAMP_TRAINING",),
                )
            )
            ramp_stage = next(
                stage for stage in result.stages if stage.stage_id == "RAMP_TRAINING"
            )
            self.assertEqual(ramp_stage.status, "READY")
            self.assertTrue(ramp_stage.result["forced"])

            manifest = result.manifest_path.read_text(encoding="utf-8")
            self.assertIn('"force_stages": [', manifest)
            self.assertIn('"RAMP_TRAINING"', manifest)

    def test_force_stage_cannot_conflict_with_skip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ValueError):
                run_full_flow(
                    FullFlowConfig(
                        config_path=Path("configs/model_small_vram.yaml"),
                        mode="plan",
                        device="cpu",
                        output_root=Path(temp_dir),
                        run_id="bad-force-skip",
                        force_stages=("FINAL_EVALUATION",),
                        skip_stages=("FINAL_EVALUATION",),
                    )
                )

    def test_force_stage_must_be_inside_selected_range(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ValueError):
                run_full_flow(
                    FullFlowConfig(
                        config_path=Path("configs/model_small_vram.yaml"),
                        mode="plan",
                        device="cpu",
                        output_root=Path(temp_dir),
                        run_id="bad-force-range",
                        from_stage="MODEL_EXPORT",
                        force_stages=("RAMP_TRAINING",),
                    )
                )


if __name__ == "__main__":
    unittest.main()
