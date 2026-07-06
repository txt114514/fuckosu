from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from traning.core.full_flow import (
    FULL_FLOW_STAGES,
    FullFlowConfig,
    load_full_flow_status,
    run_full_flow,
    stage_ids,
)
from traning.core.full_flow.orchestrator import (
    _FlowRuntime,
    _finish_stage,
    _initial_stage_states,
    _publish_initial_dashboard_stages,
    _run_ramp_section,
)
from traning.core.full_flow.result import utc_now
from visualization.lib import DashboardReporter, PipelinePhase, ResourceState


class FullFlowTests(unittest.TestCase):
    def test_stage_ids_are_unique_and_ordered(self) -> None:
        ids = stage_ids()
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(ids[0], "ENVIRONMENT_PREFLIGHT")
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

    def test_finish_stage_updates_dashboard_reporter(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runtime = _FlowRuntime(
                config=FullFlowConfig(
                    config_path=Path("configs/model_small_vram.yaml"),
                    mode="execute",
                    device="cpu",
                    output_root=root,
                    run_id="ui-stage-test",
                ),
                run_id="ui-stage-test",
                output_dir=root / "ui-stage-test",
                started_at=utc_now(),
                stages=_initial_stage_states(),
            )
            runtime.reporter = DashboardReporter(
                run_id="ui-stage-test",
                output_dir=root / "dashboard",
            )

            _finish_stage(
                runtime,
                "RAMP_TRAINING",
                "PASSED",
                result={"processed": 1, "total": 1},
                artifacts=("ramp/manifest.json",),
            )

            state = runtime.reporter.snapshot()
            stage = state.pipeline_stages["ramp_training"]
            self.assertEqual(stage.status, "passed")
            self.assertEqual(stage.processed, 1)
            self.assertEqual(stage.total, 1)
            self.assertEqual(stage.output_path, "ramp/manifest.json")

    def test_initial_dashboard_stages_are_published_as_pending(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runtime = _FlowRuntime(
                config=FullFlowConfig(
                    config_path=Path("configs/model_small_vram.yaml"),
                    mode="execute",
                    device="cpu",
                    output_root=root,
                    run_id="ui-pending-test",
                ),
                run_id="ui-pending-test",
                output_dir=root / "ui-pending-test",
                started_at=utc_now(),
                stages=_initial_stage_states(),
            )
            runtime.reporter = DashboardReporter(
                run_id="ui-pending-test",
                output_dir=root / "dashboard",
            )

            _publish_initial_dashboard_stages(runtime)

            state = runtime.reporter.snapshot()
            self.assertEqual(state.pipeline_phase, PipelinePhase.STARTUP.value)
            self.assertEqual(
                set(state.pipeline_stages),
                {stage.stage_id.lower() for stage in FULL_FLOW_STAGES},
            )
            self.assertTrue(
                all(stage.status == "pending" for stage in state.pipeline_stages.values())
            )

    def test_full_flow_reports_initial_resource_snapshot_to_dashboard(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            captured: dict[str, DashboardReporter] = {}

            class FakeDashboardHandle:
                def __init__(self, reporter: DashboardReporter) -> None:
                    self.reporter = reporter

                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, traceback):
                    self.reporter.close()
                    return None

            def fake_dashboard_reporter(**kwargs):
                reporter = DashboardReporter(
                    run_id=str(kwargs["run_id"]),
                    output_dir=Path(kwargs["output_dir"]),
                )
                captured["reporter"] = reporter
                return FakeDashboardHandle(reporter)

            with (
                patch(
                    "traning.core.full_flow.orchestrator.create_dashboard_reporter",
                    side_effect=fake_dashboard_reporter,
                ),
                patch(
                    "traning.core.full_flow.orchestrator.collect_resource_state",
                    return_value=ResourceState(
                        cpu_percent=12.5,
                        process_memory_gb=0.25,
                        disk_free_gb=42.0,
                    ),
                ),
                patch("traning.core.full_flow.orchestrator._run_startup_section"),
                patch(
                    "traning.core.full_flow.orchestrator._run_resume_section",
                    return_value=SimpleNamespace(policy="none", stage_checkpoint_paths={}),
                ),
            ):
                result = run_full_flow(
                    FullFlowConfig(
                        config_path=Path("configs/model_small_vram.yaml"),
                        mode="dry-run",
                        device="cpu",
                        output_root=root,
                        run_id="resource-snapshot-test",
                        progress_ui="rich",
                    )
                )

            self.assertEqual(result.status, "dry-run-passed")
            resources = captured["reporter"].snapshot().resources
            self.assertEqual(resources.cpu_percent, 12.5)
            self.assertEqual(resources.process_memory_gb, 0.25)
            self.assertEqual(resources.disk_free_gb, 42.0)

    def test_ramp_section_passes_formal_gallery_output_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_dir = root / "ui-gallery-test"
            output_dir.mkdir()
            runtime = _FlowRuntime(
                config=FullFlowConfig(
                    config_path=Path("configs/model_small_vram.yaml"),
                    mode="execute",
                    device="cpu",
                    output_root=root,
                    run_id="ui-gallery-test",
                    auto_launch_full=True,
                    gallery_output_root=Path("traning_example"),
                    gallery_samples_per_group=2,
                ),
                run_id="ui-gallery-test",
                output_dir=output_dir,
                started_at=utc_now(),
                stages=_initial_stage_states(),
            )
            runtime.reporter = DashboardReporter(
                run_id="ui-gallery-test",
                output_dir=root / "dashboard",
            )
            fake_ramp = SimpleNamespace(
                manifest_path=output_dir / "ramp" / "ramp" / "manifest.json",
                final_readiness_path=output_dir / "ramp" / "ramp" / "readiness.json",
                full_training_started=True,
                full_training_run_dir=output_dir / "ramp" / "ramp" / "full_training",
                as_dict=lambda: {"status": "passed"},
            )
            inheritance = SimpleNamespace(
                policy="auto",
                stage_checkpoint_paths={},
            )

            with patch(
                "traning.core.full_flow.orchestrator.run_training_ramp",
                return_value=fake_ramp,
            ) as ramp:
                _run_ramp_section(runtime, inheritance=inheritance, reporter=runtime.reporter)

            kwargs = ramp.call_args.kwargs
            self.assertEqual(kwargs["full_gallery_output_root"], Path("traning_example"))
            self.assertEqual(kwargs["full_gallery_samples_per_group"], 2)


if __name__ == "__main__":
    unittest.main()
