from __future__ import annotations

import os
import pty
import select
import subprocess
import sys
import time
from pathlib import Path
import unittest

from visualization.lib import (
    DatasetUsageState,
    PipelineStageState,
    ResourceState,
    TrainingEvent,
    TrainingStopState,
    create_dashboard_reporter,
)


class VisualizationDashboardTests(unittest.TestCase):
    def test_required_directories_and_public_api(self) -> None:
        root = Path("src/visualization")
        for name in ("conf", "core", "docs", "lib", "state", "tests"):
            self.assertTrue((root / name).is_dir(), name)
        self.assertFalse(Path("src/training_dashboard").exists())
        self.assertFalse(Path("src/traning/visualization").exists())

    def test_reporter_persists_state_events_and_best_parameters(self) -> None:
        with self.subTest("plain reporter"):
            handle = create_dashboard_reporter(
                run_id="test-run",
                output_dir=Path("/tmp/visualization_test_dashboard"),
                progress_ui="off",
            )
            self.assertEqual(handle.reporter.snapshot().status, "off")

        output = Path("/tmp/visualization_test_dashboard_active")
        with create_dashboard_reporter(
            run_id="test-run",
            output_dir=output,
            progress_ui="plain",
        ) as handle:
            reporter = handle.reporter
            reporter.update_pipeline_stage(
                PipelineStageState(
                    stage_id="raw_data",
                    name="原始数据变更检测",
                    status="passed",
                    processed=0,
                    total=0,
                    message="原始数据未发生变化，无需重新转换",
                )
            )
            reporter.report_dataset_usage(
                DatasetUsageState(total_segments=190, unique_segments=126)
            )
            reporter.report_resource(ResourceState(gpu_reserved_gb=1.0))
            reporter.emit_event(
                TrainingEvent.create(
                    event_type="raw_data",
                    severity="info",
                    message_key="raw_data_unchanged",
                )
            )
            reporter.report_score(score=0.75, trial_id="trial_1", level="A")

        self.assertTrue((output / "dashboard_state.json").exists())
        self.assertTrue((output / "events.jsonl").exists())
        self.assertTrue((output / "best_parameters.json").exists())

    def test_dataset_exhausted_stop_state_is_persisted(self) -> None:
        output = Path("/tmp/visualization_test_stop")
        with create_dashboard_reporter(
            run_id="stop-test",
            output_dir=output,
            progress_ui="plain",
        ) as handle:
            handle.reporter.request_stop(
                TrainingStopState(
                    reason="DATASET_EXHAUSTED",
                    message="训练集已用完，当前训练无法继续",
                    exit_code=3,
                    step=18,
                    target_step=50,
                    latest_checkpoint="emergency_step_18.pt",
                    inheritance_path="inheritance/latest",
                )
            )

        self.assertTrue((output / "stop_state.json").exists())
        self.assertIn(
            "DATASET_EXHAUSTED",
            (output / "stop_state.json").read_text(encoding="utf-8"),
        )

    def test_stop_summary_exits_for_tty_keys(self) -> None:
        for label, key in {
            "q": b"q",
            "Q": b"Q",
            "enter": b"\n",
            "esc": b"\x1b",
        }.items():
            with self.subTest(label=label):
                output = _run_stop_summary_in_pty(key)
                self.assertIn("训练已停止", output)

    def test_stop_summary_does_not_wait_without_tty(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                _STOP_SUMMARY_SNIPPET,
            ],
            cwd=Path.cwd(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=5,
            env=_child_env(),
        )
        self.assertEqual(completed.returncode, 0, completed.stdout)
        self.assertIn("训练已停止", completed.stdout)


_STOP_SUMMARY_SNIPPET = (
    "from visualization.core.lifecycle import show_stop_summary;"
    "from visualization.lib import TrainingStopState;"
    "show_stop_summary("
    "TrainingStopState(reason='USER_INTERRUPTED', message='用户请求停止', "
    "exit_code=2, step=3, target_step=9, latest_checkpoint='safe.pt', "
    "inheritance_path='inheritance'), wait_for_key=True)"
)


def _run_stop_summary_in_pty(key: bytes) -> str:
    master_fd, slave_fd = pty.openpty()
    process = subprocess.Popen(
        [sys.executable, "-c", _STOP_SUMMARY_SNIPPET],
        cwd=Path.cwd(),
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        close_fds=True,
        env=_child_env(),
    )
    os.close(slave_fd)
    output = bytearray()
    sent_key = False
    deadline = time.monotonic() + 5
    try:
        while time.monotonic() < deadline:
            readable, _, _ = select.select([master_fd], [], [], 0.1)
            if readable:
                try:
                    chunk = os.read(master_fd, 4096)
                except OSError:
                    break
                if not chunk:
                    break
                output.extend(chunk)
                if not sent_key and "训练已停止".encode() in output:
                    os.write(master_fd, key)
                    sent_key = True
            if sent_key and process.poll() is not None:
                break
        if not sent_key:
            os.write(master_fd, key)
        return_code = process.wait(timeout=5)
        if return_code != 0:
            raise AssertionError(
                f"stop summary exited with {return_code}: "
                f"{output.decode(errors='ignore')}"
            )
        while True:
            readable, _, _ = select.select([master_fd], [], [], 0)
            if not readable:
                break
            try:
                output.extend(os.read(master_fd, 4096))
            except OSError:
                break
    finally:
        if process.poll() is None:
            process.kill()
        os.close(master_fd)
    return output.decode(errors="ignore")


def _child_env() -> dict[str, str]:
    env = dict(os.environ)
    current = env.get("PYTHONPATH")
    prefix = "src:."
    env["PYTHONPATH"] = f"{prefix}:{current}" if current else prefix
    return env


if __name__ == "__main__":
    unittest.main()
