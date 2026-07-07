from __future__ import annotations

import subprocess
import unittest
from unittest.mock import patch

from rich.console import Console

from visualization.core.panels.resources_panel import render_resources_panel
from visualization.lib.models import ResourceState, TrainingDashboardState
from visualization.lib.resources import _MonitorProbe, collect_resource_state


class VisualizationResourceMonitorTests(unittest.TestCase):
    def test_host_exec_bridge_makes_gpu_monitor_available(self) -> None:
        def fake_which(name: str) -> str | None:
            if name in {"nvidia-smi", "host-exec"}:
                return f"/usr/bin/{name}"
            return None

        def fake_run(command, **kwargs):
            if command[0] == "nvidia-smi":
                return subprocess.CompletedProcess(command, 1, "", "直接监控不可用")
            return subprocess.CompletedProcess(
                command,
                0,
                "0, 37, 1024, 8192, 55, 20.5, NVIDIA RTX\n",
                "",
            )

        with (
            patch(
                "visualization.lib.resources.torch.cuda.is_available",
                return_value=False,
            ),
            patch(
                "visualization.lib.resources._collect_pynvml",
                return_value=_MonitorProbe(error="NVML 不可用"),
            ),
            patch("visualization.lib.resources.shutil.which", side_effect=fake_which),
            patch("visualization.lib.resources.subprocess.run", side_effect=fake_run),
        ):
            state = collect_resource_state()

        self.assertEqual(state.gpu_monitor_source, "host-exec nvidia-smi")
        self.assertEqual(state.gpu_utilization, 37.0)
        self.assertAlmostEqual(state.gpu_memory_used_gb or 0, 1.0)
        self.assertAlmostEqual(state.gpu_memory_utilization or 0, 12.5)
        self.assertIsNone(state.gpu_monitor_error)

    def test_unavailable_monitor_keeps_utilization_empty(self) -> None:
        with (
            patch(
                "visualization.lib.resources.torch.cuda.is_available",
                return_value=False,
            ),
            patch(
                "visualization.lib.resources._collect_pynvml",
                return_value=_MonitorProbe(error="NVML 不可用"),
            ),
            patch("visualization.lib.resources.shutil.which", return_value=None),
        ):
            state = collect_resource_state()

        self.assertIsNone(state.gpu_utilization)
        self.assertIn("GPU 监控不可用", state.gpu_monitor_error or "")

    def test_unavailable_monitor_panel_does_not_show_fake_zero(self) -> None:
        state = TrainingDashboardState(
            run_id="resource-panel",
            resources=ResourceState(gpu_monitor_error="GPU 监控不可用"),
        )
        console = Console(record=True, width=100)
        console.print(render_resources_panel(state))
        output = console.export_text()

        self.assertIn("不可用", output)
        self.assertNotIn("0.0%", output)


if __name__ == "__main__":
    unittest.main()
