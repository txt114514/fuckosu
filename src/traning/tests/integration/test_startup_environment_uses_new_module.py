"""验证 start 可脱离仓库根 environment 并复用新报告对象。"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

from start.checks import registry
from traning.lib.environment import collect_environment_report


def test_start_environment_check_calls_canonical_collector_once(monkeypatch) -> None:
    """启动检查只采集一次报告，details 不得再次探测 CUDA。"""

    report = collect_environment_report(cpu_mode_allowed=True)
    calls = 0

    def _collect_once():
        """返回固定报告并记录唯一 collector 调用。"""

        nonlocal calls
        calls += 1
        return report

    monkeypatch.setattr(registry, "collect_environment_report", _collect_once)
    result = registry.check_environment(require_cuda=False)

    assert calls == 1
    assert result.details["cuda_available"] is report.torch.cuda_available
    assert result.details["python"] == report.python_version


def test_start_checks_import_without_repository_root_on_pythonpath(tmp_path: Path) -> None:
    """只暴露 src 时，start 不应再因根 environment 缺失而导入失败。"""

    workspace = Path(__file__).resolve().parents[4]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(workspace / "src")
    result = subprocess.run(
        (
            sys.executable,
            "-c",
            "import start.checks.registry; import traning.lib.environment; print('ok')",
        ),
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
