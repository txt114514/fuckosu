from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from package.checks import StartupCheckReport, StartupCheckResult


REPO_ROOT = Path(__file__).resolve().parents[4]


def run_full_checks() -> StartupCheckReport:
    command = (
        sys.executable,
        "-m",
        "pytest",
        "src/before_traning/tests/full_checks",
        "-q",
    )
    completed = _run_pytest(command)
    return StartupCheckReport(
        scope="before_traning.full_checks",
        results=(
            StartupCheckResult(
                key="before_traning:full_tests",
                status="passed" if completed.returncode == 0 else "failed",
                message=f"pytest returned {completed.returncode}",
                details={
                    "command": command,
                    "stdout_tail": _tail(completed.stdout),
                    "stderr_tail": _tail(completed.stderr),
                },
            ),
        ),
    )


def _run_pytest(command: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    src_path = str(REPO_ROOT / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src_path if not existing else os.pathsep.join((src_path, existing))
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _tail(text: str, *, max_lines: int = 80) -> str:
    return "\n".join(text.splitlines()[-max_lines:])


__all__ = ["run_full_checks"]
