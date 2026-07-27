"""用 tmux 创建训练主窗格及多个只读仪表盘观察窗格。"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shlex
import shutil
import subprocess


PANELS: tuple[tuple[str, str], ...] = (
    ("current", "当前试验"),
    ("parameters", "参数"),
    ("tests", "测试"),
    ("scores", "评分"),
    ("resources", "资源"),
    ("events", "事件"),
)


@dataclass(frozen=True)
class MultiTerminalLaunchResult:
    status: str
    message: str
    session_name: str | None = None


def launch_panel_terminals(
    *,
    run_id: str,
    dashboard_dir: Path,
    cwd: Path,
) -> MultiTerminalLaunchResult:
    """在当前或新 tmux 会话中启动只读状态观察面板。"""

    tmux = shutil.which("tmux")
    if tmux is None:
        return MultiTerminalLaunchResult(
            status="unavailable",
            message="当前容器未安装 tmux，无法自动创建多个终端窗格",
        )
    state_path = dashboard_dir / "dashboard_state.json"
    session_name = _session_name(run_id)
    commands = [
        _watch_command(cwd=cwd, state_path=state_path, panel=panel, title=title)
        for panel, title in PANELS
    ]
    try:
        if os.environ.get("TMUX"):
            _launch_in_current_session(tmux, commands)
            return MultiTerminalLaunchResult(
                status="launched",
                message="已在当前 tmux 会话中创建多面板终端",
                session_name=None,
            )
        _launch_detached_session(tmux, session_name, commands)
        return MultiTerminalLaunchResult(
            status="launched",
            message=f"已创建 tmux 会话：{session_name}",
            session_name=session_name,
        )
    except Exception as error:
        return MultiTerminalLaunchResult(
            status="failed",
            message=f"创建多终端面板失败：{type(error).__name__}: {error}",
            session_name=session_name,
        )


def launch_attached_training_terminals(
    *,
    run_id: str,
    dashboard_dir: Path,
    cwd: Path,
    training_command: str,
) -> MultiTerminalLaunchResult:
    """创建包含主训练命令的独立会话，并将当前终端附加到该会话。"""

    tmux = shutil.which("tmux")
    if tmux is None:
        return MultiTerminalLaunchResult(
            status="unavailable",
            message="当前容器未安装 tmux，无法自动创建多个终端窗格",
        )
    session_name = _session_name(run_id)
    state_path = dashboard_dir / "dashboard_state.json"
    commands = [
        _titled_command(cwd=cwd, title="主训练", command=training_command),
        *(
            _watch_command(cwd=cwd, state_path=state_path, panel=panel, title=title)
            for panel, title in PANELS
        ),
    ]
    try:
        _launch_detached_session(tmux, session_name, commands)
        attached = subprocess.run([tmux, "attach-session", "-t", session_name], check=False)
        if attached.returncode != 0:
            return MultiTerminalLaunchResult(
                status="failed",
                message=f"附加到 tmux 会话失败，返回码 {attached.returncode}",
                session_name=session_name,
            )
        return MultiTerminalLaunchResult(
            status="attached",
            message=f"已启动并进入 tmux 多面板会话：{session_name}",
            session_name=session_name,
        )
    except Exception as error:
        return MultiTerminalLaunchResult(
            status="failed",
            message=f"创建并进入多终端面板失败：{type(error).__name__}: {error}",
            session_name=session_name,
        )


def _watch_command(
    *,
    cwd: Path,
    state_path: Path,
    panel: str,
    title: str,
) -> str:
    # 所有观察窗格只读取同一原子快照，不与训练进程共享内存或控制权。
    return (
        _command_prefix(cwd=cwd, title=title)
        + f"PYTHONPATH=src:. python -m visualization.core.panel_watcher "
        f"{shlex.quote(str(state_path))} {shlex.quote(panel)}"
    )


def _titled_command(
    *,
    cwd: Path,
    title: str,
    command: str,
) -> str:
    return _command_prefix(cwd=cwd, title=title) + command


def _command_prefix(*, cwd: Path, title: str) -> str:
    return f"cd {shlex.quote(str(cwd))} && printf '\\033]0;{title}\\007' && "


def _launch_in_current_session(tmux: str, commands: list[str]) -> None:
    if not commands:
        return
    _run_tmux(tmux, "split-window", "-h", commands[0])
    for command in commands[1:]:
        _run_tmux(tmux, "split-window", "-v", command)
        _run_tmux(tmux, "select-layout", "tiled")
    _run_tmux(tmux, "select-layout", "tiled")


def _launch_detached_session(tmux: str, session_name: str, commands: list[str]) -> None:
    if not commands:
        return
    _run_tmux(tmux, "new-session", "-d", "-s", session_name, commands[0])
    for command in commands[1:]:
        _run_tmux(tmux, "split-window", "-t", session_name, "-v", command)
        _run_tmux(tmux, "select-layout", "-t", session_name, "tiled")
    _run_tmux(tmux, "select-layout", "-t", session_name, "tiled")


def _run_tmux(tmux: str, *args: str) -> None:
    subprocess.run([tmux, *args], check=True)


def _session_name(run_id: str) -> str:
    safe = "".join(char if char.isalnum() else "_" for char in run_id)
    return f"osu_ui_{safe[:32]}"
