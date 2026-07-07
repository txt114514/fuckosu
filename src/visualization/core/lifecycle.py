from __future__ import annotations

import select
import sys
import termios
import tty
from contextlib import suppress

from rich.console import Console
from rich.panel import Panel

from visualization.conf.messages import display_text
from visualization.lib.models import TrainingStopState

STOP_EXIT_CODES = {
    "COMPLETED": 0,
    "USER_INTERRUPTED": 2,
    "DATASET_EXHAUSTED": 3,
    "RESOURCE_EXHAUSTED": 4,
    "NUMERIC_ERROR": 5,
    "CHECKPOINT_FAILED": 6,
    "DATA_INCOMPATIBLE": 7,
    "FATAL": 8,
}


def show_stop_summary(stop: TrainingStopState, *, wait_for_key: bool = True) -> None:
    console = Console()
    text = (
        f"原因：{display_text(stop.message)}\n"
        f"当前步数：{stop.step or 0} / {stop.target_step or '未知'}\n"
        f"已保存检查点：{stop.latest_checkpoint or '无'}\n"
        f"已生成继承状态：{stop.inheritance_path or '无'}\n\n"
        "按 Q、回车或 Esc 退出"
    )
    console.print(Panel(text, title="训练已停止", border_style="red"))
    if wait_for_key and sys.stdin.isatty() and sys.stdout.isatty():
        wait_for_exit_key()


def wait_for_exit_key() -> None:
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            readable, _, _ = select.select([sys.stdin], [], [], 0.2)
            if not readable:
                continue
            char = sys.stdin.read(1)
            if char in {"q", "Q", "\n", "\r", "\x1b"}:
                return
    finally:
        with suppress(Exception):
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
