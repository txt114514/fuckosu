"""控制训练进度日志频率，始终保留首尾步骤。"""

from __future__ import annotations


def should_report_training_step(step: int, target: int) -> bool:
    if step <= 1 or step >= target:
        return True
    interval = max(1, target // 20)
    return step % interval == 0
