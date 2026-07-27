"""按需公开训练前处理的持久化状态管理器。"""

from __future__ import annotations


__all__ = ["ProcessStatusManager"]


def __getattr__(name: str):
    if name == "ProcessStatusManager":
        from before_traning.state.process_status import ProcessStatusManager

        return ProcessStatusManager
    raise AttributeError(name)
