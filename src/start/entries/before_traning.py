"""声明 before_traning 顶层模块的启动入口元数据。"""

from start.modules import source_module_entry

ENTRY = source_module_entry("before_traning")

__all__ = ["ENTRY"]
