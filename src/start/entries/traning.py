"""声明 traning 顶层模块的启动入口元数据。"""

from start.modules import source_module_entry

ENTRY = source_module_entry("traning")

__all__ = ["ENTRY"]
