"""声明 package 顶层模块的启动入口元数据。"""

from start.modules import source_module_entry

ENTRY = source_module_entry("package")

__all__ = ["ENTRY"]
