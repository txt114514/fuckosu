# Project Instructions

## Engineering Principles

- 修改或新增代码时，优先复用项目现有 API、数据模型和已安装依赖提供的稳定 API；
  只有现有接口无法满足需求时才新增实现，并保持新增逻辑局部、可替换。

## Code Navigation

- 处理 `src/Traning` 前，先读 `project_index/PROJECT_MAP.md`。
- 定位函数优先运行 `python project_index/build_index.py --lookup 符号名`；
  也可搜索 `project_index/FUNCTION_LOCATIONS.md`，再按需读取
  `project_index/FUNCTION_INDEX.md` 的对应模块块和精确源码行。
- 不要为理解局部改动重新遍历全部 Python 文件；按 Project Map 的阶段表和影响面扩展阅读。

## Index Maintenance

- 修改任何 `src/Traning/**/*.py` 文件后，必须运行：
  `python project_index/build_index.py`
- 完成修改前必须运行：
  `python project_index/build_index.py --check`
- `FUNCTION_INDEX.md` 和 `FUNCTION_LOCATIONS.md` 是生成文件，不要手工编辑。
- 如果改动影响架构分层、阶段调用链、模块职责、配置字段、状态步骤、文件契约或跨模块影响面，
  同时人工更新 `project_index/PROJECT_MAP.md`。
- 函数/方法/类的新增、删除、改名、移动、签名变化和关键调用变化，都属于必须重建索引的修改。
