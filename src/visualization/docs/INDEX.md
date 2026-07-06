# 可视化文档索引

- [可视化模块详细介绍](README.md)
- [中文终端 UI](TERMINAL_UI.md)
- [实时 UI 架构](REALTIME_UI_ARCHITECTURE.md)

稳定公共入口位于 `visualization.lib`。训练代码只依赖 reporter、gallery API 和状态 DTO，
不直接依赖 Rich 面板、终端布局或 `visualization.core` 内部实现。
