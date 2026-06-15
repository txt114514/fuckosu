# Project Map

这是仓库级全局索引。各模块的解释性文档放在对应源码目录内，本文件只维护模块入口，
避免在仓库外部重复保存模块实现说明。

## 模块索引

| 模块 | 源码目录 | 对外说明 | Codex 索引 |
|---|---|---|---|
| 全局共享 API | `src/package` | [`README.md`](../src/package/README.md) | 公开入口：`src/package/__init__.py` |
| 训练前处理 | `src/before_traning` | [`README.md`](../src/before_traning/docs/README.md) | [`CODEX_INDEX.md`](../src/before_traning/docs/CODEX_INDEX.md) |
| 模型训练 | `src/traning` | [`README.md`](../src/traning/docs/README.md)、[`TRAINING_PLAN.md`](../src/traning/docs/TRAINING_PLAN.md)、[`TEST_PARAMETER_MECHANISM.md`](../src/traning/docs/TEST_PARAMETER_MECHANISM.md)、[`SCORING_SPEC.md`](../src/traning/docs/SCORING_SPEC.md) | [`CODEX_INDEX.md`](../src/traning/docs/CODEX_INDEX.md) |

## 全局 API 约定

- `src/package` 只存放被 `src` 下多个顶层模块共同调用的稳定 API。
- 调用方应从 `package` 的公开入口导入，不依赖 `_` 开头的名称或内部实现模块。
- 仅服务单个模块的简单局部实现留在对应模块内部。
- 子模块开始持续扩充独立功能、形成稳定契约、隔离第三方依赖或承担跨层边界时，
  即使只有一个调用方，也应迁入 `src/package`；原模块只保留编排和适配。
- 新增全局 API 时按领域建立子模块，并在 `src/package/__init__.py` 显式导出。

## before_traning 最短阅读路径

1. 先读 `src/before_traning/docs/CODEX_INDEX.md`，确认阶段、分层和改动影响面。
2. 运行 `python project_index/build_index.py --lookup 符号名` 定位函数或模块。
3. 用 `rg -n "符号名" src/before_traning/docs/CODEX_INDEX.md` 搜索索引。
4. 只按需读取命中的源码行和索引列出的相邻模块。

## traning 最短阅读路径

1. 先读 `src/traning/docs/CODEX_INDEX.md`，确认当前阶段和数据契约。
2. 需要模型目标和长期路线时读取 `src/traning/docs/TRAINING_PLAN.md`。
3. 运行 `python project_index/build_index.py --lookup 符号名` 定位实现。
4. 当前首要入口是 `core/data_input`；空间、候选缓存和时序目录暂为阶段边界。
5. 批次评估图集入口是 `save-annotation-gallery`；结果契约位于
   `state/gallery_schema.py`，按最高分 trial 和六个数据子项目生成 `passed/failed`
   标注目录。
6. 参数搜索、score 和通过机制的已实现/待实现边界见
   `src/traning/docs/TEST_PARAMETER_MECHANISM.md`。
7. 点与 slider 的 `point-slider-v2` 公式见
   `src/traning/docs/SCORING_SPEC.md`。

## 索引维护

- 生成脚本：`project_index/build_index.py`
- 重建命令：`python project_index/build_index.py`
- 校验命令：`python project_index/build_index.py --check`
- 两个模块的 `docs/CODEX_INDEX.md` 都是生成文件，不要手工编辑。
- 模块内部架构、阶段、配置、状态或文件契约变化时，更新生成脚本中的导航内容并重建。
- 新增、删除或移动模块导航入口时，更新本文件。
