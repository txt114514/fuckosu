# traning lib 结构审计

本文记录 `src/traning/lib` 的整理结果和后续放置规则。`lib` 是 `traning`
作用域内的复用实现层；只有被 `src` 下多个顶层模块共同调用、需要形成稳定契约的
结构或 API，才上移到 `src/package` 并从 `package` 公开入口导出。

## 当前保留目录

| 目录 | 定位 | 当前内容 |
|---|---|---|
| `lib/data` | 训练数据读取与几何辅助 | annotation、segment discovery、Dataset、video reader、patch tiling、sampling、collate、颜色 cue、合成测试结构。 |
| `lib/models` | 可复用模型组件 | local/global encoder、global structure head、gated sparse fusion、dense object head、temporal GRU、模型输出契约和 stack builder。 |
| `lib/training` | 训练期数学与空间候选解码 | spatial targets、multi-task loss、一致性 loss、feature canvas、dense prediction canvas、Top-K candidate 与 slider 连通域路径解码。 |
| `lib/metrics` | 评分和序列评估算法 | `point-slider-v2` 单对象评分、`click-sequence-v1` 点击序列模拟、空间/时间/决策错误归因底层字段。 |
| `lib/runtime` | CUDA 与内存运行时入口 | AMP、GradScaler、channels-last、TF32、显存/RAM 预算、OOM 建议、device/tensor/module 迁移。 |
| `lib/visualization` | 结果渲染与图集实现 | 单帧标注渲染、点击帧选择、输出目录身份、可选 ffplay 展示、最佳 trial 图集保存。 |

## 本次清理

- 删除 `lib/losses` 空壳目录。真实损失实现统一在
  `lib/training/losses.py`，公开入口是 `traning.lib.training`。
- 删除 `lib/candidates` 空壳目录。当前候选生成和 slider 路径解码统一在
  `lib/training/spatial_decode.py`，跨层稳定候选引用放在
  `package.contracts.candidate`。
- 删除空的 `lib/optimization` 残留目录。训练闭环控制的真实实现统一在
  `core/optimization`，底层评分算法位于 `lib/metrics`。
- 删除 `lib/compat` 兼容层。`traning` 包初始化不再安装旧模块别名，旧
  `traning.Lib.*`、`traning.data.*`、`traning.models.*`、`traning.training.*`
  和旧 core 路径不再作为训练稳定版入口。

## 放置规则

1. `core` 只保留阶段业务编排、CLI/service 适配和该阶段独有逻辑。
2. 被多个 `traning.core` 阶段共同调用的训练内部能力放在 `traning.lib`。
3. 被 `src/before_traning`、`src/traning` 或未来其他顶层模块共同调用的稳定结构/API
   放在 `src/package`，调用方从 `package` 公开入口导入。
4. `lib` 中新增公开 API 必须通过对应子包 `__init__.py` 导出；调用方不要依赖私有名称。
5. 旧路径不再兼容；迁移后只保留当前正式包名和目录。

## 已知边界

- `lib/models/stack.py` 依赖 `traning.conf.Settings` 来按配置构建模型栈，这是配置到模型的稳定适配入口。
- `lib/visualization/gallery.py` 依赖 `traning.state.gallery_schema` 的
  `BatchGalleryRequest`，因为图集输入契约目前只服务 `traning`。如果该契约被训练前处理
  或独立推理包复用，应把契约迁入 `package.contracts.evaluation` 或新的
  `package.contracts.gallery`，再让 `traning.state` 做适配。
- 训练稳定版删除了 compatibility layer；任何旧路径导入失败都应改为当前正式入口，
  而不是恢复别名。

## 后续建议

- 保持 `lib/training` 作为训练期数学集合，不再新增顶层 `lib/losses`、`lib/candidates`
  这类只有概念没有实现边界的目录。
- 当 `lib/data.models`、`lib/models.outputs`、`lib/metrics.sequence` 中的结构开始被
  `before_traning` 或外部推理复用时，只迁移稳定契约到 `src/package`，算法实现仍按职责
  留在 `traning.lib`。
- 如果可视化图集继续扩展为 HTML/CSV 浏览索引，可以在 `lib/visualization` 下新增实现；
  如果它变成 result_export 独有流程，则迁回 `core/result_export`。
