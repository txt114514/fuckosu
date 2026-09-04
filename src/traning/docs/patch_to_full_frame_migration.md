# Patch 到全帧主流程迁移

## 文件清单

| 状态 | 文件 | 用途 |
|---|---|---|
| 保留、deprecated/legacy | `traning/lib/data/patch_stream.py` | 历史 ROI/分块流工具，不是正式 dataset 或 model 入口 |
| 保留、deprecated/legacy | `traning/lib/data/tiling.py` | 历史窗口生成与切块工具 |
| 保留、deprecated/legacy | `traning/lib/data/coordinates.py` | 只服务 PatchMeta 的局部/全局辅助 |
| 保留、legacy fixture | `traning/lib/data/synthetic_structures.py` | 合成结构与跨 patch 回归辅助 |
| 保留、诊断参数 | `traning/lib/runtime/memory.py` 中 `patch_size` 文本 | OOM 建议兼容字段，不参与模型输入构建 |
| 已删除 | 无 | 本轮不删除可能仍被历史测试或离线工具使用的文件；通过撤出公开面和主链路停用 |

这些模块从 `traning.lib.data` 的默认公开导出中移除；需要历史 ROI 工具的调用方必须显式导入 deprecated leaf module。它们不被 `traning.main`、`core.app`、`core.data`、`core.perception` 或 `core.training` 导入。

## 默认链路核验

正式数据构建由 `core.data.segments.build_training_datasets` 读取完整帧，正式感知由 `core.perception.PerceptionModel` 消费全帧 batch，生产训练由 `core.training.ProductionTrainer` 编排。正式推理由 `core.app.V2RuntimePipeline` 将完整 `VideoFrame` 交给 perception runtime。两条入口都不构造 `PatchStream`、`PatchWindow` 或固定 canvas。

明确结论：

- 默认训练链路不经过固定 Patch。
- 默认推理链路不经过固定 Patch。
- 保留的 patch/tiling 代码只属于 legacy/可选 ROI refiner 工具，不得重新接入默认链路。

架构检查通过 AST/import 搜索持续约束默认入口；如果未来引入 ROI refiner，必须在完整帧模型之后作为显式可选阶段，并保持全帧坐标 transform fingerprint。
