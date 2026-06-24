# Model Export Plan

## 模块定位

源码入口：`src/traning/core/model_export`

模型导出迁移模块负责把训练过程中产生的 checkpoint、配置和推理依赖整理为可部署、
可迁移、可复现的 artifact。当前已实现首版 PyTorch artifact 导出器。

## 训练与部署差异

训练阶段：

- 可使用完整标签、difficulty、课程采样和评估 JSON。
- 空间训练可逐 patch backward。
- 时间模型可按窗口训练。
- 可保存 optimizer、scheduler、AMP scaler 和全局 step。

部署阶段：

- 只能读取当前帧和历史状态。
- 空间模块在线输出候选和 slider path。
- 时间模块通过 `step(frame_candidates, state)` 更新状态。
- 决策模块输出当前动作、坐标和必要的 hold/release 状态。
- 不允许依赖未来帧、训练标签或评估器 JSON。

## Artifact 组成

首版导出包建议包含：

- `settings.yaml`：完整可解析配置快照。
- `model_state.pt`：空间模型、时间模型和必要 head 的权重。
- `checkpoint_metadata.json`：来源 trial、课程阶段、父 checkpoint、step 和代码版本。
- `candidate_cache_schema.json`：若导出离线评估包，记录候选缓存版本。
- `score_schema.json`：若导出评估结果，记录 score 和 passed 的公式版本。
- `runtime_policy.json`：AMP、channels-last、TF32、输入尺寸和 patch 规格。
- `README.md`：推理入口、输入输出和限制。

训练恢复包还应额外包含 optimizer、scheduler、AMP scaler 和随机种子状态；部署包不需要。

## 兼容与迁移

导出器需要显式管理：

- `traning` 包版本或代码 commit；
- `Settings` schema 版本；
- 模型结构参数；
- 候选缓存版本；
- score 版本；
- 坐标转换版本；
- dense target 版本。

历史导入兼容只应留在训练代码内。导出后的部署入口应使用当前稳定路径：

```text
traning.lib.*
traning.core.spatial
traning.core.temporal
traning.core.decision
```

## 暂不实现

首版不要求：

- ONNX；
- TensorRT；
- 自定义 CUDA kernel；
- 多 GPU 或分布式导出；
- 自动云训练；
- 大规模模型压缩；
- 复杂 UI。

这些能力可以预留接口，但不能阻塞可复现 PyTorch artifact。

## 当前实现

- `ModelArtifactSpec` 描述导出输入、artifact 类型、版本和附加文件。
- `export_model_artifact` 会复制配置、空间 checkpoint、时间 checkpoint、metadata 和 extra files。
- `manifest.json` 记录 artifact 版本、score 版本、candidate cache 版本、代码版本、文件大小和 sha256。
- `validate_model_artifact` 会校验 manifest 版本、文件存在性和 sha256。
- 已区分 `resume` 包和 `inference` 包。

## 后续计划

- 增加 artifact smoke test：加载导出包，运行一帧空间推理和一次 temporal step。
- 增加 schema migration：旧配置字段迁移到当前 `conf` 模型，并记录迁移日志。
