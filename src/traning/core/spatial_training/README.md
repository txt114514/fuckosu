# Spatial Training Call Structure

`core/spatial_training` 只负责空间模块的业务流程编排；训练代码依赖的稳定能力统一从
`traning.Lib` 公开入口导入。

## 调用链

```text
CLI / candidate cache
  -> traning.core.spatial_training
     -> traning.Lib.runtime      # CUDA/AMP/channels-last/内存预算
     -> traning.Lib.data         # PatchStream、颜色 cue、坐标和数据结构
     -> traning.Lib.models       # local/global/fusion/spatial head 模型栈
     -> traning.Lib.training     # loss、dense target、CPU canvas、候选/slider 解码
```

## 文件职责

- `spatial_inference.py`：单帧空间推理；完整帧 global 前向与逐 patch local/fusion/head
  在 GPU 上运行，预测结果 detach 到 CPU 后做全图画布融合、NMS 和 slider 连通域解码。
- `spatial_trainer.py`：首版单帧空间训练；冻结 global/structure，串行 patch 前向并逐
  patch backward，避免保留全帧所有 patch 的计算图。
- `__init__.py`：空间流程公开入口；导出 `run_spatial_frame_inference`、
  `run_spatial_training` 和候选字典化函数。

## CPU / GPU 分工

- GPU：`global_encoder_forward`、`local_encoder_forward_per_patch`、
  `global_local_fusion_per_patch`、`spatial_prediction_head_per_patch`。
- CPU：RGB 归一化、osu 色号/数字/边缘 cue、patch 切分和 padding、预测画布融合、
  Top-K/NMS、slider path 连通域解码、JSON summary 或候选缓存写出。

## 边界约定

- 不从旧 `traning.data`、`traning.models`、`traning.training` 或 `traning.core.memory`
  导入；这些历史路径只通过 `traning.Lib.compat` 集中兼容。
- 训练运行时统一复用 `traning.Lib.runtime`，包括 `configure_torch_runtime`、
  `tensor_to_device`、`module_to_device`、`autocast_context`、`create_grad_scaler` 和
  `enforce_runtime_memory_budget`。
- 训练集导入只通过 `traning.core.dataset_import`；空间模块不直接扫描文件系统。
