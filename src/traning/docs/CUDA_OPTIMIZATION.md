# CUDA Optimization Contract

本训练项目默认面向 RTX 4060 Laptop 8GB VRAM。后续新增训练循环、评估循环或模型 smoke
路径时，应复用 `traning.Lib.runtime` 中的统一入口，而不是在各模块里散写 CUDA 开关。

## Runtime Policy

- `configure_torch_runtime`：统一设置 TF32、cuDNN benchmark、matmul precision 和
  channels-last 策略。
- `module_to_device`：把模型迁移到目标 device；CUDA 上按配置切换 channels-last。
- `tensor_to_device`：把 BCHW Tensor 迁移到目标 device；CUDA 上支持 channels-last 和
  non-blocking copy。
- `autocast_context`：根据 `memory.amp_dtype` 选择 FP16/BF16/FP32 autocast。
- `create_grad_scaler`：仅在 CUDA FP16 时默认启用 GradScaler；BF16 默认不需要 scaler。
- `collect_memory_snapshot`：记录 allocated/reserved 显存，用于 profile 和 OOM 报告。
- `enforce_runtime_memory_budget`：训练、推理和模型 smoke 入口的统一内存预算检查；
  计算 `max_vram_gib`、`reserve_vram_gib`、`max_ram_gib`、`reserve_ram_gib` 后的有效预算。
  CPU 预算会同时考虑当前可用内存：`current_rss + available_ram - reserve_ram_gib`；
  CUDA 上设置 PyTorch per-process memory fraction，给系统、驱动和桌面保留余量。

## Default Small-VRAM Choices

默认配置位于 `configs/model_small_vram.yaml`：

- AMP: `auto`，RTX 4060 上优先 BF16。
- `channels_last: true`
- `allow_tf32: true`
- `cudnn_benchmark: true`
- `grad_scaler: auto`
- `compile_model: false`
- `max_vram_gib: 6.5`
- `reserve_vram_gib: 1.0`
- `max_ram_gib: 24.0`
- `reserve_ram_gib: 4.0`
- `pin_memory: true`
- `patch_batch_size: 1`
- `backward_per_patch: true`

`torch.compile` 默认关闭，因为首版需要优先保证稳定和可诊断；后续可在固定训练阶段单独测速后开启。
RTX 4060 Laptop 当前可见总显存约 7.62 GiB；默认有效训练显存预算为
`min(6.5, total_vram - 1.0)`，避免把显存吃到系统/驱动无法正常运行。

## Training Step Rules

训练 step 应遵循：

```python
runtime = configure_torch_runtime(...)
budget = enforce_runtime_memory_budget(...)
model = module_to_device(model, device, channels_last=runtime.channels_last)
scaler = create_grad_scaler(...)
optimizer.zero_grad(set_to_none=True)

with autocast_context(device, settings.memory.amp_dtype):
    loss = compute_loss(...)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

不要在 step 中长期保存 GPU Tensor list；日志、图集、候选缓存和画布数据应先 `detach()` 并迁移到 CPU。
除阶段切换、评测结束或 OOM 恢复外，不要频繁调用 `torch.cuda.empty_cache()`。

## Spatial Inference Split

单帧空间推理入口是 `traning.core.spatial_training.run_spatial_frame_inference`。这条路径保持清晰分工：

- GPU：完整帧 global encoder 前向、逐 patch local encoder 前向、global-local fusion 和 spatial
  head 前向；全部通过 `configure_torch_runtime`、`module_to_device`、`tensor_to_device` 和
  `autocast_context` 进入 CUDA runtime。
- CPU：RGB 归一化、osu 色号/白色数字/边缘 cue、PatchStream 切分和 padding、预测
  `detach().cpu()` 后的全图画布融合、Top-K/局部最大值/NMS 候选解码、slider 连通域
  polyline 恢复，以及 JSON summary 或候选缓存写出。

离线缓存命令 `build-candidate-cache` 已复用该入口。后续评估命令也应复用同一入口，
避免在 CLI 或训练脚本中再次手写一套 patch 前向与候选融合循环。
