# CUDA Optimization Contract

本训练项目默认面向 RTX 4060 Laptop 8GB VRAM。后续新增训练循环、评估循环或模型 smoke
路径时，应复用 `traning.core.memory` 中的统一入口，而不是在各模块里散写 CUDA 开关。

## Runtime Policy

- `configure_torch_runtime`：统一设置 TF32、cuDNN benchmark、matmul precision 和
  channels-last 策略。
- `module_to_device`：把模型迁移到目标 device；CUDA 上按配置切换 channels-last。
- `tensor_to_device`：把 BCHW Tensor 迁移到目标 device；CUDA 上支持 channels-last 和
  non-blocking copy。
- `autocast_context`：根据 `memory.amp_dtype` 选择 FP16/BF16/FP32 autocast。
- `create_grad_scaler`：仅在 CUDA FP16 时默认启用 GradScaler；BF16 默认不需要 scaler。
- `collect_memory_snapshot`：记录 allocated/reserved 显存，用于 profile 和 OOM 报告。

## Default Small-VRAM Choices

默认配置位于 `configs/model_small_vram.yaml`：

- AMP: `auto`，RTX 4060 上优先 BF16。
- `channels_last: true`
- `allow_tf32: true`
- `cudnn_benchmark: true`
- `grad_scaler: auto`
- `compile_model: false`
- `pin_memory: true`
- `patch_batch_size: 1`
- `backward_per_patch: true`

`torch.compile` 默认关闭，因为首版需要优先保证稳定和可诊断；后续可在固定训练阶段单独测速后开启。

## Training Step Rules

训练 step 应遵循：

```python
runtime = configure_torch_runtime(...)
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
