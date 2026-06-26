# Environment

## 运行目标

训练环境默认面向 RTX 4060 Laptop 8GB VRAM。核心策略是保留原始图像精度，同时通过
patch 串行、AMP、channels-last 和 CPU offload 控制显存。

普通 Codex sandbox 可能看不到 `/dev/nvidia*`。不要因为 sandbox 看不到 CUDA 就重装
PyTorch 或修改 CUDA 镜像。需要 GPU/CUDA 时通过 host bridge 进入正常容器 namespace：

```bash
host-exec docker exec -u dev osu_ai_dev bash -lc 'cd /home/dev/workspace && bash environment/check_gpu.sh'
```

训练 CLI 的 CUDA 严格检查示例：

```bash
host-exec docker exec -u dev osu_ai_dev bash -lc 'cd /home/dev/workspace && PYTHONPATH=src:. python -m traning.main env-check --strict --require-cuda'
```

## 环境检查

常用命令：

```bash
PYTHONPATH=src:. python -m traning.main env-check
PYTHONPATH=src:. python -m traning.main model-smoke --config configs/model_small_vram.yaml
PYTHONPATH=src python -m pytest src/traning/tests -q
python project_index/build_index.py --check
```

`env-check` 汇总 Python、ffmpeg、nvidia-smi、torch、torchvision、CUDA、GPU 名称、cuDNN、
BF16、显存和依赖导入状态。

## Runtime Policy

新增训练、评估或 smoke 入口必须复用 `traning.lib.runtime`：

- `configure_torch_runtime`
- `module_to_device`
- `tensor_to_device`
- `autocast_context`
- `create_grad_scaler`
- `collect_memory_snapshot`
- `enforce_runtime_memory_budget`
- `format_oom_guidance`

不要在各模块里散写 CUDA 开关。

## 小显存默认配置

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

`torch.compile` 默认关闭，首版优先稳定和可诊断。

## 训练 step 规则

训练路径默认使用：

- `optimizer.zero_grad(set_to_none=True)`
- AMP 和必要时 GradScaler；
- channels-last；
- TF32 和 cuDNN benchmark；
- pinned memory；
- non-blocking GPU copy；
- patch 完成后 detach 并迁移到 CPU。

不要在 step 中长期保存 GPU Tensor list；除阶段切换、评测结束或 OOM 恢复外，不要频繁
调用 `torch.cuda.empty_cache()`。

## GPU / CPU 分工

空间推理中：

- GPU：完整帧 global encoder、逐 patch local encoder、fusion、spatial head。
- CPU：RGB 归一化、颜色 cue、PatchStream、padding、画布融合、NMS、slider 连通域、
  JSON summary 和候选缓存写出。

离线缓存、评估和部署应复用同一分工，避免重复实现 patch 前向和融合循环。

## OOM 处理原则

- 先降低 `patch_limit`、通道数、feature channels 或候选数。
- 再降低 batch/window 相关设置。
- 保持输入原始分辨率作为优先目标，不把整体 resize 当作首选解决方案。
- OOM 报告应包含显存 snapshot、有效预算、当前 patch 规格和建议动作。
