# 训练环境与 CUDA 规则

正式配置默认使用 CUDA。Codex 普通 sandbox 可能看不到 `/dev/nvidia*`，即使开发容器已经
正常挂载 GPU；不要因此重装 PyTorch、改 CUDA 镜像或给生产代码增加 CPU 静默 fallback。

## 主机桥检查

```bash
host-exec docker exec -u dev osu_ai_dev bash -lc \
  'cd /home/dev/workspace && bash src/traning/lib/environment/check_gpu.sh'

host-exec docker exec -u dev osu_ai_dev bash -lc \
  'cd /home/dev/workspace && PYTHONPATH=src python -m traning env-check --config configs/traning.yaml --strict'
```

完整训练同样从主机桥运行：

```bash
host-exec docker exec -u dev osu_ai_dev bash -lc \
  'cd /home/dev/workspace && PYTHONPATH=src python src/start/main.py run --config configs/traning.yaml --device cuda --resume'
```

CPU 仅用于明确的 dry-run 或测试。若要直接运行模型 CPU smoke，配置必须同时设置：

```yaml
runtime:
  device: cpu
  require_cuda: false
  amp: false
```

## 统一 runtime API

新增或修改训练 step 必须复用 `traning.lib.runtime`：

- `configure_torch_runtime`
- `module_to_device`
- `tensor_to_device`
- `autocast_context`
- `create_grad_scaler`
- `collect_memory_snapshot`
- `enforce_runtime_memory_budget`
- `format_oom_guidance`

不要在不同模型或阶段各写一套设备选择、AMP 和 CUDA 开关。

## CUDA step 约束

- `optimizer.zero_grad(set_to_none=True)`；
- CUDA 使用 AMP，float16 时启用 GradScaler，bfloat16 通常不缩放；
- 图像模型使用 channels-last，非图像二维张量不强制转换；
- 启用 TF32 与 cuDNN benchmark；
- DataLoader 使用 pinned memory，GPU copy 使用 non-blocking；
- 不在 step 中长期保留无用 GPU Tensor 列表；
- 不频繁调用 `torch.cuda.empty_cache()`；
- model forward 不做磁盘 IO，Dataset 不产生 UI side effect。

## OOM 处理顺序

1. 根据 `collect_memory_snapshot` 和 `format_oom_guidance` 记录当前峰值；
2. 降低 batch、候选上限或 feature/hidden channels；
3. 缩短一次处理的 sequence/window；
4. 必要时减少输入/patch 规格，避免先用整体 resize 隐藏几何问题；
5. 保留失败 trial 和搜索状态，让下一组合法参数继续，而不是把 OOM 伪装成 passed。

## 常用非 GPU 验证

```bash
PYTHONPATH=src python -m traning config-check --config configs/traning.yaml
PYTHONPATH=src:. python -m pytest -q src/traning/tests
PYTHONPATH=src:. python -m pytest -q src/start/tests
python project_index/build_index.py --check
```
