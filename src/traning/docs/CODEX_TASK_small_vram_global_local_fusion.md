# CODEX_TASK.md

# Codex 工程修改任务：小显存全局—局部图像特征提取与跨 Patch 融合

本文件用于指导 Codex 直接审计并修改当前仓库。

## 总要求

请直接检查、修改并测试当前工程，不要只给出建议或伪代码。

当前工程已经运行在 CUDA 容器中，并且已有部分 PyTorch 环境。必须在现有工程上增量修改。

### Dockerfile 限制

**不得重构或替换现有 Dockerfile。**

允许的操作只有：

1. 在现有 Dockerfile 中追加缺失的系统依赖安装；
2. 在现有 Dockerfile 中追加缺失的 Python 依赖安装；
3. 合并重复的安装命令，但不得改变原基础镜像；
4. 不得改变 CUDA、cuDNN、PyTorch 基础镜像版本；
5. 不得改变已有用户、UID/GID、工作目录、挂载、ENTRYPOINT、CMD；
6. 不得删除已有依赖；
7. 不得为了安装新包而重新安装 `torch`；
8. 不得修改 `docker-compose.yml`、Dev Container GPU 配置或 NVIDIA Runtime，除非现有配置存在明确语法错误，并且必须在报告中说明。

如果依赖可以通过 `pyproject.toml`、`requirements.txt` 或容器内 pip 安装解决，优先修改 Python 依赖文件，仅在需要 `ffmpeg`、系统动态库等依赖时向 Dockerfile 追加安装项。

---

# 1. 项目目标

将现有 osu! 视频训练工程改造成适合小显存训练的全局—局部视觉结构：

```text
当前高分辨率帧
    ├── 高分辨率重叠 Patch 局部分支
    │       提取点击圆、数字、圆弧、缩圈边缘、slider 局部路径
    │
    ├── 低分辨率完整画面全局分支
    │       提取完整缩圈、长 slider、spinner、跨 Patch 对象关系
    │
    └── 历史帧低分辨率时序分支
            提取缩圈变化、对象出现/消失、按下/保持/释放状态
                         ↓
                全局上下文门控注入
                         ↓
               稀疏跨区域特征融合
                         ↓
                多任务结构预测头
                         ↓
      no_op / press / hold / release + x, y, time
```

主要运行设备：

```text
RTX 4060 Laptop
8 GB VRAM
CUDA 容器
```

主要输入：

```text
约 1484 × 846
60 FPS
```

优先级：

```text
精度 > 训练速度
```

允许串行计算和 CPU 缓存，但不允许通过整体降低当前帧分辨率来替代高分辨率识别。

---

# 2. 必须支持的跨 Patch 内容

跨 Patch 融合不能只处理 slider。

必须同时支持：

- approach circle / 缩圈；
- hit circle；
- slider head；
- slider body；
- slider tail；
- slider repeat point；
- spinner；
- 大半径圆环；
- 被 Patch 边缘切开的圆弧；
- 同一对象分布在多个 Patch 中的局部结构；
- 多个相邻对象发生重叠；
- 点、划、缩圈同时出现；
- 跨帧缩圈半径变化；
- 跨帧 slider 状态变化。

禁止仅使用以下方法作为跨 Patch 主方案：

- 最终检测框 NMS；
- 最终坐标聚类；
- 最终候选列表去重；
- Patch 独立分类后投票；
- 对被切开的结构直接丢弃。

NMS 和聚类只能作为最后的候选去重手段，真正的对象统合必须发生在特征层。

---

# 3. 第一步：审计现有工程

在修改代码前，先检查仓库结构并生成：

```text
docs/codex_refactor_audit.md
```

至少检查：

```bash
find . -maxdepth 5 -type f \
  | sort \
  | grep -v -E '(^|/)(\.git|\.venv|__pycache__|node_modules|build|dist)(/|$)'
```

检查所有 Python 导入：

```bash
grep -R --line-number \
  --exclude-dir=.git \
  --exclude-dir=.venv \
  --exclude-dir=__pycache__ \
  -E '^(from|import) |torch|typer|prefect|pydantic|opencv|av|einops' .
```

重点确认：

1. Python 包实际名称是 `traning`、`Traning` 还是其他名称；
2. 是否存在大小写混用；
3. 当前 CLI 入口；
4. 当前配置读取方式；
5. 当前数据 schema；
6. 当前训练入口；
7. 当前模型结构；
8. 当前 Patch 切分代码；
9. 当前候选提取代码；
10. 当前时序模型；
11. 当前测试目录；
12. 当前 PyTorch、TorchVision、CUDA 版本；
13. Dockerfile 中已经安装的系统依赖；
14. `pyproject.toml` 或 requirements 文件；
15. 是否已有 Ruff、pytest、mypy 配置；
16. 是否已有 Prefect 工作流；
17. 是否已有 DINO、timm、Transformers 或 Hugging Face 依赖；
18. 是否已有视频解码依赖；
19. 是否已有显存监控；
20. 是否已有断点恢复。

审计报告必须列出：

- 可直接复用的模块；
- 应修改的模块；
- 应新增的模块；
- 不应修改的模块；
- 缺失依赖；
- 包名冲突；
- 兼容性风险；
- 最终实际修改清单。

不要在审计完成前批量重命名目录。

---

# 4. 环境补全

## 4.1 PyTorch 与 CUDA

不得重新安装 PyTorch。

不得执行：

```bash
pip install torch
pip install --upgrade torch
pip install torch torchvision torchaudio
```

除非现有容器中根本没有 PyTorch，并且审计报告明确说明；正常情况下基础 CUDA 镜像已经提供 PyTorch。

新增环境检查模块：

```text
environment/env_check.py
```

若实际包名不同，使用工程现有包名。

环境检查至少输出：

- Python 版本；
- PyTorch 版本；
- TorchVision 版本；
- `torch.version.cuda`；
- `torch.cuda.is_available()`；
- GPU 名称；
- GPU Compute Capability；
- cuDNN 版本；
- BF16 是否支持；
- GPU 总显存；
- GPU 当前可用显存；
- FFmpeg 是否存在；
- OpenCV 是否可导入；
- PyAV 是否可导入；
- NumPy 是否可导入；
- SciPy 是否可导入；
- Pillow 是否可导入；
- Typer 是否可导入；
- Pydantic Settings 是否可导入；
- Prefect 是否可导入；
- Optuna 是否可导入；
- timm 是否可导入；
- safetensors 是否可导入。

接入现有 Typer CLI，提供：

```bash
python -m traning.cli env-check
```

如果当前工程使用：

```bash
python main.py ...
```

则应新增：

```bash
python main.py env-check
```

不得创建第二套重复 CLI。

## 4.2 Python 依赖

优先沿用仓库已有依赖管理方式。

若已有 `pyproject.toml`，在其中补全；若只有 requirements 文件，则补全 requirements。

建议核心依赖：

```toml
numpy
pillow
pyyaml
pydantic>=2
pydantic-settings>=2
typer
rich
loguru
opencv-python-headless
av
einops
scipy
safetensors
tqdm
psutil
```

建议训练依赖：

```toml
tensorboard
optuna
```

建议开发依赖：

```toml
pytest
pytest-cov
ruff
mypy
types-PyYAML
```

可选视觉依赖：

```toml
timm
huggingface-hub
```

要求：

- `timm` 和 `huggingface-hub` 必须是可选依赖；
- 默认 smoke test 不能要求联网；
- 默认 smoke test 不能要求下载预训练权重；
- Prefect 不得成为模型导入的硬依赖；
- SQLModel 不得成为模型导入的硬依赖；
- 不添加 Node/npm；
- 不安装 `flash-attn`；
- 不安装 `xformers`；
- 不安装需要自行编译 CUDA 扩展的包；
- 不引入与当前 CUDA 版本不确定兼容的第三方算子。

## 4.3 Dockerfile 允许的最小追加

仅当当前 Dockerfile 缺少以下系统环境时，允许追加：

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    build-essential \
    pkg-config \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*
```

必须先检查是否已经安装，避免重复。

若 Python 依赖由 requirements 或 pyproject 管理，可以在现有安装区域增加：

```dockerfile
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e ".[train,dev]"
```

但必须适配当前 Dockerfile 原有目录结构，不能盲目复制这两行。

若现有 Dockerfile 已经在容器启动后挂载仓库，则不要添加会导致源码被镜像中的旧版本覆盖的 COPY 逻辑。

## 4.4 GPU 检查脚本

新增：

```text
environment/check_gpu.sh
```

内容应等价于：

```bash
#!/usr/bin/env bash
set -euo pipefail

nvidia-smi

python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("torch_cuda:", torch.version.cuda)

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available inside the container")

print("device:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
print("bf16:", torch.cuda.is_bf16_supported())

free_bytes, total_bytes = torch.cuda.mem_get_info()
print("free_vram_gib:", free_bytes / 1024**3)
print("total_vram_gib:", total_bytes / 1024**3)
PY
```

---

# 5. 配置系统

将散落的配置逐步收敛到 Pydantic，但不得一次删除旧配置加载逻辑。

新增或扩展：

```text
traning/conf.py
```

建议结构：

```python
from pathlib import Path

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class RuntimeSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="OSU_AI_",
        env_file=".env",
        extra="ignore",
    )

    target_root: Path
    cache_root: Path = Path(".cache/osu_ai")
    overwrite: bool = False
    continue_on_error: bool = False
    global_offset_ms: float = 0.0
    device: str = "cuda"


class TilingConfig(BaseModel):
    patch_height: int = 512
    patch_width: int = 512
    overlap_y: int = 128
    overlap_x: int = 128
    patch_batch_size: int = 1
    serial: bool = True


class LocalEncoderConfig(BaseModel):
    stem_channels: int = 8
    feature_channels: int = 48
    output_stride: int = 8
    embedding_dim: int = 96


class GlobalEncoderConfig(BaseModel):
    input_height: int = 360
    input_width: int = 640
    feature_channels: int = 64
    backbone: str = "lightweight_cnn"
    pretrained: bool = False
    frozen: bool = False


class FusionConfig(BaseModel):
    mode: str = "gated_sparse_sampling"
    heads: int = 4
    sampling_points: int = 4
    layers: int = 2
    hidden_dim: int = 96


class TemporalConfig(BaseModel):
    enabled: bool = True
    model_type: str = "causal_gru"
    hidden_size: int = 256
    layers: int = 2
    history_frames: int = 8


class MemoryConfig(BaseModel):
    amp_dtype: str = "auto"
    gradient_checkpointing: bool = True
    backward_per_patch: bool = True
    cache_global_features: bool = True
    offload_candidates_to_cpu: bool = True
    max_vram_gib: float = 7.5


class SMETConfig(BaseModel):
    enabled: bool = False
```

必须校验：

- overlap 小于 patch 尺寸；
- patch 尺寸大于 0；
- hidden dim 可被 heads 整除；
- sampling points 大于 0；
- batch size 大于 0；
- `serial=True` 时默认 patch batch size 为 1；
- 默认关闭预训练权重下载；
- 默认关闭 SMET。

配置支持：

- YAML；
- Pydantic 校验；
- CLI 覆盖；
- 默认配置文件。

新增：

```text
configs/model_small_vram.yaml
```

建议默认值：

```yaml
input:
  width: 1484
  height: 846
  resize: false

tiling:
  patch_height: 512
  patch_width: 512
  overlap_y: 128
  overlap_x: 128
  patch_batch_size: 1
  serial: true

local_encoder:
  stem_channels: 8
  feature_channels: 48
  output_stride: 8
  embedding_dim: 96

global_encoder:
  input_height: 360
  input_width: 640
  feature_channels: 64
  backbone: lightweight_cnn
  pretrained: false
  frozen: false

fusion:
  mode: gated_sparse_sampling
  heads: 4
  sampling_points: 4
  layers: 2
  hidden_dim: 96

temporal:
  enabled: true
  model_type: causal_gru
  hidden_size: 256
  layers: 2
  history_frames: 8

memory:
  amp_dtype: auto
  gradient_checkpointing: true
  backward_per_patch: true
  cache_global_features: true
  offload_candidates_to_cpu: true
  max_vram_gib: 7.5

smet:
  enabled: false
```

---

# 6. Patch 流与坐标系统

优先复用已有 Patch 代码。若没有则新增：

```text
traning/data/patch_stream.py
traning/data/coordinates.py
```

实现：

```python
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PatchMeta:
    index: int
    x0: int
    y0: int
    x1: int
    y1: int
    frame_width: int
    frame_height: int
    valid_width: int
    valid_height: int
```

实现：

```python
class PatchStream:
    def iter_patches(
        self,
        frame: torch.Tensor,
    ) -> Iterator[tuple[torch.Tensor, PatchMeta]]:
        ...
```

要求：

- 输入格式有明确约定；
- 推荐内部统一 `CHW`；
- 支持任意宽高；
- 支持边缘 padding；
- 覆盖完整图像；
- 支持 512×512、overlap 128；
- 不允许出现未覆盖像素；
- 不允许重复 Patch 坐标；
- local 坐标可转换为 global；
- global 坐标可查找所属 Patch；
- image 坐标可转换到 feature-grid；
- feature-grid 可转换回 image；
- 单元测试覆盖不能整除的尺寸；
- Patch 生成阶段在 CPU；
- 支持 pinned memory；
- 允许 GPU 非阻塞传输；
- 不在 PatchStream 中调用模型。

---

# 7. 高分辨率局部编码器

新增或改造：

```text
traning/models/local_encoder.py
```

第一版使用纯 PyTorch 轻量 CNN。

建议：

```text
输入 Patch 512×512
    ↓
stride 1 stem，8 channels
    ↓
depthwise separable residual block
    ↓
256×256，16 channels
    ↓
128×128，32 channels
    ↓
64×64，48 channels
    ↓
轻量 FPN
    ↓
stride 8 local feature
```

要求：

- 使用 GroupNorm，不依赖大 batch；
- 使用 SiLU 或 GELU；
- 使用 depthwise separable convolution；
- 使用残差连接；
- 支持 gradient checkpointing；
- 高分辨率阶段通道数必须较小；
- 不使用全局 self-attention；
- 输出 dataclass；
- 不返回无说明 tuple；
- 支持 batch size 1；
- 支持 FP16/BF16 autocast；
- 保留足够细节用于点击圆中心和圆环边缘。

建议输出：

```python
@dataclass
class LocalFeatures:
    dense: torch.Tensor
    pyramid: dict[str, torch.Tensor]
    stride: int
```

---

# 8. 低分辨率全局编码器

新增或改造：

```text
traning/models/global_encoder.py
```

第一版必须提供无网络依赖的轻量 CNN 全局编码器。

输入：

```text
完整当前帧
→ 缩放到 640×360
→ 全局编码器
```

该分支负责：

- 完整缩圈；
- 大圆环；
- 长 slider；
- spinner；
- 多目标关系；
- Patch 之间的对象归属；
- 全局几何上下文。

要求：

- 当前帧局部分支仍使用原始分辨率；
- 低分辨率全图只能作为全局上下文；
- 不能用低分辨率结果直接替代最终精确坐标；
- 输出多尺度特征；
- 默认 backbone 为 `lightweight_cnn`；
- 可选支持 `timm`；
- 可选支持冻结的预训练 encoder；
- 权重下载必须显式启用；
- 离线模式必须能运行；
- 无预训练权重时 smoke test 必须通过。

可选 backbone 名称：

```text
lightweight_cnn
mobilenet_v3_small
convnext_atto
dinov3_external
```

其中 `dinov3_external` 只保留适配接口，不要求 Codex 自动下载或实现完整 DINOv3。

---

# 9. 全局结构头

新增：

```text
traning/models/global_structure_head.py
```

全局结构头不能只输出类别。

至少输出：

```python
@dataclass
class GlobalStructurePrediction:
    objectness: torch.Tensor
    center_heatmap: torch.Tensor
    ring_likelihood: torch.Tensor
    slider_likelihood: torch.Tensor
    spinner_likelihood: torch.Tensor
    coarse_radius: torch.Tensor
    context_tokens: torch.Tensor
```

目的：

- 给局部 Patch 提供“该区域属于哪个全局对象”的信息；
- 为缩圈估计粗略圆心和半径；
- 为 slider 提供粗略整体方向；
- 为 spinner 提供完整结构上下文。

---

# 10. 跨 Patch 特征融合

新增或改造：

```text
traning/models/gated_sparse_fusion.py
```

第一版不要依赖自定义 CUDA deformable attention 扩展。

使用纯 PyTorch 实现以下两步。

## 10.1 全局门控注入

将全局特征采样到局部 Patch 对应位置：

```python
global_context = sample_global_feature(
    global_feature,
    patch_meta,
    local_feature_shape,
)
```

融合：

```python
gate = torch.sigmoid(gate_projection(global_context))
fused = local_feature * (1.0 + gate) + context_projection(global_context)
```

要求：

- 使用 `grid_sample` 或等价纯 PyTorch 算子；
- Patch 与全局特征坐标严格对齐；
- 支持 padding；
- 支持非方形图像；
- 支持非方形 Patch；
- 提供坐标测试。

## 10.2 稀疏跨区域采样

对每个局部查询位置，仅采样少量全局位置。

建议参数：

```text
heads = 4
sampling_points = 4
layers = 2
```

形式：

```text
局部 query
    ↓
预测归一化采样偏移
    ↓
在低分辨率全局特征图上采样 K 个位置
    ↓
加权求和
    ↓
与局部特征融合
```

要求：

- 不创建 `N×N` 全局注意力矩阵；
- 不把所有高分辨率 Token 做 self-attention；
- 使用 `grid_sample`；
- 采样坐标限制在合法范围；
- 支持梯度反向传播；
- 可通过配置关闭；
- 单元测试验证跨 Patch 圆环两侧可以产生关联；
- 单元测试验证长 slider 两端可以通过全局分支关联。

融合输出：

```python
@dataclass
class FusedPatchFeatures:
    dense: torch.Tensor
    patch_meta: PatchMeta
    global_context: torch.Tensor
```

---

# 11. 特征画布与流式显存控制

新增：

```text
traning/training/feature_canvas.py
traning/Lib/runtime/memory.py
```

不要长期保存所有 Patch 的完整计算图。

推荐训练方式：

```text
全局低分辨率分支前向一次
    ↓
串行取 Patch
    ↓
局部编码
    ↓
全局上下文注入
    ↓
局部结构损失
    ↓
逐 Patch backward
    ↓
释放 Patch 激活
```

示意：

```python
optimizer.zero_grad(set_to_none=True)

global_features = global_encoder(global_frame)

patch_count = patch_stream.count(frame)

for patch, meta in patch_stream.iter_patches(frame):
    patch = patch.to(device, non_blocking=True)

    with autocast_context:
        local_features = local_encoder(patch)
        fused_features = fusion(
            local_features=local_features,
            global_features=global_features,
            patch_meta=meta,
        )
        prediction = patch_heads(fused_features)
        loss = compute_patch_loss(prediction, labels, meta)
        loss = loss / patch_count

    scaler.scale(loss).backward()

    del patch
    del local_features
    del fused_features
    del prediction
    del loss

scaler.step(optimizer)
scaler.update()
```

注意：

- 若 `global_features` 需要接收所有 Patch 的梯度，则逐 Patch backward 时要正确处理计算图；
- 第一版可以先冻结全局编码器，避免 `retain_graph=True`；
- 后续训练全局编码器时，可以：
  1. 缓存 detached 全局特征并单独训练；
  2. 对全局分支单独计算全局结构损失；
  3. 分阶段训练；
- 禁止默认每个 Patch backward 都保留同一个全局大图；
- 禁止默认 `retain_graph=True`；
- 禁止一次保存全部 Patch 激活。

推荐训练阶段：

```text
阶段 A：训练局部编码器和局部预测头
阶段 B：训练低分辨率全局结构头
阶段 C：冻结两个编码器，训练融合层
阶段 D：小学习率联合微调
阶段 E：训练因果时序模型
```

---

# 12. 输出头

新增或改造：

```text
traning/models/object_heads.py
traning/models/outputs.py
```

空间模型输出至少包含：

```python
@dataclass
class SpatialPrediction:
    center_heatmap: torch.Tensor
    visible_heatmap: torch.Tensor
    xy_offset: torch.Tensor
    object_type_logits: torch.Tensor
    ring_mask: torch.Tensor
    ring_radius: torch.Tensor
    slider_mask: torch.Tensor
    slider_direction: torch.Tensor
    spinner_mask: torch.Tensor
    candidate_embedding: torch.Tensor
```

对象类型建议：

```text
background
hit_circle
approach_circle
slider_head
slider_body
slider_tail
slider_repeat
spinner
```

不要把 approach circle 直接合并成 background。

## 缩圈输出

至少支持：

```text
center_x
center_y
current_radius
radius_change
time_to_hit
confidence
```

## Slider 输出

至少支持：

```text
head
body mask
tail
local direction
path confidence
object embedding
```

第一版不要求精确恢复完整贝塞尔曲线，但必须保留足够特征，使跨 Patch slider 片段可以归属于同一对象。

---

# 13. 损失函数

新增：

```text
traning/training/losses.py
```

至少包含：

```text
center heatmap loss
visible heatmap loss
xy offset loss
object type loss
ring segmentation loss
ring radius regression loss
slider segmentation loss
slider direction loss
spinner segmentation loss
global-local consistency loss
cross-patch embedding consistency loss
temporal consistency loss
```

推荐：

```python
total_loss = (
    w_center * center_loss
    + w_offset * offset_loss
    + w_type * type_loss
    + w_ring * ring_loss
    + w_radius * radius_loss
    + w_slider * slider_loss
    + w_spinner * spinner_loss
    + w_global_local * global_local_consistency
    + w_cross_patch * cross_patch_consistency
)
```

## 跨 Patch 一致性损失

同一个对象在不同 Patch 中的 embedding 应接近：

```text
same object → embedding distance 小
different object → embedding distance 大
```

可使用：

- supervised contrastive loss；
- cosine embedding loss；
- triplet loss。

第一版优先实现简单且稳定的 cosine embedding loss。

## 缩圈几何约束

对于同一 approach circle：

```text
不同 Patch 中检测到的圆弧
应该对应近似相同的圆心和半径
```

实现几何一致性损失：

```text
center consistency
radius consistency
ring mask consistency
```

---

# 14. 因果时序模型

新增或改造：

```text
traning/models/temporal_model.py
```

输入：

```text
当前空间候选
+
历史隐藏状态
+
历史低分辨率全局特征
```

禁止输入未来帧。

第一版使用：

```text
Causal GRU
```

建议：

```yaml
hidden_size: 256
layers: 2
history_frames: 8
```

输出：

```python
@dataclass
class ActionPrediction:
    action_logits: torch.Tensor
    selected_candidate_logits: torch.Tensor
    x: torch.Tensor
    y: torch.Tensor
    time_offset_ms: torch.Tensor
    next_hidden_state: torch.Tensor
```

动作类别：

```text
no_op
press
hold
release
```

时序模型必须支持：

```python
output, state = model.step(current_features, previous_state)
```

并提供：

```python
state = model.initial_state(batch_size, device)
```

测试必须验证：

- 同一输入前缀得到相同输出；
- 添加未来帧不能改变过去输出；
- reset 后状态清空；
- batch size 1 可运行；
- CPU 和 CUDA 接口一致。

---

# 15. 数据标签扩展

检查现有数据 schema，尽量兼容已有字段。

需要支持：

- 圆心；
- hit circle；
- approach circle；
- 缩圈当前半径；
- 缩圈目标半径；
- time-to-hit；
- slider mask；
- slider head/tail；
- spinner；
- 当前动作；
- object identity；
- Patch 内和全局坐标。

若原始 `.osu` 文件不能直接提供 approach circle 每帧半径，则根据：

- ApproachRate；
- 对象开始时间；
- 当前帧时间；
- osu! 规则；

生成近似监督标签。

必须把公式和假设写入：

```text
docs/label_generation.md
```

禁止把估计值伪装成原始标注。

---

# 16. 合成跨 Patch 测试数据

新增：

```text
traning/data/synthetic_structures.py
```

用于生成无需真实视频即可测试的图像：

1. 跨两个 Patch 的圆；
2. 跨四个 Patch 的缩圈；
3. 位于 Patch 边界的 hit circle；
4. 跨多个 Patch 的曲线；
5. 长直线 slider；
6. 弯曲 slider；
7. spinner；
8. 多个重叠圆；
9. 两个不同圆弧但不属于同一圆；
10. 噪声背景。

合成数据只用于测试和 smoke test，不用于替代正式训练标签。

---

# 17. 必须新增的测试

新增或扩展测试：

```text
tests/test_env_check.py
tests/test_patch_stream.py
tests/test_coordinates.py
tests/test_local_encoder.py
tests/test_global_encoder.py
tests/test_global_sampling.py
tests/test_gated_fusion.py
tests/test_cross_patch_ring.py
tests/test_cross_patch_slider.py
tests/test_spatial_model.py
tests/test_causal_temporal.py
tests/test_memory_smoke.py
```

至少验证：

## Patch

- 1484×846 完整覆盖；
- 1920×1080 完整覆盖；
- 任意奇数宽高；
- 边缘 padding；
- local/global 坐标往返；
- overlap 合法；
- 非法 overlap 抛异常。

## 缩圈

- 圆环分布在四个 Patch；
- 每个 Patch 只能看到局部圆弧；
- 全局分支可看到完整结构；
- 融合后各 Patch 的对象 embedding 更一致；
- 圆心坐标在容差内一致；
- 半径在容差内一致。

## Slider

- slider 横跨多个 Patch；
- slider head 和 tail 不在同一 Patch；
- 全局分支能提供同一对象上下文；
- 融合后同一 slider 的 embedding 接近。

## 显存

在 CUDA 可用时运行 smoke test：

```text
batch size 1
512×512 Patch
640×360 global frame
FP16/BF16
forward
backward
optimizer step
```

记录：

```text
torch.cuda.max_memory_allocated()
torch.cuda.max_memory_reserved()
```

不得要求 CI 一定有 GPU；无 GPU 时跳过 CUDA 测试。

---

# 18. CLI

接入现有 Typer 应用，不创建重复应用。

至少新增：

```bash
python -m traning.cli env-check
python -m traning.cli model-smoke --config configs/model_small_vram.yaml
python -m traning.cli memory-profile --config configs/model_small_vram.yaml
python -m traning.cli visualize-patches --input IMAGE
python -m traning.cli visualize-fusion --input IMAGE
python -m traning.cli train-spatial --config CONFIG
python -m traning.cli train-fusion --config CONFIG
python -m traning.cli train-temporal --config CONFIG
```

若已有 `main.py`，则保留原用法。

CLI 要求：

- 错误信息清晰；
- 文件不存在时立即报错；
- CUDA 不可用时给出明确提示；
- 可选择 CPU smoke test；
- 不静默降级；
- Rich 输出表格；
- 日志写入 `runs/<run_id>/`。

---

# 19. 显存策略

必须落实：

- Patch 串行；
- patch batch size 1；
- AMP；
- gradient checkpointing；
- GroupNorm；
- 小通道高分辨率特征；
- 低分辨率全局分支；
- 稀疏全局采样；
- 禁止高分辨率全局注意力；
- `optimizer.zero_grad(set_to_none=True)`；
- CPU 缓存候选；
- 训练阶段按模块分阶段；
- 可冻结全局分支；
- 可缓存全局特征；
- 避免无必要的 `.clone()`；
- 避免保留 Python list 中的 GPU Tensor；
- 日志只保存 detached 标量；
- 可选 channels-last；
- 数据加载使用 pinned memory；
- GPU 拷贝使用 non-blocking；
- 必须提供显存统计。

不要频繁调用：

```python
torch.cuda.empty_cache()
```

仅允许在阶段切换、评测结束或 OOM 恢复后使用。

---

# 20. OOM 处理

在训练入口中捕获 CUDA OOM，输出：

- 当前 Patch 尺寸；
- global 输入尺寸；
- batch size；
- AMP dtype；
- 最大已分配显存；
- 最大保留显存；
- 当前配置路径；
- 建议的降显存顺序。

建议降显存顺序：

1. 保持原图分辨率；
2. 将局部分支 feature channels 从 48 降到 32；
3. 将 global 输入从 640×360 降到 512×288；
4. 将 fusion hidden dim 从 96 降到 64；
5. 将 fusion layers 从 2 降到 1；
6. 将 sampling points 从 4 降到 2；
7. 开启或增加 checkpointing；
8. 减少 history frames；
9. 最后才减小 Patch 尺寸。

不要自动静默修改配置后继续训练。

---

# 21. 代码质量

运行：

```bash
ruff check .
ruff format --check .
pytest -q
```

尽可能运行：

```bash
mypy traning
```

若历史代码无法一次全部通过 Ruff：

- 不要对整个仓库做无关格式化；
- 只保证新增和修改文件通过；
- 在 `docs/codex_refactor_audit.md` 中列出历史遗留问题。

公共 API 要求：

- 类型标注；
- docstring；
- 明确 tensor shape；
- 明确 dtype；
- 明确 device；
- 明确坐标系；
- 不使用裸 `except Exception: pass`；
- 不静默吞异常；
- 不使用全局可变状态；
- 不在 import 时启动 GPU；
- 不在 import 时下载权重。

---

# 22. 分阶段实施顺序

Codex 必须按以下顺序执行。

## 阶段 1：审计和环境

- 生成审计报告；
- 补全依赖；
- 最小追加 Dockerfile 环境；
- 环境检查；
- GPU 检查脚本；
- Ruff/pytest 基础配置。

## 阶段 2：Patch 和坐标

- PatchStream；
- 坐标转换；
- overlap 融合权重；
- 合成结构数据；
- Patch 测试。

## 阶段 3：局部和全局编码器

- LocalEncoder；
- GlobalEncoder；
- 输出 dataclass；
- CPU smoke test；
- CUDA smoke test。

## 阶段 4：融合

- 全局门控注入；
- 稀疏全局采样；
- 跨 Patch ring 测试；
- 跨 Patch slider 测试。

## 阶段 5：输出头和损失

- 多任务输出；
- 缩圈标签；
- slider 标签；
- 一致性损失；
- 单帧训练 smoke test。

## 阶段 6：时序

- Causal GRU；
- 状态接口；
- no_op/press/hold/release；
- 因果性测试。

## 阶段 7：整合和文档

- 接入 CLI；
- 接入配置；
- 接入 pipeline；
- 运行测试；
- 生成修改报告。

---

# 23. 暂不实现的内容

第一轮不要实现：

- SMET 动态参数拓扑；
- 自定义 CUDA kernel；
- FlashAttention；
- xFormers；
- 完整 DINOv3 下载和训练；
- V-JEPA 训练；
- 全图高分辨率 Transformer；
- 多 GPU；
- 分布式训练；
- 大规模 Optuna 搜索；
- 自动云训练；
- TensorRT；
- ONNX 导出；
- 复杂 UI。

只保留接口或 TODO，不要让这些内容阻塞第一版。

---

# 24. 验收标准

修改完成后必须满足：

## 环境

```bash
python -m traning.cli env-check
```

能够明确显示环境状态。

## 单元测试

```bash
pytest -q
```

新增测试全部通过。

## Ruff

```bash
ruff check <新增和修改文件>
```

通过。

## 模型 smoke test

```bash
python -m traning.cli model-smoke \
  --config configs/model_small_vram.yaml
```

至少完成：

```text
global forward
one patch local forward
fusion forward
prediction forward
loss calculation
backward
optimizer step
```

## 跨 Patch 缩圈测试

合成一个跨四个 Patch 的圆环：

- 单个 Patch 只含局部圆弧；
- 全局分支看到完整圆；
- 融合输出在不同 Patch 中预测相近圆心；
- 融合输出在不同 Patch 中预测相近半径；
- 同一圆对象 embedding 相似。

## 跨 Patch slider 测试

合成一个跨多个 Patch 的 slider：

- head 和 tail 位于不同 Patch；
- 融合后属于同一对象；
- 局部方向连续；
- 对象 embedding 一致。

## 因果测试

历史输入固定时：

- 当前输出不依赖未来帧；
- 加入未来帧不能改变过去输出。

## 显存

在 8GB GPU 上，默认 smoke test 不发生 OOM。

---

# 25. 最终交付文件

Codex 完成后至少交付：

```text
docs/codex_refactor_audit.md
docs/architecture_small_vram.md
docs/environment_setup.md
docs/label_generation.md
docs/codex_changes.md
configs/model_small_vram.yaml
environment/check_gpu.sh
```

以及对应的：

- 模型代码；
- Patch 代码；
- 融合代码；
- 时序代码；
- CLI；
- 测试；
- 依赖配置。

`docs/codex_changes.md` 必须包含：

1. 修改过的文件；
2. 新增的文件；
3. 删除的文件；
4. Dockerfile 追加的内容；
5. 新增依赖；
6. 运行命令；
7. 测试结果；
8. Ruff 结果；
9. CUDA smoke test 结果；
10. 最大显存；
11. 尚未完成的问题；
12. 后续建议。

---

# 26. Codex 最终回复格式

Codex 完成工程修改后，不要只回复“完成”。

最终回复必须包含：

```text
1. 审计结论
2. 实际修改
3. 环境补全
4. Dockerfile 最小追加
5. 模型结构
6. 跨 Patch 融合实现
7. 测试结果
8. 显存测试结果
9. 运行命令
10. 未完成事项
```

若因仓库缺少数据或模型入口而无法完整集成，也必须：

- 完成能够独立运行的模块；
- 完成测试；
- 清楚指出阻塞位置；
- 不得伪造成功结果。
