# Codex Refactor Audit: Small VRAM Global/Local Fusion

本审计对应 `CODEX_TASK_small_vram_global_local_fusion.md`。阶段 1 已完成环境检查与补全；本轮在此基础上继续进入小显存全局-局部模型的第一版工程骨架。

## 当前结论

- 实际 Python 包名是 `traning`，当前 CLI 入口是 `src/traning/main.py` 中的 Typer app。
- `src/Traning` 是大小写遗留目录，本轮前序清理中已从工作区移除；后续不应恢复或新增大小写混用导入。
- 当前配置读取集中在 `src/traning/conf/settings.py` 与 `src/traning/conf/config.yaml`，已使用 Pydantic settings。
- 训练数据 schema 是 `video.mp4` + `beatmap.json`，由 `SegmentFrameDataset` 通过 OpenCV 解码视频帧；训练输入不读取音频。
- 当前训练阶段注册表只登记 `data_input`。`spatial`、`candidate_cache`、`temporal`、`evaluation`、`export` 目录是边界占位。
- 现有 patch 能力在 `src/traning/Lib/data/tiling.py`，只提供窗口与 Tensor view，不是任务文档要求的完整 `PatchStream`。
- 当前已有评分、点击序列模拟、错误归因和可视化图集契约。
- 本轮新增了 PatchStream、坐标转换、合成跨 Patch 结构、局部/全局 encoder、全局结构头、grid_sample 融合、空间输出头、因果 GRU、损失函数、CPU feature canvas 和 smoke/profile CLI。
- 候选缓存、正式训练循环、真实标签光栅化、checkpoint 训练恢复和导出器仍待后续实现。
- 当前 Dockerfile 基于 `pytorch/pytorch:2.9.0-cuda13.0-cudnn9-devel`，不得替换 base image，也不得重新安装 torch。

## 可直接复用模块

- `traning.Lib.data.discovery`：扫描 `video.mp4` 与 `beatmap.json`，并支持 item split/filter。
- `traning.Lib.data.dataset`、`sampling`、`collate`、`video_reader`：当前视频帧 Dataset 基础。
- `traning.Lib.data.tiling`：可作为后续 PatchStream 的窗口生成基础。
- `src/package/coordinates.py`：osu 坐标与屏幕/视频坐标转换的稳定 API。
- `traning.Lib.metrics.scoring`、`sequence`：点、slider 与点击序列计分。
- `traning.state.gallery_schema`：trial/frame 评估与错误归因契约。
- `traning.core.visualization` 与 `traning.Lib.visualization`：标注预览和批次图集。
- `before_traning` 的 segment planner 与 ffmpeg 分段输出：用于保证训练集片段来源一致。

## 应修改或新增模块

- 保留 `traning.core.env_check` 和现有 Typer app；本轮继续扩展 `model-smoke`、`memory-profile`、`visualize-patches`、`visualize-fusion`、`train-spatial`、`train-fusion`、`train-temporal`。
- 新增 `traning.data`：`PatchStream`、坐标转换和合成结构。
- 新增 `traning.models`：local/global encoder、全局结构头、融合、空间输出头和因果时序模型。
- 新增 `traning.training`：损失函数和 CPU feature canvas。
- 扩展 `traning.conf.settings` 和 `configs/model_small_vram.yaml`，保留旧 `data_input` 配置兼容。
- 更新 `project_index/build_index.py` 的训练模块导航说明，并重建自动索引。

## 不应在本阶段修改

- 不在本轮实现正式训练循环、候选缓存、checkpoint 恢复或导出器；训练 CLI 目前是显式阶段入口，会报告 optimizer loop 尚未实现。
- 不引入 `flash-attn`、`xformers`、自定义 CUDA extension 或需要编译 CUDA 算子的第三方包。
- 不自动下载预训练权重，不把 `timm`、`huggingface-hub` 作为默认硬依赖。
- 不改 `.devcontainer/docker-compose.yml` 的 GPU 运行参数，除非后续环境检查证明 compose 配置本身有问题。
- 不创建第二套 CLI；`traning.cli` 只复用 `traning.main.app`。

## 缺失依赖

当前重装后容器检查到的缺失包包括：

- 可选视觉依赖：`timm`、`huggingface-hub`

Dockerfile 已安装或已有的关键依赖包括：`torch`、`torchvision`、`torchaudio`、`opencv-python-headless`、`numpy`、`scipy`、`pillow`、`typer`、`pydantic-settings`、`prefect`、`tensorboard`、`ruff`、`pytest`、`mypy`、`av`、`einops`、`optuna`、`safetensors`、`ffmpeg`、`git`、`libglib2.0-0`、`libgl1`。

本阶段 Dockerfile 只追加默认需要的缺失依赖；`timm` 与 `huggingface-hub` 保持可选，不默认安装。

## 包名冲突

- 工程实际包名为 `traning`。
- 遗留 `src/Traning` 不能继续使用；大小写混用在 Linux 下会形成两个不同包，容易导致导入和索引不一致。
- 任务文档示例中的 `python -m traning.cli env-check` 与当前 `python src/traning/main.py env-check` 都应指向同一个 Typer app。

## 兼容性风险

- 当前重装后容器内 `torch 2.9.0+cu130` 与 `torchvision 0.24.0+cu130` 存在，但 `torch.cuda.is_available()` 为 `False`，`nvidia-smi` 无法与 NVIDIA driver 通信；这仍是当前运行环境 GPU 透传或驱动可见性问题，不应通过重装 torch 解决。
- `host-exec` 会以 root 进入主机 namespace 执行命令，权限很高；应只用于显式诊断和主机侧检查，普通代码修改和测试仍在容器内执行。
- 文档给出的全仓库 `grep` 会扫入 `.vscode-server`、训练视频、压缩包和二进制资产；后续检查应限定 `src`、`.devcontainer`、`project_index`、`scripts` 等项目源码范围。
- 当前仓库没有 `pyproject.toml` 或 requirements 文件；依赖管理沿用现有 Dockerfile 的 pip 安装区域。
- 直接从仓库根运行 `python -m traning.cli ...` 需要 `PYTHONPATH=src`，或继续使用 `python src/traning/main.py ...`；当前没有 pyproject 安装入口。
- 全仓库 `ruff check .` 仍有历史遗留：`before_traning` 的 Prefect home 初始化触发 E402，若干旧 import 未使用；本轮新增/修改范围的 `ruff check` 已通过。
- 全仓库 `ruff format --check .` 仍会要求重排大量历史文件；本轮新增/修改 Python 文件和 `project_index/build_index.py` 的格式检查已通过。

## 本阶段实际修改清单

- 扩展 `src/traning/conf/settings.py` 和 `src/traning/conf/__init__.py`
- 新增 `configs/model_small_vram.yaml`
- 更新 `.gitignore`，允许 `configs/` 下的默认模型配置入库
- 新增 `src/traning/data/{patch_stream.py,coordinates.py,synthetic_structures.py}`
- 新增 `src/traning/models/{local_encoder.py,global_encoder.py,global_structure_head.py,gated_sparse_fusion.py,object_heads.py,outputs.py,temporal_model.py}`
- 新增 `src/traning/training/{losses.py,feature_canvas.py}`
- 新增 `src/traning/core/memory.py`
- 扩展 `src/traning/main.py` 的 smoke/profile/visualize/train 阶段命令
- 新增 `src/traning/docs/label_generation.md`
- 新增 Patch、坐标、encoder、fusion、跨 Patch、空间头、时序和 memory smoke 测试
- 更新 `project_index/build_index.py` 并重建 `src/traning/docs/CODEX_INDEX.md`
