# Codex Refactor Audit: Small VRAM Global/Local Fusion

本审计对应 `CODEX_TASK_small_vram_global_local_fusion.md` 的阶段 1。当前只处理检查与环境补全，不进入 PatchStream、模型结构、融合网络或训练循环实现。

## 当前结论

- 实际 Python 包名是 `traning`，当前 CLI 入口是 `src/traning/main.py` 中的 Typer app。
- `src/Traning` 是大小写遗留目录，本轮前序清理中已从工作区移除；后续不应恢复或新增大小写混用导入。
- 当前配置读取集中在 `src/traning/conf/settings.py` 与 `src/traning/conf/config.yaml`，已使用 Pydantic settings。
- 训练数据 schema 是 `video.mp4` + `beatmap.json`，由 `SegmentFrameDataset` 通过 OpenCV 解码视频帧；训练输入不读取音频。
- 当前训练阶段注册表只登记 `data_input`。`spatial`、`candidate_cache`、`temporal`、`evaluation`、`export` 目录是边界占位。
- 现有 patch 能力在 `src/traning/Lib/data/tiling.py`，只提供窗口与 Tensor view，不是任务文档要求的完整 `PatchStream`。
- 当前已有评分、点击序列模拟、错误归因和可视化图集契约；尚无空间模型、全局编码器、融合网络、候选缓存、时序模型、训练循环和 checkpoint 训练恢复。
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

- 新增 `traning.core.env_check`，输出 Python、PyTorch、CUDA、GPU、FFmpeg 和关键包状态。
- 扩展现有 Typer CLI，增加 `env-check`；新增 `traning.cli` 仅作为同一 app 的 `python -m` 薄入口。
- 最小追加 `.devcontainer/Dockerfile` 缺失依赖，不改变 CUDA/cuDNN/PyTorch 基础镜像。
- 新增 `scripts/check_gpu.sh`，用于 Docker 重建后做 CUDA smoke test。
- 新增 `.devcontainer/bin/host-exec`，作为显式主机侧诊断命令桥。
- 更新 `project_index/build_index.py` 的训练模块导航说明，并重建自动索引。

## 不应在本阶段修改

- 不实现 `PatchStream`、local/global encoder、fusion、temporal model、candidate cache、训练循环或导出器。
- 不引入 `flash-attn`、`xformers`、自定义 CUDA extension 或需要编译 CUDA 算子的第三方包。
- 不自动下载预训练权重，不把 `timm`、`huggingface-hub` 作为默认硬依赖。
- 不改 `.devcontainer/docker-compose.yml` 的 GPU 运行参数，除非后续环境检查证明 compose 配置本身有问题。
- 不创建第二套 CLI；`traning.cli` 只复用 `traning.main.app`。

## 缺失依赖

当前运行容器检查到的缺失包包括：

- 运行/训练依赖：`av`、`einops`、`optuna`、`safetensors`
- 可选视觉依赖：`timm`
- 开发依赖：`pytest`、`mypy`

Dockerfile 已安装或已有的关键依赖包括：`torch`、`torchvision`、`torchaudio`、`opencv-python-headless`、`numpy`、`scipy`、`pillow`、`typer`、`pydantic-settings`、`prefect`、`tensorboard`、`ruff`、`ffmpeg`、`git`、`libglib2.0-0`、`libgl1`。

本阶段 Dockerfile 只追加默认需要的缺失依赖；`timm` 与 `huggingface-hub` 保持可选，不默认安装。

## 包名冲突

- 工程实际包名为 `traning`。
- 遗留 `src/Traning` 不能继续使用；大小写混用在 Linux 下会形成两个不同包，容易导致导入和索引不一致。
- 任务文档示例中的 `python -m traning.cli env-check` 与当前 `python src/traning/main.py env-check` 都应指向同一个 Typer app。

## 兼容性风险

- 当前容器内 `torch 2.9.0+cu130` 存在，但 `torch.cuda.is_available()` 为 `False`，`nvidia-smi`/NVML 初始化不可用；这通常是当前运行环境 GPU 透传或驱动可见性问题，不应通过重装 torch 解决。
- `host-exec` 会以 root 进入主机 namespace 执行命令，权限很高；应只用于显式诊断和主机侧检查，普通代码修改和测试仍在容器内执行。
- 文档给出的全仓库 `grep` 会扫入 `.vscode-server`、训练视频、压缩包和二进制资产；后续检查应限定 `src`、`.devcontainer`、`project_index`、`scripts` 等项目源码范围。
- 当前仓库没有 `pyproject.toml` 或 requirements 文件；依赖管理沿用现有 Dockerfile 的 pip 安装区域。
- `pytest` 当前容器缺失，所以本阶段本地验证继续使用 `python -m unittest`；Docker 重建后再运行 `pytest`。

## 本阶段实际修改清单

- 新增 `src/traning/core/env_check.py`
- 新增 `src/traning/cli.py`
- 扩展 `src/traning/main.py` 的 `env-check` 命令
- 新增 `scripts/check_gpu.sh`
- 新增 `.devcontainer/bin/host-exec` 与 `.devcontainer/HOST_EXEC.md`
- 最小追加 `.devcontainer/Dockerfile` 的系统依赖与 Python 依赖
- 更新 `src/traning/docs/README.md` 的环境检查命令
- 更新索引生成脚本并重建 `CODEX_INDEX.md`
