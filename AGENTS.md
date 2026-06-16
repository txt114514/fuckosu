# Project Instructions

## Engineering Principles

- 修改或新增代码时，优先复用项目现有 API、数据模型和已安装依赖提供的稳定 API；
  只有现有接口无法满足需求时才新增实现，并保持新增逻辑局部、可替换。
- 多个同构步骤、字段映射、状态转换或函数调用应优先定义为分组表、注册表或规格表，
  再由统一循环完成选择、转换和执行；避免用大量并列 `if`、重复字典构造或逐项
  硬编码调用。仅当步骤确有不同控制流或错误语义时才单独实现。
- 被 `src` 下多个顶层模块共同调用的稳定 API 必须放在 `src/package` 中，并通过
  `package` 的公开入口导出；实现按领域拆分为子模块，禁止调用方依赖其内部私有名称。
  仅服务单个顶层模块的局部实现通常保留在该模块内部，不为“可能复用”提前迁入全局包。
- 即使暂时只有一个调用方，当子模块需要持续扩充独立功能、形成稳定数据/API 契约、
  隔离第三方依赖或明确管理跨层边界时，也应将该领域能力迁入 `src/package`；
  原模块只保留业务编排和适配层。

## Code Navigation

- 处理 `src/before_traning` 前，先读全局索引 `project_index/PROJECT_MAP.md`，
  再按其中链接读取 `src/before_traning/docs/CODEX_INDEX.md`。
- 处理 `src/traning` 前，先读全局索引 `project_index/PROJECT_MAP.md`，
  再按其中链接读取 `src/traning/docs/CODEX_INDEX.md` 和训练目标
  `src/traning/docs/TRAINING_PLAN.md`。
- 定位函数优先运行 `python project_index/build_index.py --lookup 符号名`；
  也可搜索对应模块 `docs/CODEX_INDEX.md` 的模块块和精确源码行。
- 不要为理解局部改动重新遍历全部 Python 文件；按 Project Map 的阶段表和影响面扩展阅读。

## GPU Command Execution

- Codex 的普通 `exec_command` sandbox 可能看不到 `/dev/nvidia*`，即使 devcontainer 本体
  已经能正常使用 CUDA；不要因此重装 PyTorch 或修改 CUDA 镜像。
- 需要运行 GPU/CUDA 命令时，优先通过主机桥进入正常容器 namespace：
  `host-exec docker exec -u dev osu_ai_dev bash -lc 'cd /home/dev/workspace && <command>'`
- GPU 可用性验证命令：
  `host-exec docker exec -u dev osu_ai_dev bash -lc 'cd /home/dev/workspace && bash scripts/check_gpu.sh'`
- 训练 CLI 的 CUDA 验证示例：
  `host-exec docker exec -u dev osu_ai_dev bash -lc 'cd /home/dev/workspace && PYTHONPATH=src python -m traning.cli env-check --strict --require-cuda'`
- 后续新增或修改 `src/traning` 训练代码时，必须优先复用 `traning.core.memory` 的 CUDA
  运行时入口：`configure_torch_runtime`、`module_to_device`、`tensor_to_device`、
  `autocast_context`、`create_grad_scaler` 和 `collect_memory_snapshot`。
- CUDA 训练路径默认使用 `optimizer.zero_grad(set_to_none=True)`、AMP、必要时 GradScaler、
  channels-last、TF32/cuDNN benchmark、pinned memory 和 non-blocking GPU copy；不要在训练
  step 中保留无用 GPU Tensor 列表，也不要频繁调用 `torch.cuda.empty_cache()`。

## Index Maintenance

- 修改任何 `src/before_traning/**/*.py` 或 `src/traning/**/*.py` 文件后，必须运行：
  `python project_index/build_index.py`
- 完成修改前必须运行：
  `python project_index/build_index.py --check`
- 两个模块的 `docs/CODEX_INDEX.md` 都是生成文件，不要手工编辑。
- 如果改动影响架构分层、阶段调用链、模块职责、配置字段、状态步骤、文件契约或跨模块影响面，
  同时更新 `project_index/build_index.py` 中对应模块的 Codex 导航内容；
  模块入口变化时再更新 `project_index/PROJECT_MAP.md`。
- 函数/方法/类的新增、删除、改名、移动、签名变化和关键调用变化，都属于必须重建索引的修改。
