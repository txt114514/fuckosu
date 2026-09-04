# structure / environment / types 一次性迁移审计

## 审计依据和范围

本审计以用户提供的《迁移完成的硬验收标准》第 21.1 至 21.18 节为准。仓库根目录在执行审计时没有 `CODEX_ONE_SHOT_STRUCTURE_ENV_TYPES_REFACTOR.md` 实体文件，因此没有臆测缺失的第 1 至第 20 节；附件中的硬验收文本是本轮可核验的唯一规范。

审计覆盖 `src/traning`、`src/start`、`src/package`、仓库根 `environment`、项目索引、启动文档、训练/推理调用链以及现有测试。迁移保留本轮开始前工作区中尚未提交的训练调度、课程数据、难例反馈和生产恢复修改，结构移动不改变这些实现的业务语义。

## 迁移前结论

- `traning` 仍以 `app/config/contracts/data/perception/tracking/belief/outcome/decision/training` 等顶层扁平目录作为权威实现，缺少要求的 `main.py`、`__main__.py`、`conf`、`core`、`state`、`lib/environment` 和 `lib/validation`。
- 仓库根 `environment/env_check.py` 是环境检测实现，`start.checks.registry` 直接依赖它；仅设置 `PYTHONPATH=src` 并离开仓库当前目录时会失去该包。
- `environment.env_check.EnvironmentReport` 与旧 `traning.app.environment.EnvironmentReport` 是两个不同结构的同名类型；`Point2D` 也在 `package` 与旧 `traning.contracts` 各有定义。
- 环境 CLI 只输出配置、设备和坐标检查项，没有输出硬标准列出的完整 Python、依赖、Torch/CUDA、GPU 和显存字段。
- primitive 与 Tensor 检查散布在 config、contracts、perception、belief、outcome、evaluation 和 training 中；部分内部链路对同一 Tensor 重复执行 rank、shape、dtype、device 和 finite 全扫描。
- 固定 patch/tiling 实现仍存在于 `traning.lib.data` 的历史工具中，但生产数据集、正式训练和正式推理入口没有导入这些模块。

## 权威边界决定

- 配置权威：`traning.conf`。
- 业务编排与模型权威：`traning.core`。
- 环境、校验、IO、运行时和辅助能力权威：`traning.lib`。
- 训练专用 DTO、Enum、Protocol 和类型注册权威：`traning.state`；真正跨顶层模块复用的几何类型继续由 `package` 定义并在 state 直接复用。
- 旧顶层路径仅作 deprecated wrapper；wrapper 不定义业务类、函数、dataclass 或 Enum。
- 外部 YAML/JSON/CLI、原始帧、artifact、public model API 和 loss 是严格边界。边界内部默认不重新执行同一份完整 Tensor 契约；`TRANING_STRICT_INTERNAL_CHECKS=1` 仅用于额外诊断。

## 重复检查审计与处理分类

迁入 `traning.lib.validation` 的通用检查包括整数、实数、布尔、路径、非空字符串、Enum、有限性、Tensor、shape、同 device 和同 dtype。配置解析与 state DTO 继续调用这些单一 primitive，而不是保留局部同义函数。

默认内部路径关闭的检查包括 feature/output dataclass 对同一 Tensor 的完整 finite 扫描、perception 输出进入 tracking 后的候选全量重扫、belief feature 在同一步中的二次完整检查、outcome loss 前后对同一批 prediction/label 的重复全契约扫描。调试严格模式可恢复诊断检查。

始终保留的检查包括：配置跨字段约束；原始图像字节长度和 frame identity；DataSplit、坐标 transform fingerprint 与数据质量门；artifact schema、SHA-256、checkpoint/context identity；模型公开入口最低 rank/channel/feature guard；loss 边界的一次 prediction/label 对齐；frame 单调性、track id 唯一性、关联覆盖、CLICK 引用和数值算法的索引/空 mask/分母保护。这些检查分别属于不可信输入边界、状态机不变量或必要的 cheap invariant，不受内部严格模式开关影响。

## 类型合并审计

已合并或设为规范别名的硬标准类型由 `TYPE_REGISTRY` 枚举。`RuntimeFrame`、`CandidateObservation`、`TrackLifecycle`、`OutcomeDistribution`、`DecisionAction`、`DecisionResult` 和 `MemorySnapshot` 作为旧名称只指向规范的 `VideoFrame`、`Candidate`、`TrackState`、`OutcomePrediction`、`ActionType`、`ActionPrediction` 和 `MemoryReport` 对象。

共享 `Point2D`、`Size2D`、矩形几何继续由 `package.contracts.geometry` 定义；`traning.state.geometry` 只复用这些对象。训练专用 frame/batch、prediction、outcome、decision、environment 与 telemetry 类型位于 `traning.state`。

没有强行合并的相近类型及原因：`OsuPoint` 是有界 osu! 坐标，不能退化成任意二维点；`FramePixelPoint` 与可越界的投影点具有不同安全语义；不同 artifact manifest 带各自 schema/identity；局部/全局 feature 输出具有不同 stride 契约；配置类的目录职责不同。它们不是同义类型，保留能防止错误空间或错误 artifact 被静默混用。

## 风险控制

最高风险是移动后模块以新旧名称各加载一次而产生 class identity 分裂。兼容 leaf wrapper 因此转发到 canonical module，测试必须断言旧类型与新类型使用 `is` 保持同一对象。第二个风险是把 tagged `Point2D` 的默认坐标空间误用于视频像素；跨坐标边界仍由专用 `OsuPoint`、`FramePixelPoint`、transform fingerprint 和坐标测试保护。第三个风险是以“去重”为由删除 public boundary；本迁移只省略可信内部重复扫描，不降低外部输入、artifact、状态机与 loss 边界。
