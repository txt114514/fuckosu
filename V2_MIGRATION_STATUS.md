# OSU Decision Model V2 迁移状态

本文件是 `新修改/CODEX_REFACTOR_INSTRUCTIONS.md` 指定的阶段门禁。阶段必须按顺序完成；当前阶段测试未通过时，后续阶段不得开始。

## 当前状态

- 当前阶段：Phase 11 — cleanup legacy adapters
- 状态：覆盖迁移实现已完成；最终验证结果记录于本文件 Phase 11
- V2 包根：`src/traning`
- legacy 活动代码：已从 `src/traning` 和独立 `src/visualization` 退役，不提供兼容入口
- legacy 冻结参考：`src/traning.zip` 与 Phase 0 golden 证据
- 详细盘点：`src/traning/docs/MIGRATION_INVENTORY.md`

## 阶段门禁

| 阶段 | 内容 | 状态 | 验证 |
|---|---|---|---|
| Phase 0 | legacy freeze + golden baseline | 已通过 | `229 passed, 2 skipped, 29 subtests`；golden `4 passed` |
| Phase 1 | contracts + config + infrastructure | 已通过 | Phase 1 + golden：`24 passed` |
| Phase 2 | data pipeline / cache / quality gate | 已通过 | V2 + golden：`34 passed` |
| Phase 3 | perception migration | 已通过 | V2 + golden：`41 passed` |
| Phase 4 | tracking | 已通过 | V2 + golden：`55 passed` |
| Phase 5 | temporal belief | 已通过 | V2 + golden：`63 passed` |
| Phase 6 | outcome oracle + dataset | 已通过 | V2 + golden：`111 passed` |
| Phase 7 | outcome model + calibration | 已通过 | V2 + golden：`147 passed` |
| Phase 8 | decision / optimal stopping | 已通过 | V2 + golden：`182 passed` |
| Phase 9 | training orchestration / optimization | 已通过 | V2 + golden：`216 passed` |
| Phase 10 | telemetry / visualization | 已通过 | V2 + golden：`250 passed` |
| Phase 11 | cleanup legacy adapters | 已通过 | 2026-09-04 全仓 `396 passed`；训练包 `371 passed`；Ruff、索引、CUDA strict env-check 与真实 GPU smoke 通过 |

## 架构决定

V2 使用单一顶层包 `src/traning`，并已物理覆盖旧训练实现，而不是把
`contracts`、`data`、`training` 等通用名称散落为多个顶层包或保留并行
`osu_v2` 命名空间。领域边界由 `traning` 下的 typed 子包表达；跨顶层模块确需共享的
稳定 API 仍按工程规则迁入 `src/package` 并公开导出。

## Phase 0 基线

- Git 基线：`9ed1486`（`添加注释`）
- 冻结包 SHA-256：`143dce9980b75fc17a2babc3b9434c5ca20dbdee87b241923c6fbdfac6101e28`
- 解压后的 `traning` 源码与工作树比较：排除 `__pycache__`/`*.pyc` 后无差异
- 进入任务前已有改动：`src/traning/lib/metrics/scoring.py` 的空行差异；本阶段不覆盖该用户改动
- Golden 覆盖：Perception recall、candidate geometry、point/slider oracle、final sequence score

Phase 0 门禁于 2026-08-24 通过。V2 实现可进入 Phase 1；任何模型迁移仍需等待 Phase 1 和 Phase 2 各自通过。

## Phase 1 结果

- `src/traning/contracts`：canonical typed contracts；runtime/inference schema 从结构上排除 GT-only 字段。
- `src/traning/config`：单一 V2 配置、严格未知键/类型/schema 校验及 JSON/YAML round-trip。
- `src/traning/infrastructure`：原子 bytes/text/JSON/JSONL 发布、SHA-256、严格 JSON 读取和确定性 seed。
- 门禁：Phase 1 单元测试与 Phase 0 golden 合计 `24 passed`。

Phase 1 门禁于 2026-08-24 通过，现按顺序进入 Phase 2；Perception 仍未开始迁移。

## Phase 2 结果

- Candidate cache 使用不可变 records generation，manifest 最后原子提交；严格校验 schema、dataset/producer identity、SHA-256 与 row count，并拒绝 GT 字段注入。
- Preprocessing metadata 与 dataset catalog 只通过 typed Repository 暴露；SQLite table/column/schema version 均封装在 adapter 内。
- DataQuality 由 registry 统一产生 canonical issue，`report.ok` 只依赖 `blocks_training`；pipeline 不重新解释 severity。
- 门禁：V2 单元测试与 golden 合计 `34 passed`，包括发布失败保留旧 cache、checksum/row mismatch、错误 schema 和 blocking 语义负测。

Phase 2 门禁于 2026-08-24 通过，现按顺序进入 Phase 3。

## Phase 3 结果

- 迁移 Local Encoder、Global Encoder、Gated Fusion 与 Spatial Head；未迁入 dead `GlobalStructureHead`，也未引入 SMET。
- `global_frozen` 与实际 `requires_grad` 一致；未冻结 global 的端到端 loss 反传已验证所有参数获得非零梯度。
- Dense head 以明确 `instance_ids` 做 identity prototype/cosine-margin 监督，不再使用全图 embedding variance。
- runtime 只接受 `RuntimeFrame`，只输出 `CandidateObservation[]`；没有 GT Top-K 注入。
- 候选坐标统一采用 `(cell + 0.5 + offset) * (frame_size / map_size)`，x/y 分别缩放并在像素边界截断；ring 半径也沿同一尺度链映射。
- 门禁：V2 单元测试与 golden 合计 `41 passed`。

Phase 3 门禁于 2026-08-24 通过，现按顺序进入 Phase 4。

## Phase 4 结果

- Association 将归一化空间距离、cosine embedding 距离和物体类型距离按配置权重统一成可解释成本，并以 `(cost, track_id, candidate_id)` 稳定决胜。
- Tracker 维护独立稳定 ID 和 `NEW → ACTIVE → MISSING → EXPIRED` 生命周期；过期事件只输出一次，后续同一候选必须获得新 ID。
- `missed_frames` 明确定义为 tracker 成功处理但目标未匹配的帧次数；原始 `frame_index` 可因抽帧跳号，`time_since_seen_ms` 仍按真实时间戳计算。
- 非法类型、重复 ID、混合帧、时间倒退及 embedding 不一致均在状态变更前硬失败；候选反序、confidence slot 变化和交叉目标不会交换轨迹身份。
- 门禁：V2 单元测试与 golden 合计 `55 passed`。

Phase 4 门禁于 2026-08-24 通过，现按顺序进入 Phase 5。

## Phase 5 结果

- Temporal Belief 使用 projection + 显式多层 `GRUCell`，每个 `track_id` 独立递推；全部层 hidden 都进入 `BeliefState.belief_embedding`，不存在模型私有时序 side state。
- 正式 belief 路径不包含 action/candidate logits，也不导入 GT、future、legacy 或 SMET；训练使用 tensor `forward_step`，runtime 只收 `TrackedObservation`。
- Runtime 对同帧轨迹稳定排序并原子提交状态与时钟；MISSING 保留历史，EXPIRED 当帧输出后清理，遗漏现存轨迹或非法批次不会污染 state。
- 因果门禁同时验证值与梯度：改变 future suffix 不改变任何 prefix belief，且当前 belief 对未来输入梯度严格为零；逐轨迹隔离、输入反序、分段执行和 reset 重放均通过。
- 回补 Perception 零向量边界：identity embedding 与 slider direction 在极端零 head 输出时仍发布确定性单位向量，`CandidateObservation` 拒绝零 embedding，避免 tracking cosine 未定义。
- 门禁：V2 单元测试与 golden 合计 `63 passed`。

Phase 5 门禁于 2026-08-24 通过，现按顺序进入 Phase 6。

## Phase 6 结果

- 将 legacy point、slider path、slider 和 click sequence 的成熟算法迁到唯一 `traning.evaluation` 实现；V2 Oracle、离线 evaluation 与后续 attribution 不再各自复制公式。
- `OutcomeOracle.evaluate` 把 `(oracle state, track, position, horizon)` 映成固定五类标签、归一化 score、validity 与 expiry；同一实例的 `evaluate_sequence` 只委托 canonical sequence scorer。
- Counterfactual builder 按 `sample_id → track_id → horizon` 稳定生成 typed `OutcomeTrainingSample`，输入反序不改变记录顺序或 JSONL bytes。
- Outcome dataset 复用 canonical `ArtifactManifest`，采用 immutable generation + manifest-last；严格校验 schema、dataset/split/producer、oracle/scoring version、row count、SHA-256、重复 ID 与未知/GT 字段，提交失败保留旧制品。
- 已将用户点名样本的原始冲突记录在 `src/traning/docs/LEGACY_FAILURE_DIAGNOSIS.md`：frame 105 实际是高概率 `no_op`，gallery 标 `decision`，旧 attribution 却改投 `spatial`；V2 以后只允许共享 typed 结果作为归因权威。
- 门禁：V2 单元测试与 golden 合计 `90 passed`。

Phase 6 初测于 2026-08-24 达到 `90 passed`；最终审计发现并修复了 sample identity、
split provenance、label invariant 与 unresolved attribution 四项 P0。Counterfactual dataset
现在使用 typed wrapper 绑定 split 与 lineage，sample ID 使用无歧义 length-prefix 编码；
Outcome category 由 canonical contract 唯一定义，slider 起点与阈值边界均有硬校验；
evaluation 只发布 canonical typed attribution，零点击未解决目标不会再被图片后处理误投为空间模块。
最终门禁为 V2 单元/回归测试 `111 passed`，于 2026-08-24 通过，现进入 Phase 7。

## Phase 7 结果

- `DenseOutcomeModel` 使用 belief embedding、规范化 horizon 和显式 CLICK 条件的普通
  dense MLP，输出固定五分类与独立 expiry 概率；没有 SMET、oracle、GT 或 legacy 依赖。
- 五分类分布统一推导 expected score 与 variance；同一 belief 的不同 horizon 具有可学习且
  已验证的分布差异，trunk/category/expiry 所有参数都进入反向图。
- Outcome batch 保留 split 与完整样本 lineage，以分类 CE 和 expiry BCE 为主任务，score
  仅作显式低权重辅助项；优化步骤使用 `zero_grad(set_to_none=True)`。
- canonical metrics 提供 NLL、Brier、top-label ECE、expected-score MAE 和 expiry 指标；
  温度校准只在给定 validation logits 上确定性拟合，并显式包含 T=1，保证 NLL 不退化。
- 门禁：V2 单元/回归测试 `147 passed`，Ruff、compileall 与 diff-check 全通过。

Phase 7 门禁于 2026-08-24 通过，现按顺序进入 Phase 8。

## Phase 8 结果

- `compute_click_utility` 只有一套可审计公式：expected score 减风险、点击、invalid、
  miss 与 expiry 成本；success confidence 直接来自 low/medium/high 概率和。
- `OptimalStoppingPlanner` 只接收同一时刻的 typed `BeliefState[]` 与完整
  `OutcomeDistribution[]`，严格拒绝图像、GT、oracle、action/candidate logits 和缺失预测。
- planner 显式比较 horizon 0 的 `Q_CLICK` 与最小正 horizon 的 `Q_WAIT`；未来收益更高时
  选择 WAIT，效用相同时稳定优先 CLICK_NOW，轨迹/输入反序不改变结果。
- Decision contract 强制 CLICK 为 horizon 0 且其效用不低于 wait；WAIT 必须是正 horizon、
  不绑定目标/outcome 且 selected utility 等于 wait utility。
- 门禁：V2 单元/回归测试 `182 passed`，Ruff、compileall 与 diff-check 全通过。

Phase 8 门禁于 2026-08-24 通过，现按顺序进入 Phase 9。

## Phase 9 结果

- 训练编排先消费 canonical `DataQualityReport`，阻断时 runner 零调用；随后严格按
  Perception → Tracking → Belief → Outcome → Decision → Evaluation 执行。首个 gate
  失败会立即结束当前 trial，再由搜索控制器选择下一组合法 proposal，而不是结束整个程序。
- Evaluation stage 直接发布唯一 typed `TrialAcceptance`；搜索、编排与最终状态共享该
  对象，只有 data/perception/tracking/belief/outcome/decision/golden 七个 gate 全部通过
  才可成为 `PASSED`。
- 参数由统一 registry 校验、clamp 与量化；旧制品中的 `score_threshold=-0.01` 不会被
  发布。seed 确定性搜索不重复提案，`max_trials=None` 时失败后持续选择新参数，直到
  全通过或有限空间显式 `EXHAUSTED`；预算耗尽也只产生 typed error，不伪装成功。
- Curriculum、ASHA 和 hard-example 已迁入 typed 规则与路由层：curriculum 固定
  BASIC → MULTI_OBJECT → COMPLEX → FULL，ASHA 先应用严格 gate，hard-example contract
  只允许 TRAIN。Phase 11 的生产收口进一步把这些规则接入真实资源 job、持久化恢复与训练
  消费路径，不再停留在共享 consumer view。
- 门禁：V2 单元/回归测试 `216 passed`，Ruff、format、compileall 与 diff-check 全通过。

Phase 9 门禁于 2026-08-24 通过，现按顺序进入 Phase 10。

## Phase 10 结果

- typed telemetry 固定为 `metrics.jsonl`、`resources.jsonl`、`evaluation.jsonl` 与
  `events.jsonl` 四通道；统一 schema version、run identity、严格 JSON、线程锁、
  `O_APPEND` 与 `fsync`，损坏尾行、重复键、串台、缺失通道和跨 run 污染均显式拒绝。
- `Reporter → StateStore → Renderer` 已成为正式边界；Store 先耐久化后推进不可变
  snapshot/history，Reporter 不维护第二份 mutable live state，Rich/Qt 只消费带版本的
  `DashboardSnapshot` 并产出 frozen typed view-model。
- 指标覆盖 loss/step/score、perception recall、tracking ID switches、Outcome
  NLL/Brier/ECE/expected-score error、decision utility、wait/click ratio、GPU、VRAM 与
  throughput；GPU ratio 到百分比只在展示规格表统一格式化。
- 契约级 frame 105 集成回归证明 hard-example consumer view、telemetry 与只读 dashboard
  renderer 不复制或改写 canonical `SequenceEvaluationEvent`；零点击 unresolved 始终是
  Decision，展示层不能改投 Spatial。Phase 11 收口又验证了该事件进入持久化 Decision
  权重时仍精确绑定原 sequence/frame，而不是由图片目录反推归因。
- 门禁：V2 单元/集成/回归测试 `250 passed`，Ruff、format、compileall 与 diff-check
  全通过。

Phase 10 门禁于 2026-08-24 通过，现按顺序进入 Phase 11。

## Phase 11 结果

- `src/start/main.py` 保持总启动入口和原始数据准备顺序，但训练部分已直接替换为 V2；
  无参数启动和 `run` 都执行 raw scan → conversion → split → checks → production training
  → report。不存在 `v2` 兼容子命名空间。
- 正式 runtime 只能从经 manifest 校验的三模型 bundle 装配。checkpoint 使用
  generation-first/manifest-last 原子发布，加载时共同校验 dataset identity、三模型契约
  SHA-256、权重 SHA-256、strict state dict 和坐标变换指纹；仅部署目录变化不会误拒绝权重。
- 训练 CLI 直接从真实 typed segment dataset 构建六阶段 runner，不再接收外部
  `module:factory` evaluator。普通阶段或最终 gate 未通过会继续提出合法且未重复参数；
  默认 `max_trials: null` 不再复现 legacy 两轮停止。固定 blocking
  DataQuality 在 runner 构造前终止，避免把不可由参数修复的数据错误变成无限搜索；进入搜索
  执行后或 winning checkpoint 复验期间的不可恢复异常产生 `search.failed`，预算/空间耗尽产生
  `search.exhausted`，全通过产生 `search.passed`。初始化及恢复身份校验可在 telemetry 建立前
  直接失败。
- 训练 target 使用实际 dense rasterizer 与 production decoder 的互逆方程；预测、canonical
  scorer、evaluation event、telemetry 和 production gallery overlay 共用相同原帧尺寸与
  transform fingerprint。PNG 逐帧原子写入，manifest-last 记录其 SHA-256、事件、尺寸和
  坐标指纹。frame 36 固定映射为 `(411.75, 230.40)`；原帧边缘点击逆映射到 playfield 外时
  由 scorer 记为空间 miss，而不是中断批次。
- frame 105 的零点击事实通过唯一 canonical event 保持 `Decision/unresolved`；hard-example
  路由、telemetry 和 gallery 不再根据覆盖层重新归因。GT/candidate 覆盖不能冒充实际 CLICK
  或命中。production gallery 现在按当前帧事件分入 `failed/decision`，而已通过的其他帧仍在
  `passed/none`；TRAIN 事件同时可被持久化为精确 Decision 帧权重，validation/test 仍被
  强制排除。
- 当前 affine 方程经 5 个独立 ROI 控制点验证，平均残差约 `0.449 px`、最大约
  `1.394 px`。审计确认 legacy 提交未保存原始 passed train/validation 拟合集，因此 V2
  使用诚实身份 `legacy-control-validated-v1`，证据制品显式记录
  `fit_reproducible=false`；`coordinate-audit --require-refit-provenance` 会硬失败，不伪造来源。
- Candidate cache、Outcome batch、sequence score/event、telemetry 与 checkpoint 都强制保留
  dataset/artifact/坐标 lineage；旧 schema、错误尺寸、错误 identity、未知字段、摘要漂移和
  跨变换混批均有负向测试。
- 全仓 Python 模块均有中文用途 docstring；`traning` 生产和测试的全部公开定义均有中文
  docstring，并由 AST 架构测试持续门禁。Phase 0 冻结制品不为机械补注释而改写。
- 覆盖迁移加入真实 dataset、Belief/Outcome 训练、production stage runner、原子搜索恢复、
  winning checkpoint 复验和 start 总流程 dry-run。生产收口进一步接通累计 curriculum 数据、
  同 cohort/stage/rung ASHA、增量父 checkpoint、TRAIN-only hard-example 加权消费和 job 级
  恢复账本；普通 prune 后会继续提出未重复参数。2026-09-04 的最终验证为：全仓
  `396 passed`、训练包 `371 passed`（普通沙箱仅有预期 CUDA 可见性 warning）、`ruff check`
  通过、索引一致、start dry-run 通过；RTX 4060 Laptop 上 strict env-check 与真实
  BF16/channels-last Perception forward、backward、GradScaler/AdamW step 通过，最大 PyTorch
  allocation 约 `0.143 GiB`。

Phase 0 → Phase 11 已按顺序完成；legacy 只保留冻结 reference/golden baseline，唯一正式
训练与推理路径为 `src/traning`，唯一总启动路径为 `src/start/main.py`。
