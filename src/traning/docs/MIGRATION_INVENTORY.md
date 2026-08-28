# OSU Decision Model V2 内部迁移清单

本清单是 Phase 0 的输出。分类只描述迁移策略，不代表把 legacy 接口暴露给 V2：

- `KEEP_ALGORITHM`：算法语义可保留，但必须通过 V2 typed contract 重新承载。
- `MIGRATE`：迁移成熟实现并调整职责边界。
- `REWRITE`：旧契约或控制流与 V2 冲突，需要重新实现。
- `DELETE`：V2 正式路径禁止保留。
- `LEGACY_ONLY`：仅留作冻结基线、回归对照或短期迁移适配。

## 模块分类

| 领域 | Legacy 位置 | 分类 | 证据与 V2 处理 |
|---|---|---|---|
| Local Encoder | `src/traning/lib/models/local_encoder.py` | `MIGRATE` | 深度可分离卷积、残差块和 stride 语义成熟；迁入 Perception，输入/输出改为 V2 模型结构。 |
| Global Encoder | `src/traning/lib/models/global_encoder.py` | `MIGRATE` | 多尺度完整帧编码可复用；V2 必须保留 `pretrained`/`frozen` 的显式配置，并测试未冻结时梯度可达。 |
| Gated Fusion | `src/traning/lib/models/gated_sparse_fusion.py` | `KEEP_ALGORITHM` | 位置采样与门控融合属于有效算法；移除对 legacy feature dataclass 的依赖。 |
| Global Structure Head | `src/traning/lib/models/global_structure_head.py`、`stack.py` | `REWRITE` | `build_model_stack` 构建 `structure`，但 spatial trainer 又无条件冻结 `global` 与 `structure`，正式 forward/loss 没有形成可靠闭环；V2 要么接入图并监督，要么删除。 |
| Spatial Prediction Head | `src/traning/lib/models/object_heads.py`、`outputs.py` | `MIGRATE` | 稠密中心、类型、几何头可复用；输出需收敛为 `CandidateObservation[]`，不向 runtime 暴露训练标签。 |
| Spatial target/loss | `src/traning/lib/training/spatial_targets.py`、`losses.py` | `MIGRATE` | 热图与几何监督可用于训练专属路径；旧 embedding 一致性目标不能作为 identity 主监督，改为 instance/temporal identity loss。 |
| Spatial trainer | `src/traning/core/spatial/spatial_trainer.py` | `REWRITE` | 编排、checkpoint 和 CUDA runtime 用法可参考，但第 149 行附近无条件冻结 global/structure 与配置冲突；训练入口改用 V2 配置和 typed batch。 |
| Spatial inference | `src/traning/core/spatial/spatial_inference.py` | `MIGRATE` | patch 流、canvas 融合可迁移；`spatial_candidate_to_dict` 等宽松字典边界不能成为 V2 核心接口。 |
| Candidate geometry/decode | `src/traning/lib/training/spatial_decode.py` | `KEEP_ALGORITHM` | cell-center + offset、像素 NMS、slider path 解码进入 golden baseline；迁入 Perception 并返回 typed observation。 |
| Candidate generator | `src/traning/core/decision/generator.py` | `REWRITE` | 同一 record 同时写 candidates 与由 `hit_objects` 生成的 `temporal_target`，混合 runtime 与 GT；V2 分离训练/推理 record。 |
| Candidate cache schema | `src/traning/state/candidate_cache_schema.py` | `LEGACY_ONLY` | 只维护版本常量，无法证明完整发布；V2 使用 manifest + checksum + row count + atomic publish。 |
| Candidate cache writer/reader | `src/traning/core/decision/generator.py`、`src/traning/core/temporal/dataset.py` | `REWRITE` | JSONL 字典被下游直接解析，schema/version mismatch 存在 fallback；V2 cache repository 必须校验后才返回 typed records。 |
| Temporal model | `src/traning/lib/models/temporal_model.py` | `LEGACY_ONLY` | 因果 GRU 的时间方向可参考，但它产生共享序列的 `action_logits` 与 candidate logits，不是 per-track belief。 |
| Temporal dataset | `src/traning/core/temporal/dataset.py` | `REWRITE` | `_temporal_slot_candidates` 会把 GT `selected_candidate_id` 强制塞进 Top-K（约第 410–444 行），推理候选语义被训练标签污染。 |
| Temporal target generator | `src/traning/core/decision/generator.py::_build_temporal_target` | `LEGACY_ONLY` | 基于 beatmap GT 生成 action/candidate/time/xy imitation 标签；只可作为旧基线或训练侧迁移素材，不能进入 runtime。 |
| Temporal trainer | `src/traning/core/temporal/trainer.py` | `REWRITE` | 主要优化 action imitation、selected candidate CE、xy/time 回归；V2 改为 per-track belief 训练，imitation 最多为 auxiliary。 |
| Tracking / association | Legacy 无独立模块 | `REWRITE` | 旧 pipeline 依赖每帧 candidate slot，没有稳定 `track_id` 或生命周期；V2 新建确定性 association 与 per-track 状态。 |
| Decision runner | `src/traning/core/decision/runner.py` | `DELETE` | `_decision_row` 对 `action_probs.argmax()` 和 candidate logits `argmax()` 做正式决定；与 optimal stopping 明确冲突。仅冻结在 legacy 中。 |
| Point/slider oracle | `src/traning/lib/metrics/scoring.py` | `KEEP_ALGORITHM` | `score_point`、`score_slider` 已有连续空间/时间与路径判定并进入 golden；迁为唯一共享 oracle，不复制公式。 |
| Sequence oracle | `src/traning/lib/metrics/sequence.py` | `KEEP_ALGORITHM` | `score_click_sequence` 已处理频率限制、目标一次消费和错误归因；迁为同一 oracle 的序列入口。 |
| Trial evaluator/gallery | `src/traning/core/optimization/scoring/*` | `MIGRATE` | 成熟错误归因与样本导出可复用；输入改为 V2 `DecisionResult`/oracle report，禁止从 cache 中读取 GT 来修补 runtime 输出。 |
| Canonical dataset split | `src/package/dataset_split/*`、`src/package/contracts/dataset/*` | `MIGRATE` | 已是跨模块稳定 API；V2 contract 复用单一 `DataSplit`，不再维护字符串别名或隐式 fallback。 |
| Manifest / SQLite status | `src/before_traning/state/*`、`src/traning/lib/data/preprocessing_metadata.py` | `REWRITE` | 训练层当前直接 `sqlite3.connect` 并理解表结构；V2 repository 隐藏 SQLite，返回领域对象而非 row。 |
| Dataset import / preflight | `src/traning/core/dataset_import/*` | `MIGRATE` | discovery/loader 算法可迁移；`DataInputReport.issues` 目前把 severity/blocking 编进字符串，改为 canonical `DataQualityIssue`。 |
| Data quality semantics | `src/traning/core/dataset_import/preflight.py`、`full_flow/stages.py`、`training_ramp.py`、`src/visualization/*` | `REWRITE` | severity、blocking 和 UI 展示由多层各自解释；V2 只在领域层判定 blocking，UI 只读。 |
| Full training pipeline | `src/traning/core/decision/pipeline.py`、`full_flow/orchestrator.py` | `REWRITE` | 当前把 spatial→candidate cache→temporal imitation→decision argmax 串成正式路径；V2 按 Perception→Tracking→Belief→Outcome→Decision 重组。 |
| Curriculum | `src/traning/core/optimization/parameter_search/curriculum.py` | `MIGRATE` | 分阶段难度规格可保留，适配 V2 objective 和 artifact contract。 |
| ASHA / parameter search | `src/traning/core/optimization/parameter_search/{planner,executor,objectives}.py` | `MIGRATE` | 试验规划与 early-stop 机制可复用；完成条件必须是全套验收通过，不能因进程仍开启却无候选参数而静默结束。 |
| Hard-example mining | `src/traning/core/optimization/{attribution,parameter_search/hard_examples.py}` | `MIGRATE` | 归因和权重算法可保留；样本身份改用 canonical key/track，并禁止泄漏验证/测试 split。 |
| Checkpoint/artifact | `src/traning/core/model_export/artifact.py`、`training_inheritance/*`、`state/versioning.py` | `MIGRATE` | 版本检查和继承诊断成熟；统一为 `ArtifactManifest`，schema/hash 不匹配必须硬失败。 |
| Telemetry reporter/store | `src/visualization/lib/reporter.py`、`state/*` | `MIGRATE` | 事件存储与快照可复用；新增 perception/tracking/belief/outcome/decision 分层事件，单向 reporter→store。 |
| Rich dashboard | `src/visualization/core/renderers/rich_renderer.py`、`panels/*` | `MIGRATE` | 仅消费 V2 telemetry projection；不得反向修改训练状态或重新判定质量门禁。 |
| Qt GUI | `src/visualization/core/gui*.py` | `MIGRATE` | 保留只读展示与生命周期管理；不导入训练内部模型/SQLite schema。 |
| Environment checker / CUDA runtime | `environment/*`、`src/traning/lib/runtime/*` | `MIGRATE` | 环境诊断与 CUDA 辅助 API 成熟；训练路径继续复用 runtime API，V2 配置负责严格 CUDA 门禁。 |
| SMET / DynamicSparseLinear | `src/traning/lib/models/smet.py` | `LEGACY_ONLY` | V2 Outcome baseline 明确使用 dense MLP；不把动态稀疏层带入 Outcome 或 Decision。 |
| Legacy action output | `src/traning/lib/models/outputs.py::ActionPrediction` | `DELETE` | `action_logits` 不能作为正式 runtime contract；V2 以 `OutcomeDistribution` 与 `DecisionResult` 取代。 |

## 已确认的 P0 风险

1. **GT leakage**：`TemporalCandidateWindowDataset` 的候选槽位集合依赖 GT selected id；这不是单纯的 loss 标签，而是改变了模型可见输入。
2. **错误决策边界**：`run_temporal_decision` 把动作分类概率最大项直接发布为动作，无法显式比较 CLICK 与 WAIT 的未来价值。
3. **状态身份缺失**：候选 ID 是帧内槽位/缓存 ID，没有跨帧稳定 track lifecycle，GRU 隐状态因而无法表达独立目标 belief。
4. **冻结配置失效**：spatial trainer 无条件冻结 global 与 structure，覆盖了 `global_encoder.frozen=False` 的用户意图。
5. **dead/孤立 head**：GlobalStructureHead 被构建和冻结，但没有证据表明其输出参与主 spatial forward/loss。
6. **缓存完整性不足**：legacy schema 版本常量不能检测截断、错行数或内容篡改。
7. **质量语义分裂**：严重度与 blocking 以字符串、stage flag、UI severity 多种形式存在，容易把应阻塞的数据错误降级为展示警告。
8. **持久层泄漏**：训练数据读取函数直接打开 preprocessing SQLite，调用方因而依赖旧表结构。

## Phase 0 门禁证据

- Legacy 源码冻结清单：`src/traning/legacy/legacy_freeze.json`
- Golden fixture：`src/traning/tests/regression/fixtures/legacy_golden_v1.json`
- Golden test：`src/traning/tests/regression/test_legacy_golden_baseline.py`
- 任务开始时全仓测试：`229 passed, 2 skipped, 29 subtests passed`
- Phase 0 golden tests：`4 passed`

后续阶段不得修改这些 expected 值来掩盖行为变化；若 oracle 规格本身升级，必须新增版本化 fixture，并保留旧版本回归。
