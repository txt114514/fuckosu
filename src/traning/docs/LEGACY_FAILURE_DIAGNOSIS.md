# Legacy 准确画面被归入错误模块的证据

## 样本

`item_000001/long_sequence_000008` 的 frame 105–107 在旧输出中位于
`failed/long_sequence`。frame 105 的图片看起来“打击位置很准”，但图片上的
目标、候选和路径覆盖层不等于程序实际执行了点击。

## 原始制品中的一致事实

- `decision/decisions.jsonl` 的 frame 105 动作为 `no_op`，概率
  `0.8981025218963623`；虽然模型还输出了 candidate slot 3，但没有执行动作，
  因此 `predicted_video_xy` 为 `null`。
- `evaluation/trial_score_report.json` 记录 `target_count=1`、`click_count=0`、
  `unresolved_count=1`、`quality_score=0`。所以序列级 `failed` 本身有依据：
  该帧没有真实点击，不能把覆盖层中心当作命中。

## 真正的错误模块归因冲突

同一个 frame 105 在两个旧制品中得到不同归因：

- gallery manifest：`primary_error="decision"`，tag 为 `unresolved_target`；
- `evaluation/attribution.json`：`primary_error="spatial"`，tag 额外包含
  `candidate_match_failed` 与 `nearest_candidate_outside_radius`。

也就是说，旧程序先因 `no_op` 留下 unresolved target，随后另一套归因逻辑又
根据 target-candidate 半径匹配把它改投 spatial；gallery 和优化计划并不共享
一个权威结论。这会把本应检查决策/等待策略的样本送进 spatial hard-example
与 loss 调权，正是“看起来很准却被分到错误模块”的直接原因。

## V2 修复约束

V2 不修改覆盖层来掩盖问题，而是在完整数据流中统一修复：

1. Perception 坐标、Tracking identity、OutcomeOracle 评分和 Decision 动作分别
   产生 typed 结果；覆盖层不得反向定义通过与否。
2. point、slider、sequence 只保留一套 canonical scoring，Oracle、evaluation、
   hard-example attribution 和 gallery 必须消费同一结果对象。
3. “未执行 CLICK”不能因为存在 candidate slot 就当作命中；其失败首先属于
   Decision。只有 Perception/Tracking 的独立门禁明确失败时，才记录对应层的
   次级原因，不能覆盖主因。
4. Phase 9/10 必须用自动化测试固定：同一 evaluation event 在训练调参、遥测和
   gallery 中具有完全相同的 `primary_error` 与 pass/fail。

## 为什么该次训练在进程未被手动关闭时停止

`artifacts/training_runs/20260727T122556Z` 与用户点名的 gallery 属于同一次运行。
它不是因为没有下一组参数而自然完成，而是命中了旧配置中的硬 trial 预算：

- `resolved_config.yaml` 与 level A 的 `resolved_level_config.yaml` 都记录
  `execute_generated_jobs: true`，但同时记录 `max_trials: 2`。
- events 先记录 `ramp.continue ... trial=1/2`，第二轮后明确记录
  `[RAMP][FAILED] level=a trials=2/2`，随后 full flow 以 `RAMP_FAILED` 结束。
- 第二轮的 `evaluation/next_training_job.json` 实际已经存在，说明优化器已经提出
  第三组参数；旧 ramp 只是因为预算为 2 而没有执行它。
- 两轮的聚合 quality score 都是 `0.824 >= 0.8`，但严格结果仍是
  `hits=0, unresolved=88`，因此不能把聚合阈值当成“完全通过”。

该制品还暴露出另一个边界问题：未执行的下一 job 中 `score_threshold=-0.01`，超出
概率阈值范围。Phase 9 的 V2 搜索必须在发布 job 前用 typed parameter spec 校验并
截断/拒绝越界提案；只允许“所有验收门禁通过”成为成功完成，预算耗尽必须发布
明确的 `EXHAUSTED` 状态而不是伪装成完成。无预算限制时，失败 trial 必须继续产生并
执行新的、未重复且合法的参数提案，直到通过、用户中断或出现显式不可恢复错误。

## long_sequence frame 36 坐标偏移的多控制点证据

旧覆盖图把 osu 坐标 `(79.89, 101.22)` 画在视频坐标 `(354, 223)`；这不是该帧
目标中心。legacy 基线提交 `9ed1486` 引入、且由现存独立 ROI 控制点验证的 affine 关系为：

```text
video_x = 2.115860914627143 * osu_x
        + 0.0011971920855575358 * osu_y
        + 242.59057485632047

video_y = 0.0003418231662923798 * osu_x
        + 2.1166805757239477 * osu_y
        + 16.12108357719331
```

代入 frame 36 得到 `(411.75, 230.40)`，即应落在 `(412, 230)` 附近。独立原视频 ROI
控制点 `(80,101)->(412,230)`、`(395,215)->(1078.5,471.5)`、
`(213,179)->(693.5,395)`、`(256,183)->(785.5,404.5)` 与
`(508,237)->(1317.5,517.5)` 的最大残差约 `1.394 px`，低于 4 px 门限。

证据边界必须明确：仓库没有保存该提交所称的完整 passed train/validation 拟合点、
RANSAC inlier 清单或点集摘要；现有 5 个点只能验证方程，无法唯一重建它。V2 因此不再
使用 `pass-sample-ransac-v1` 这个过度声明的身份，而采用
`legacy-control-validated-v1`，并把 `fit_reproducible=false`、控制点、来源和残差门限保存到
`configs/traning_coordinate_evidence.json`。`v2 coordinate-audit` 默认复算控制点；若要求
完整拟合来源，则 `--require-refit-provenance` 会按设计失败，绝不伪造缺失证据。

这组系数不是针对一张图做平移，也不能只写进 renderer。V2 的离线标注适配、训练样本、
Oracle/evaluation、gallery 与命中位置必须共同调用 `package.coordinates` 的公开 affine
transform；原始帧尺寸不匹配时必须显式拒绝或先使用对应 transform，禁止静默复用上述
1484×846 标定。Phase 11 的端到端适配测试会把这些控制点和 frame 36 固定为回归门禁。
