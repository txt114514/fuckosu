# 单次 validation boundary

## 默认调用链

```text
配置文件/CLI
→ ConfigBoundary（严格一次）
→ V2Config
→ 原始样本/数据集
→ DataBoundary（严格一次）
→ 已验证 VideoFrame / TrainingSample / TrainingBatch
→ ModelInputBoundary（公开模型入口最低 guard）
→ perception
→ TrackingBoundary
→ belief
→ outcome
→ DecisionBoundary
→ action
```

训练 loss 在 `LossBoundary` 对 prediction、label、weight 的 shape、dtype 兼容性、device、finite 和范围做一次完整对齐。loss 内核与多个指标共享可信结果，不为同一 batch 反复同步 GPU 或遍历所有字段。

## 集中 API

`traning.lib.validation.primitives` 提供 `require_int`、`require_real`、`require_bool`、`require_path`、`require_non_empty_str`、`require_enum` 和标量 `require_finite`。`tensors` 提供 `TensorSpec`、`require_tensor`、`require_shape`、`require_same_device`、`require_same_dtype` 和 Tensor finite 检查。`boundaries` 公开 Config/Data/ModelInput/Perception/Tracking/Belief/Outcome/Decision/TrainingBatch/Loss 十个明确边界。

边界返回 `Validated`/可信包装或 canonical state 对象。可信标记不绕过外部 API；只有同一内部链路明确接收边界产物时才省略完整复查。

## 严格内部诊断

默认值不读取模糊布尔字符串：只有环境变量 `TRANING_STRICT_INTERNAL_CHECKS=1` 开启内部诊断。开启后，feature/output dataclass、内部 tracker handoff、belief hidden state 和 outcome 中间结果可以执行额外完整检查；关闭时只保留必要的 shape guard 和 cheap invariant。

## 保留检查及原因

- `ConfigBoundary`：YAML/JSON/CLI 是不可信输入，保留所有类型、范围、schema 和跨字段检查。
- `DataBoundary`：原始 RGB byte length、尺寸、identity、split、坐标 fingerprint 和质量门必须严格。
- artifact/IO：schema、hash、原子提交、checkpoint/context 是恢复安全边界，永远严格。
- public model API：保留 tensor/rank/channel/feature 最低 guard，使独立调用能快速失败。
- `LossBoundary`：prediction/label 只对齐一次；AMP 下允许模型浮点 dtype 与 float32 target 的受支持组合，但必须同 device。
- 状态机：frame 单调、track identity、association coverage、因果时间和 CLICK 引用是业务不变量，不属于可删的重复类型检查。
- 数值内核：空集合、索引范围、除零保护是局部 cheap invariant。

## 已收敛的典型重复点

- 配置和多个领域局部 `_require_int/_require_real/_require_finite` 改用集中 primitive。
- perception target 与 loss 不再各自完整扫描同一批 Tensor。
- belief 同一步不再在 `step` 和 feature 构建两次校验相同 observation。
- runtime 的 Candidate batch 在 perception/tracking handoff 校验一次，association 内核不再第三次全量复查。
- outcome 训练 batch、forward output、loss 和多个指标共享边界结果；默认内部 dataclass 不执行重复 GPU `.item()` finite 同步。

独立公开 metric/calibrator API 仍严格，因为调用者未必来自可信训练链路；内部 trusted kernel 才能免复查。
