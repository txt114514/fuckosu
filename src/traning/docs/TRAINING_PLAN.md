# osu! 视频识别与点击决策训练方案总结

## 1. 项目目标

本项目希望从 osu! 游戏视频中逐帧识别当前应执行的操作，并输出：

- 当前帧是否需要操作；
- 操作位置 `x, y`；
- 操作状态：`no-op / press / hold / release`；
- 必要时输出目标类型：`circle / slider / spinner`；
- 对 slider 进一步输出路径、持续时间和重复状态。

实际运行时只能逐帧读取当前画面，因此模型必须采用因果、流式结构：

```text
当前帧 + 上一时刻状态
        ↓
空间识别与候选提取
        ↓
时序状态更新
        ↓
输出当前动作和坐标
```

模型不能依赖未来帧。

## 2. 核心约束

### 2.1 输入分辨率

原始视频约为：

```text
1484 × 846
60 FPS
```

为了保留圆点边缘、数字、slider 路径和重叠关系，训练阶段不整体缩放输入图像。

### 2.2 显存

主要训练设备：

```text
RTX 4060 Laptop
8 GB VRAM
```

训练显存的主要来源不是参数本身，而是中间激活、反向传播计算图和优化器状态。

### 2.3 速度与精度取舍

当前目标：

```text
优先保证精度，以时间换空间
```

可以接受训练较慢，但需要：

- 保留原始像素分辨率；
- 降低峰值显存；
- 支持较长时序；
- 最终能够部署为逐帧推理模型。

## 3. 总体架构

```text
原始视频帧
    ↓
空间识别模块
    ↓
候选目标列表
    ↓
全局时序决策模块
    ↓
no-op / press / hold / release + x,y
```

推荐完整流程：

```text
原始帧 1484×846
    ↓
重叠分块
    ↓
patch 串行空间识别
    ↓
局部热力图、坐标、类型、embedding
    ↓
映射回全图坐标
    ↓
候选去重与 Top-K
    ↓
流式时序模型
    ↓
当前动作判断
```

## 4. 激活图显存优化：原图分块串行训练

### 4.1 基本方法

将完整帧切成多个重叠 patch：

```yaml
patch_width: 512
patch_height: 512
overlap_x: 96-128
overlap_y: 96-128
patch_batch_size: 1
```

每次 GPU 只处理一个 patch：

```text
patch 1 → 前向 → 反向 → 释放
patch 2 → 前向 → 反向 → 释放
...
```

峰值显存接近单个 patch 的训练显存，而不是整帧或全部 patch 的总和。

### 4.2 必须使用重叠

不能完全硬切，因为：

- circle 可能位于边界；
- slider 可能跨 patch；
- 点划重叠可能被截断；
- 边缘区域缺少上下文。

建议重叠宽度为 `96～128 px`。

### 4.3 patch 输出

每个 patch 不长期保存大型深层特征，只保留小型结果：

```python
{
    "heatmap": ...,
    "xy_offset": ...,
    "type_probability": ...,
    "candidates": [
        {
            "x": local_x,
            "y": local_y,
            "score": confidence,
            "embedding": [...]
        }
    ]
}
```

局部坐标映射为全局坐标：

```text
x_global = patch_offset_x + x_local
y_global = patch_offset_y + y_local
```

### 4.4 重叠区域融合

可采用：

- Hann window 加权；
- 高斯权重；
- 热力图加权平均；
- 坐标聚类；
- NMS 去重。

patch 中心权重大，边缘权重小。

### 4.5 逐 patch 反向传播

不应先保留所有 patch 的计算图再统一 backward。推荐：

```python
optimizer.zero_grad(set_to_none=True)

for patch in patches:
    with torch.autocast("cuda", dtype=torch.float16):
        output = model(patch)
        loss = compute_patch_loss(output) / len(patches)

    scaler.scale(loss).backward()

scaler.step(optimizer)
scaler.update()
```

这样每个 patch 完成 backward 后，其大部分激活可以释放。

### 4.6 跨 patch slider 统合

首版需要支持一条 slider 跨越多个 patch，但不考虑两条 slider 路径交叉或接触后产生
分叉的情况。

支持范围：

- 单条 slider 跨多个 patch；
- 多条彼此不接触、中心线连通域相互独立的 slider；
- slider head 和几何尾端位于不同 patch。

不支持范围：

- 两条 slider 中心线交叉；
- 两条 slider 路径接触后形成同一连通域；
- 需要在分叉点判断实例归属的路径。

交叉或接触样本应在 target 构建或数据检查阶段过滤，不进入首版训练和正式评估。

每个 patch 不独立输出完整 slider，而输出局部稠密信息：

```python
{
    "slider_path_logits": ...,
    "slider_head_logits": ...,
    "slider_orientation": ...,
}
```

方向场使用无方向切线 `(cos 2θ, sin 2θ)`，避免相邻 patch 对同一条线给出相反行进
方向时在融合中互相抵消。当前范围不需要 `slider_instance_embedding`，也不需要交叉点
图匹配。

训练标签必须先在完整视频坐标系生成，再裁剪到各 patch：

```text
osu! slider path
→ 视频全局坐标
→ 曲线插值与中心线光栅化
→ 全局 head heatmap 和 orientation
→ 按 patch 窗口裁剪标签
```

推理时按以下流程统合：

```text
patch 局部稠密预测
→ 使用 Hann/高斯权重写回 CPU 全局画布
→ 全局 path/head/orientation 加权融合
→ path 阈值化
→ 按连通域分离互不接触的 slider
→ 中心线细化
→ 从 head 沿 orientation 追踪单条无分叉路径
→ 等距采样为固定长度 polyline
```

`512 × 512` patch 与 `128 px` overlap 可视为每侧约 `64 px` halo。融合时优先信任
patch 中心区域，降低边缘截断造成的路径断裂。

路径恢复后执行结构校验：

- 每个 slider 连通分量只能关联一个 head；
- 中心线除端点外不应出现度数大于 2 的分叉节点；
- 出现分叉、交叉或多个 head 竞争同一连通分量时，将样本标记为超出支持范围并跳过。

## 5. 空间模型设计

### 5.1 保留原始输入像素

输入 patch 不缩放，例如：

```text
512×512 原始像素 patch
```

但网络内部允许适度降采样：

```text
512×512
→ 浅层 stride 1
→ 主体 stride 2
→ 深层 stride 4/8
→ 多尺度融合
→ stride 2 热力图
→ 局部 offset 精修
```

最终坐标通过热力图单元和 offset 恢复：

```text
x = 2 × (cell_x + Δx)
y = 2 × (cell_y + Δy)
```

计划中的高精度空间输出目标为 `stride=2`。当前工程第一版小显存骨架先使用
`SmallLocalEncoder.output_stride=8`，对于 `512 × 512` patch 输出 `64 × 64`
dense target/prediction，用于打通真实标签光栅化和串行 patch 训练；后续候选精修网络再降到
`stride=2/1`。

目标版固定空间输出 `stride=2`。对于 `512 × 512` patch，输出热力图为
`256 × 256`，已能提供接近原像素级的定位。不要让整个深层主干保持 stride 1；
如果后续显存仍有余量，只允许候选局部精修网络输出 stride 1。深层主干仍保持
stride 4/8 的语义特征，再通过多尺度融合供 stride 2 主输出使用。

### 5.2 色号和数字辅助通道

打击目标的颜色集合有限，且内部数字/白色纹理稳定。当前工程提供可开关的
`input.color_cues: osu_basic` 输入增强：

- 有限色号/高饱和目标响应；
- 白色数字和内纹响应；
- 目标相关边缘响应。

这些 cue 追加到 RGB 后作为模型输入通道，不直接替代模型预测。这样可以降低背景颜色干扰，
同时保留网络对换 skin、相近背景色和遮挡情况的学习能力。

### 5.3 高分辨率少通道

推荐：

```text
512×512：4～8 通道
256×256：8～16 通道
128×128：24～48 通道
```

避免高分辨率特征长期维持大量通道。

### 5.4 推荐模块

- depthwise separable convolution；
- residual block；
- GroupNorm；
- SiLU；
- FPN 或轻量多尺度融合。

### 5.5 空间输出

```python
{
    "visible_heatmap": ...,
    "center_heatmap": ...,
    "xy_offset": ...,
    "object_type": ...,
    "candidate_embedding": ...,
    "slider_path_logits": ...,
    "slider_head_logits": ...,
    "slider_orientation": ...,
}
```

空间模型负责发现全部可见候选，不直接决定是否点击。首版 slider 不输出 instance
embedding；跨 patch 路径在全局稠密画布融合后恢复。

### 5.6 两轮扫描与条件复查

空间推理采用级联方式，用额外计算时间换取较低峰值显存和更高局部精度。

第一轮是完整全图扫描：

```text
所有 512 × 512 重叠 patch
→ stride 2 空间模型
→ 局部热力图、Top-K 候选和 slider 稠密图
→ CPU 全局融合
```

第一轮必须覆盖所有 patch，负责召回全部潜在目标和重建跨 patch slider。

第二轮是候选区域精修：

```text
全局候选
→ 从原始帧裁剪 128 / 192 / 256 像素局部区域
→ 更深的局部精修网络
→ 更新中心、类型、可见性、遮挡和 slider 局部方向
```

精修裁剪尺寸按候选尺度从规格表选择，不为不同类型维护重复控制流。精修模型可以使用
stride 1 输出，但只处理少量候选区域。

局部精修 stride 由训练前显存 dry-run 决定：

```text
用最大精修 crop 和目标 batch 做前向/反向 dry-run
→ 峰值显存低于预算：refiner stride 1
→ 超出预算或 OOM：refiner 回退 stride 2
→ 将最终选择写入 trial architecture 参数并冻结
```

同一 trial 训练过程中不得根据某个 batch 动态切换 stride，否则 checkpoint、评估指标和
候选坐标精度不可直接比较。

第三轮是条件歧义复查，不对所有候选执行。触发条件包括：

- 两个候选置信度接近；
- 点与 slider 视觉重叠；
- 重复点位于相同坐标；
- slider 全局融合后出现低连续性或端点置信度不足；
- 当前动作置信度低于复查阈值。

跨 patch slider 本身不自动触发第三轮。它先通过全局稠密图融合处理，只有融合结果不连续
或结构校验失败时才复查。已明确不支持的 slider 交叉/接触分叉样本仍直接过滤，不进入
第三轮尝试消歧。

级联输出覆盖规则：

```text
第一轮提供候选召回和全局路径
第二轮覆盖候选坐标、类型与局部属性
第三轮只覆盖被标记为歧义的字段
全局 slider polyline 始终以第一轮全局画布为主
```

## 6. GPU 与 CPU 协同

### GPU 负责

- 当前 patch 的空间模型前向；
- 当前 patch 的反向传播；
- 小型时序模型训练；
- 当前活跃参数优化。

### CPU 负责

- 视频解码；
- patch 切分；
- 已完成 patch 的结果缓存；
- 全图候选融合；
- NMS、聚类和排序；
- 长期候选序列存储；
- 数据预取；
- 必要时保存优化器状态或非活跃参数。

### patch 结果转移到 CPU

每个 patch 完成后，GPU 只允许短暂保留当前 patch 的计算图。反向传播或推理结束后，只把
下列精简结果转移到 CPU：

- 局部中心/可见热力图；
- Top-K 候选坐标、分数和类型概率；
- 小型候选 embedding；
- slider path/head/orientation 局部稠密图。

```python
candidate_cpu = {
    "xy": candidate_xy.detach().cpu(),
    "score": score.detach().cpu(),
    "embedding": embedding.detach().cpu(),
    "type_probability": type_probability.detach().cpu(),
}
```

然后删除 GPU 中间结果：

```python
del patch_features
del patch_output
```

禁止为等待整帧融合而保存所有 patch 的深层 backbone/FPN 特征图。CPU 全局画布只保存
输出 stride 下的累积图、权重和精简候选。以每 patch 16 个候选、每候选 128 维为例，
整帧候选缓存远小于深层特征图。

### 推荐流水线

```text
CPU 解码下一帧
    ↓
CPU 切分 patch
    ↓
GPU 处理当前 patch
    ↓
CPU 接收上一 patch 输出并融合
    ↓
GPU 继续处理下一 patch
```

可使用 pinned memory、non-blocking copy、数据预取和独立 CUDA stream。

## 7. 空间与时序分阶段训练

### 7.1 阶段一：空间模型训练

输入单帧 patch，输出：

- 候选中心；
- 坐标偏移；
- 类型；
- embedding；
- slider 局部路径信息。

特点：

- 不展开长时序；
- 逐 patch backward；
- 原始像素分辨率；
- 重点优化检测与定位。

当前实现状态：`train-spatial` 已接入 `build_spatial_loss_targets` 和
`run_spatial_training`，采用全帧 global encoder 冻结、patch 串行 local/fusion/head
反传的首版训练路径。

候选解码状态：`SpatialPredictionCanvas` 已能把每个 patch 的 center、visible、type、
ring、slider、spinner 和 embedding 输出写回全图概率画布；`decode_spatial_candidates`
已提供 Top-K、局部最大值和半径 NMS 的首版点候选解码。slider 连通域、中心线细化和候选缓存
仍属于下一阶段。

### 7.2 离线缓存候选

空间模型训练完成后，离线处理全部视频帧：

```text
视频
→ 每帧分块串行识别
→ CPU 全局融合
→ 候选局部精修
→ 条件歧义复查
→ 保存候选序列
```

示例：

```python
{
    "timestamp_ms": 600.0,
    "candidates": [
        {
            "x": 508.1,
            "y": 237.3,
            "score": 0.98,
            "type_logits": [...],
            "embedding": [...]
        }
    ]
}
```

### 7.3 阶段二：时序决策模型训练

输入：

```text
[T, K, D]
```

其中：

- `T`：时间长度；
- `K`：每帧最大候选数；
- `D`：候选特征维度。

输出：

```python
{
    "action": "no_op / press / hold / release",
    "selected_candidate": ...,
    "time_offset": ...
}
```

推荐：

```yaml
sequence_length: 128-512
max_candidates_per_frame: 16-32
candidate_dim: 64-128
```

## 8. 流式时序模型

实际运行接口：

```python
output, state = model.step(frame_candidates, state)
```

不能依赖未来帧。

可选模型：

- GRU / LSTM；
- causal Transformer；
- state-space model；
- 候选间 attention + 因果状态。

第一版建议：

```yaml
type: GRU
hidden_size: 128-256
layers: 1-2
```

复杂后可升级为因果 Transformer。

## 9. 动作输出设计

正式动作类别：

```text
0: no-op
1: press
2: hold
3: release
```

普通点：

```text
no-op → press → release → no-op
```

Slider：

```text
no-op → press → hold → hold → release
```

重复点：

```text
press → release → press → release
```

坐标损失只在有效动作时计算。

## 10. SMET 参数优化

### 10.1 作用

SMET 属于动态稀疏训练：

```text
预定义较大的潜在参数空间
→ 训练时只激活部分连接
→ 定期剪除不重要连接
→ 激活新的连接
→ 活跃参数预算保持固定
```

它主要节省：

- 参数；
- 梯度；
- Adam 一阶矩；
- Adam 二阶矩；
- 参数相关训练显存。

它不直接解决高分辨率激活图显存。

### 10.2 适用模块

适合：

- 候选 embedding 投影；
- 大型 MLP；
- GRU/LSTM 大矩阵；
- Transformer；
- 候选交互层；
- 动作专家；
- 大型 1×1 通道混合层。

不优先用于：

- 第一层高分辨率卷积；
- 小型 depthwise convolution；
- 很小的动作头。

### 10.3 推荐配置

```yaml
smet:
  enabled: true
  pattern: block
  block_size: 16
  density: 0.25
  topology_update_interval: 250
  prune_regrow_ratio: 0.10
  new_parameter_warmup_steps: 10-20
```

推荐结构化稀疏单位：

- 通道；
- 16×16 参数块；
- attention head；
- residual block；
- expert；
- temporal block。

### 10.4 渐增密度扩展

可在原始 SMET 基础上增加：

```text
初始密度较低
→ 模型长期欠拟合
→ 增加活跃密度
→ 新生参数使用局部 warm-up
```

示例：

```yaml
initial_density: 0.25
density_increment: 0.10
maximum_density: 1.00
growth_patience: 5
```

模块级增长：

```text
单点定位不足 → 增加空间模块密度
多点/重复点不足 → 增加时序模块密度
slider 路径不足 → 增加路径分支密度
```

## 11. 推荐搜索与课程组合

首版不自行实现搜索算法，采用：

```text
TPE / 随机参数生成
→ ASHA / Successive Halving 阶段剪枝
→ Curriculum Learning 逐级增加数据难度
→ Hard Example Mining 提高失败样本权重
→ 少量候选完整训练与模拟排名
```

TPE 负责利用历史 trial 生成参数，ASHA 负责控制训练预算，课程学习负责定义晋级难度，
难例挖掘只改变训练采样分布，不替代独立评估集。

### 11.1 三层参数搜索

必须分开管理三类参数：

| 层级 | 参数示例 | 是否重新训练 |
|---|---|---|
| 外层：模型结构 | 通道数、层数、hidden size、output stride、候选数、patch 大小 | 是 |
| 中层：训练参数 | learning rate、weight decay、loss 权重、课程预算、采样比例 | 通常是 |
| 内层：推理参数 | press/release 阈值、cooldown、时间 offset、平滑、NMS 阈值 | 否 |

首版固定完整输入为 `1484 × 846` 且不 resize，因此不搜索输入分辨率。显存和感受野相关
参数应搜索 `patch_width`、`patch_height`、overlap 和模型 output stride。

内层推理参数必须对同一个 checkpoint 快速搜索，禁止因为改变点击阈值或 cooldown
重复训练神经网络。

### 11.2 Trial 晋级流程

每个 trial 具有稳定的 `trial_id`、参数快照、课程阶段、已消费 step、checkpoint 和评估记录。

```text
阶段 0：TPE 或随机生成结构参数和训练参数
  ↓
阶段 1：训练 1,000～5,000 step
        单点 / 单划固定评估集
        未达到连续通过阈值 → prune
  ↓
阶段 2：从阶段 1 checkpoint 继续训练
        加入多点 / 多划，累计约 5,000～20,000 step
        未达到晋级阈值 → prune
  ↓
阶段 3：继续同一 checkpoint
        加入重复点、点划重叠、复杂 slider、高密度场景
        输出复杂阶段分数与失败类型
  ↓
阶段 4：少量合格 trial 完整训练
        对每个 checkpoint 独立搜索推理参数
        进行完整模拟和最终排名
```

预算是阶段上限，不要求所有 trial 使用相同 step。ASHA 可根据同一 rung 中的标准化指标
异步晋级，避免等待最慢 trial。

### 11.3 Checkpoint 继承规则

- 同一 trial 晋级课程阶段时，必须继承模型、优化器、scheduler、AMP scaler 和全局 step。
- 晋级只增加训练预算和数据难度，不重新初始化模型。
- 结构参数变化会改变参数拓扑，必须创建新 trial 并从头训练。
- 普通训练参数变化默认创建新 trial；只有明确采用 PBT 时，才允许记录 mutation 后继续。
- 推理参数变化只生成新的 evaluation run，不产生训练 checkpoint。
- TPE 在高分区域生成的“附近变体”是新 trial，必须从基础课程开始，不能继承另一 trial
  的 checkpoint。

### 11.4 复杂阶段反馈 TPE

复杂阶段不手工定义固定“加密半径”。将以下指标作为 trial 观测值交给 TPE：

- 基础阶段是否通过；
- 达到各阶段所需 step；
- 复杂阶段综合分；
- 误点击、提前点击、重复点漏检、slider 中断等分项；
- 峰值显存与推理延迟。

达到 `densify_score_threshold` 表示该区域值得继续采样；达到
`pass_score_threshold` 才能进入最终候选集合。TPE 应利用全部历史 trial，而不是只围绕
第一个可行参数点生成邻域。

### 11.5 难例挖掘

训练采样初始建议：

```text
50% 普通随机样本
30% 最近失败样本
20% 稀有复杂样本
```

难例池记录失败类型、checkpoint、首次发现阶段和最近一次损失。评估集保持固定且不进入
难例采样，避免搜索过程通过反复训练测试样本获得虚假提升。

### 11.6 可比性约束

- 每个 rung 使用固定、分层且顺序确定的评估集。
- 连续通过测试的样本顺序由评估版本固定，不随 trial 改变。
- 每个重要结构至少使用多个随机种子复验，最终排名报告均值和方差。
- 剪枝指标必须同时包含质量和资源约束，不能只比较单一 loss。
- 所有 trial 保存配置、代码版本、数据版本和父 checkpoint/mutation 关系。

## 12. 当前阶段测试逻辑

单个点与 slider 的空间、时间、膨胀路径覆盖和组合 score 使用
[`SCORING_SPEC.md`](SCORING_SPEC.md) 的 `point-slider-v2`。其中 slider 参考和预测路径
先分别膨胀为半径 `1.5x` 的容差走廊，再计算双向覆盖率。

### 12.1 容错连续通过

```yaml
single_point:
  required_consecutive_passes: 15
  max_failures: 2
  max_cases: 40

single_slider:
  required_consecutive_passes: 10
  max_failures: 2
  max_cases: 35

multi_point:
  required_consecutive_passes: 8
  max_failures: 3
  max_cases: 35

multi_slider:
  required_consecutive_passes: 6
  max_failures: 3
  max_cases: 30
```

失败重置连续通过计数，但只要总失败数未超限仍可继续。

### 12.2 重复与复杂阶段

```yaml
repeat_complex:
  densify_score_threshold: 0.72
  pass_score_threshold: 0.86
```

逻辑：

```text
低于加密阈值 → 不加密
达到加密阈值 → 将高价值观测反馈给 TPE
达到通过阈值 → 加入最终合格参数集合
```

TPE 生成的任何新参数组合都是新 trial，必须从基础阶段重新开始；原 trial 的课程晋级则
继续训练自己的 checkpoint。

### 12.3 最佳参数标注图集

每个训练批次完成当前已到达阶段的评估后，评估器输出 trial 总分和逐帧
`passed / failed` 结果。保存标注命令只选择总分最高的 trial，并按以下六个数据子项目
归档：

```text
single_point / slider / multi_point / point_slider / spinner / long_sequence
```

对最佳 trial 已评估到的每个子项目，分别从通过和不通过结果中随机抽取最多 10 帧；
少于 10 帧时全部输出。随机种子随批次结果保存以保证可复现。未进入的课程子项目没有
评估记录，因此不创建目录。

```text
traning_example/
  output_<次数>__<UTC时间>__<batch>__<best_trial>/
  best_parameters.json
  manifest.json
  passed/<subproject>/*.png
  failed/<subproject>/*.png
```

输出次数在 `traning_example/.output_counter` 中持久递增；UTC 时间、次数同时进入目录名、
图片文字和 JSON 清单。同一批次重复输出不会覆盖历史图集。

帧记录使用稳定的 `sample_key + frame_index`，不能依赖本次 Dataset 的顺序索引。
图集渲染失败不得改变训练状态；训练内服务捕获异常，只返回一次 warning，之后静默
禁用对应可视化能力。

## 13. 参数分类

### 模型结构参数

- 空间通道；
- 网络深度；
- 时序隐藏维度；
- Transformer 层数；
- 候选数量；
- patch 宽高与 overlap；
- 输出 stride；
- SMET 最大参数空间。

### 训练参数

- learning rate；
- weight decay；
- loss 权重；
- 课程阶段长度；
- 数据采样比例；
- SMET topology update interval；
- 活跃密度。

### 推理参数

- press threshold；
- release threshold；
- no-op threshold；
- cooldown；
- 时间偏移；
- NMS 半径；
- 候选置信度阈值。

推荐嵌套搜索：

```text
外层：结构搜索，决定独立训练 lineage
中层：训练参数与 ASHA 预算
训练：课程晋级并继承同一 checkpoint
内层：固定 checkpoint 的推理参数快速搜索
```

## 14. 推荐首版配置

### 空间训练

```yaml
input:
  width: 1484
  height: 846
  resize: false

tiling:
  patch_size: [512, 512]
  overlap: 128
  serial: true
  patch_batch_size: 1

spatial_model:
  high_resolution_channels: 8
  output_stride: 2
  feature_channels: 32-48
  top_k_per_patch: 16
  embedding_dim: 64-128

refinement:
  enabled: auto
  crop_sizes: [128, 192, 256]
  preferred_output_stride: 1
  fallback_output_stride: 2
  memory_dry_run: true
  peak_vram_budget_mb: 7168
  max_candidates_per_frame: 32

ambiguity_review:
  enabled: true
  max_regions_per_frame: 8
  low_confidence_threshold: 0.60
  close_score_margin: 0.05
  slider_continuity_threshold: 0.80

offload:
  patch_outputs_to_cpu: true
  retain_backbone_features: false
  non_blocking: true

training:
  batch_size: 1
  fp16: true
  backward_per_patch: true
  gradient_checkpointing: true
```

### 候选缓存

```yaml
candidate_cache:
  max_candidates_per_frame: 32
  embedding_dim: 128
  save_dtype: float16
  storage: disk
```

### 时序训练

```yaml
temporal_model:
  type: causal_gru
  hidden_size: 256
  layers: 2
  sequence_length: 128-256

output:
  action_classes:
    - no_op
    - press
    - hold
    - release
  selected_candidate: true
  time_offset: true
```

### SMET

第一版先关闭：

```yaml
smet:
  enabled: false
```

时序模型明显增大后再开启：

```yaml
smet:
  enabled: true
  modules:
    - temporal_projection
    - temporal_blocks
    - action_experts
  pattern: block
  block_size: 16
  initial_density: 0.40
  topology_update_interval: 250
  prune_regrow_ratio: 0.10
  new_parameter_warmup_steps: 20
```

## 15. 训练与部署区别

### 训练阶段

允许：

- 串行 patch；
- 多轮空间精修；
- CPU offload；
- 候选缓存；
- 动态稀疏拓扑更新；
- 很慢的完整评测；
- 多模型对照。

### 部署阶段

不需要：

- backward；
- 梯度；
- Adam 状态；
- 拓扑更新；
- 新生参数 warm-up。

SMET训练结束后：

```text
固定最终 mask
→ 结构化裁剪
→ 导出最终子模型
```

实际接口：

```python
output, state = model.step(current_frame, state)
```

## 16. 最终实施路线

```text
第一阶段：原图分块训练空间模型
第二阶段：离线生成帧级候选缓存
第三阶段：训练因果时序决策模型
第四阶段：进行阶段化参数搜索
第五阶段：根据欠拟合证据增加容量或 SMET 密度
第六阶段：固定最终活跃结构并导出部署模型
```

## 17. 核心原则

1. 不缩放输入，不等于所有层都保持原分辨率。
2. 串行 patch 主要解决激活显存。
3. CPU/GPU 协同用于缓存、融合和隐藏中间结果。
4. SMET主要解决参数、梯度和优化器状态显存。
5. 空间识别与时序判断应优先分开训练。
6. `no-op` 必须是正式动作类别。
7. 复杂阶段高分用于参数区域加密，不能掩盖基础能力不足。
8. 模型容量应根据欠拟合证据增加，而不是按显存上限盲目放大。
9. 训练可以非常慢，但部署应固定最终活跃子网络。
10. 先完成稳定基础版本，再加入动态稀疏和专家机制。
11. 首版 slider 只支持无交叉、无接触分叉路径；跨 patch 通过全局稠密图融合解决。
