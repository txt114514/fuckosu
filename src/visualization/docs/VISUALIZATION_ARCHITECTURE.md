# visualization 架构

`visualization` 是与 `traning` 平级的中文训练控制台模块。训练核心只依赖
`visualization.lib` 中的结构化 reporter，不依赖 Rich、Panel 或具体终端布局。

数据流：

```text
traning.core -> visualization.lib.TrainingReporter
             -> visualization.state.DashboardStateStore
             -> visualization.core.renderers
```

Gallery 导出实现位于 `visualization.core.gallery.exporter`。旧
`traning.lib.visualization.gallery` 只保留兼容转发。

`traning.core.full_flow` 会把 raw-data、before_traning、split、preflight、resume、
ramp、readiness、full training、evaluation、artifact、inheritance 和 report 阶段统一上报到
`PipelineStageState`。进入真实 spatial/temporal 训练后，训练 loop 继续逐 step 上报 loss、样本和资源状态。
