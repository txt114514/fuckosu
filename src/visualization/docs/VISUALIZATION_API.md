# visualization 公共 API

外部模块优先使用：

```python
from visualization.lib import create_dashboard_reporter, NullReporter
```

核心状态模型包括 `TrainingDashboardState`、`TrainingEvent`、
`PipelineStageState`、`DatasetUsageState`、`ResourceState` 和
`TrainingStopState`。

Gallery 公共 API：

```python
from visualization.lib.gallery_api import export_best_trial_gallery
```
