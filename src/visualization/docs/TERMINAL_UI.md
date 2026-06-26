# 中文终端 UI

CLI 参数：

```text
--progress-ui auto|rich|plain|off
--progress-language zh-CN
```

TTY 默认使用 Rich 动态界面，非 TTY 使用 plain。Rich 初始化失败会写入事件并降级 plain，
不得中止训练。`off` 关闭界面但仍允许训练业务继续运行。

停止摘要在保存状态后显示。交互式 TTY 支持 `Q`、`Enter` 和 `Esc` 退出；非 TTY 不等待按键。
该行为由 `src/visualization/tests/test_dashboard.py` 使用真实 pseudo-terminal 覆盖。
