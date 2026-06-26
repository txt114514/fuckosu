# start 启动入口

`src/start` 固定仓库内 `src` 顶层模块入口和启动前自检：

- `start.main`：总 CLI 入口，可列出模块和执行启动前检查。
- `start.modules`：登记 `src/package`、`src/before_traning`、`src/traning`
  以及 `src/start` 自身的稳定入口。
- `start.entries`：按模块拆分的入口对象，供外部或脚本稳定引用。
- `start.checks`：渐进自检与训练启动前检查。
- `start.flow`：完整启动编排；只调用 before_traning/traning 各自检测包的启动文件，
  再按返回值决定是否继续。

常用命令：

```bash
PYTHONPATH=src python -m start modules
PYTHONPATH=src python -m start check
PYTHONPATH=src python -m start check --config configs/model_small_vram.yaml --device cpu
PYTHONPATH=src python -m start run --training-config configs/model_small_vram.yaml --device cpu --dry-run --test-level quick --no-before-match-probe
```

完整启动顺序：

1. 调用 `before_traning.tests.startup_checks.runner.run_startup_checks`。
   其中 `before_traning:raw_data` 会只读检测未匹配的新 `.osz`、待匹配视频、
   已导入但未 `video_matched` 的样本和启动层 matched manifest；已匹配或已打包的样本不再检测、不修改。
2. 如果 `before_traning:raw_data.details.should_run_before_traning` 为 true，
   且未传 `--dry-run` / `--skip-before-traning`，运行 before_traning 七阶段流水线更新训练集。
3. 同步 `training_package/splits/dataset_split_manifest.json`。已有 item 的 split 冻结，
   只给新增且已打包完成的 item 做增量分配；默认不自动扩充 test split。
4. 调用 `traning.tests.startup_checks.runner.run_startup_checks` 检查训练配置、设备、数据输入和 core 入口。
5. `--test-level quick` 复用两个启动检测报告；`--test-level full` 调用
   `before_traning.tests.full_checks.runner.run_full_checks` 和
   `traning.tests.full_checks.runner.run_full_checks`。
6. 检测通过后执行完整训练。

推荐训练入口现在是 `PYTHONPATH=src:. python -m traning.main full-flow`；
`start run` 仍保留为启动前处理和固定小预算训练的组合入口。

完整训练入口 `traning.core.decision.run_full_training_pipeline` 每次启动都会先调用
`start.checks.run_training_startup_checks`。自检失败时训练不会继续；自检结果写入
`full_training_summary.json`。
