# start 启动入口

`src/start` 固定仓库内 `src` 顶层模块入口和启动前自检：

- `start.main`：总 CLI 入口，可列出模块和执行启动前检查。
- `start.modules`：登记 `src/package`、`src/before_traning`、`src/traning`
  以及 `src/start` 自身的稳定入口。
- `start.entries`：按模块拆分的入口对象，供外部或脚本稳定引用。
- `start.checks`：渐进自检与训练启动前检查。
- `start.flow`：旧的启动编排业务函数，保留给适配层测试和诊断使用。
- `start run`：当前推荐总入口，直接进入 `traning.core.full_flow`，启动中文 UI，
  先展示启动检查/渐进训练准备 UI，再在真实训练阶段切换到正式训练 UI。

常用命令：

```bash
PYTHONPATH=src python -m start modules
PYTHONPATH=src python -m start check
PYTHONPATH=src python -m start check --config configs/model_small_vram.yaml --device cpu
PYTHONPATH=src python src/start/main.py run --config configs/model_full_small_vram.yaml --device cuda --progress-ui rich --auto-launch-full
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
6. 检测通过后执行受控渐进训练；进入 `RAMP_TRAINING` 后 UI 自动切换为正式训练 UI。
7. 渐进训练通过后，默认 `--auto-launch-full` 启动正式训练。
8. 正式训练评估后，最佳参数对应的测试图集默认输出到 `traning_example/`，目录结构遵循
   `src/traning/docs/RESULT_EXPORT_PLAN.md`。

推荐总入口现在是：

```bash
PYTHONPATH=src:. python src/start/main.py run --config configs/model_full_small_vram.yaml --device cuda --progress-ui rich
```

容器内 CUDA 运行应使用项目约定的 host bridge：

```bash
host-exec docker exec -t -u dev osu_ai_dev bash -lc 'cd /home/dev/workspace && PYTHONPATH=src:. python src/start/main.py run --config configs/model_full_small_vram.yaml --device cuda --progress-ui rich --auto-launch-full'
```

常用 UI/训练参数：

- `--progress-ui auto|rich|plain|off`：TTY 下 `auto` 会启用 Rich UI。
- `--test-level none|quick|full`：`none` 跳过 ramp preflight 中的 pytest full checks；
  默认 `quick` 会执行训练 full checks。
- `--auto-launch-full/--no-auto-launch-full`：是否在渐进训练通过后进入正式训练，默认开启。
- `--gallery-output-root traning_example`：正式训练最佳参数图集输出根目录，默认就是
  `traning_example/`。

完整训练入口 `traning.core.decision.run_full_training_pipeline` 每次启动都会先调用
`start.checks.run_training_startup_checks`。自检失败时训练不会继续；自检结果写入
`full_training_summary.json`。
