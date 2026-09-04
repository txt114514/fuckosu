# start 总启动入口

`src/start` 是仓库唯一完整生命周期入口。它保留原来的总启动顺序，但训练阶段已经
整体替换为 `src/traning` 的 typed V2 生产服务，不再调用旧 `traning.core`、独立
`visualization` 或外部 evaluator 工厂。

## 执行顺序

1. 只读扫描未匹配谱面、视频和待处理样本。
2. 确有新原始数据时运行 `before_traning` 七阶段转换；否则明确跳过。
3. 增量同步 canonical dataset split manifest，既有 item 的 split 不漂移。
4. 加载 `configs/traning.yaml`，检查设备、坐标证据和 canonical 数据质量报告。
5. `--dry-run` 到此生成报告；正式模式进入可恢复的 curriculum/ASHA 参数搜索和六阶段训练。
6. 只有全部门禁通过且 runtime checkpoint 复验成功，整个流程才返回 `passed`。

默认配置不限制 trial 数。普通阶段门禁失败或 ASHA prune 会保存资源 job 与 observation 并
继续选择未重复参数；固定数据质量错误、显式 trial 预算/空间耗尽和不可恢复异常才会终止，
并各自留下明确终态。

## 命令

```bash
PYTHONPATH=src python src/start/main.py run \
  --config configs/traning.yaml \
  --device cuda \
  --resume

# 同一入口的模块形式；无子命令也执行完整流程
PYTHONPATH=src python -m start

# CPU 只读预检，不启动训练
PYTHONPATH=src python -m start run \
  --config configs/traning.yaml \
  --device cpu \
  --dry-run

# 诊断
PYTHONPATH=src python -m start modules
PYTHONPATH=src python -m start config-check --config configs/traning.yaml
PYTHONPATH=src python -m start coordinate-audit --config configs/traning.yaml
PYTHONPATH=src python -m start env-check --config configs/traning.yaml --strict
```

CUDA 必须通过工程约定的主机桥运行：

```bash
host-exec docker exec -u dev osu_ai_dev bash -lc \
  'cd /home/dev/workspace && PYTHONPATH=src python src/start/main.py run --config configs/traning.yaml --device cuda --resume'
```

每次运行在 `artifacts/training_runs/<run_id>/start_flow_report.json` 保存完整阶段证据；
各 trial 的 curriculum/rung checkpoint、TRAIN-only hard-example feedback 与
`search_state.json` 位于同一 run 目录。重启时使用 `--resume` 接续，会先校验全部制品摘要，
不会重复已原子提交的 job。
