# `traning` 目录结构迁移

## `before_traning` 当前结构

`before_traning` 以 `main.py` 为入口，按 `conf/`、`core/`、`Lib/`、`state/`、`tests/` 和 `docs/` 分层。其核心特点是入口只编排、配置与状态有明确权威目录、领域流程不散落在包根。

## `traning` 迁移前结构

迁移前的权威实现直接位于 `app/`、`config/`、`contracts/`、`data/`、`perception/`、`tracking/`、`belief/`、`outcome/`、`decision/`、`evaluation/`、`training/`、`telemetry/`、`visualization/` 和 `infrastructure/`。包根没有统一的 `main.py`/`__main__.py`，工具型环境实现还位于仓库根 `environment/`。

## `traning` 迁移后结构

```text
src/traning/
├── main.py
├── __main__.py
├── conf/
├── core/
│   ├── app/
│   ├── data/
│   ├── perception/
│   ├── tracking/
│   ├── belief/
│   ├── outcome/
│   ├── decision/
│   ├── evaluation/
│   └── training/
├── lib/
│   ├── data/
│   ├── environment/
│   ├── infrastructure/
│   ├── runtime/
│   ├── telemetry/
│   ├── validation/
│   └── visualization/
├── state/
├── tests/
└── docs/
```

## 目录映射表

| 迁移前 | 权威位置 | 说明 |
|---|---|---|
| `traning/app` | `traning/core/app` | CLI、runtime、factory 和训练搜索编排 |
| `traning/config` | `traning/conf` | 唯一严格配置模型与版本 |
| `traning/contracts` | `traning/state` | DTO、Enum、Protocol 和 registry |
| `traning/data` | `traning/core/data` | 数据集、坐标、质量和领域仓储编排；已有底层视频工具仍在 `lib/data` |
| `traning/perception` | `traning/core/perception` | 全帧感知模型、decode、runtime 和训练 |
| `traning/tracking` | `traning/core/tracking` | 关联与有状态 tracker |
| `traning/belief` | `traning/core/belief` | 因果 belief 编码和训练 |
| `traning/outcome` | `traning/core/outcome` | outcome 模型、oracle、dataset 与训练 |
| `traning/decision` | `traning/core/decision` | optimal stopping 与 utility |
| `traning/evaluation` | `traning/core/evaluation` | 评分、归因、序列指标 |
| `traning/training` | `traning/core/training` | 生产调度、优化、checkpoint 与难例反馈 |
| `traning/infrastructure` | `traning/lib/infrastructure` | 持久化、错误与确定性工具 |
| `traning/telemetry` | `traning/lib/telemetry` | reporter/store；DTO 权威位于 state |
| `traning/visualization` | `traning/lib/visualization` | gallery/dashboard 渲染辅助 |
| 根 `environment` | `traning/lib/environment` | 完整环境与 GPU 报告；根路径降级为 wrapper |

## 保留 wrapper 表

| 旧入口 | 新入口 | 兼容策略 |
|---|---|---|
| `traning.app[.*]` | `traning.core.app[.*]` | deprecated re-export |
| `traning.config[.*]` | `traning.conf[.*]` | deprecated re-export |
| `traning.contracts[.*]` | `traning.state[.*]` | deprecated re-export，类型 identity 不变 |
| `traning.data[.*]` | `traning.core.data[.*]` | deprecated re-export |
| `traning.perception/tracking/belief/outcome/decision/evaluation/training[.*]` | 对应 `traning.core.*` | deprecated re-export |
| `traning.infrastructure[.*]` | `traning.lib.infrastructure[.*]` | deprecated re-export |
| `traning.telemetry[.*]` | `traning.lib.telemetry[.*]` | deprecated re-export |
| `traning.visualization[.*]` | `traning.lib.visualization[.*]` | deprecated re-export |
| 根 `environment[.*]` | `traning.lib.environment[.*]` | deprecated re-export/脚本转发 |

## deprecated 文件表

上述旧包内的 `__init__.py`、旧 leaf module wrapper、根 `environment/__init__.py`、`environment/env_check.py`、`environment/check_gpu.sh` 与根环境 README 均是 deprecated 兼容入口。它们不得再承载实现；新源码、启动注册和文档示例均使用 canonical 路径。

`traning.lib.data.patch_stream`、`traning.lib.data.tiling` 与 patch 坐标辅助模块是 legacy/可选 ROI 工具，不从 `lib.data` 默认公开面导出，也不在生产主链路中。

## 无法迁移的文件表

| 文件 | 原因 |
|---|---|
| `traning/legacy/legacy_freeze.json` | Phase 0 回归冻结制品，路径本身是已有 golden baseline 契约；不是活动 Python 架构 |
| `traning/tests/regression/fixtures/*` | 回归 fixture 的相对路径是测试制品契约，仍归 tests 管理 |

除此之外，没有把旧扁平业务实现作为“无法迁移”遗留；保留的旧 Python 路径均只作 wrapper。
