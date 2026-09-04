"""将 start 的生命周期协议适配到 traning 生产训练服务。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from start.flow import TrainingExecutionResult
from traning.conf import V2Config
from traning.state import DataQualityReport, DataSplit
from traning.core.data import TrainingDatasetBundle, build_training_datasets
from traning.core.training import ProductionGateSpec, ProductionTrainer


@dataclass(slots=True)
class ProductionTrainingExecutor:
    """复用 inspect 构建的同一 bundle，避免检查与训练读取两份数据结论。"""

    gates: ProductionGateSpec = ProductionGateSpec()
    _config: V2Config | None = field(default=None, init=False, repr=False)
    _datasets: TrainingDatasetBundle | None = field(
        default=None,
        init=False,
        repr=False,
    )

    def inspect(
        self,
        config: V2Config,
        *,
        split: DataSplit,
    ) -> DataQualityReport:
        """构建惰性 typed 数据集，并返回它携带的 canonical 质量报告。"""

        if not isinstance(config, V2Config):
            raise TypeError("config 必须是 V2Config")
        if not isinstance(split, DataSplit) or split is DataSplit.ALL:
            raise ValueError("split 必须是具体 DataSplit")
        if self._datasets is None or self._config != config:
            self._datasets = build_training_datasets(config)
            self._config = config
        return self._datasets.quality_report

    def run(
        self,
        config: V2Config,
        *,
        split: DataSplit,
        quality_report: DataQualityReport,
        run_dir: Path,
        run_id: str,
        resume: bool,
    ) -> TrainingExecutionResult:
        """执行可恢复搜索；只有 winning checkpoint 通过复验后才返回 passed。"""

        if split is not DataSplit.TRAIN:
            raise ValueError("生产模型训练入口只接受 split=train")
        inspected = self.inspect(config, split=split)
        if quality_report is not inspected:
            raise ValueError(
                "start 必须把 inspect 返回的同一 DataQualityReport 交回 run"
            )
        datasets = self._datasets
        if datasets is None:  # pragma: no cover - inspect 已保证
            raise RuntimeError("训练数据 bundle 未初始化")
        result = ProductionTrainer(config, datasets, self.gates).run(
            run_dir=run_dir,
            run_id=run_id,
            resume=resume,
        )
        parameters = asdict(result.observation.parameters)
        metrics = None if result.metrics is None else asdict(result.metrics)
        return TrainingExecutionResult(
            status="passed",
            message=(
                f"trial {result.observation.trial_index} 全门禁通过；"
                "runtime checkpoint 已完成摘要复验"
            ),
            trial_index=result.observation.trial_index,
            objective=result.observation.objective,
            checkpoint_path=result.checkpoint_directory,
            details={
                "parameters": parameters,
                "metrics": metrics,
                "resumed": result.resumed,
            },
        )


__all__ = ("ProductionTrainingExecutor",)
