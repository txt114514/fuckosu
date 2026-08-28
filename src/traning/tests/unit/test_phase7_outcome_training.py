"""Phase 7 Outcome 训练批次、损失与优化步骤验收。"""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from traning.config import OutcomeConfig
from traning.contracts import (
    BeliefState,
    DataSplit,
    DecisionAction,
    ObjectTypeDistribution,
    OutcomeCategory,
    OutcomeTrainingSample,
    Point2D,
)
from traning.outcome.model import DenseOutcomeModel
from traning.outcome.dataset import CounterfactualOutcomeDataset
from traning.outcome.training import (
    OutcomeLossWeights,
    collate_outcome_samples,
    compute_outcome_loss,
    evaluate_outcome_batch,
    train_outcome_step,
)


_TRAINING_PATH = Path(__file__).resolve().parents[2] / "outcome/training.py"


def _config() -> OutcomeConfig:
    return replace(OutcomeConfig(), hidden_dims=(8, 4), horizons_ms=(0, 16, 32))


def _belief(index: int) -> BeliefState:
    return BeliefState(
        track_id=f"track-{index}",
        timestamp_ms=1000.0 + index,
        belief_embedding=(0.1 * index, -0.2, 0.3),
        position_mean=Point2D(10.0, 20.0),
        position_uncertainty=Point2D(1.0, 1.0),
        visibility_probability=0.8,
        object_type_distribution=ObjectTypeDistribution(0.7, 0.1, 0.1, 0.1),
        age=2,
        time_since_seen_ms=0.0,
        uncertainty=0.1,
    )


def _sample(
    index: int,
    category: OutcomeCategory,
    score: float,
    *,
    split: DataSplit = DataSplit.TRAIN,
) -> OutcomeTrainingSample:
    invalid = category is OutcomeCategory.INVALID
    belief = _belief(index)
    return OutcomeTrainingSample(
        sample_id=f"outcome-{index}",
        split=split,
        source_sample_id=f"source-{index}",
        oracle_state_id=f"oracle-{index}",
        belief=belief,
        action=DecisionAction.CLICK,
        action_track_id=belief.track_id,
        horizon_ms=float(index * 16),
        target_category=category,
        target_score=score,
        valid=not invalid,
        expires=invalid,
        target_object_id=None if invalid else f"object-{index}",
    )


def _samples() -> tuple[OutcomeTrainingSample, ...]:
    return (
        _sample(1, OutcomeCategory.INVALID, 0.0),
        _sample(2, OutcomeCategory.MISS, 0.0),
        _sample(3, OutcomeCategory.LOW, 0.25),
        _sample(4, OutcomeCategory.MEDIUM, 0.65),
        _sample(5, OutcomeCategory.HIGH, 0.9),
    )


def _dataset(
    records: tuple[OutcomeTrainingSample, ...] | None = None,
) -> CounterfactualOutcomeDataset:
    """把测试 records 绑定到一个明确坐标指纹，模拟正式 artifact loader。"""

    selected = _samples() if records is None else records
    return CounterfactualOutcomeDataset(
        split=selected[0].split,
        records=selected,
        transform_fingerprint="transform-0123456789abcdef",
    )


def test_collate_preserves_lineage_and_tensor_contract() -> None:
    """批处理拼装必须保留 lineage 并产生规定张量契约。"""

    batch = collate_outcome_samples(_dataset(), belief_embedding_dim=3)

    assert batch.split is DataSplit.TRAIN
    assert batch.transform_fingerprint == "transform-0123456789abcdef"
    assert batch.sample_ids == tuple(f"outcome-{index}" for index in range(1, 6))
    assert batch.source_sample_ids[0] == "source-1"
    assert batch.oracle_state_ids[-1] == "oracle-5"
    assert batch.track_ids[2] == "track-3"
    assert batch.belief_embeddings.shape == (5, 3)
    assert batch.belief_embeddings.dtype is torch.float32
    assert batch.category_targets.dtype is torch.long
    assert batch.valid_targets.dtype is torch.bool
    assert batch.expiry_targets.tolist() == [1.0, 0.0, 0.0, 0.0, 0.0]
    assert batch.valid_targets.tolist() == [False, True, True, True, True]
    assert batch.category_targets.tolist() == [0, 1, 2, 3, 4]


def test_collate_rejects_naked_records_and_wrong_embedding_dim() -> None:
    """批处理只能接收带指纹 dataset，并拒绝错误 embedding 维度。"""

    with pytest.raises(TypeError, match="CounterfactualOutcomeDataset"):
        collate_outcome_samples((), belief_embedding_dim=3)  # type: ignore[arg-type]
    mixed = (_samples()[0], _sample(9, OutcomeCategory.HIGH, 0.9, split=DataSplit.TEST))
    with pytest.raises(ValueError, match="dataset.split"):
        CounterfactualOutcomeDataset(
            DataSplit.TRAIN,
            mixed,
            "transform-0123456789abcdef",
        )
    with pytest.raises(ValueError, match="维度"):
        collate_outcome_samples(_dataset(), belief_embedding_dim=2)
    duplicate = (_samples()[0], _samples()[0])
    with pytest.raises(ValueError, match="重复"):
        _dataset(duplicate)


def test_loss_is_finite_and_backward_reaches_all_model_groups() -> None:
    """训练损失必须有限，且反向传播覆盖所有模型参数组。"""

    torch.manual_seed(3)
    model = DenseOutcomeModel(_config(), belief_embedding_dim=3)
    batch = collate_outcome_samples(_dataset(), belief_embedding_dim=3)
    output = model(batch.belief_embeddings, batch.horizon_ms)
    loss = compute_outcome_loss(output, batch)

    assert all(
        torch.isfinite(value).item()
        for value in (loss.total, loss.category, loss.expiry, loss.score)
    )
    loss.total.backward()
    for module in (model.trunk, model.category_head, model.expiry_head):
        parameters = tuple(module.parameters())
        assert parameters
        assert all(parameter.grad is not None for parameter in parameters)


def test_evaluation_reuses_canonical_metrics_and_returns_finite_values() -> None:
    """评估必须复用 canonical 指标并返回有限数值。"""

    model = DenseOutcomeModel(_config(), belief_embedding_dim=3)
    batch = collate_outcome_samples(_dataset(), belief_embedding_dim=3)
    metrics = evaluate_outcome_batch(
        model(batch.belief_embeddings, batch.horizon_ms), batch, calibration_bins=5
    )

    assert all(
        torch.isfinite(value).item()
        for value in (
            metrics.multiclass_nll,
            metrics.multiclass_brier,
            metrics.calibration_error,
            metrics.expected_score_mae,
            metrics.expiry_brier,
        )
    )


def test_real_optimizer_step_clears_grads_and_updates_parameters() -> None:
    """真实优化步骤必须清除梯度并更新模型参数。"""

    class RecordingSgd(torch.optim.SGD):
        """记录 zero_grad 参数的 SGD 测试替身。"""

        used_set_to_none: bool = False

        def zero_grad(self, set_to_none: bool = True) -> None:
            """记录并透传 set_to_none 选项。"""

            self.used_set_to_none = set_to_none
            super().zero_grad(set_to_none=set_to_none)

    torch.manual_seed(7)
    model = DenseOutcomeModel(_config(), belief_embedding_dim=3)
    batch = collate_outcome_samples(_dataset(), belief_embedding_dim=3)
    optimizer = RecordingSgd(model.parameters(), lr=0.1)
    before = tuple(parameter.detach().clone() for parameter in model.parameters())

    loss = train_outcome_step(model, batch, optimizer)
    after = tuple(parameter.detach() for parameter in model.parameters())

    assert optimizer.used_set_to_none is True
    assert torch.isfinite(loss.total).item()
    assert any(
        not torch.equal(old, new) for old, new in zip(before, after, strict=True)
    )
    assert all(parameter.grad is not None for parameter in model.parameters())


@pytest.mark.parametrize(
    "weights",
    (
        OutcomeLossWeights(category=1.0, expiry=1.0, score=0.0),
        OutcomeLossWeights(category=0.5, expiry=2.0, score=0.2),
    ),
)
def test_primary_loss_weights_cannot_be_replaced_by_score(
    weights: OutcomeLossWeights,
) -> None:
    """分数辅助项不得替代分类与过期两个主损失。"""

    assert weights.category > 0
    assert weights.expiry > 0
    assert weights.score >= 0


def test_invalid_primary_loss_weights_are_rejected() -> None:
    """任一主损失权重为零时必须拒绝配置。"""

    with pytest.raises(ValueError, match="主任务"):
        OutcomeLossWeights(category=0.0, expiry=1.0, score=10.0)
    with pytest.raises(ValueError, match="主任务"):
        OutcomeLossWeights(category=1.0, expiry=0.0, score=10.0)


def test_training_module_has_no_runtime_oracle_ground_truth_or_any_dependency() -> None:
    """训练模块不得依赖 runtime oracle、GT 或宽泛 Any。"""

    tree = ast.parse(_TRAINING_PATH.read_text(encoding="utf-8"))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }

    assert "Any" not in imports
    assert not any("oracle" in name.lower() for name in imports)
    assert not any("groundtruth" in name.lower() for name in imports)
    assert not any(name == "traning" or name.startswith("traning.") for name in imports)
