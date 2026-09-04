"""Phase 7 Outcome 训练批次、损失与优化步骤验收。"""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from traning.conf import OutcomeConfig
from traning.state import (
    BeliefState,
    DataSplit,
    DecisionAction,
    ObjectTypeDistribution,
    OutcomeCategory,
    OutcomeTrainingSample,
    Point2D,
)
from traning.core.outcome.model import DenseOutcomeModel
from traning.core.outcome.dataset import CounterfactualOutcomeDataset
from traning.core.outcome.training import (
    OutcomeLossWeights,
    collate_outcome_samples,
    compute_outcome_loss,
    evaluate_outcome_batch,
    train_outcome_step,
)


_TRAINING_PATH = Path(__file__).resolve().parents[2] / "core/outcome/training.py"


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


def test_sample_weighted_loss_matches_manual_normalized_components() -> None:
    """三个逐样本损失必须共享同一权重并除以权重总和。"""

    torch.manual_seed(11)
    model = DenseOutcomeModel(_config(), belief_embedding_dim=3)
    batch = collate_outcome_samples(_dataset(), belief_embedding_dim=3)
    output = model(batch.belief_embeddings, batch.horizon_ms)
    sample_weights = torch.tensor((1.0, 2.0, 4.0, 8.0, 16.0))

    loss = compute_outcome_loss(
        output,
        batch,
        sample_weights=sample_weights,
    )

    category_per_record = -torch.log_softmax(output.category_logits, dim=1)[
        torch.arange(len(batch.sample_ids)), batch.category_targets
    ]
    expiry_per_record = F.softplus(output.expiry_logits) - (
        batch.expiry_targets * output.expiry_logits
    )
    score_delta = (output.expected_score - batch.score_targets).abs()
    score_per_record = torch.where(
        score_delta < 1.0,
        0.5 * score_delta.square(),
        score_delta - 0.5,
    )
    expected_components = tuple(
        (component * sample_weights).sum() / sample_weights.sum()
        for component in (
            category_per_record,
            expiry_per_record,
            score_per_record,
        )
    )

    assert torch.allclose(loss.category, expected_components[0])
    assert torch.allclose(loss.expiry, expected_components[1])
    assert torch.allclose(loss.score, expected_components[2])
    assert torch.allclose(
        loss.total,
        expected_components[0] + expected_components[1] + 0.1 * expected_components[2],
    )


def test_sample_weights_change_model_gradient_without_scaling_all_samples_equally() -> (
    None
):
    """非均匀 hard-example 权重必须真实改变模型梯度。"""

    torch.manual_seed(19)
    model = DenseOutcomeModel(_config(), belief_embedding_dim=3)
    batch = collate_outcome_samples(_dataset(), belief_embedding_dim=3)

    unweighted = compute_outcome_loss(
        model(batch.belief_embeddings, batch.horizon_ms),
        batch,
    )
    unweighted.total.backward()
    unweighted_gradient = model.category_head.weight.grad
    assert unweighted_gradient is not None
    unweighted_snapshot = unweighted_gradient.detach().clone()

    model.zero_grad(set_to_none=True)
    weighted = compute_outcome_loss(
        model(batch.belief_embeddings, batch.horizon_ms),
        batch,
        sample_weights=torch.tensor((20.0, 1.0, 1.0, 1.0, 1.0)),
    )
    weighted.total.backward()
    weighted_gradient = model.category_head.weight.grad

    assert weighted_gradient is not None
    assert bool(torch.isfinite(weighted_gradient).all().item())
    assert not torch.allclose(unweighted_snapshot, weighted_gradient)


def test_sample_weights_none_preserves_original_mean_reductions() -> None:
    """省略权重或显式传 None 都必须精确保留原始 mean reduction。"""

    torch.manual_seed(23)
    model = DenseOutcomeModel(_config(), belief_embedding_dim=3)
    batch = collate_outcome_samples(_dataset(), belief_embedding_dim=3)
    output = model(batch.belief_embeddings, batch.horizon_ms)

    implicit = compute_outcome_loss(output, batch)
    explicit = compute_outcome_loss(output, batch, sample_weights=None)
    legacy_components = (
        F.cross_entropy(output.category_logits, batch.category_targets),
        F.binary_cross_entropy_with_logits(
            output.expiry_logits,
            batch.expiry_targets,
        ),
        F.smooth_l1_loss(output.expected_score, batch.score_targets),
    )

    assert torch.equal(implicit.category, explicit.category)
    assert torch.equal(implicit.expiry, explicit.expiry)
    assert torch.equal(implicit.score, explicit.score)
    assert torch.equal(implicit.category, legacy_components[0])
    assert torch.equal(implicit.expiry, legacy_components[1])
    assert torch.equal(implicit.score, legacy_components[2])


def test_sample_weights_reject_invalid_type_shape_device_dtype_and_values() -> None:
    """权重边界不得隐式转换、reshape、搬运或接受非正非有限值。"""

    model = DenseOutcomeModel(_config(), belief_embedding_dim=3)
    batch = collate_outcome_samples(_dataset(), belief_embedding_dim=3)
    output = model(batch.belief_embeddings, batch.horizon_ms)
    batch_size = len(batch.sample_ids)

    with pytest.raises(TypeError, match="torch.Tensor"):
        compute_outcome_loss(output, batch, sample_weights=(1.0,) * batch_size)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"\[batch\]"):
        compute_outcome_loss(
            output,
            batch,
            sample_weights=torch.ones(batch_size, 1),
        )
    with pytest.raises(ValueError, match=r"\[batch\]"):
        compute_outcome_loss(
            output,
            batch,
            sample_weights=torch.ones(batch_size - 1),
        )
    with pytest.raises(ValueError, match="同一设备"):
        compute_outcome_loss(
            output,
            batch,
            sample_weights=torch.ones(batch_size, device="meta"),
        )
    with pytest.raises(TypeError, match="dtype"):
        compute_outcome_loss(
            output,
            batch,
            sample_weights=torch.ones(batch_size, dtype=torch.float64),
        )

    invalid_values = (
        (float("nan"), 1.0, 1.0, 1.0, 1.0),
        (float("inf"), 1.0, 1.0, 1.0, 1.0),
        (0.0, 1.0, 1.0, 1.0, 1.0),
        (-1.0, 1.0, 1.0, 1.0, 1.0),
    )
    for values in invalid_values:
        with pytest.raises(ValueError, match="有限|严格大于"):
            compute_outcome_loss(
                output,
                batch,
                sample_weights=torch.tensor(values),
            )

    with pytest.raises(ValueError, match="权重和必须有限"):
        compute_outcome_loss(
            output,
            batch,
            sample_weights=torch.full(
                (batch_size,),
                torch.finfo(torch.float32).max,
            ),
        )


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
