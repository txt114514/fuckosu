"""Phase 7 dense Outcome 模型验收。"""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from traning.conf import OutcomeConfig
from traning.state import (
    BeliefState,
    ObjectTypeDistribution,
    OutcomeDistribution,
    Point2D,
)
from traning.core.outcome.model import (
    SCORE_REPRESENTATIVES,
    DenseOutcomeModel,
    OutcomeTensorOutput,
)


_MODEL_PATH = Path(__file__).resolve().parents[2] / "core/outcome/model.py"


def _config() -> OutcomeConfig:
    return replace(OutcomeConfig(), hidden_dims=(6, 4), horizons_ms=(0, 16, 32))


def _belief() -> BeliefState:
    return BeliefState(
        track_id="track-1",
        timestamp_ms=1000.0,
        belief_embedding=(0.2, -0.1, 0.4),
        position_mean=Point2D(100.0, 80.0),
        position_uncertainty=Point2D(1.0, 2.0),
        visibility_probability=0.8,
        object_type_distribution=ObjectTypeDistribution(0.7, 0.1, 0.1, 0.1),
        age=3,
        time_since_seen_ms=0.0,
        uncertainty=0.2,
    )


def _constant_parameters(model: DenseOutcomeModel, value: float) -> None:
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(value)


def test_forward_probabilities_expected_score_and_variance_are_exact() -> None:
    """五分类概率及代表分数矩必须由同一分布推导。"""

    model = DenseOutcomeModel(_config(), belief_embedding_dim=3)
    output = model(torch.tensor([[0.2, -0.1, 0.4]]), torch.tensor([16.0]))

    assert isinstance(output, OutcomeTensorOutput)
    assert output.category_logits.shape == (1, 5)
    assert output.expiry_logits.shape == (1,)
    assert torch.allclose(
        output.category_probabilities.sum(dim=1), torch.ones(1), atol=1e-7
    )
    representatives = torch.tensor(SCORE_REPRESENTATIVES)
    expected = (output.category_probabilities[0] * representatives).sum()
    variance = (
        output.category_probabilities[0] * (representatives - expected).square()
    ).sum()
    assert output.expected_score[0].item() == pytest.approx(expected.item())
    assert output.variance[0].item() == pytest.approx(variance.item())
    assert 0.0 <= output.expiry_probability[0] <= 1.0


def test_horizon_feature_deterministically_changes_distribution() -> None:
    """同一 belief 的不同 horizon 必须有可学习且可证明的分布差异。"""

    config = replace(_config(), hidden_dims=(1,))
    model = DenseOutcomeModel(config, belief_embedding_dim=3)
    with torch.no_grad():
        trunk_linear = model.trunk[0]
        assert isinstance(trunk_linear, nn.Linear)
        trunk_linear.weight.zero_()
        trunk_linear.bias.zero_()
        # 输入倒数第二维是 normalized horizon，最后一维是固定 CLICK=1。
        trunk_linear.weight[0, -2] = 2.0
        model.category_head.weight.copy_(torch.arange(5.0).unsqueeze(1))
        model.category_head.bias.zero_()
        model.expiry_head.weight.fill_(1.0)
        model.expiry_head.bias.zero_()
    belief = torch.zeros(2, 3)
    output = model(belief, torch.tensor([0.0, 32.0]))

    assert not torch.equal(
        output.category_probabilities[0], output.category_probabilities[1]
    )
    assert output.expiry_probability[0] != output.expiry_probability[1]


def test_all_trunk_category_and_expiry_parameters_receive_nonzero_gradient() -> None:
    """所有构建参数都必须进入同一训练 forward。"""

    model = DenseOutcomeModel(_config(), belief_embedding_dim=3)
    _constant_parameters(model, 0.1)
    output = model(
        torch.tensor([[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]]),
        torch.tensor([16.0, 32.0]),
    )
    category_weights = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0]])
    loss = (
        output.category_logits * category_weights
    ).sum() + output.expiry_logits.sum()
    loss.backward()

    parameter_groups = (
        tuple(model.trunk.named_parameters()),
        tuple(model.category_head.named_parameters()),
        tuple(model.expiry_head.named_parameters()),
    )
    for group in parameter_groups:
        assert group
        for _name, parameter in group:
            assert parameter.grad is not None
            assert torch.count_nonzero(parameter.grad).item() > 0


def test_predict_returns_canonical_outcome_distribution() -> None:
    """runtime 入口只接受 BeliefState+horizon，并保留 track identity。"""

    model = DenseOutcomeModel(_config(), belief_embedding_dim=3)
    result = model.predict(_belief(), 16.0)

    assert isinstance(result, OutcomeDistribution)
    assert result.track_id == "track-1"
    assert result.horizon_ms == 16.0
    assert sum(
        (
            result.p_invalid,
            result.p_miss,
            result.p_low_score,
            result.p_medium_score,
            result.p_high_score,
        )
    ) == pytest.approx(1.0)
    assert result.expected_score >= 0.0
    assert result.variance >= 0.0


@pytest.mark.parametrize(
    ("belief", "horizon", "error"),
    (
        (torch.zeros(3), torch.zeros(1), ValueError),
        (torch.zeros(1, 2), torch.zeros(1), ValueError),
        (torch.zeros(1, 3), torch.zeros(1, 1), ValueError),
        (torch.zeros(1, 3), torch.tensor([-1.0]), ValueError),
        (torch.zeros(1, 3), torch.tensor([float("nan")]), ValueError),
        (torch.zeros(1, 3, dtype=torch.float64), torch.zeros(1), TypeError),
        (torch.zeros(1, 3), torch.zeros(1, dtype=torch.float64), TypeError),
    ),
)
def test_forward_rejects_invalid_shape_dtype_and_values(
    belief: torch.Tensor,
    horizon: torch.Tensor,
    error: type[Exception],
) -> None:
    """forward 必须拒绝错误形状、dtype 与非有限输入。"""

    model = DenseOutcomeModel(_config(), belief_embedding_dim=3)
    with pytest.raises(error):
        model(belief, horizon)


def test_predict_rejects_wrong_embedding_and_horizon() -> None:
    """predict 必须拒绝维度不符的 belief 和非法 horizon。"""

    model = DenseOutcomeModel(_config(), belief_embedding_dim=4)
    with pytest.raises(ValueError):
        model.predict(_belief(), 16.0)
    model = DenseOutcomeModel(_config(), belief_embedding_dim=3)
    with pytest.raises(ValueError):
        model.predict(_belief(), -1.0)
    with pytest.raises(TypeError):
        model.predict(_belief(), True)


def test_model_ast_has_no_legacy_oracle_gt_smet_or_any() -> None:
    """runtime 模型不得依赖训练 oracle、GT 或 legacy 稀疏实现。"""

    tree = ast.parse(_MODEL_PATH.read_text(encoding="utf-8"))
    forbidden_names = {"Any", "SMET", "GroundTruthObject", "OutcomeOracle"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(
                not alias.name.startswith(("osu_v2", "traning.core.outcome.oracle"))
                for alias in node.names
            )
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert not module.startswith(("osu_v2", "traning.core.outcome.oracle"))
        if isinstance(node, ast.Name):
            assert node.id not in forbidden_names
        if isinstance(node, ast.Attribute):
            assert node.attr not in {"ground_truth", "target_category"}
