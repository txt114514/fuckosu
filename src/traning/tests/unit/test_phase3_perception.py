"""验证 Perception 的梯度、坐标契约、identity 监督和无 GT runtime。"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from traning.conf import PerceptionConfig
from traning.state import RuntimeFrame
from traning.core.perception import (
    DensePerceptionOutput,
    PerceptionLossWeights,
    PerceptionModel,
    PerceptionRuntime,
    PerceptionTargets,
    compute_perception_loss,
    decode_candidates,
)
from traning.core.perception.models import FusedFeatureOutput, SpatialHead


def _dense_output(
    *,
    height: int = 4,
    width: int = 6,
    embedding_dim: int = 3,
) -> DensePerceptionOutput:
    """构造可精确控制峰值与坐标的稠密输出。"""

    scalar = torch.full((1, 1, height, width), -20.0)
    type_logits = torch.full((1, 4, height, width), -20.0)
    type_logits[:, 3] = 20.0
    embedding = torch.zeros((1, embedding_dim, height, width))
    embedding[:, 0] = 1.0
    return DensePerceptionOutput(
        center_logits=scalar.clone(),
        visibility_logits=torch.full_like(scalar, 20.0),
        type_logits=type_logits,
        xy_offsets=torch.zeros((1, 2, height, width)),
        ring_logits=scalar.clone(),
        ring_radius=torch.ones_like(scalar),
        slider_logits=scalar.clone(),
        slider_direction=torch.zeros((1, 2, height, width)),
        spinner_logits=scalar.clone(),
        identity_embedding=embedding,
    )


def _set_ring_peak(
    output: DensePerceptionOutput,
    *,
    row: int,
    column: int,
    offset_x: float,
    offset_y: float,
) -> None:
    """只在测试中原地设置一个强 ring 峰值。"""

    output.center_logits[0, 0, row, column] = 20.0
    output.type_logits[0, :, row, column] = torch.tensor((20.0, -20.0, -20.0, -20.0))
    output.xy_offsets[0, :, row, column] = torch.tensor((offset_x, offset_y))
    output.ring_logits[0, 0, row, column] = 20.0


def _targets_for(output: DensePerceptionOutput) -> PerceptionTargets:
    """构造两个显式实例 ID，覆盖所有 dense loss 入口。"""

    batch, _, height, width = output.center_logits.shape
    scalar = torch.zeros((batch, 1, height, width), dtype=output.center_logits.dtype)
    type_indices = torch.full((batch, height, width), -1, dtype=torch.long)
    instance_ids = torch.full((batch, height, width), -1, dtype=torch.long)
    for batch_index in range(batch):
        type_indices[batch_index, 0, 0] = 0
        type_indices[batch_index, 0, 1] = 1
        instance_ids[batch_index, 0, 0] = 10
        instance_ids[batch_index, 0, 1] = 11
    center = scalar.clone()
    center[:, :, 0, :2] = 1.0
    visibility = center.clone()
    ring = scalar.clone()
    ring[:, :, 0, 0] = 1.0
    slider = scalar.clone()
    slider[:, :, 0, 1] = 1.0
    return PerceptionTargets(
        center_heatmap=center,
        visibility=visibility,
        type_indices=type_indices,
        xy_offsets=torch.zeros((batch, 2, height, width)),
        ring=ring,
        ring_radius=torch.ones_like(scalar),
        slider=slider,
        slider_direction=torch.zeros((batch, 2, height, width)),
        spinner=scalar.clone(),
        instance_ids=instance_ids,
    )


def test_unfrozen_global_encoder_receives_end_to_end_gradients() -> None:
    """global_frozen=False 必须真实改变优化图，而不只是保存配置值。"""

    config = PerceptionConfig(
        frame_width=32,
        frame_height=32,
        embedding_dim=8,
        global_frozen=False,
    )
    model = PerceptionModel(config)
    prediction = model(torch.rand((2, 3, 32, 32)))
    result = compute_perception_loss(
        prediction,
        _targets_for(prediction),
        PerceptionLossWeights(),
    )
    result.total.backward()
    global_parameters = tuple(model.global_encoder.parameters())
    assert global_parameters
    assert all(parameter.requires_grad for parameter in global_parameters)
    assert all(parameter.grad is not None for parameter in global_parameters)
    assert (
        sum(float(parameter.grad.abs().sum()) for parameter in global_parameters) > 0.0
    )
    assert not any("structure" in name for name, _ in model.named_modules())


def test_frozen_global_encoder_has_no_grad_but_local_branch_trains() -> None:
    """冻结只作用于 global，不得意外冻结整个 Perception。"""

    config = PerceptionConfig(
        frame_width=32,
        frame_height=32,
        embedding_dim=8,
        global_frozen=True,
    )
    model = PerceptionModel(config)
    prediction = model(torch.rand((1, 3, 32, 32)))
    prediction.center_logits.mean().backward()
    assert all(
        not parameter.requires_grad for parameter in model.global_encoder.parameters()
    )
    assert any(
        parameter.grad is not None and bool(parameter.grad.abs().sum())
        for parameter in model.local_encoder.parameters()
    )


def test_decode_uses_one_anisotropic_cell_mapping_equation() -> None:
    """候选点和 ring 半径均由特征网格统一映到原始帧，而非渲染补丁偏移。"""

    output = _dense_output(height=4, width=6)
    _set_ring_peak(output, row=1, column=1, offset_x=0.25, offset_y=-0.125)
    config = PerceptionConfig(
        frame_width=192,
        frame_height=128,
        score_threshold=0.1,
        nms_radius_px=8.0,
    )
    candidates = decode_candidates(
        output,
        frame_id="frame-1",
        frame_index=1,
        timestamp_ms=16.0,
        frame_width=192,
        frame_height=128,
        config=config,
    )
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.x == pytest.approx(56.0)
    assert candidate.y == pytest.approx(44.0)
    assert candidate.ring is not None
    assert candidate.ring.radius_px == pytest.approx(32.0)

    # 同一网格映到两倍宽、不同高度时，x/y 分别使用各自尺度。
    remapped = decode_candidates(
        output,
        frame_id="frame-2",
        frame_index=2,
        timestamp_ms=32.0,
        frame_width=384,
        frame_height=160,
        config=config,
    )[0]
    assert remapped.x == pytest.approx(112.0)
    assert remapped.y == pytest.approx(55.0)


def test_decode_clamps_extreme_edge_offset_to_pixel_domain() -> None:
    """最后一个 cell 的 +0.5 offset 不得产生等于 frame size 的越界坐标。"""

    output = _dense_output(height=2, width=2)
    _set_ring_peak(output, row=1, column=1, offset_x=0.5, offset_y=0.5)
    candidate = decode_candidates(
        output,
        frame_id="edge",
        frame_index=0,
        timestamp_ms=0.0,
        frame_width=20,
        frame_height=10,
        config=PerceptionConfig(
            frame_width=20,
            frame_height=10,
            score_threshold=0.1,
            nms_radius_px=0.0,
        ),
    )[0]
    assert (candidate.x, candidate.y) == (19.0, 9.0)


def test_perception_runtime_accepts_only_runtime_frame_fields() -> None:
    """正式入口从 RuntimeFrame 到 candidates，全程没有训练 label 参数。"""

    config = PerceptionConfig(
        frame_width=16,
        frame_height=16,
        score_threshold=0.1,
        nms_radius_px=0.0,
    )
    output = _dense_output(height=2, width=2, embedding_dim=config.embedding_dim)
    _set_ring_peak(output, row=0, column=0, offset_x=0.0, offset_y=0.0)

    class FixedModel:
        """返回固定稠密输出的最小感知测试模型。"""

        def __call__(self, image: torch.Tensor) -> DensePerceptionOutput:
            assert image.shape == (1, 3, 16, 16)
            return output

    frame = RuntimeFrame(
        frame_id="runtime-frame",
        frame_index=3,
        timestamp_ms=48.0,
        width=8,
        height=4,
        image_bytes=bytes(8 * 4 * 3),
    )
    candidates = PerceptionRuntime(FixedModel(), config).infer(frame)
    assert len(candidates) == 1
    assert candidates[0].frame_id == frame.frame_id
    assert candidates[0].candidate_id.startswith("runtime-frame:candidate:")


def test_perception_source_has_no_legacy_or_gt_runtime_dependency() -> None:
    """静态阻止旧接口或 GT-only 名称重新进入正式 Perception 源码。"""

    package_root = Path(__file__).resolve().parents[2] / "core/perception"
    forbidden_runtime_names = {
        "temporal_target",
        "selected_candidate_id",
        "hit_objects",
    }
    for path in package_root.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.module is None or not node.module.startswith("osu_v2"), (
                    path
                )
            if isinstance(node, ast.Import):
                assert all(
                    not alias.name.startswith("osu_v2") for alias in node.names
                ), path
        if "/runtime/" in path.as_posix() or "/decode/" in path.as_posix():
            assert forbidden_runtime_names.isdisjoint(source.split()), path


def test_pretrained_global_without_weights_is_rejected() -> None:
    """禁止随机初始化后冒充 pretrained，或随即被错误冻结。"""

    with pytest.raises(ValueError, match="权重来源"):
        PerceptionModel(PerceptionConfig(global_pretrained=True))


def test_spatial_head_never_publishes_zero_identity_or_direction_vectors() -> None:
    """零输出极端情况也必须提供 tracking 可计算 cosine 的单位向量。"""

    head = SpatialHead(in_channels=4, embedding_dim=3)
    with torch.no_grad():
        head.identity_head.weight.zero_()
        head.identity_head.bias.zero_()
        slider_head = head.heads["slider_direction"]
        slider_head.weight.zero_()
        slider_head.bias.zero_()
    output = head(
        FusedFeatureOutput(
            dense=torch.zeros((1, 4, 2, 2)),
            global_context=torch.zeros((1, 4, 2, 2)),
        )
    )
    identity_norms = torch.linalg.vector_norm(output.identity_embedding, dim=1)
    direction_norms = torch.linalg.vector_norm(output.slider_direction, dim=1)
    assert torch.allclose(identity_norms, torch.ones_like(identity_norms))
    assert torch.allclose(direction_norms, torch.ones_like(direction_norms))


def test_identity_loss_pulls_same_instance_across_temporal_batch() -> None:
    """同一 object_id 在相邻帧的 embedding 不同，必须产生跨帧 pull 损失。"""

    scalar = torch.zeros((2, 1, 1, 1))
    embedding = torch.tensor(
        (
            (((1.0,),), ((0.0,),)),
            (((0.0,),), ((1.0,),)),
        )
    )
    prediction = DensePerceptionOutput(
        center_logits=scalar.clone(),
        visibility_logits=scalar.clone(),
        type_logits=torch.zeros((2, 4, 1, 1)),
        xy_offsets=torch.zeros((2, 2, 1, 1)),
        ring_logits=scalar.clone(),
        ring_radius=scalar.clone(),
        slider_logits=scalar.clone(),
        slider_direction=torch.ones((2, 2, 1, 1)),
        spinner_logits=scalar.clone(),
        identity_embedding=embedding,
    )
    targets = PerceptionTargets(
        center_heatmap=torch.ones_like(scalar),
        visibility=torch.ones_like(scalar),
        type_indices=torch.zeros((2, 1, 1), dtype=torch.long),
        xy_offsets=torch.zeros((2, 2, 1, 1)),
        ring=torch.ones_like(scalar),
        ring_radius=torch.ones_like(scalar),
        slider=torch.zeros_like(scalar),
        slider_direction=torch.zeros((2, 2, 1, 1)),
        spinner=torch.zeros_like(scalar),
        instance_ids=torch.full((2, 1, 1), 7, dtype=torch.long),
    )

    loss = compute_perception_loss(
        prediction,
        targets,
        PerceptionLossWeights(),
    )
    assert float(loss.identity) > 0.1
