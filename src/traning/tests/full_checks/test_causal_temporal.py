"""验证时序模型的因果性、状态传递与未来信息隔离。"""

from __future__ import annotations

import unittest

import torch

from traning.lib.models import CausalTemporalModel, DynamicSparseLinear


class CausalTemporalTests(unittest.TestCase):
    def test_future_frames_do_not_change_past_outputs(self) -> None:
        torch.manual_seed(123)
        model = CausalTemporalModel(
            input_size=5, hidden_size=7, layers=2, candidate_slots=3
        )
        sequence = torch.randn(5, 1, 5)
        # 同一前缀分别独立执行和作为完整序列的一部分执行，可直接捕获
        # attention/RNN 意外读取未来帧的因果性回归。
        prefix_outputs, _ = model(sequence[:3])
        full_outputs, _ = model(sequence)
        for left, right in zip(prefix_outputs, full_outputs[:3]):
            self.assertTrue(torch.allclose(left.action_logits, right.action_logits))
            self.assertTrue(torch.allclose(left.x, right.x))

    def test_reset_state_repeats_output(self) -> None:
        model = CausalTemporalModel(
            input_size=4, hidden_size=6, layers=1, candidate_slots=2
        )
        features = torch.randn(1, 4)
        state = model.initial_state(1, "cpu")
        first, _ = model.step(features, state)
        reset, _ = model.step(features, model.initial_state(1, "cpu"))
        self.assertTrue(torch.allclose(first.action_logits, reset.action_logits))

    def test_batch_size_one_runs(self) -> None:
        model = CausalTemporalModel(input_size=4)
        state = model.initial_state(1, "cpu")
        output, next_state = model.step(torch.randn(1, 4), state)
        self.assertEqual(output.next_hidden_state.shape, next_state.shape)

    def test_smet_sparse_heads_run(self) -> None:
        model = CausalTemporalModel(
            input_size=4,
            hidden_size=6,
            layers=1,
            candidate_slots=2,
            smet_enabled=True,
            smet_sparsity=0.50,
            smet_update_interval=1,
        )
        self.assertTrue(
            any(isinstance(module, DynamicSparseLinear) for module in model.modules())
        )
        outputs, _ = model(torch.randn(3, 1, 4))
        self.assertEqual(outputs[-1].selected_candidate_logits.shape, (1, 2))

    def test_smet_sparse_heads_backward_after_dynamic_updates(self) -> None:
        torch.manual_seed(321)
        model = CausalTemporalModel(
            input_size=4,
            hidden_size=6,
            layers=1,
            candidate_slots=2,
            smet_enabled=True,
            smet_sparsity=0.50,
            smet_update_interval=1,
        )
        model.train()
        outputs, _ = model(torch.randn(4, 1, 4))
        loss = sum(
            output.action_logits.sum()
            + output.selected_candidate_logits.sum()
            + output.x.sum()
            + output.y.sum()
            + output.time_offset_ms.sum()
            for output in outputs
        )

        loss.backward()

        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_smet_training_forward_does_not_mutate_mask_buffers(self) -> None:
        torch.manual_seed(654)
        model = CausalTemporalModel(
            input_size=4,
            hidden_size=6,
            layers=1,
            candidate_slots=2,
            smet_enabled=True,
            smet_sparsity=0.50,
            smet_update_interval=1,
        )
        model.train()
        sparse_layers = [
            module for module in model.modules() if isinstance(module, DynamicSparseLinear)
        ]
        # autograd 会用 Tensor version 检测原地修改；显式记录版本可在错误
        # 尚未触发反向异常前定位动态稀疏 mask 的隐式 mutation。
        versions = [layer.mask._version for layer in sparse_layers]

        outputs, _ = model(torch.randn(4, 1, 4))
        loss = sum(output.action_logits.sum() for output in outputs)
        loss.backward()

        self.assertEqual(
            [layer.mask._version for layer in sparse_layers],
            versions,
        )

    def test_mutating_future_window_does_not_change_prefix(self) -> None:
        torch.manual_seed(456)
        model = CausalTemporalModel(
            input_size=6, hidden_size=8, layers=2, candidate_slots=4
        )
        sequence = torch.randn(8, 2, 6)
        mutated = sequence.clone()
        # 大幅放大未来扰动，避免正常随机噪声过小而掩盖微弱的信息泄漏。
        mutated[4:] = torch.randn_like(mutated[4:]) * 100.0
        original_outputs, _ = model(sequence)
        mutated_outputs, _ = model(mutated)
        for left, right in zip(original_outputs[:4], mutated_outputs[:4]):
            self.assertTrue(torch.allclose(left.action_logits, right.action_logits))
            self.assertTrue(torch.allclose(left.selected_candidate_logits, right.selected_candidate_logits))

    def test_segmented_execution_matches_continuous_and_batch_isolated(self) -> None:
        torch.manual_seed(789)
        model = CausalTemporalModel(
            input_size=3, hidden_size=5, layers=1, candidate_slots=2
        )
        sequence = torch.randn(6, 2, 3)
        continuous, _ = model(sequence)
        state = model.initial_state(2, "cpu")
        segmented = []
        for frame in sequence[:3]:
            output, state = model.step(frame, state)
            segmented.append(output)
        for frame in sequence[3:]:
            output, state = model.step(frame, state)
            segmented.append(output)
        for left, right in zip(continuous, segmented):
            self.assertTrue(torch.allclose(left.action_logits, right.action_logits))

        # 只改 batch 中第二条序列，第一条输出必须保持不变，防止隐藏状态
        # 在 batch 维度被错误共享。
        changed = sequence.clone()
        changed[:, 1] = torch.randn_like(changed[:, 1]) * 50.0
        changed_outputs, _ = model(changed)
        for left, right in zip(continuous, changed_outputs):
            self.assertTrue(torch.allclose(left.action_logits[0], right.action_logits[0]))


if __name__ == "__main__":
    unittest.main()
