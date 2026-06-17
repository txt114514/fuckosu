from __future__ import annotations

import unittest

import torch

from traning.Lib.models import CausalTemporalModel


class CausalTemporalTests(unittest.TestCase):
    def test_future_frames_do_not_change_past_outputs(self) -> None:
        torch.manual_seed(123)
        model = CausalTemporalModel(
            input_size=5, hidden_size=7, layers=2, candidate_slots=3
        )
        sequence = torch.randn(5, 1, 5)
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


if __name__ == "__main__":
    unittest.main()
