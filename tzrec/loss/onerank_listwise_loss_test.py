# Copyright (c) 2025, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for ``tzrec.loss.onerank_listwise_loss`` (CPU only)."""

import math
import unittest
from typing import List

import torch
from parameterized import parameterized

from tzrec.loss.onerank_listwise_loss import OneRankListwiseLoss

# (lengths, labels) pairs covering every masking branch of the loss.
_CASES = [
    ([4], [0, 1, 0, 0]),  # single positive
    ([5], [1, 0, 1, 1, 0]),  # multiple positives
    ([3], [1, 1, 1]),  # no negative -> dropped
    ([3], [0, 0, 0]),  # no positive -> dropped
    ([0, 4], [1, 0, 0, 0]),  # empty request
    ([6, 1, 3], [0, 1, 0, 0, 1, 0, 1, 0, 1, 0]),  # length-1 request -> dropped
    ([2, 5, 3, 4], [1, 0] + [0, 1, 1, 0, 0] + [1, 1, 1] + [0, 0, 0, 1]),
]


def _reference_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    lengths: List[int],
    scale: torch.Tensor,
) -> torch.Tensor:
    """Straight transcription of the definition, one request at a time.

    Requests with no positive or no negative are dropped; the rest average
    the negated log-softmax of their positives.
    """
    terms = []
    start = 0
    for length in lengths:
        end = start + length
        seg_logits, seg_labels = logits[start:end], labels[start:end]
        start = end
        num_pos = int((seg_labels != 0).sum())
        if num_pos == 0 or num_pos == length:
            continue
        log_probs = torch.log_softmax(seg_logits * scale, dim=0)
        terms.append(-log_probs[seg_labels != 0].mean())
    if not terms:
        return logits.new_zeros(())
    return torch.stack(terms).sum() / len(terms)


class OneRankListwiseLossTest(unittest.TestCase):
    """Tests for the per-request list-wise InfoNCE term."""

    @parameterized.expand([(0.07, True), (1.0, True), (0.5, False)])
    def test_matches_per_request_reference(
        self, temperature: float, learnable: bool
    ) -> None:
        """Value and gradient agree with a per-request loop reference.

        The gradient half is the real assertion: it is what proves the
        detached per-segment max shift is exactly neutral.
        """
        module = OneRankListwiseLoss(
            temperature_init=temperature, learnable_temperature=learnable
        )
        scale = module.logit_scale.detach().clamp(max=math.log(100)).exp().double()
        for lengths_list, labels_list in _CASES:
            with self.subTest(lengths=lengths_list):
                total = sum(lengths_list)
                torch.manual_seed(total * 17 + len(lengths_list))
                logits = torch.randn(total, dtype=torch.float64, requires_grad=True)
                labels = torch.tensor(labels_list, dtype=torch.float64)
                lengths = torch.tensor(lengths_list, dtype=torch.int64)

                got = module(logits, labels, lengths)
                want = _reference_loss(logits, labels, lengths_list, scale)
                torch.testing.assert_close(got, want, rtol=1e-10, atol=1e-12)

                got_grad = torch.autograd.grad(got, logits, retain_graph=True)[0]
                if want.requires_grad:
                    want_grad = torch.autograd.grad(want, logits)[0]
                else:
                    # Every request was dropped; the reference is a constant.
                    want_grad = torch.zeros_like(logits)
                torch.testing.assert_close(got_grad, want_grad, rtol=1e-9, atol=1e-11)

    def test_dropped_requests_leak_no_gradient(self) -> None:
        """An all-positive or all-negative request contributes nothing.

        Both are masked, not skipped, so this is what verifies the mask
        actually zeroes their contribution rather than merely shrinking it.
        """
        module = OneRankListwiseLoss(temperature_init=1.0, learnable_temperature=False)
        lengths = torch.tensor([3, 3, 3], dtype=torch.int64)
        # usable | all positive | all negative
        labels = torch.tensor([1.0, 0, 0, 1, 1, 1, 0, 0, 0])
        logits = torch.randn(9, requires_grad=True)

        module(logits, labels, lengths).backward()
        self.assertEqual(logits.grad[3:].abs().max().item(), 0.0)
        self.assertGreater(logits.grad[:3].abs().max().item(), 0.0)

        # And the value equals that of the usable request on its own.
        detached = logits.detach()
        torch.testing.assert_close(
            module(detached, labels, lengths),
            module(detached[:3], labels[:3], torch.tensor([3])),
        )

    def test_all_dropped_batch_is_finite_zero(self) -> None:
        """A batch with nothing to learn from must be 0, not 0/0 -> NaN."""
        module = OneRankListwiseLoss(temperature_init=1.0)
        lengths = torch.tensor([0, 2, 2], dtype=torch.int64)
        labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
        logits = torch.randn(4, requires_grad=True)

        value = module(logits, labels, lengths)
        self.assertTrue(torch.isfinite(value))
        self.assertEqual(value.item(), 0.0)
        value.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(logits.grad.abs().max().item(), 0.0)
        self.assertTrue(torch.isfinite(module.logit_scale.grad))

    def test_temperature_clamp_keeps_gradients_finite(self) -> None:
        """A runaway temperature must not overflow to +Inf -> NaN grad."""
        module = OneRankListwiseLoss(temperature_init=0.07)
        with torch.no_grad():
            module.logit_scale.fill_(1e4)
        lengths = torch.tensor([4, 4], dtype=torch.int64)
        labels = torch.tensor([1.0, 0, 0, 0, 0, 1, 0, 0])
        logits = (torch.randn(8) * 3.0).requires_grad_(True)

        value = module(logits, labels, lengths)
        self.assertTrue(torch.isfinite(value))
        value.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        # Clamped, so the scale itself is out of the gradient path.
        self.assertEqual(module.logit_scale.grad.item(), 0.0)

    def test_per_request_max_shift(self) -> None:
        """A request far below the batch max must still be well defined.

        With a single batch-wide max shift its exponentials underflow to
        zero and the softmax denominator becomes 0/0.
        """
        module = OneRankListwiseLoss(temperature_init=1.0, learnable_temperature=False)
        lengths = torch.tensor([3, 3], dtype=torch.int64)
        labels = torch.tensor([1.0, 0, 0, 1, 0, 0])
        logits = torch.tensor(
            [500.0, 499.0, 498.0, -500.0, -501.0, -502.0], requires_grad=True
        )

        got = module(logits, labels, lengths)
        want = _reference_loss(logits, labels, [3, 3], torch.tensor(1.0))
        torch.testing.assert_close(got, want)
        got.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_temperature_parameterization(self) -> None:
        """``learnable_temperature`` picks parameter vs buffer."""
        learnable = OneRankListwiseLoss(temperature_init=0.07)
        self.assertEqual([n for n, _ in learnable.named_parameters()], ["logit_scale"])
        torch.testing.assert_close(
            learnable.logit_scale.detach(), torch.tensor(math.log(1 / 0.07))
        )

        fixed = OneRankListwiseLoss(temperature_init=0.5, learnable_temperature=False)
        self.assertEqual(list(fixed.named_parameters()), [])
        self.assertEqual([n for n, _ in fixed.named_buffers()], ["logit_scale"])

    def test_non_positive_temperature_raises(self) -> None:
        """``log(1 / T)`` is undefined for T <= 0, so reject it up front."""
        for bad in (0.0, -1.0):
            with self.assertRaisesRegex(ValueError, "temperature_init"):
                OneRankListwiseLoss(temperature_init=bad)

    def test_loss_weight_is_a_scalar_multiplier(self) -> None:
        """``loss_weight`` carries the global-average-loss rescaling."""
        module = OneRankListwiseLoss(temperature_init=1.0, learnable_temperature=False)
        lengths = torch.tensor([3, 4], dtype=torch.int64)
        labels = torch.tensor([1.0, 0, 0, 0, 1, 0, 0])
        logits = torch.randn(7)

        torch.testing.assert_close(
            module(logits, labels, lengths, torch.tensor(2.5)),
            module(logits, labels, lengths) * 2.5,
        )

    def test_integer_labels_are_accepted(self) -> None:
        """Labels arrive as ints from the bitmask decode in some configs."""
        module = OneRankListwiseLoss(temperature_init=1.0, learnable_temperature=False)
        lengths = torch.tensor([4], dtype=torch.int64)
        logits = torch.randn(4)
        torch.testing.assert_close(
            module(logits, torch.tensor([0, 1, 0, 0]), lengths),
            module(logits, torch.tensor([0.0, 1.0, 0.0, 0.0]), lengths),
        )


if __name__ == "__main__":
    unittest.main()
