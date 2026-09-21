# Copyright (c) 2024, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import unittest

import torch
from parameterized import parameterized

from tzrec.loss.jrc_loss import JRCLoss
from tzrec.utils.test_util import parameterized_name_func

_LOGITS = torch.tensor(
    [
        [0.9, 0.1],
        [0.5, 0.5],
        [0.3, 0.7],
        [0.2, 0.8],
        [0.8, 0.2],
        [0.55, 0.45],
        [0.33, 0.67],
        [0.55, 0.45],
    ],
    dtype=torch.float32,
)
_LABELS = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1])
_LENGTHS = torch.tensor([4, 4])


def _reference_session_loss(
    logits: torch.Tensor, labels: torch.Tensor, session_ids: torch.Tensor
) -> torch.Tensor:
    """Per-sample session term as the original batch-mask formulation.

    Each positive competes with the negatives of its session on the positive
    logit, each negative with the positives of its session on the negative
    logit, the sample itself being the softmax target.
    """
    mask = session_ids.unsqueeze(1) == session_ids.unsqueeze(0)
    is_pos = labels == 1
    same_and_other_class = mask & (is_pos.unsqueeze(1) != is_pos.unsqueeze(0))
    keep = same_and_other_class | torch.eye(labels.numel(), dtype=torch.bool)
    own_channel = torch.where(is_pos, 1, 0)
    rows = logits[:, own_channel].T
    rows = rows.masked_fill(~keep, -1e9)
    return torch.nn.functional.cross_entropy(
        rows, torch.arange(labels.numel()), reduction="none"
    )


class JRCLossTest(unittest.TestCase):
    def test_jrc_loss(self) -> None:
        loss = JRCLoss()(_LOGITS, _LABELS, _LENGTHS)
        self.assertEqual(0.7199, round(loss.item(), 4))

    def test_jrc_loss_reduce_none(self) -> None:
        loss = JRCLoss(reduction="none")(_LOGITS, _LABELS, _LENGTHS)
        self.assertEqual((8,), tuple(loss.shape))
        self.assertEqual(0.7199, round(torch.mean(loss).item(), 4))

    @parameterized.expand(
        [
            # sessions with both classes, all positives, all negatives, one row
            [
                [5, 3, 4, 1, 6],
                [1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1],
            ],
            [[19], [0] * 18 + [1]],
            [[7, 7, 5], [1] * 7 + [0] * 7 + [1, 0, 1, 0, 1]],
        ],
        name_func=parameterized_name_func,
    )
    def test_matches_batch_mask_formulation(self, lengths, labels) -> None:
        torch.manual_seed(0)
        lengths_t = torch.tensor(lengths)
        labels_t = torch.tensor(labels)
        logits = torch.randn(labels_t.numel(), 2, requires_grad=True)
        session_ids = torch.repeat_interleave(torch.arange(len(lengths)), lengths_t)

        loss = JRCLoss(alpha=0.3, reduction="none")(logits, labels_t, lengths_t)
        ce = torch.nn.functional.cross_entropy(logits, labels_t, reduction="none")
        expected = 0.3 * ce + 0.7 * _reference_session_loss(
            logits, labels_t, session_ids
        )
        torch.testing.assert_close(loss, expected)

        (grad,) = torch.autograd.grad(loss.sum(), logits)
        (expected_grad,) = torch.autograd.grad(expected.sum(), logits)
        torch.testing.assert_close(grad, expected_grad)

    def test_segment_ids_in_any_order(self) -> None:
        torch.manual_seed(0)
        lengths = torch.tensor([5, 3, 4])
        labels = torch.tensor([1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1])
        logits = torch.randn(labels.numel(), 2)
        perm = torch.randperm(labels.numel())
        segment_ids = torch.repeat_interleave(torch.arange(3), lengths)[perm]

        loss = JRCLoss(reduction="none")(
            logits[perm], labels[perm], lengths, segment_ids
        )
        expected = JRCLoss(reduction="none")(logits, labels, lengths)[perm]
        torch.testing.assert_close(loss, expected)


if __name__ == "__main__":
    unittest.main()
