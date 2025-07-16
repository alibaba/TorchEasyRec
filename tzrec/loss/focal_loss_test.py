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

import unittest

import torch

from tzrec.loss.focal_loss import BinaryFocalLoss


class BinaryFocalLossTest(unittest.TestCase):
    def test_binary_focal_loss(self) -> None:
        loss_class = BinaryFocalLoss(gamma=2.0, alpha=0.5)
        logits = torch.tensor(
            [0.9, 0.5, 0.3, 0.2, 0.8, 0.5, 0.3, 0.5],
            dtype=torch.float32,
        )
        labels = torch.tensor([1, 1, 1, 0, 1, 0, 1, 0], dtype=torch.float)
        loss = loss_class(logits, labels)
        self.assertEqual(0.083, round(loss.item(), 4))


class BinaryFocalLossTestReduceNone(unittest.TestCase):
    def test_binary_focal_loss_reduce_none(self) -> None:
        loss_class = BinaryFocalLoss(gamma=2.0, alpha=0.5, reduction="none")
        logits = torch.tensor(
            [0.9, 0.5, 0.3, 0.2, 0.8, 0.5, 0.3, 0.5],
            dtype=torch.float32,
        )
        labels = torch.tensor([1, 1, 1, 0, 1, 0, 1, 0], dtype=torch.float)
        loss = loss_class(logits, labels)

        self.assertEqual(0.083, round(torch.mean(loss).item(), 4))


if __name__ == "__main__":
    unittest.main()
