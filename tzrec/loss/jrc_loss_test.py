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

from tzrec.loss.jrc_loss import JRCLoss


class JRCLossTest(unittest.TestCase):
    def test_jrc_loss(self) -> None:
        loss_class = JRCLoss()
        logits = torch.tensor(
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
        labels = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1])
        session_ids = torch.tensor([1, 1, 1, 1, 2, 2, 2, 2], dtype=torch.int8)
        loss = loss_class(logits, labels, session_ids)
        self.assertEqual(0.7199, round(loss.item(), 4))


class JRCLossTestReduceNone(unittest.TestCase):
    def test_jrc_loss_reduce_none(self) -> None:
        loss_class = JRCLoss(reduction="none")
        logits = torch.tensor(
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
        labels = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1])
        session_ids = torch.tensor([1, 1, 1, 1, 2, 2, 2, 2], dtype=torch.int8)
        loss = loss_class(logits, labels, session_ids)

        self.assertEqual(0.7199, round(torch.mean(loss).item(), 4))


if __name__ == "__main__":
    unittest.main()
