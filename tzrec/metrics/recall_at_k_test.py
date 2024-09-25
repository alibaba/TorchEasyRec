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

from tzrec.metrics.recall_at_k import RecallAtK


class RecallAtKTest(unittest.TestCase):
    def test_recall_at_k(self):
        metric = RecallAtK(top_k=2)
        preds = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]])
        target = torch.tensor(
            [[True, False, False, False], [True, False, False, False]]
        )
        metric.update(preds, target)
        value = metric.compute()
        torch.testing.assert_close(value, torch.tensor(0.5))


if __name__ == "__main__":
    unittest.main()
