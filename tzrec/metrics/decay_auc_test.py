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

from tzrec.metrics.decay_auc import DecayAUC


class DecayAUCTest(unittest.TestCase):
    def test_decay_auc(self):
        metric = DecayAUC(decay=0.9, thresholds=10)
        preds = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        target = torch.tensor([1, 0, 1, 0, 0, 0, 1, 1])
        target2 = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0])
        metric.update(preds, target)
        value = metric.compute()
        torch.testing.assert_close(value, torch.tensor(0.5625))
        metric.update(preds, target)
        value = metric.compute()
        torch.testing.assert_close(value, torch.tensor(0.5625))
        metric.update(preds, target2)
        value = metric.compute()
        torch.testing.assert_close(value, torch.tensor(0.5437), rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
