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

import numpy as np
import torch

from tzrec.metrics.xauc import XAUC


class XAUCTest(unittest.TestCase):
    def test_xauc_1(self):
        metric = XAUC(sample_ratio=1.0, in_batch=True)
        preds = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        target = torch.tensor([0.2, 0.3, 0.5, 0.4, 0.3, 0.2, 0.8, 0.9])
        metric.update(preds, target)
        value = metric.compute()
        torch.testing.assert_close(value, torch.tensor(0.6786), rtol=1e-4, atol=1e-4)

    def test_xauc_2(self):
        res = []

        for _ in range(200):
            metric = XAUC(sample_ratio=1.0)
            preds = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
            target = torch.tensor([0.2, 0.3, 0.5, 0.4, 0.3, 0.2, 0.8, 0.9])
            metric.update(preds, target)
            value = metric.compute()
            res.append(value)
        mean_value = np.mean(res)

        torch.testing.assert_close(mean_value, 0.6786, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    unittest.main()
