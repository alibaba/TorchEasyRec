# Copyright (c) 2026, Alibaba Group;
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
import torchmetrics

from tzrec.models.sid_model import BaseSidModel


class UpdateUniqueSidRatioTest(unittest.TestCase):
    """Unit-test the codebook-coverage metric helper.

    ``_update_unique_sid_ratio`` is pure tensor logic over
    ``self._metric_modules`` (no proto dependency), so it is testable without
    a full ``BaseSidModel`` config — full model coverage waits on the concrete
    subclasses in the follow-up PRs.
    """

    def _bare_model(self) -> BaseSidModel:
        # Bypass __init__ (which needs a pipeline config); only the metric
        # module the helper touches needs to exist.
        model = BaseSidModel.__new__(BaseSidModel)
        model._metric_modules = {"unique_sid_ratio": torchmetrics.MeanMetric()}
        return model

    def test_empty_batch_is_noop(self) -> None:
        model = self._bare_model()
        model._update_unique_sid_ratio(torch.empty(0, 3, dtype=torch.long))
        # The B == 0 guard returns early -> no sample recorded.
        self.assertEqual(model._metric_modules["unique_sid_ratio"].weight.item(), 0.0)

    def test_ratio_on_known_duplicates(self) -> None:
        model = self._bare_model()
        # 3 unique rows out of 4 -> ratio 0.75.
        codes = torch.tensor([[1, 2], [1, 2], [3, 4], [5, 6]])
        model._update_unique_sid_ratio(codes)
        self.assertAlmostEqual(
            model._metric_modules["unique_sid_ratio"].compute().item(), 0.75, places=6
        )


if __name__ == "__main__":
    unittest.main()
