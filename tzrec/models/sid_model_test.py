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

from tzrec.models.sid_model import _masked_mean


class MaskedMeanTest(unittest.TestCase):
    """Tests for the shared ``BaseSidModel`` masked-mean reduction."""

    def test_no_mask_is_plain_mean(self) -> None:
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        torch.testing.assert_close(_masked_mean(x), x.mean())

    def test_mask_averages_over_valid_rows_only(self) -> None:
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        mask = torch.tensor([True, False, True, False])
        torch.testing.assert_close(_masked_mean(x, mask), torch.tensor(2.0))  # (1+3)/2

    def test_empty_mask_is_zero_not_nan(self) -> None:
        out = _masked_mean(
            torch.tensor([1.0, 2.0, 3.0]), torch.zeros(3, dtype=torch.bool)
        )
        self.assertEqual(out.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
