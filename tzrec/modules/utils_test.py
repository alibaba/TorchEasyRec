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

from tzrec.modules.utils import div_no_nan


class ModuleUtilsTest(unittest.TestCase):
    def test_div_no_nan(self):
        torch.testing.assert_close(
            div_no_nan(
                torch.tensor([1, 1, 1], dtype=torch.float32),
                torch.tensor([2, 4, 8], dtype=torch.float32),
            ),
            torch.tensor([0.5, 0.25, 0.125], dtype=torch.float32),
        )

    def test_div_no_nan_by_zero(self):
        torch.testing.assert_close(
            div_no_nan(
                torch.tensor([1, 1, 1], dtype=torch.float32),
                torch.tensor([0, 0, 0], dtype=torch.float32),
            ),
            torch.tensor([0, 0, 0], dtype=torch.float32),
        )


if __name__ == "__main__":
    unittest.main()
