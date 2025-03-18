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

from tzrec.datasets.utils import calc_slice_position


class DatasetUtilsTest(unittest.TestCase):
    def test_calc_slice_position(self):
        num_tables = 81
        num_workers = 8
        batch_size = 10
        remain_row_counts = [0] * num_workers
        worker_row_counts = [0] * num_workers
        for i in range(num_tables):
            for j in range(num_workers):
                start, end, remain_row_counts[j] = calc_slice_position(
                    row_count=81,
                    slice_id=j,
                    slice_count=num_workers,
                    batch_size=batch_size,
                    drop_redundant_bs_eq_one=True if i == num_tables - 1 else False,
                    pre_total_remain=remain_row_counts[j],
                )
                worker_row_counts[j] += end - start
        self.assertTrue(np.all(np.ceil(np.array(worker_row_counts) / batch_size) == 82))
        self.assertEqual(sum(worker_row_counts), num_tables * 81 - 1)


if __name__ == "__main__":
    unittest.main()
