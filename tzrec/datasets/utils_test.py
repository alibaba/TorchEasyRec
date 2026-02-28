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
import pyarrow as pa
import pyarrow.compute as pc

from tzrec.datasets.utils import (
    calc_remaining_intervals,
    calc_slice_position,
    process_hstu_neg_sample,
    process_hstu_seq_data,
    redistribute_intervals,
)


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

    def test_process_hstu_seq_data(self):
        """Test processing sequence data for HSTU match model."""
        input_data = {"sequence": pa.array(["1;2;3;4", "5;6;7;8", "9;10;11;12"])}

        split, slice_result, training_seq = process_hstu_seq_data(
            input_data=input_data, seq_attr="sequence", seq_str_delim=";"
        )

        # Verify results
        # Test original split sequences
        expected_split_values = [
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "11",
            "12",
        ]
        self.assertEqual(pc.list_flatten(split).to_pylist(), expected_split_values)

        # Test sliced sequences (target items)
        expected_slice_values = ["2", "3", "4", "6", "7", "8", "10", "11", "12"]
        self.assertEqual(slice_result.to_pylist(), expected_slice_values)

        # Test training sequences
        expected_training_seqs = ["1;2;3", "5;6;7", "9;10;11"]
        self.assertEqual(training_seq.to_pylist(), expected_training_seqs)

    def test_process_hstu_neg_sample(self):
        """Test processing negative samples for HSTU match model."""
        input_data = {"sequence": pa.array(["1", "2", "3"])}
        neg_samples = pa.array(["101", "102", "103", "104", "105", "106"])

        result = process_hstu_neg_sample(
            input_data=input_data,
            v=neg_samples,
            neg_sample_num=2,
            seq_str_delim=";",
            seq_attr="sequence",
        )

        expected_results = [
            "1;101;102",
            "2;103;104",
            "3;105;106",
        ]
        self.assertEqual(result.to_pylist(), expected_results)

    def test_process_hstu_neg_sample_with_different_delim(self):
        """Test negative sampling with different delimiter."""
        input_data = {"sequence": pa.array(["1", "2", "3"])}

        neg_samples = pa.array(["101", "102", "103", "104", "105", "106"])

        result = process_hstu_neg_sample(
            input_data=input_data,
            v=neg_samples,
            neg_sample_num=2,
            seq_str_delim="|",
            seq_attr="sequence",
        )

        expected_results = [
            "1|101|102",
            "2|103|104",
            "3|105|106",
        ]
        self.assertEqual(result.to_pylist(), expected_results)


if __name__ == "__main__":
    unittest.main()
