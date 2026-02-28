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


class CheckpointUtilsTest(unittest.TestCase):
    def test_calc_remaining_intervals_no_checkpoint(self):
        """Test remaining intervals when no checkpoint exists."""
        result = calc_remaining_intervals(
            checkpoint_state=None,
            input_path="/data/test.parquet",
            total_rows=1000,
        )
        self.assertEqual(result, [(0, 1000)])

    def test_calc_remaining_intervals_empty_checkpoint(self):
        """Test remaining intervals with empty checkpoint state."""
        result = calc_remaining_intervals(
            checkpoint_state={},
            input_path="/data/test.parquet",
            total_rows=1000,
        )
        self.assertEqual(result, [(0, 1000)])

    def test_calc_remaining_intervals_single_worker(self):
        """Test remaining intervals with single worker checkpoint."""
        # Worker consumed rows 0-499 (checkpoint at row 499)
        checkpoint_state = {"/data/test.parquet:0": 499}
        result = calc_remaining_intervals(
            checkpoint_state=checkpoint_state,
            input_path="/data/test.parquet",
            total_rows=1000,
        )
        # Remaining: [500, 1000)
        self.assertEqual(result, [(500, 1000)])

    def test_calc_remaining_intervals_multiple_workers(self):
        """Test remaining intervals with multiple workers' checkpoints."""
        # 2 workers: worker0 [0, 500), worker1 [500, 1000)
        # Worker0 consumed up to row 299
        # Worker1 consumed up to row 799
        checkpoint_state = {
            "/data/test.parquet:0": 299,
            "/data/test.parquet:500": 799,
        }
        result = calc_remaining_intervals(
            checkpoint_state=checkpoint_state,
            input_path="/data/test.parquet",
            total_rows=1000,
        )
        # Remaining: [300, 500), [800, 1000)
        self.assertEqual(result, [(300, 500), (800, 1000)])

    def test_calc_remaining_intervals_fully_consumed(self):
        """Test remaining intervals when all data is consumed."""
        # Worker consumed all rows up to 999 (last row)
        checkpoint_state = {"/data/test.parquet:0": 999}
        result = calc_remaining_intervals(
            checkpoint_state=checkpoint_state,
            input_path="/data/test.parquet",
            total_rows=1000,
        )
        # No remaining intervals
        self.assertEqual(result, [])

    def test_calc_remaining_intervals_unrelated_path(self):
        """Test remaining intervals when checkpoint is for different path."""
        checkpoint_state = {"/data/other.parquet:0": 499}
        result = calc_remaining_intervals(
            checkpoint_state=checkpoint_state,
            input_path="/data/test.parquet",
            total_rows=1000,
        )
        # Unrelated checkpoint, return full range
        self.assertEqual(result, [(0, 1000)])

    def test_redistribute_intervals_single_worker(self):
        """Test redistribute intervals with single worker."""
        intervals = [(100, 500), (600, 1000)]
        result = redistribute_intervals(intervals, worker_id=0, num_workers=1)
        self.assertEqual(result, [(100, 500), (600, 1000)])

    def test_redistribute_intervals_two_workers(self):
        """Test redistribute intervals among two workers."""
        # Total remaining: 800 rows (400 + 400)
        intervals = [(100, 500), (600, 1000)]

        # Worker 0 gets first half of total rows
        result_w0 = redistribute_intervals(intervals, worker_id=0, num_workers=2)
        # Worker 1 gets second half
        result_w1 = redistribute_intervals(intervals, worker_id=1, num_workers=2)

        # Combined should cover all intervals
        total_rows_w0 = sum(end - start for start, end in result_w0)
        total_rows_w1 = sum(end - start for start, end in result_w1)
        self.assertEqual(total_rows_w0 + total_rows_w1, 800)

    def test_redistribute_intervals_empty_intervals(self):
        """Test redistribute with empty intervals."""
        result = redistribute_intervals([], worker_id=0, num_workers=2)
        self.assertEqual(result, [])

    def test_redistribute_intervals_topology_change(self):
        """Test redistribute when changing from 2 to 3 workers."""
        # Original 2 workers, remaining intervals from their checkpoints
        intervals = [(300, 500), (800, 1000)]  # 400 rows total

        # Now redistribute among 3 workers
        total_rows = 0
        for worker_id in range(3):
            result = redistribute_intervals(intervals, worker_id=worker_id, num_workers=3)
            for start, end in result:
                total_rows += end - start

        self.assertEqual(total_rows, 400)  # All remaining rows accounted for


if __name__ == "__main__":
    unittest.main()
