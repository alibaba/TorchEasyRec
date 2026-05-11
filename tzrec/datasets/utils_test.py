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

from tzrec.datasets.utils import (
    _normalize_type_str,
    build_sampler_input,
    calc_remaining_intervals,
    calc_slice_intervals,
    calc_slice_position,
    combine_candidate_sequence_block,
    get_input_fields_proto,
)
from tzrec.protos import data_pb2
from tzrec.protos.data_pb2 import FieldType


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

    def test_calc_slice_intervals_single_worker(self):
        """Test calc_slice_intervals with single worker."""
        # Simulate intervals [(100, 500), (600, 1000)] via checkpoint_state
        # checkpoint at row 99 means rows 0-99 consumed, remaining starts at 100
        # checkpoint at row 599 means rows 500-599 consumed, remaining starts at 600
        checkpoint_state = {
            "/data/test.parquet:0": 99,
            "/data/test.parquet:500": 599,
        }
        result, _ = calc_slice_intervals(
            total_rows=1000,
            worker_id=0,
            num_workers=1,
            checkpoint_state=checkpoint_state,
            input_path="/data/test.parquet",
        )
        self.assertEqual(result, [(100, 500), (600, 1000)])

    def test_calc_slice_intervals_two_workers(self):
        """Test calc_slice_intervals among two workers."""
        # Total remaining: 800 rows (400 + 400)
        # Intervals: [(100, 500), (600, 1000)]
        checkpoint_state = {
            "/data/test.parquet:0": 99,
            "/data/test.parquet:500": 599,
        }

        # Worker 0 gets first half of total rows
        result_w0, _ = calc_slice_intervals(
            total_rows=1000,
            worker_id=0,
            num_workers=2,
            checkpoint_state=checkpoint_state,
            input_path="/data/test.parquet",
        )
        # Worker 1 gets second half
        result_w1, _ = calc_slice_intervals(
            total_rows=1000,
            worker_id=1,
            num_workers=2,
            checkpoint_state=checkpoint_state,
            input_path="/data/test.parquet",
        )

        # Combined should cover all intervals
        total_rows_w0 = sum(end - start for start, end in result_w0)
        total_rows_w1 = sum(end - start for start, end in result_w1)
        self.assertEqual(total_rows_w0 + total_rows_w1, 800)

    def test_calc_slice_intervals_empty_intervals(self):
        """Test calc_slice_intervals with empty intervals (fully consumed)."""
        # All data consumed: checkpoint at row 999 (last row)
        checkpoint_state = {"/data/test.parquet:0": 999}
        result, _ = calc_slice_intervals(
            total_rows=1000,
            worker_id=0,
            num_workers=2,
            checkpoint_state=checkpoint_state,
            input_path="/data/test.parquet",
        )
        self.assertEqual(result, [])

    def test_calc_slice_intervals_topology_change(self):
        """Test calc_slice_intervals when changing from 2 to 3 workers."""
        # Original 2 workers, remaining intervals from their checkpoints
        # Intervals: [(300, 500), (800, 1000)] = 400 rows total
        checkpoint_state = {
            "/data/test.parquet:0": 299,
            "/data/test.parquet:500": 799,
        }

        # Now redistribute among 3 workers
        total_rows = 0
        for worker_id in range(3):
            result, _ = calc_slice_intervals(
                total_rows=1000,
                worker_id=worker_id,
                num_workers=3,
                checkpoint_state=checkpoint_state,
                input_path="/data/test.parquet",
            )
            for start, end in result:
                total_rows += end - start

        self.assertEqual(total_rows, 400)  # All remaining rows accounted for

    def test_build_sampler_input_string_item_id_no_user_id(self):
        # NegativeSampler-style: no user_id_field; item_id is delimited string.
        input_data = {
            "item_id": pa.array(["1;2", "3"]),
            "label": pa.array([1, 0]),
        }
        out = build_sampler_input(
            input_data,
            item_id_field="item_id",
            user_id_field=None,
            seq_field_delims={"item_id": ";"},
        )
        # input_data is not mutated
        self.assertEqual(input_data["item_id"].to_pylist(), ["1;2", "3"])
        # output is flat
        self.assertEqual(out["item_id"].to_pylist(), ["1", "2", "3"])
        # other keys carry through
        self.assertEqual(out["label"].to_pylist(), [1, 0])
        # returned dict is a different object
        self.assertIsNot(out, input_data)

    def test_build_sampler_input_list_item_id_expands_user_id(self):
        # NegativeSamplerV2 / HardNeg style: item_id arrives as list<int64>;
        # user_id is expanded by per-row positive count.
        input_data = {
            "item_id": pa.array([[1, 2], [3]], type=pa.list_(pa.int64())),
            "user_id": pa.array(["u0", "u1"]),
        }
        out = build_sampler_input(
            input_data,
            item_id_field="item_id",
            user_id_field="user_id",
            seq_field_delims={"item_id": ";"},
        )
        self.assertEqual(out["item_id"].to_pylist(), [1, 2, 3])
        self.assertEqual(out["user_id"].to_pylist(), ["u0", "u0", "u1"])
        # original unchanged
        self.assertEqual(input_data["item_id"].to_pylist(), [[1, 2], [3]])
        self.assertEqual(input_data["user_id"].to_pylist(), ["u0", "u1"])

    def test_build_sampler_input_scalar_item_id_passthrough(self):
        # DSSM-style: item_id is scalar int64 (single positive per row).
        # No flatten, no user_id expansion.
        input_data = {
            "item_id": pa.array([1, 2, 3], type=pa.int64()),
            "user_id": pa.array(["u0", "u1", "u2"]),
        }
        out = build_sampler_input(
            input_data,
            item_id_field="item_id",
            user_id_field="user_id",
            seq_field_delims={"item_id": ";"},
        )
        self.assertEqual(out["item_id"].to_pylist(), [1, 2, 3])
        self.assertEqual(out["user_id"].to_pylist(), ["u0", "u1", "u2"])

    def test_build_sampler_input_item_id_not_in_seq_field_delims(self):
        # If item_id_field has no seq_delim, treat as scalar and pass through.
        input_data = {"item_id": pa.array(["1", "2"])}
        out = build_sampler_input(
            input_data,
            item_id_field="item_id",
            user_id_field=None,
            seq_field_delims={},
        )
        self.assertEqual(out["item_id"].to_pylist(), ["1", "2"])

    def test_build_sampler_input_no_item_id_field(self):
        # Sampler config without item_id_field at all (defensive path).
        input_data = {"a": pa.array([1, 2])}
        out = build_sampler_input(
            input_data,
            item_id_field=None,
            user_id_field=None,
            seq_field_delims={},
        )
        self.assertIsNot(out, input_data)  # still shallow-copied
        self.assertEqual(out["a"].to_pylist(), [1, 2])

    def test_combine_candidate_sequence_block_single_pos(self):
        pos_data = pa.array(["1", "2", "3"])
        negs = pa.array(["10", "20", "30"])
        result, pos_lengths = combine_candidate_sequence_block(
            pos_data=pos_data, negs=negs, seq_delim=";"
        )
        # Last row carries the shared neg pool; others hold pos only.
        self.assertEqual(result.to_pylist(), ["1", "2", "3;10;20;30"])
        np.testing.assert_array_equal(pos_lengths, np.array([1, 1, 1], dtype=np.int32))

    def test_combine_candidate_sequence_block_multivalue_pos(self):
        pos_data = pa.array(["1;2", "3;4;5"])  # 5 positives total
        negs = pa.array(["10", "20"])
        result, pos_lengths = combine_candidate_sequence_block(
            pos_data=pos_data, negs=negs, seq_delim=";"
        )
        # row 0 (i<B-1): "1;2"
        # row 1 (last):  "3;4;5;10;20"
        self.assertEqual(result.to_pylist(), ["1;2", "3;4;5;10;20"])
        np.testing.assert_array_equal(pos_lengths, np.array([2, 3], dtype=np.int32))

    def test_combine_candidate_sequence_block_list_pos(self):
        # pos_data may arrive already as a list array (e.g. from Parquet).
        pos_data = pa.array([[1, 2], [3]], type=pa.list_(pa.int64()))
        negs = pa.array(["10", "20"])
        result, pos_lengths = combine_candidate_sequence_block(
            pos_data=pos_data, negs=negs, seq_delim=";"
        )
        self.assertEqual(result.to_pylist(), ["1;2", "3;10;20"])
        np.testing.assert_array_equal(pos_lengths, np.array([2, 1], dtype=np.int32))

    def test_combine_candidate_sequence_block_list_typed_negs(self):
        # The sampler emits list<T> for negs when the attr's field schema is
        # list-typed (_to_arrow_array wraps each scalar in a 1-element list).
        # combine must flatten that shape.
        pos_data = pa.array(["1", "2"])
        negs = pa.array([[10], [20], [30]], type=pa.list_(pa.int64()))
        result, pos_lengths = combine_candidate_sequence_block(
            pos_data=pos_data, negs=negs, seq_delim=";"
        )
        self.assertEqual(result.to_pylist(), ["1", "2;10;20;30"])
        np.testing.assert_array_equal(pos_lengths, np.array([1, 1], dtype=np.int32))

    def test_combine_candidate_sequence_block_empty_negs(self):
        pos_data = pa.array(["1", "2"])
        negs = pa.array([], type=pa.string())
        result, pos_lengths = combine_candidate_sequence_block(
            pos_data=pos_data, negs=negs, seq_delim=";"
        )
        self.assertEqual(result.to_pylist(), ["1", "2"])
        np.testing.assert_array_equal(pos_lengths, np.array([1, 1], dtype=np.int32))

    def test_normalize_type_str_basic_types(self):
        """Test normalizing basic types."""
        self.assertEqual(_normalize_type_str("int32"), "INT32")
        self.assertEqual(_normalize_type_str("INT64"), "INT64")
        self.assertEqual(_normalize_type_str("string"), "STRING")
        self.assertEqual(_normalize_type_str("float"), "FLOAT")
        self.assertEqual(_normalize_type_str("double"), "DOUBLE")

    def test_normalize_type_str_aliases(self):
        """Test ODPS aliases: BIGINT -> INT64, INT -> INT32."""
        self.assertEqual(_normalize_type_str("BIGINT"), "INT64")
        self.assertEqual(_normalize_type_str("bigint"), "INT64")
        self.assertEqual(_normalize_type_str("INT"), "INT32")
        self.assertEqual(_normalize_type_str("int"), "INT32")

    def test_normalize_type_str_array_types(self):
        """Test array types with aliases."""
        self.assertEqual(_normalize_type_str("ARRAY<BIGINT>"), "ARRAY<INT64>")
        self.assertEqual(_normalize_type_str("ARRAY<INT>"), "ARRAY<INT32>")
        self.assertEqual(_normalize_type_str("ARRAY<INT64>"), "ARRAY<INT64>")
        self.assertEqual(_normalize_type_str("ARRAY<INT32>"), "ARRAY<INT32>")
        self.assertEqual(_normalize_type_str("ARRAY<STRING>"), "ARRAY<STRING>")
        self.assertEqual(_normalize_type_str("array<float>"), "ARRAY<FLOAT>")

    def test_normalize_type_str_nested_array_types(self):
        """Test nested array types."""
        self.assertEqual(
            _normalize_type_str("ARRAY<ARRAY<BIGINT>>"), "ARRAY<ARRAY<INT64>>"
        )
        self.assertEqual(
            _normalize_type_str("ARRAY<ARRAY<INT>>"), "ARRAY<ARRAY<INT32>>"
        )

    def test_normalize_type_str_map_types(self):
        """Test map types with aliases."""
        self.assertEqual(_normalize_type_str("MAP<STRING,BIGINT>"), "MAP<STRING,INT64>")
        self.assertEqual(_normalize_type_str("MAP<STRING,INT>"), "MAP<STRING,INT32>")
        self.assertEqual(_normalize_type_str("MAP<BIGINT,STRING>"), "MAP<INT64,STRING>")
        self.assertEqual(_normalize_type_str("MAP<INT,STRING>"), "MAP<INT32,STRING>")
        self.assertEqual(_normalize_type_str("MAP<BIGINT,BIGINT>"), "MAP<INT64,INT64>")

    def test_normalize_type_str_spaces(self):
        """Test handling of spaces around commas and angle brackets."""
        self.assertEqual(_normalize_type_str("MAP<STRING, INT>"), "MAP<STRING,INT32>")
        self.assertEqual(
            _normalize_type_str("MAP< STRING , BIGINT >"), "MAP<STRING,INT64>"
        )
        self.assertEqual(_normalize_type_str("ARRAY< INT >"), "ARRAY<INT32>")
        self.assertEqual(_normalize_type_str("  BIGINT  "), "INT64")

    def test_get_input_fields_proto_basic_types(self):
        """Test parsing basic types from input_fields_str."""
        data_config = data_pb2.DataConfig()
        data_config.input_fields_str = "user_id:BIGINT;item_id:INT64;label:FLOAT"

        fields = get_input_fields_proto(data_config)

        self.assertEqual(len(fields), 3)
        self.assertEqual(fields[0].input_name, "user_id")
        self.assertEqual(fields[0].input_type, FieldType.INT64)
        self.assertEqual(fields[1].input_name, "item_id")
        self.assertEqual(fields[1].input_type, FieldType.INT64)
        self.assertEqual(fields[2].input_name, "label")
        self.assertEqual(fields[2].input_type, FieldType.FLOAT)

    def test_get_input_fields_proto_array_types(self):
        """Test parsing array types."""
        data_config = data_pb2.DataConfig()
        data_config.input_fields_str = "ids:ARRAY<BIGINT>;values:ARRAY<FLOAT>"

        fields = get_input_fields_proto(data_config)

        self.assertEqual(len(fields), 2)
        self.assertEqual(fields[0].input_name, "ids")
        self.assertEqual(fields[0].input_type, FieldType.ARRAY_INT64)
        self.assertEqual(fields[1].input_name, "values")
        self.assertEqual(fields[1].input_type, FieldType.ARRAY_FLOAT)

    def test_get_input_fields_proto_map_types(self):
        """Test parsing map types."""
        data_config = data_pb2.DataConfig()
        data_config.input_fields_str = "feat_map:MAP<STRING, BIGINT>"

        fields = get_input_fields_proto(data_config)

        self.assertEqual(len(fields), 1)
        self.assertEqual(fields[0].input_name, "feat_map")
        self.assertEqual(fields[0].input_type, FieldType.MAP_STRING_INT64)

    def test_get_input_fields_proto_nested_array(self):
        """Test parsing nested array types."""
        data_config = data_pb2.DataConfig()
        data_config.input_fields_str = "nested:ARRAY<ARRAY<INT>>"

        fields = get_input_fields_proto(data_config)

        self.assertEqual(len(fields), 1)
        self.assertEqual(fields[0].input_name, "nested")
        self.assertEqual(fields[0].input_type, FieldType.ARRAY_ARRAY_INT32)

    def test_get_input_fields_proto_empty_string(self):
        """Test empty input_fields_str returns empty list."""
        data_config = data_pb2.DataConfig()
        data_config.input_fields_str = ""

        fields = get_input_fields_proto(data_config)
        self.assertEqual(len(fields), 0)

    def test_get_input_fields_proto_trailing_semicolon(self):
        """Test handling of trailing semicolon."""
        data_config = data_pb2.DataConfig()
        data_config.input_fields_str = "user_id:BIGINT;item_id:INT64;"

        fields = get_input_fields_proto(data_config)

        self.assertEqual(len(fields), 2)
        self.assertEqual(fields[0].input_name, "user_id")
        self.assertEqual(fields[1].input_name, "item_id")

    def test_get_input_fields_proto_fallback_to_input_fields(self):
        """Test fallback to input_fields when input_fields_str is not set."""
        data_config = data_pb2.DataConfig()
        field1 = data_config.input_fields.add()
        field1.input_name = "test_field"
        field1.input_type = FieldType.STRING

        fields = get_input_fields_proto(data_config)

        self.assertEqual(len(fields), 1)
        self.assertEqual(fields[0].input_name, "test_field")
        self.assertEqual(fields[0].input_type, FieldType.STRING)

    def test_get_input_fields_proto_invalid_format(self):
        """Test error handling for invalid format."""
        data_config = data_pb2.DataConfig()
        data_config.input_fields_str = "invalid_field"

        with self.assertRaises(ValueError) as context:
            get_input_fields_proto(data_config)
        self.assertIn("Invalid input_fields_str format", str(context.exception))

    def test_get_input_fields_proto_unknown_type(self):
        """Test error handling for unknown type."""
        data_config = data_pb2.DataConfig()
        data_config.input_fields_str = "field1:UNKNOWN_TYPE"

        with self.assertRaises(ValueError) as context:
            get_input_fields_proto(data_config)
        self.assertIn("Unknown field type", str(context.exception))


if __name__ == "__main__":
    unittest.main()
