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
from parameterized import parameterized

from tzrec.datasets.utils import (
    _normalize_type_str,
    build_sampler_input,
    calc_remaining_intervals,
    calc_slice_intervals,
    calc_slice_position,
    combine_negs_to_candidate_sequence,
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

    # Single parameterized test for build_sampler_input covering the
    # three transform branches (string flatten + user_id expand, list
    # flatten + user_id expand, scalar pass-through) plus two defensive
    # pass-through cases. Every case uniformly verifies:
    #   - output dict equals expected_output
    #   - input_data is not mutated (shallow-copy contract)
    #   - returned dict is a different object from input_data
    @parameterized.expand(
        [
            # (name, input_data, item_id_field, user_id_field,
            #  seq_field_delims, expected_output)
            (
                # NegativeSampler-style: no user_id_field; item_id is
                # delimited string; gets flattened.
                "string_item_id_no_user_id",
                {"item_id": pa.array(["1;2", "3"]), "label": pa.array([1, 0])},
                "item_id",
                None,
                {"item_id": ";"},
                {"item_id": ["1", "2", "3"], "label": [1, 0]},
            ),
            (
                # NegativeSamplerV2 / HardNeg style: item_id arrives as
                # list<int64>; user_id is expanded by per-row pos count.
                "list_item_id_expands_user_id",
                {
                    "item_id": pa.array([[1, 2], [3]], type=pa.list_(pa.int64())),
                    "user_id": pa.array(["u0", "u1"]),
                },
                "item_id",
                "user_id",
                {"item_id": ";"},
                {"item_id": [1, 2, 3], "user_id": ["u0", "u0", "u1"]},
            ),
            (
                # DSSM-style: item_id is scalar int64. No flatten, no
                # user_id expand.
                "scalar_item_id_passthrough",
                {
                    "item_id": pa.array([1, 2, 3], type=pa.int64()),
                    "user_id": pa.array(["u0", "u1", "u2"]),
                },
                "item_id",
                "user_id",
                {"item_id": ";"},
                {"item_id": [1, 2, 3], "user_id": ["u0", "u1", "u2"]},
            ),
            (
                # item_id_field has no seq_delim entry -> pass through.
                "item_id_not_in_seq_field_delims",
                {"item_id": pa.array(["1", "2"])},
                "item_id",
                None,
                {},
                {"item_id": ["1", "2"]},
            ),
            (
                # Sampler config without item_id_field at all -> still
                # shallow-copied, no transformation.
                "no_item_id_field",
                {"a": pa.array([1, 2])},
                None,
                None,
                {},
                {"a": [1, 2]},
            ),
        ]
    )
    def test_build_sampler_input(
        self,
        _name,
        input_data,
        item_id_field,
        user_id_field,
        seq_field_delims,
        expected_output,
    ):
        # Snapshot input_data so we can verify the function didn't mutate it.
        input_snapshot = {k: v.to_pylist() for k, v in input_data.items()}

        out = build_sampler_input(
            input_data,
            item_id_field=item_id_field,
            user_id_field=user_id_field,
            seq_field_delims=seq_field_delims,
        )

        # Contract 1: output equals expected (per-column pylist compare).
        self.assertEqual({k: v.to_pylist() for k, v in out.items()}, expected_output)
        # Contract 2: input_data is not mutated.
        self.assertEqual(
            {k: v.to_pylist() for k, v in input_data.items()}, input_snapshot
        )
        # Contract 3: returned dict is a different object (shallow copy).
        self.assertIsNot(out, input_data)

    # Single parameterized test for combine_negs_to_candidate_sequence
    # exercising both the string path (output type preserved as string) and
    # the list path (output type preserved as list<T> / large_list<T>).
    @parameterized.expand(
        [
            # name, pos_data, negs, expected_rows, expected_pos_lengths, expected_type
            (
                "single_pos_string",
                pa.array(["1", "2", "3"]),
                pa.array(["10", "20", "30"]),
                ["1", "2", "3;10;20;30"],
                [1, 1, 1],
                pa.string(),
            ),
            (
                "multivalue_pos_string",
                pa.array(["1;2", "3;4;5"]),
                pa.array(["10", "20"]),
                ["1;2", "3;4;5;10;20"],
                [2, 3],
                pa.string(),
            ),
            (
                # String pos + list<T> negs: still uses string path; combine
                # flattens the list-wrapped negs before joining.
                "string_pos_list_negs",
                pa.array(["1", "2"]),
                pa.array([[10], [20], [30]], type=pa.list_(pa.int64())),
                ["1", "2;10;20;30"],
                [1, 1],
                pa.string(),
            ),
            (
                "empty_negs_string",
                pa.array(["1", "2"]),
                pa.array([], type=pa.string()),
                ["1", "2"],
                [1, 1],
                pa.string(),
            ),
            (
                # list<T> pos + flat T negs -> list<T> out (no string round-trip).
                "list_pos_flat_negs",
                pa.array([[1, 2], [3]], type=pa.list_(pa.int64())),
                pa.array([10, 20], type=pa.int64()),
                [[1, 2], [3, 10, 20]],
                [2, 1],
                pa.list_(pa.int64()),
            ),
            (
                # The sampler emits list<T> of 1-element lists when the
                # attr's field schema is list-typed (see _to_arrow_array in
                # sampler.py:168); combine flattens that shape.
                "list_pos_list_negs",
                pa.array([[1, 2], [3]], type=pa.list_(pa.int64())),
                pa.array([[10], [20], [30]], type=pa.list_(pa.int64())),
                [[1, 2], [3, 10, 20, 30]],
                [2, 1],
                pa.list_(pa.int64()),
            ),
            (
                "list_pos_empty_negs",
                pa.array([[1, 2], [3]], type=pa.list_(pa.int64())),
                pa.array([], type=pa.int64()),
                [[1, 2], [3]],
                [2, 1],
                pa.list_(pa.int64()),
            ),
            (
                # large_list<T> pos -> large_list<T> out.
                "large_list_pos",
                pa.array([["a", "b"], ["c"]], type=pa.large_list(pa.string())),
                pa.array(["d", "e"], type=pa.string()),
                [["a", "b"], ["c", "d", "e"]],
                [2, 1],
                pa.large_list(pa.string()),
            ),
        ]
    )
    def test_combine_negs_to_candidate_sequence(
        self,
        _name,
        pos_data,
        negs,
        expected_rows,
        expected_pos_lengths,
        expected_type,
    ):
        result, pos_lengths = combine_negs_to_candidate_sequence(
            pos_data=pos_data, negs=negs, seq_delim=";"
        )
        self.assertEqual(result.to_pylist(), expected_rows)
        np.testing.assert_array_equal(
            pos_lengths, np.array(expected_pos_lengths, dtype=np.int32)
        )
        self.assertTrue(result.type.equals(expected_type))

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
