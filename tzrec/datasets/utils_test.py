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
    _normalize_type_str,
    calc_slice_position,
    get_input_fields_proto,
    process_hstu_neg_sample,
    process_hstu_seq_data,
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
