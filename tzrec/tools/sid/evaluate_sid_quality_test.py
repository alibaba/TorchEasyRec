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

import argparse
import math
import os
import shutil
import unittest
from collections import OrderedDict
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
from parameterized import parameterized

from tzrec.tools.sid import evaluate_sid_quality as esq
from tzrec.utils import misc_util
from tzrec.utils.sid.quality import (
    SidLayerQualityMetrics,
    SidQualityMetrics,
)
from tzrec.utils.test_util import make_test_dir, parameterized_name_func

_CODEBOOK = [4, 8, 16]
_ROWS = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [1, 1, 1],
    [1, 2, 2],
    [2, 3, 3],
    [3, 7, 15],
]


def _list_array(rows):
    return pa.array(rows, type=pa.list_(pa.int64()))


def _run_single(rows, codebook=None, **kwargs):
    effective_codebook = _CODEBOOK if codebook is None else codebook
    batch = {"codes": _list_array(rows)}
    return esq.run_evaluation([batch], "codes", effective_codebook, **kwargs)


class DecodeCodesTest(unittest.TestCase):
    def test_list_and_string_columns_decode_identically(self) -> None:
        list_result = esq.decode_codes(_list_array([[1, 2, 3], [4, 5, 6]]), 3)
        string_result = esq.decode_codes(pa.array(["1,2,3", "4,5,6"]), 3)
        np.testing.assert_array_equal(list_result.values, string_result.values)
        np.testing.assert_array_equal(list_result.valid_rows, [True, True])
        np.testing.assert_array_equal(string_result.valid_rows, [True, True])
        self.assertEqual(list_result.malformed_rows, 0)

    def test_malformed_rows_remain_aligned(self) -> None:
        list_result = esq.decode_codes(
            _list_array([[1, 2, 3], None, [4, 5], [6, None, 7], [8, 9, 10]]),
            3,
        )
        np.testing.assert_array_equal(
            list_result.valid_rows, [True, False, False, False, True]
        )
        np.testing.assert_array_equal(
            list_result.values[[0, 4]], [[1, 2, 3], [8, 9, 10]]
        )
        self.assertEqual(list_result.malformed_rows, 3)

        string_result = esq.decode_codes(
            pa.array(
                [
                    "1,2,3",
                    None,
                    "1,x,3",
                    "1,2",
                    "1,2,,3",
                    "1,99999999999999999999,3",
                ]
            ),
            3,
        )
        np.testing.assert_array_equal(
            string_result.valid_rows, [True, False, False, False, False, False]
        )
        self.assertEqual(string_result.malformed_rows, 5)

    def test_large_list_and_sliced_list_are_supported(self) -> None:
        large_list = pa.array(
            [[1, 2], [3, 4], [5, 6]], type=pa.large_list(pa.int32())
        ).slice(1)
        result = esq.decode_codes(large_list, 2)
        np.testing.assert_array_equal(result.values, [[3, 4], [5, 6]])
        np.testing.assert_array_equal(result.valid_rows, [True, True])

    def test_non_integer_list_is_a_schema_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-integer"):
            esq.decode_codes(pa.array([[1.0, 2.0]], type=pa.list_(pa.float64())), 2)

    def test_unsigned_value_outside_int64_is_malformed(self) -> None:
        codes = pa.array([[1], [np.iinfo(np.uint64).max]], type=pa.list_(pa.uint64()))
        result = esq.decode_codes(codes, 1)
        np.testing.assert_array_equal(result.valid_rows, [True, False])
        np.testing.assert_array_equal(result.values[0], [1])

    def test_nonpositive_layer_count_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            esq.decode_codes(pa.array(["1"]), 0)


class RunEvaluationTest(unittest.TestCase):
    def test_single_field_builds_long_schema_and_metadata(self) -> None:
        evaluation = _run_single(_ROWS, source="input", log_top_sids=5)
        self.assertEqual(len(evaluation.summary_rows), 1)
        summary = evaluation.summary_rows[0]
        self.assertEqual(
            list(summary),
            [
                "source",
                "view",
                "sid_field",
                "codebook",
                "input_rows",
                "evaluated_items",
                "invalid_pair_rows",
                "total",
                "unique_sid",
                "no_collision_rate",
                "uniquely_identified_item_rate",
                "max_collision",
                "gini",
                "entropy",
                "max_entropy",
                "entropy_ratio",
            ],
        )
        self.assertEqual(summary["source"], "input")
        self.assertEqual(summary["view"], "single")
        self.assertEqual(summary["sid_field"], "codes")
        self.assertEqual(summary["codebook"], "4,8,16")
        self.assertEqual(summary["input_rows"], 7)
        self.assertEqual(summary["evaluated_items"], 7)
        self.assertEqual(summary["invalid_pair_rows"], 0)
        self.assertEqual(summary["total"], 7)
        self.assertEqual(summary["unique_sid"], 5)
        self.assertEqual(summary["max_collision"], 3)
        self.assertEqual(len(evaluation.layer_rows), 3)
        self.assertEqual(
            [row["layer"] for row in evaluation.layer_rows],
            [0, 1, 2],
        )
        self.assertEqual(evaluation.top_sids["single"][0], ("0,0,0", 3))

    def test_single_uses_explicit_custom_field_and_ignores_other_columns(self) -> None:
        batch = {
            "final_code": _list_array([[0, 0, 0], [1, 1, 1]]),
            "origin": pa.array(
                [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
                type=pa.list_(pa.float64()),
            ),
            "item_id": pa.array([1, 2]),
            "itemids": pa.array([[10], [20]], type=pa.list_(pa.int64())),
            "codebook": pa.array(["unrelated", "unrelated"]),
        }
        evaluation = esq.run_evaluation([batch], "final_code", [2, 2, 2])
        self.assertEqual(evaluation.summary_rows[0]["sid_field"], "final_code")
        self.assertEqual(evaluation.summary_rows[0]["unique_sid"], 2)

    def test_single_drops_malformed_and_out_of_range_rows(self) -> None:
        rows = [
            [1, 2, 3],
            [1, 2, 3],
            None,
            [4, 5],
            [1, None, 3],
            [9, 9, 9],
            [20, 1, 1],
        ]
        evaluation = _run_single(rows, [16, 16, 16])
        summary = evaluation.summary_rows[0]
        self.assertEqual(summary["input_rows"], 7)
        self.assertEqual(summary["evaluated_items"], 3)
        self.assertEqual(summary["invalid_pair_rows"], 4)
        self.assertEqual(summary["total"], 3)
        self.assertEqual(summary["unique_sid"], 2)

    def test_compare_uses_one_common_valid_cohort(self) -> None:
        batch = {
            "origin": _list_array([[0, 0], None, [0, 1], [2, 0], [1, 1], [0, 0]]),
            "final": _list_array([[0, 0], [0, 1], [0, 2], [1, 0], [1], [1, 1]]),
        }
        evaluation = esq.run_evaluation(
            [batch],
            "final",
            [2, 2],
            origin_codes_field="origin",
        )
        self.assertEqual(
            [row["view"] for row in evaluation.summary_rows],
            ["before", "after", "delta"],
        )
        before, after, delta = evaluation.summary_rows
        for row in evaluation.summary_rows:
            self.assertEqual(row["input_rows"], 6)
            self.assertEqual(row["evaluated_items"], 2)
            self.assertEqual(row["invalid_pair_rows"], 4)
        self.assertEqual(before["total"], 2)
        self.assertEqual(before["unique_sid"], 1)
        self.assertEqual(before["max_collision"], 2)
        self.assertEqual(after["total"], 2)
        self.assertEqual(after["unique_sid"], 2)
        self.assertEqual(after["max_collision"], 1)
        self.assertEqual(delta["unique_sid"], 1)
        self.assertEqual(before["sid_field"], "origin")
        self.assertEqual(after["sid_field"], "final")
        self.assertEqual(delta["sid_field"], "origin->final")

    def test_compare_counts_rows_invalid_on_both_sides_once(self) -> None:
        batch = {
            "origin": pa.array(["0", None, "3", "0"]),
            "final": pa.array(["0", None, "3", "1"]),
        }

        evaluation = esq.run_evaluation(
            [batch], "final", [2], origin_codes_field="origin"
        )

        for row in evaluation.summary_rows:
            self.assertEqual(row["input_rows"], 4)
            self.assertEqual(row["evaluated_items"], 2)
            self.assertEqual(row["invalid_pair_rows"], 2)

    def test_compare_rejects_an_entirely_invalid_pair_cohort(self) -> None:
        batch = {
            "origin": pa.array([None, "0"]),
            "final": pa.array(["0", None]),
        }

        with self.assertRaisesRegex(ValueError, "no valid rows"):
            esq.run_evaluation([batch], "final", [2], origin_codes_field="origin")

    def test_compare_top_sids_excludes_delta(self) -> None:
        batch = {
            "origin": _list_array([[0, 0, 0], [0, 0, 0]]),
            "final": _list_array([[0, 0, 0], [0, 0, 1]]),
        }
        evaluation = esq.run_evaluation(
            [batch],
            "final",
            [2, 2, 2],
            origin_codes_field="origin",
            log_top_sids=1,
        )
        self.assertEqual(set(evaluation.top_sids), {"before", "after"})

    def test_error_policy_rejects_invalid_rows(self) -> None:
        with self.assertRaisesRegex(ValueError, "batch 0 contains 1"):
            _run_single(
                [[0, 0, 0], [9, 0, 0]],
                [2, 2, 2],
                invalid_row_policy="error",
            )

    def test_all_invalid_rows_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "no valid rows"):
            _run_single([None, [1, 2]], [2, 2, 2])

    def test_invalid_policy_and_field_configuration_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid_row_policy"):
            _run_single([[0, 0, 0]], invalid_row_policy="ignore")
        with self.assertRaisesRegex(ValueError, "must be different"):
            esq.run_evaluation([], "codes", [2], origin_codes_field="codes")

    def test_missing_selected_fields_are_rejected(self) -> None:
        with self.assertRaisesRegex(KeyError, "codes field"):
            esq.run_evaluation([{"other": pa.array(["0"])}], "codes", [2])
        with self.assertRaisesRegex(KeyError, "origin codes field"):
            esq.run_evaluation(
                [{"final": pa.array(["0"])}],
                "final",
                [2],
                origin_codes_field="origin",
            )

    def test_compare_rejects_unaligned_columns(self) -> None:
        with self.assertRaisesRegex(ValueError, "same number"):
            esq.run_evaluation(
                [
                    {
                        "origin": pa.array(["0", "1"]),
                        "final": pa.array(["0"]),
                    }
                ],
                "final",
                [2],
                origin_codes_field="origin",
            )

    def test_order_and_batch_boundaries_do_not_change_results(self) -> None:
        origin = [[0, 0], [0, 0], [0, 1], [1, 0], [1, 1]]
        final = [[0, 1], [0, 0], [0, 1], [1, 1], [1, 0]]
        one_batch = [{"origin": _list_array(origin), "final": _list_array(final)}]
        reordered_batches = [
            {
                "origin": _list_array([origin[4], origin[1]]),
                "final": _list_array([final[4], final[1]]),
            },
            {
                "origin": _list_array([origin[3], origin[0], origin[2]]),
                "final": _list_array([final[3], final[0], final[2]]),
            },
        ]
        first = esq.run_evaluation(
            one_batch, "final", [2, 2], origin_codes_field="origin"
        )
        second = esq.run_evaluation(
            reordered_batches, "final", [2, 2], origin_codes_field="origin"
        )
        self.assertEqual(first.summary_rows, second.summary_rows)
        self.assertEqual(first.layer_rows, second.layer_rows)


class SidQualityCliTest(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = make_test_dir()

    def tearDown(self) -> None:
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_parser_exposes_explicit_field_contract(self) -> None:
        args = esq.build_parser().parse_args(
            [
                "--input_path",
                "input.parquet",
                "--codebook",
                "4,8,16",
                "--codes_field",
                "final",
                "--origin_codes_field",
                "origin",
            ]
        )
        self.assertEqual(args.codes_field, "final")
        self.assertEqual(args.origin_codes_field, "origin")
        self.assertEqual(args.invalid_row_policy, "drop")

    def test_parse_codebook_and_positive_int(self) -> None:
        self.assertEqual(esq._parse_codebook("256, 256,256"), [256, 256, 256])
        self.assertEqual(esq._positive_int("5"), 5)

    @parameterized.expand(
        [
            ("noninteger_codebook", esq._parse_codebook, "16,abc", "must be"),
            ("nonpositive_codebook", esq._parse_codebook, "0,16", "positive"),
            ("empty_codebook", esq._parse_codebook, "", "must be"),
            ("nonpositive_integer", esq._positive_int, "0", "positive"),
        ],
        name_func=parameterized_name_func,
    )
    def test_cli_value_parsers_reject_invalid_values(
        self, _name, parser, value, expected_error
    ) -> None:
        with self.assertRaisesRegex(argparse.ArgumentTypeError, expected_error):
            parser(value)

    @parameterized.expand(
        [
            ("single", ["--codes_field", "final"], ["final"]),
            (
                "compare",
                [
                    "--origin_codes_field",
                    "origin",
                    "--codes_field",
                    "final",
                ],
                ["origin", "final"],
            ),
        ],
        name_func=parameterized_name_func,
    )
    def test_main_projects_only_explicit_sid_fields(
        self, _name, field_args, selected_fields
    ) -> None:
        batch = {field: pa.array(["0"]) for field in selected_fields}
        reader = SimpleNamespace(
            schema=pa.schema(
                [pa.field(field, pa.string()) for field in selected_fields]
            ),
            to_batches=lambda: iter([batch]),
        )
        with mock.patch.object(esq, "create_reader", return_value=reader) as create:
            esq.main(
                [
                    "--input_path",
                    "input.parquet",
                    "--codebook",
                    "2",
                    *field_args,
                ]
            )
        self.assertEqual(create.call_args.kwargs["selected_cols"], selected_fields)

    @parameterized.expand(
        [
            (
                "missing_field",
                [
                    "--origin_codes_field",
                    "origin",
                    "--codes_field",
                    "final",
                ],
                ["final"],
                "selected SID fields .*origin.* not found",
                True,
            ),
            (
                "duplicate_fields",
                ["--origin_codes_field", "codes"],
                ["codes"],
                "must be different",
                False,
            ),
        ],
        name_func=parameterized_name_func,
    )
    def test_main_reports_field_errors(
        self, _name, field_args, schema_fields, expected_error, expects_reader
    ) -> None:
        reader = SimpleNamespace(
            schema=pa.schema([pa.field(field, pa.string()) for field in schema_fields])
        )

        def parser_error(message):
            raise ValueError(message)

        with (
            mock.patch.object(esq, "create_reader", return_value=reader) as create,
            mock.patch.object(
                argparse.ArgumentParser, "error", side_effect=parser_error
            ),
            self.assertRaisesRegex(ValueError, expected_error),
        ):
            esq.main(
                [
                    "--input_path",
                    "input.parquet",
                    "--codebook",
                    "2",
                    *field_args,
                ]
            )
        if expects_reader:
            create.assert_called_once()
        else:
            create.assert_not_called()

    def test_rows_to_arrow_columns_validates_schema(self) -> None:
        rows = (OrderedDict(a=1, b="x"), OrderedDict(a=2, b="y"))
        columns = esq._rows_to_arrow_columns(rows)
        self.assertEqual(columns["a"].to_pylist(), [1, 2])
        self.assertEqual(columns["b"].to_pylist(), ["x", "y"])
        with self.assertRaisesRegex(ValueError, "empty"):
            esq._rows_to_arrow_columns([])
        with self.assertRaisesRegex(ValueError, "same ordered schema"):
            esq._rows_to_arrow_columns([OrderedDict(a=1, b=2), OrderedDict(b=2, a=1)])

    def test_summary_and_layer_row_builders_preserve_view_metadata(self) -> None:
        metrics = SidQualityMetrics(
            total=2,
            unique_sid=1,
            no_collision_rate=0.5,
            uniquely_identified_item_rate=0.0,
            max_collision=2,
            gini=0.0,
            entropy=0.0,
            max_entropy=math.log(4),
            entropy_ratio=0.0,
        )
        summary = esq._build_summary_row(
            "input", "before", "origin", [2, 2], 3, 2, 1, metrics
        )
        self.assertEqual((summary["view"], summary["sid_field"]), ("before", "origin"))
        self.assertEqual(
            (
                summary["input_rows"],
                summary["evaluated_items"],
                summary["invalid_pair_rows"],
            ),
            (3, 2, 1),
        )
        layers = esq._build_layer_rows(
            "input",
            "before",
            "origin",
            [2, 2],
            3,
            2,
            1,
            [SidLayerQualityMetrics(0, 2, 0.5, 1, 1.0)],
        )
        self.assertEqual(len(layers), 1)
        self.assertEqual(
            (layers[0]["view"], layers[0]["layer"], layers[0]["coverage"]),
            ("before", 0, 0.5),
        )

    def test_writer_is_closed_when_write_fails(self) -> None:
        writer = mock.Mock()
        writer.write.side_effect = RuntimeError("write failed")
        with mock.patch.object(esq, "create_writer", return_value=writer):
            with self.assertRaisesRegex(RuntimeError, "write failed"):
                esq._write_rows(
                    "output",
                    "ParquetWriter",
                    "quota",
                    [OrderedDict(value=1)],
                )
        writer.close.assert_called_once_with()

    def test_end_to_end_compare_writes_long_format_outputs(self) -> None:
        input_dir = os.path.join(self.test_dir, "input")
        os.makedirs(input_dir)
        pq.write_table(
            pa.table(
                {
                    "item_id": pa.array(["a", "b", "c"]),
                    "origin": _list_array([[0, 0, 0], [0, 0, 0], [1, 1, 1]]),
                    "final": _list_array([[0, 0, 0], [0, 0, 1], [1, 1, 1]]),
                }
            ),
            os.path.join(input_dir, "part-0.parquet"),
        )
        summary_output = os.path.join(self.test_dir, "summary")
        layer_output = os.path.join(self.test_dir, "layer")
        command = (
            "PYTHONPATH=. python -m tzrec.tools.sid.evaluate_sid_quality "
            f"--input_path {input_dir}/*.parquet --origin_codes_field origin "
            "--codes_field final --codebook 2,2,2 "
            f"--summary_output {summary_output} "
            f"--layer_stats_output {layer_output}"
        )
        ran = misc_util.run_cmd(
            command,
            os.path.join(self.test_dir, "evaluate_sid_quality.log"),
            timeout=600,
        )
        self.assertTrue(ran)

        summary_rows = pq.read_table(
            os.path.join(summary_output, "part-0.parquet")
        ).to_pylist()
        self.assertEqual(
            [row["view"] for row in summary_rows],
            ["before", "after", "delta"],
        )
        self.assertEqual(summary_rows[0]["unique_sid"], 2)
        self.assertEqual(summary_rows[1]["unique_sid"], 3)
        self.assertEqual(summary_rows[2]["unique_sid"], 1)

        layer_rows = pq.read_table(
            os.path.join(layer_output, "part-0.parquet")
        ).to_pylist()
        self.assertEqual(len(layer_rows), 9)
        self.assertEqual(
            {row["view"] for row in layer_rows}, {"before", "after", "delta"}
        )

    def test_csv_compare_writes_long_format_summary(self) -> None:
        input_path = os.path.join(self.test_dir, "input.csv")
        pacsv.write_csv(
            pa.table(
                {
                    "origin": ["0,0", "0,0", "1,1"],
                    "final": ["0,0", "0,1", "1,1"],
                }
            ),
            input_path,
        )
        summary_output = os.path.join(self.test_dir, "summary")

        esq.main(
            [
                "--input_path",
                input_path,
                "--reader_type",
                "CsvReader",
                "--writer_type",
                "CsvWriter",
                "--origin_codes_field",
                "origin",
                "--codes_field",
                "final",
                "--codebook",
                "2,2",
                "--summary_output",
                summary_output,
            ]
        )

        summary = pacsv.read_csv(os.path.join(summary_output, "part-0.csv"))
        self.assertEqual(summary["view"].to_pylist(), ["before", "after", "delta"])
        self.assertEqual(summary["unique_sid"].to_pylist(), [2, 3, 1])


if __name__ == "__main__":
    unittest.main()
