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

import json
import os
import shutil
import unittest
from collections import Counter
from unittest import mock

import numpy as np
import pyarrow as pa
from parameterized import parameterized
from pyarrow import csv, parquet

from tzrec.tools.sid import resolve_sid_collisions
from tzrec.tools.sid.resolve_sid_collisions import (
    CollisionResolutionRunner,
    ResolveSidCollisionsConfig,
)
from tzrec.utils.sid.collision import stable_order_hash
from tzrec.utils.test_util import make_test_dir, parameterized_name_func


def _parquet(
    path,
    item_ids,
    codes,
    candidate_codes=None,
    item_id_type=None,
):
    if item_id_type is None:
        item_id_type = pa.int64()
    cols = {
        "item_id": pa.array(item_ids, type=item_id_type),
        "codes": pa.array(codes, type=pa.list_(pa.int64())),
    }
    if candidate_codes is not None:
        # flatten each row's [[c0, ..], ..] (topk x n_layers) to a flat list<int>
        flat = [[code for cand in row for code in cand] for row in candidate_codes]
        cols["candidate_codes"] = pa.array(flat, type=pa.list_(pa.int64()))
    parquet.write_table(pa.table(cols), path)


def _csv(path, item_ids, codes, candidate_codes=None):
    cols = {
        "item_id": item_ids,
        "codes": [",".join(map(str, row)) for row in codes],
    }
    if candidate_codes is not None:
        cols["candidate_codes"] = [
            ",".join(str(code) for candidate in row for code in candidate)
            for row in candidate_codes
        ]
    csv.write_csv(pa.table(cols), path)


class ResolveSidCollisionsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = make_test_dir()

    def tearDown(self) -> None:
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def _runner(self, inp, out, include_original=False, **kw):
        original_groups, resolved_groups = self._group_paths(out)
        config = dict(
            input_path=inp,
            output_path=out,
            original_sid_groups_output_path=(
                original_groups if include_original else None
            ),
            resolved_sid_groups_output_path=resolved_groups,
            reader_type=None,
            writer_type=None,
            batch_size=100000,
            progress_interval=1_000_000,
            item_id_field="item_id",
            code_field="codes",
            candidate_codes_field="candidate_codes",
            layer_sizes=(8, 8),
            max_items_per_codebook=2,
            strategy="candidate",
            random_num_candidates=64,
            rate_only=False,
            odps_data_quota_name="pay-as-you-go",
        )
        config.update(kw)
        return CollisionResolutionRunner(ResolveSidCollisionsConfig(**config))

    def _run(self, inp, out, **kw):
        return self._runner(inp, out, **kw).run()

    @staticmethod
    def _group_paths(out):
        return f"{out}_original_groups", f"{out}_resolved_groups"

    def _read_parquet(self, out_dir):
        return parquet.read_table(os.path.join(out_dir, "part-0.parquet")).to_pydict()

    def _assert_map_matches_resolved_groups(self, out):
        item_map = self._read_parquet(out)
        _, resolved_path = self._group_paths(out)
        resolved = self._read_parquet(resolved_path)
        resolved_items = {
            tuple(codebook): item_group
            for codebook, item_group in zip(resolved["codebook"], resolved["itemids"])
        }
        grouped_item_ids = [
            item_id for item_group in resolved["itemids"] for item_id in item_group
        ]
        self.assertCountEqual(grouped_item_ids, item_map["item_id"])
        for item_id, codebook, index in zip(
            item_map["item_id"], item_map["codebook"], item_map["index"]
        ):
            self.assertEqual(resolved_items[tuple(codebook)][index - 1], item_id)

    # ---- candidate strategy ----

    def test_candidate_reassigns_within_band(self) -> None:
        inp = os.path.join(self.test_dir, "in.parquet")
        out = os.path.join(self.test_dir, "out")
        # 5 items in bucket (0,0); candidates vary only the last layer within band 0.
        _parquet(inp, list(range(5)), [[0, 0]] * 5, [[[0, 1], [0, 2], [0, 3]]] * 5)
        stats = self._run(inp, out, max_items_per_codebook=2)
        self.assertEqual(stats.total_items, 5)
        self.assertEqual(stats.relocated_count, 3)
        self.assertEqual(stats.unresolved_count, 0)
        self.assertEqual(stats.max_final_bucket_size, 2)
        d = self._read_parquet(out)
        # every final SID keeps prefix 0 (stays in band) ...
        self.assertTrue(all(cb[0] == 0 for cb in d["codebook"]))
        # ... and no bucket exceeds the cap.
        self.assertLessEqual(max(Counter(map(tuple, d["codebook"])).values()), 2)

    def test_three_layer_candidate_reassigns_within_band(self) -> None:
        inp = os.path.join(self.test_dir, "in.parquet")
        out = os.path.join(self.test_dir, "out")
        # 3-layer SIDs: band = (code_0, code_1). Three items collide in bucket
        # (0, 0, 0); candidates keep the (0, 0) band and vary the last layer, so
        # the flat topk*3 candidate column must be split at stride 3.
        _parquet(
            inp,
            [0, 1, 2],
            [[0, 0, 0]] * 3,
            [[[0, 0, 1], [0, 0, 2], [0, 0, 3]]] * 3,
        )
        stats = self._run(inp, out, layer_sizes=(2, 3, 4), max_items_per_codebook=2)
        self.assertEqual(stats.total_items, 3)
        self.assertEqual(stats.relocated_count, 1)
        self.assertEqual(stats.unresolved_count, 0)
        self.assertEqual(stats.max_final_bucket_size, 2)
        d = self._read_parquet(out)
        # The relocated item took the first free candidate last code (1); every
        # SID keeps its (0, 0) band prefix.
        self.assertEqual(
            sorted(tuple(cb) for cb in d["codebook"]),
            [(0, 0, 0), (0, 0, 0), (0, 0, 1)],
        )

    def test_keep_original_when_only_origin_candidate(self) -> None:
        inp = os.path.join(self.test_dir, "in.parquet")
        out = os.path.join(self.test_dir, "out")
        # candidate == origin last -> skipped -> nothing to place; kept over cap.
        _parquet(inp, list(range(5)), [[0, 0]] * 5, [[[0, 0]]] * 5)
        stats = self._run(inp, out, max_items_per_codebook=2)
        self.assertEqual(stats.relocated_count, 0)
        self.assertEqual(stats.unresolved_count, 3)
        d = self._read_parquet(out)
        self.assertEqual(len(d["item_id"]), 5)  # every item preserved
        self.assertTrue(all(tuple(cb) == (0, 0) for cb in d["codebook"]))
        self._assert_map_matches_resolved_groups(out)

    @parameterized.expand(
        [
            (
                "candidate_overflow",
                "candidate",
                [[0, 0]] * 3,
                [[[0, 1]]] * 3,
                2,
                [["item_id", "codes"], ["item_id", "candidate_codes"]],
                1,
            ),
            (
                "candidate_no_overflow",
                "candidate",
                [[0, 0], [0, 1], [0, 2]],
                None,
                1,
                [["item_id", "codes"]],
                0,
            ),
            (
                "random_overflow",
                "random",
                [[0, 0]] * 3,
                None,
                2,
                [["item_id", "codes"]],
                1,
            ),
        ],
        name_func=parameterized_name_func,
    )
    def test_reader_projection_for_strategy_and_overflow(
        self,
        _name,
        strategy,
        codes,
        candidate_codes,
        max_items_per_codebook,
        expected_projections,
        expected_relocated,
    ) -> None:
        inp = os.path.join(self.test_dir, "in.parquet")
        out = os.path.join(self.test_dir, "out")
        _parquet(inp, [0, 1, 2], codes, candidate_codes)

        with mock.patch.object(
            resolve_sid_collisions,
            "create_reader",
            wraps=resolve_sid_collisions.create_reader,
        ) as create_reader:
            stats = self._run(
                inp,
                out,
                strategy=strategy,
                max_items_per_codebook=max_items_per_codebook,
            )

        self.assertEqual(
            [call.kwargs["selected_cols"] for call in create_reader.call_args_list],
            expected_projections,
        )
        self.assertEqual(stats.relocated_count, expected_relocated)
        self.assertEqual(len(self._read_parquet(out)["item_id"]), 3)

    def test_raises_when_overflow_but_no_candidates(self) -> None:
        inp = os.path.join(self.test_dir, "in.parquet")
        out = os.path.join(self.test_dir, "out")
        _parquet(inp, list(range(5)), [[0, 0]] * 5)  # no candidate_codes column
        with self.assertRaisesRegex(ValueError, "candidate_codes field .* is missing"):
            self._run(inp, out, max_items_per_codebook=2)

    def test_multi_process_launch_rejected(self) -> None:
        inp = os.path.join(self.test_dir, "in.parquet")
        out = os.path.join(self.test_dir, "out")
        _parquet(inp, list(range(3)), [[0, 0]] * 3, [[[0, 1]]] * 3)
        with (
            mock.patch.dict(os.environ, {"WORLD_SIZE": "2"}),
            self.assertRaisesRegex(RuntimeError, "single-process"),
        ):
            self._run(inp, out)

    def test_candidates_align_across_batches(self) -> None:
        inp = os.path.join(self.test_dir, "in.parquet")
        out = os.path.join(self.test_dir, "out")
        candidates = [[[0, code]] for code in [1, 2, 5, 3, 6]]
        _parquet(inp, list(range(5)), [[0, 0]] * 5, candidates)
        self._run(inp, out, batch_size=2, max_items_per_codebook=2)
        result = self._read_parquet(out)
        by_id = dict(zip(result["item_id"], result["codebook"]))
        self.assertEqual(by_id[0], [0, 1])
        self.assertEqual(by_id[1], [0, 2])
        self.assertEqual(by_id[3], [0, 3])

    def test_group_outputs_follow_sid_and_slot_order(self) -> None:
        inp = os.path.join(self.test_dir, "in.parquet")
        out = os.path.join(self.test_dir, "out")
        item_ids = list(range(10))
        # 4 items in bucket (0,0), 1 in (0,1), 2 in (0,2), 3 in (1,0).
        codes = [[0, 0]] * 4 + [[0, 1]] + [[0, 2]] * 2 + [[1, 0]] * 3
        candidates = [[[0, 0]] for _ in item_ids]
        candidates[1] = [[0, 0], [0, 1], [0, 2]]
        candidates[3] = [[0, 1], [0, 2], [0, 3]]
        candidates[8] = [[1, 0], [1, 1], [1, 2]]
        _parquet(inp, item_ids, codes, candidates)

        self._run(inp, out, include_original=True, max_items_per_codebook=2)

        original_path, resolved_path = self._group_paths(out)
        original = parquet.read_table(os.path.join(original_path, "part-0.parquet"))
        resolved = parquet.read_table(os.path.join(resolved_path, "part-0.parquet"))
        self.assertEqual(original.column_names, ["codebook", "itemids"])
        self.assertEqual(resolved.column_names, ["codebook", "itemids"])
        original_groups = original.to_pydict()
        self.assertCountEqual(
            [item_id for group in original_groups["itemids"] for item_id in group],
            item_ids,
        )
        original_code_by_item = dict(zip(item_ids, codes))
        for codebook, item_group in zip(
            original_groups["codebook"], original_groups["itemids"]
        ):
            for item_id in item_group:
                self.assertEqual(codebook, original_code_by_item[item_id])
        self._assert_map_matches_resolved_groups(out)

    def test_no_overflow_reuses_groups_with_gapped_prefixes(self) -> None:
        inp = os.path.join(self.test_dir, "in.parquet")
        out = os.path.join(self.test_dir, "out")
        _parquet(
            inp,
            list(range(6)),
            [[7, 10], [7, 2], [12, 1], [7, 2], [12, 1], [7, 10]],
        )

        self._run(
            inp,
            out,
            include_original=True,
            layer_sizes=(16, 16),
            max_items_per_codebook=2,
        )

        original_path, resolved_path = self._group_paths(out)
        original = parquet.read_table(
            os.path.join(original_path, "part-0.parquet")
        ).to_pydict()
        resolved = parquet.read_table(
            os.path.join(resolved_path, "part-0.parquet")
        ).to_pydict()
        self.assertEqual(original, resolved)
        self.assertEqual(original["codebook"], [[7, 2], [7, 10], [12, 1]])

    # ---- random strategy ----

    def test_random_reassigns_within_band(self) -> None:
        inp = os.path.join(self.test_dir, "in.parquet")
        out = os.path.join(self.test_dir, "out")
        _parquet(inp, list(range(5)), [[0, 0]] * 5)
        stats = self._run(
            inp,
            out,
            max_items_per_codebook=2,
            strategy="random",
        )
        self.assertEqual(stats.relocated_count, 3)
        d = self._read_parquet(out)
        self.assertTrue(all(cb[0] == 0 for cb in d["codebook"]))
        self.assertLessEqual(max(Counter(map(tuple, d["codebook"])).values()), 2)
        self._assert_map_matches_resolved_groups(out)

    def test_single_layer_sid(self) -> None:
        inp = os.path.join(self.test_dir, "in.parquet")
        out = os.path.join(self.test_dir, "out")
        _parquet(inp, list(range(4)), [[0]] * 4)
        stats = self._run(
            inp,
            out,
            max_items_per_codebook=1,
            strategy="random",
            layer_sizes=(16,),
        )
        self.assertEqual(stats.relocated_count, 3)
        d = self._read_parquet(out)
        self.assertEqual(len({tuple(cb) for cb in d["codebook"]}), 4)

    # ---- CSV backend ----

    @parameterized.expand(
        [
            ("csv_to_csv", "csv", "CsvReader", "CsvWriter"),
            ("parquet_to_csv", "parquet", None, "CsvWriter"),
            ("csv_to_parquet", "csv", "CsvReader", "ParquetWriter"),
        ],
        name_func=parameterized_name_func,
    )
    def test_reader_writer_format_matrix(
        self, _name, input_format, reader_type, writer_type
    ) -> None:
        inp = os.path.join(self.test_dir, f"in.{input_format}")
        out = os.path.join(self.test_dir, "out")
        item_ids = ["a", "b", "c"] if input_format == "csv" else [0, 1, 2]
        codes = [[0, 0, 0]] * 3
        candidates = [[[0, 0, 1], [0, 0, 2]]] * 3
        if input_format == "csv":
            _csv(inp, item_ids, codes, candidates)
        else:
            _parquet(inp, item_ids, codes, candidates)

        stats = self._run(
            inp,
            out,
            include_original=True,
            reader_type=reader_type,
            writer_type=writer_type,
            layer_sizes=(2, 2, 3),
            max_items_per_codebook=2,
        )

        self.assertEqual(stats.relocated_count, 1)
        original_path, resolved_path = self._group_paths(out)
        if writer_type == "CsvWriter":
            result = csv.read_csv(os.path.join(out, "part-0.csv"))
            self.assertEqual(result.schema.field("codebook").type, pa.string())
            self.assertEqual(result.schema.field("origin_codebook").type, pa.string())
            self.assertCountEqual(
                result["codebook"].to_pylist(),
                ["0,0,0", "0,0,0", "0,0,1"],
            )
            self.assertEqual(result["origin_codebook"].to_pylist(), ["0,0,0"] * 3)
            original = csv.read_csv(os.path.join(original_path, "part-0.csv"))
            resolved = csv.read_csv(os.path.join(resolved_path, "part-0.csv"))
            self.assertEqual(original["codebook"].to_pylist(), ["0,0,0"])
            self.assertCountEqual(resolved["codebook"].to_pylist(), ["0,0,0", "0,0,1"])
            for groups in (original, resolved):
                self.assertEqual(groups.column_names, ["codebook", "itemids"])
                self.assertEqual(groups.schema.field("itemids").type, pa.string())
                self.assertTrue(
                    all(
                        isinstance(json.loads(itemids), list)
                        for itemids in groups["itemids"].to_pylist()
                    )
                )
        else:
            result = parquet.read_table(os.path.join(out, "part-0.parquet"))
            self.assertEqual(result.schema.field("codebook").type, pa.list_(pa.int64()))
            self.assertEqual(
                result.schema.field("origin_codebook").type, pa.list_(pa.int64())
            )
            self.assertCountEqual(
                result["codebook"].to_pylist(),
                [[0, 0, 0], [0, 0, 0], [0, 0, 1]],
            )
            self.assertEqual(result["origin_codebook"].to_pylist(), [[0, 0, 0]] * 3)
            for path in (original_path, resolved_path):
                groups = parquet.read_table(os.path.join(path, "part-0.parquet"))
                self.assertEqual(groups.column_names, ["codebook", "itemids"])
                self.assertEqual(
                    groups.schema.field("itemids").type, pa.list_(pa.string())
                )

    def test_csv_group_itemids_use_json_escaping(self) -> None:
        inp = os.path.join(self.test_dir, "in.csv")
        out = os.path.join(self.test_dir, "out")
        item_ids = ["a;b", "x|y", "with,comma", 'with"quote', "back\\slash"]
        _csv(inp, item_ids, [[0, 0]] * len(item_ids))

        self._run(
            inp,
            out,
            include_original=True,
            reader_type="CsvReader",
            writer_type="CsvWriter",
            max_items_per_codebook=len(item_ids),
        )

        original_path, _ = self._group_paths(out)
        groups = csv.read_csv(os.path.join(original_path, "part-0.csv"))
        decoded_ids = json.loads(groups["itemids"][0].as_py())
        self.assertCountEqual(decoded_ids, item_ids)

    def test_preserves_integer_item_id_type(self) -> None:
        inp = os.path.join(self.test_dir, "in.parquet")
        out = os.path.join(self.test_dir, "out")
        parquet.write_table(
            pa.table(
                {
                    "item_id": pa.array([0, 1, 2], type=pa.int32()),
                    "codes": pa.array(
                        [[0, 0], [0, 1], [0, 2]],
                        type=pa.list_(pa.int64()),
                    ),
                }
            ),
            inp,
        )
        self._run(inp, out, include_original=True, max_items_per_codebook=1)
        schema = parquet.read_table(os.path.join(out, "part-0.parquet")).schema
        self.assertEqual(schema.field("item_id").type, pa.int32())
        original_path, resolved_path = self._group_paths(out)
        for path in (original_path, resolved_path):
            group_schema = parquet.read_table(
                os.path.join(path, "part-0.parquet")
            ).schema
            self.assertEqual(group_schema.field("itemids").type, pa.list_(pa.int32()))

    # ---- misc ----

    def test_io_large_loops_report_sample_progress(self) -> None:
        inp = os.path.join(self.test_dir, "in.parquet")
        out = os.path.join(self.test_dir, "out")
        _parquet(
            inp,
            list(range(8)),
            [[0, 0]] * 8,
            [[[0, 1], [0, 2], [0, 3]]] * 8,
        )
        progress_by_description = {}

        def make_progress(description, **_kwargs):
            progress = mock.Mock()
            progress_by_description.setdefault(description, []).append(progress)
            return progress

        with (
            mock.patch.object(
                resolve_sid_collisions,
                "ProgressLogger",
                side_effect=make_progress,
            ),
            mock.patch.object(resolve_sid_collisions, "_MAP_WRITE_ROWS", 2),
            mock.patch.object(resolve_sid_collisions, "_GROUP_WRITE_ITEMS", 2),
        ):
            self._run(
                inp,
                out,
                batch_size=2,
                progress_interval=3,
                include_original=True,
                max_items_per_codebook=2,
            )

        expected_progress = [
            mock.call(4, suffix="4 samples processed"),
            mock.call(8, suffix="8 samples processed"),
        ]
        self.assertEqual(
            progress_by_description["Reading SID input"][0].log.call_args_list,
            expected_progress,
        )
        candidate_calls = progress_by_description["Scanning candidate input"][
            0
        ].log.call_args_list
        self.assertEqual(candidate_calls, expected_progress)
        for description in (
            "Writing resolved item map",
            "Writing resolved SID item groups",
        ):
            self.assertEqual(
                progress_by_description[description][0].log.call_args_list,
                expected_progress,
            )
        self.assertEqual(
            progress_by_description["Writing original SID item groups"][
                0
            ].log.call_args_list,
            [mock.call(8, suffix="8 samples processed")],
        )

    @parameterized.expand(
        [
            (
                "resolved_groups",
                {"resolved_sid_groups_output_path": None},
                "resolved_sid_groups_output_path",
            ),
            ("map", {"output_path": None}, "output_path"),
        ],
        name_func=parameterized_name_func,
    )
    def test_normal_runs_require_operational_outputs(
        self, _name, overrides, expected_field
    ) -> None:
        with self.assertRaisesRegex(ValueError, expected_field):
            self._runner("input", "map", **overrides)

    def test_original_group_output_is_optional(self) -> None:
        inp = os.path.join(self.test_dir, "in.parquet")
        out = os.path.join(self.test_dir, "out")
        original_path, resolved_path = self._group_paths(out)
        _parquet(inp, [0, 1, 2], [[0, 0]] * 3, [[[0, 1]]] * 3)

        with mock.patch.object(
            resolve_sid_collisions,
            "build_original_item_grouping",
            wraps=resolve_sid_collisions.build_original_item_grouping,
        ) as original_builder:
            self._run(
                inp,
                out,
                max_items_per_codebook=2,
            )

        original_builder.assert_not_called()
        self.assertFalse(os.path.exists(original_path))
        self.assertTrue(os.path.exists(os.path.join(out, "part-0.parquet")))
        self.assertTrue(os.path.exists(os.path.join(resolved_path, "part-0.parquet")))

    def test_resolved_groups_are_written_without_original_output_or_overflow(
        self,
    ) -> None:
        inp = os.path.join(self.test_dir, "in.parquet")
        out = os.path.join(self.test_dir, "out")
        _, resolved_path = self._group_paths(out)
        _parquet(inp, [0, 1], [[0, 0], [0, 1]])

        self._run(
            inp,
            out,
            max_items_per_codebook=1,
        )

        resolved = self._read_parquet(resolved_path)
        self.assertEqual(resolved["codebook"], [[0, 0], [0, 1]])
        self.assertEqual(resolved["itemids"], [[0], [1]])

    def test_parser_allows_rate_only_without_outputs(self) -> None:
        args = resolve_sid_collisions.build_parser().parse_args(
            [
                "--input_path",
                "input",
                "--codebook",
                "8,8",
                "--max_items_per_codebook",
                "2",
                "--rate_only",
            ]
        )

        config = ResolveSidCollisionsConfig.from_namespace(args)

        self.assertIsNone(config.output_path)
        self.assertIsNone(config.original_sid_groups_output_path)
        self.assertIsNone(config.resolved_sid_groups_output_path)
        self.assertEqual(config.progress_interval, 1_000_000)

    def test_rejects_invalid_progress_interval(self) -> None:
        with self.assertRaisesRegex(ValueError, "progress_interval must be >= 1"):
            self._runner("input", "map", progress_interval=0)

    def test_odps_group_writes_use_native_array_columns(self) -> None:
        class OdpsWriter:
            def __init__(self) -> None:
                self.writes = []
                self.closed = False

            def write(self, output_dict) -> None:
                self.writes.append(output_dict)

            def close(self) -> None:
                self.closed = True

        inp = os.path.join(self.test_dir, "in.parquet")
        out = os.path.join(self.test_dir, "map")
        parquet.write_table(
            pa.table(
                {
                    "item_id": pa.array([1, 2], type=pa.int32()),
                    "codes": pa.array([[0, 0], [0, 1]], type=pa.list_(pa.int64())),
                }
            ),
            inp,
        )
        created_writers = []

        def make_writer(output_path):
            writer = OdpsWriter()
            created_writers.append((output_path, writer))
            return writer

        with mock.patch.object(
            CollisionResolutionRunner, "_make_writer", side_effect=make_writer
        ):
            self._run(
                inp,
                out,
                include_original=True,
                writer_type="OdpsWriter",
                max_items_per_codebook=1,
            )

        original_path, resolved_path = self._group_paths(out)
        self.assertEqual(
            [path for path, _ in created_writers],
            [original_path, resolved_path, out],
        )
        self.assertTrue(all(writer.closed for _, writer in created_writers))
        for _, writer in created_writers[:2]:
            columns = writer.writes[0]
            self.assertEqual(list(columns), ["codebook", "itemids"])
            self.assertEqual(columns["codebook"].type, pa.list_(pa.int64()))
            self.assertEqual(columns["itemids"].type, pa.list_(pa.int32()))

    def test_writer_closes_when_write_fails(self) -> None:
        class FailingWriter:
            def __init__(self) -> None:
                self.closed = False

            def write(self, output_dict) -> None:
                raise RuntimeError("write failed")

            def close(self) -> None:
                self.closed = True

        inp = os.path.join(self.test_dir, "in.parquet")
        out = os.path.join(self.test_dir, "out")
        _parquet(inp, [0], [[0, 0]])
        writer = FailingWriter()

        with (
            mock.patch.object(
                CollisionResolutionRunner, "_make_writer", return_value=writer
            ),
            self.assertRaisesRegex(RuntimeError, "write failed"),
        ):
            self._run(inp, out, max_items_per_codebook=1)

        self.assertTrue(writer.closed)

    def test_group_writer_chunks_by_item_count_without_splitting_groups(self) -> None:
        inp = os.path.join(self.test_dir, "in.parquet")
        out = os.path.join(self.test_dir, "out")
        _parquet(inp, list(range(4)), [[0, 0], [0, 0], [0, 0], [0, 1]])

        with mock.patch("tzrec.tools.sid.resolve_sid_collisions._GROUP_WRITE_ITEMS", 2):
            self._run(inp, out, include_original=True, max_items_per_codebook=3)

        original_path, _ = self._group_paths(out)
        output_file = os.path.join(original_path, "part-0.parquet")
        self.assertEqual(parquet.ParquetFile(output_file).metadata.num_row_groups, 2)
        groups = parquet.read_table(output_file).to_pydict()
        self.assertEqual([len(group) for group in groups["itemids"]], [3, 1])

    def test_map_writer_chunks_by_row_count(self) -> None:
        inp = os.path.join(self.test_dir, "in.parquet")
        out = os.path.join(self.test_dir, "out")
        _parquet(inp, list(range(5)), [[0, code] for code in range(5)])

        with mock.patch("tzrec.tools.sid.resolve_sid_collisions._MAP_WRITE_ROWS", 2):
            self._run(inp, out, max_items_per_codebook=1)

        output_file = os.path.join(out, "part-0.parquet")
        self.assertEqual(parquet.ParquetFile(output_file).metadata.num_row_groups, 3)

    def test_writers_bound_codebook_list_offsets(self) -> None:
        inp = os.path.join(self.test_dir, "in.parquet")
        out = os.path.join(self.test_dir, "out")
        _parquet(inp, [0, 1, 2], [[0, 0], [0, 1], [0, 2]])

        with (
            mock.patch(
                "tzrec.tools.sid.resolve_sid_collisions._ARROW_LIST_OFFSET_MAX", 4
            ),
            mock.patch("tzrec.tools.sid.resolve_sid_collisions._MAP_WRITE_ROWS", 100),
            mock.patch(
                "tzrec.tools.sid.resolve_sid_collisions._GROUP_WRITE_ITEMS", 100
            ),
        ):
            self._run(inp, out, include_original=True, max_items_per_codebook=1)

        original_path, resolved_path = self._group_paths(out)
        for path in (out, original_path, resolved_path):
            output_file = os.path.join(path, "part-0.parquet")
            self.assertEqual(
                parquet.ParquetFile(output_file).metadata.num_row_groups, 2
            )

    def test_rate_only_with_overflow_allows_omitted_outputs_and_skips_writes(
        self,
    ) -> None:
        inp = os.path.join(self.test_dir, "in.parquet")
        out = os.path.join(self.test_dir, "out")
        _parquet(inp, list(range(5)), [[0, 0]] * 5, [[[0, 1], [0, 2], [0, 3]]] * 5)
        runner = self._runner(
            inp,
            out,
            output_path=None,
            original_sid_groups_output_path=None,
            resolved_sid_groups_output_path=None,
            max_items_per_codebook=2,
            rate_only=True,
        )
        with (
            mock.patch.object(
                runner._resolver, "resolve", wraps=runner._resolver.resolve
            ) as resolve,
            mock.patch.object(runner, "_make_writer") as make_writer,
        ):
            stats = runner.run()

        self.assertFalse(resolve.call_args.kwargs["collect_grouping"])
        make_writer.assert_not_called()
        self.assertEqual(stats.total_items, 5)
        self.assertEqual(stats.relocated_count, 3)
        self.assertEqual(stats.unresolved_count, 0)
        self.assertFalse(os.path.exists(os.path.join(out, "part-0.parquet")))
        original_path, resolved_path = self._group_paths(out)
        self.assertFalse(os.path.exists(original_path))
        self.assertFalse(os.path.exists(resolved_path))
        self.assertFalse(os.path.exists(out))

    def test_empty_input_raises(self) -> None:
        inp = os.path.join(self.test_dir, "in.parquet")
        out = os.path.join(self.test_dir, "out")
        _parquet(inp, [], [])
        with self.assertRaisesRegex(ValueError, "SID input is empty"):
            self._run(inp, out, max_items_per_codebook=2)

    @parameterized.expand(
        [("same_batch", 100000), ("across_batches", 1)],
        name_func=parameterized_name_func,
    )
    def test_candidate_tolerates_duplicate_item_ids(
        self, _case_name, batch_size
    ) -> None:
        inp = os.path.join(self.test_dir, "in.parquet")
        out = os.path.join(self.test_dir, "out")
        distinct_ids = np.asarray(["a", "b"], dtype=object)
        hash_order = np.argsort(stable_order_hash(distinct_ids))
        keeper = distinct_ids[hash_order[0]]
        duplicate = distinct_ids[hash_order[1]]
        _parquet(
            inp,
            [keeper, duplicate, duplicate],
            [[0, 0]] * 3,
            [
                [[0, 1], [0, 2]],
                [[0, 1], [0, 2]],
                [[0, 2], [0, 3]],
            ],
            item_id_type=pa.string(),
        )

        stats = self._run(
            inp,
            out,
            batch_size=batch_size,
            include_original=True,
            max_items_per_codebook=1,
        )

        self.assertEqual(stats.total_items, 3)
        self.assertEqual(stats.relocated_count, 2)
        result = self._read_parquet(out)
        self.assertEqual(result["item_id"], [keeper, duplicate, duplicate])
        self.assertEqual(result["item_id"].count(duplicate), 2)
        self._assert_map_matches_resolved_groups(out)
        original_path, _ = self._group_paths(out)
        original_groups = self._read_parquet(original_path)
        self.assertCountEqual(
            [item_id for group in original_groups["itemids"] for item_id in group],
            [keeper, duplicate, duplicate],
        )

    def test_item_id_lookup_broadcasts_all_duplicate_targets(self) -> None:
        lookup = resolve_sid_collisions._ItemIdLookup(
            np.asarray(["b", "a", "b", "b"], dtype=object)
        )
        source_rows, target_rows = lookup.match(
            np.asarray(["b", "missing", "a"], dtype=object)
        )
        np.testing.assert_array_equal(source_rows, [0, 2])
        np.testing.assert_array_equal(target_rows, [0, 1])

        values = np.asarray([[10, 11], [20, 21], [-1, -1], [-1, -1]])
        lookup.broadcast_duplicate_targets(values)
        np.testing.assert_array_equal(
            values,
            [[10, 11], [20, 21], [10, 11], [10, 11]],
        )

    def test_random_tolerates_duplicate_item_ids(self) -> None:
        inp = os.path.join(self.test_dir, "in.parquet")
        out = os.path.join(self.test_dir, "out")
        _parquet(
            inp,
            ["b", "a", "a"],
            [[0, 0]] * 3,
            item_id_type=pa.string(),
        )

        stats = self._run(
            inp,
            out,
            strategy="random",
            random_num_candidates=8,
            max_items_per_codebook=1,
        )

        self.assertEqual(stats.total_items, 3)
        result = self._read_parquet(out)
        self.assertEqual(result["item_id"], ["b", "a", "a"])
        self._assert_map_matches_resolved_groups(out)

    def test_empty_codebook_token_raises(self) -> None:
        args = resolve_sid_collisions.build_parser().parse_args(
            [
                "--input_path",
                "input",
                "--codebook",
                "8,,8",
                "--max_items_per_codebook",
                "2",
                "--rate_only",
            ]
        )
        with self.assertRaisesRegex(ValueError, "codebook"):
            ResolveSidCollisionsConfig.from_namespace(args)


if __name__ == "__main__":
    unittest.main()
