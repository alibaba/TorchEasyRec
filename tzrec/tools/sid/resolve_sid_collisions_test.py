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

import os
import shutil
import subprocess
import sys
import unittest
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
from tzrec.utils.sid import bundle
from tzrec.utils.sid.collision import (
    CollisionResolutionConfig,
    prepare_collision_plan,
)
from tzrec.utils.sid.uniform_collision import UniformCollisionResolver
from tzrec.utils.test_util import make_test_dir, parameterized_name_func

ITEM_TO_SID = bundle.ITEM_TO_SID_ARTIFACT
SID_TO_ITEMS = bundle.SID_TO_ITEMS_ARTIFACT
DELTA_ITEM_TO_SID = bundle.DELTA_ITEM_TO_SID_ARTIFACT
DELTA_SID_TO_ITEMS = bundle.DELTA_SID_TO_ITEMS_ARTIFACT


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
        # (topk x n_layers) per row, flattened to one list<int>
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


def _write_part(path, table):
    """Write one ``part-0.parquet`` shard, the shape every writer emits."""
    os.makedirs(path, exist_ok=True)
    parquet.write_table(table, os.path.join(path, "part-0.parquet"))


def _item_to_sid_parquet(path, rows):
    """Write item_to_sid from (item_id, origin_codebook, codebook, index)."""
    _write_part(
        path,
        pa.table(
            {
                "item_id": pa.array([r[0] for r in rows], type=pa.int64()),
                "origin_codebook": pa.array(
                    [r[1] for r in rows], type=pa.list_(pa.int64())
                ),
                "codebook": pa.array([r[2] for r in rows], type=pa.list_(pa.int64())),
                "index": pa.array([r[3] for r in rows], type=pa.int64()),
            }
        ),
    )


def _sid_to_items_parquet(path, rows):
    """Write sid_to_items from (codebook, item_ids) in SID-key order."""
    _write_part(
        path,
        pa.table(
            {
                "codebook": pa.array([r[0] for r in rows], type=pa.list_(pa.int64())),
                "itemids": pa.array([r[1] for r in rows], type=pa.list_(pa.int64())),
            }
        ),
    )


def _item_to_sid_csv(path, rows):
    """Write item_to_sid in the CSV encoding this tool emits."""
    os.makedirs(path, exist_ok=True)
    csv.write_csv(
        pa.table(
            {
                "item_id": [r[0] for r in rows],
                "origin_codebook": [",".join(map(str, r[1])) for r in rows],
                "codebook": [",".join(map(str, r[2])) for r in rows],
                "index": [r[3] for r in rows],
            }
        ),
        os.path.join(path, "part-0.csv"),
    )


class ResolveSidCollisionsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = make_test_dir()

    def tearDown(self) -> None:
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def _runner(self, inp, out, **kw):
        config = dict(
            input_path=inp,
            output_path=out,
            item_to_sid_root=None,
            generation="v1",
            reader_type=None,
            item_to_sid_writer_type=None,
            batch_size=100000,
            progress_interval=1_000_000,
            item_id_field="item_id",
            code_field="codes",
            candidate_codes_field="candidate_codes",
            layer_sizes=(8, 8),
            max_items_per_codebook=2,
            strategy="candidate",
            rate_only=False,
            odps_data_quota_name="pay-as-you-go",
        )
        config.update(kw)
        return CollisionResolutionRunner(ResolveSidCollisionsConfig(**config))

    def _run(self, inp, out, **kw):
        return self._runner(inp, out, **kw).run()

    def _part(self, out, artifact, generation="v1", ext="parquet"):
        """The single part file one artifact of one generation is written to."""
        location = bundle.artifact_location(out, generation, artifact)
        return os.path.join(location, f"part-0.{ext}")

    def _read(self, out, artifact, generation="v1"):
        """Read one parquet artifact of one generation."""
        return parquet.read_table(self._part(out, artifact, generation)).to_pydict()

    def _manifest(self, out, generation="v1"):
        return bundle.read_manifest(bundle.manifest_path(out, generation))

    def _resolve(self, item_ids, codes, candidates=None, **kw):
        """Write one input, resolve it, and return the stats and the bundle root."""
        inp = os.path.join(self.test_dir, "in.parquet")
        out = os.path.join(self.test_dir, "out")
        _parquet(inp, item_ids, codes, candidates)
        return self._run(inp, out, **kw), out

    def _assert_item_to_sid_matches_sid_to_items(self, out, generation="v1"):
        item_map = self._read(out, ITEM_TO_SID, generation)
        resolved = self._read(out, SID_TO_ITEMS, generation)
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

    def _item_to_sid_rows(self, out, generation="v1"):
        """Return {item_id: (origin_codebook, codebook, index)} keyed by item."""
        item_map = self._read(out, ITEM_TO_SID, generation)
        return {
            item_id: (tuple(origin), tuple(codebook), index)
            for item_id, origin, codebook, index in zip(
                item_map["item_id"],
                item_map["origin_codebook"],
                item_map["codebook"],
                item_map["index"],
            )
        }

    def _append(self, inp, out, previous="v1", generation="v2", **kw):
        """Append onto a published generation under the same root."""
        return self._run(
            inp, out, generation=generation, from_generation=previous, **kw
        )

    def _seed_prior(self, out, map_rows, bucket_rows, generation="v1", csv=False):
        """Publish a hand-built generation at the locations the tool derives."""
        map_dir = bundle.artifact_location(out, generation, ITEM_TO_SID)
        bucket_dir = bundle.artifact_location(out, generation, SID_TO_ITEMS)
        (_item_to_sid_csv if csv else _item_to_sid_parquet)(map_dir, map_rows)
        _sid_to_items_parquet(bucket_dir, bucket_rows)
        manifest = bundle.BundleManifest(
            bundle_uuid=f"uuid-{generation}",
            codebook=[8, 8],
            capacity=2,
            max_observed_items_per_sid=len(map_rows),
            artifacts={
                SID_TO_ITEMS: bundle.ArtifactEntry(
                    save_type="parquet", rows=len(bucket_rows), location=bucket_dir
                ),
                ITEM_TO_SID: bundle.ArtifactEntry(
                    save_type="csv" if csv else "parquet",
                    rows=len(map_rows),
                    location=map_dir,
                ),
            },
        )
        with open(bundle.manifest_path(out, generation), "w") as handle:
            handle.write(manifest.to_json())
        return out

    def _prepare_single(self, item_id):
        path = os.path.join(self.test_dir, f"single_{item_id}.parquet")
        _parquet(path, [item_id], [[0, 0]], [[[0, 2]]])
        return path

    def _assert_dense_indices(self, out, generation="v1"):
        """Every bucket's index multiset must be exactly 1..count."""
        by_bucket = {}
        for _item, (_origin, codebook, index) in self._item_to_sid_rows(
            out, generation
        ).items():
            by_bucket.setdefault(codebook, []).append(index)
        for codebook, indices in by_bucket.items():
            self.assertEqual(
                sorted(indices),
                list(range(1, len(indices) + 1)),
                f"non-dense index set for {codebook}",
            )

    @parameterized.expand(
        [("candidate", "candidate", 1, 1), ("iterative", "iterative", 2, 0)],
        name_func=parameterized_name_func,
    )
    def test_strategy_selects_candidate_arbitration(
        self, _name, strategy, expected_relocated, expected_unresolved
    ) -> None:
        stats, out = self._resolve(
            [10, 11, 20, 21, 30],
            [[0], [0], [1], [1], [2]],
            [[[2], [3], [4]]] * 2 + [[[3], [2], [2]]] * 2 + [[[0], [0], [0]]],
            layer_sizes=(5,),
            max_items_per_codebook=1,
            strategy=strategy,
        )

        self.assertEqual(stats.relocated_count, expected_relocated)
        self.assertEqual(stats.unresolved_count, expected_unresolved)
        self._assert_item_to_sid_matches_sid_to_items(out)

    def test_candidate_scan_is_a_second_projected_pass(self) -> None:
        inp = os.path.join(self.test_dir, "in.parquet")
        _parquet(inp, [0, 1, 2], [[0, 0]] * 3, [[[0, 1]]] * 3)

        with mock.patch.object(
            resolve_sid_collisions,
            "create_reader",
            wraps=resolve_sid_collisions.create_reader,
        ) as create_reader:
            self._run(inp, os.path.join(self.test_dir, "out"))

        self.assertEqual(
            [call.kwargs["selected_cols"] for call in create_reader.call_args_list],
            [["item_id", "codes"], ["item_id", "candidate_codes"]],
        )

    def test_raises_when_overflow_but_no_candidates(self) -> None:
        inp = os.path.join(self.test_dir, "in.parquet")
        out = os.path.join(self.test_dir, "out")
        _parquet(inp, list(range(5)), [[0, 0]] * 5)
        with self.assertRaisesRegex(
            ValueError, "columns .*candidate_codes.* are missing"
        ):
            self._run(inp, out, max_items_per_codebook=2)

    def test_missing_input_rejected_before_reading(self) -> None:
        with self.assertRaisesRegex(ValueError, "hold no files"):
            self._runner(
                os.path.join(self.test_dir, "absent.parquet"),
                os.path.join(self.test_dir, "out"),
            )

    def test_partition_does_not_drop_invalid_codes(self) -> None:
        inp = os.path.join(self.test_dir, "in.parquet")
        codes = np.array([[-1, 0, 0], [0, 0, 0], [4, 0, 0]])
        _parquet(inp, [0, 1, 2], codes.tolist())
        for rank in range(2):
            with mock.patch.dict(os.environ, {"WORLD_SIZE": "2", "RANK": str(rank)}):
                runner = self._runner(
                    inp, os.path.join(self.test_dir, "out"), layer_sizes=(4, 4, 4)
                )
            owned = codes[runner._owned_rows(codes)]
            with self.assertRaisesRegex(ValueError, "out-of-range"):
                prepare_collision_plan(
                    np.arange(len(owned)), owned, runner._config.resolution_config
                )

    def _torchrun(self, inp, out, codebook, strategy, ranks=2, extra=()):
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={ranks}",
            "-m",
            "tzrec.tools.sid.resolve_sid_collisions",
            "--input_path",
            inp,
            "--output_path",
            out,
            "--generation",
            "v1",
            "--codebook",
            codebook,
            "--strategy",
            strategy,
            "--max_items_per_codebook",
            "1",
            "--batch_size",
            "2",
            *extra,
        ]
        log = os.path.join(self.test_dir, "torchrun.log")
        with open(log, "w") as handle:
            completed = subprocess.run(
                command, stdout=handle, stderr=subprocess.STDOUT, timeout=180
            )
        with open(log) as handle:
            output = handle.read()
        self.assertEqual(completed.returncode, 0, output)
        return output

    def _assert_distributed_matches_single(self, single, distributed, generation):
        expected_manifest = self._manifest(single, generation)
        actual_manifest = self._manifest(distributed, generation)
        for artifact in expected_manifest.artifacts:
            key = "item_id" if artifact in bundle.ITEM_TO_SID_ARTIFACTS else "codebook"
            expected = parquet.read_table(
                bundle.artifact_location(single, generation, artifact)
            ).to_pylist()
            actual = parquet.read_table(
                bundle.artifact_location(distributed, generation, artifact)
            ).to_pylist()
            self.assertEqual(
                sorted(actual, key=lambda row: row[key]),
                sorted(expected, key=lambda row: row[key]),
            )
        self.assertEqual(
            {name: entry.rows for name, entry in actual_manifest.artifacts.items()},
            {name: entry.rows for name, entry in expected_manifest.artifacts.items()},
        )
        self.assertEqual(
            actual_manifest.max_observed_items_per_sid,
            expected_manifest.max_observed_items_per_sid,
        )

    def test_torchrun_matches_single_and_appends_with_a_new_rank_count(self) -> None:
        strategy = "iterative"
        input_dir = os.path.join(self.test_dir, "input")
        os.makedirs(input_dir)
        codes = [[2, 0, 0], [0, 0, 0], [2, 0, 0], [0, 0, 0], [3, 1, 0], [1, 3, 2]]
        candidates = [
            [[*code[:-1], (code[-1] + offset) % 4] for offset in range(4)]
            for code in codes
        ]
        for first, stop in ((0, 3), (3, 6)):
            _parquet(
                os.path.join(input_dir, f"part-{first}.parquet"),
                list(range(first, stop)),
                codes[first:stop],
                candidates[first:stop],
            )
        inp = os.path.join(input_dir, "*.parquet")
        single = os.path.join(self.test_dir, "single")
        distributed = os.path.join(self.test_dir, "distributed")
        self._run(
            inp,
            single,
            layer_sizes=(4, 4, 4),
            max_items_per_codebook=1,
            strategy=strategy,
        )
        self._torchrun(inp, distributed, "4,4,4", strategy)
        self._assert_distributed_matches_single(single, distributed, "v1")

        appended = os.path.join(self.test_dir, "append.parquet")
        codes = [[0, 0, 0], [2, 0, 0], [3, 1, 0]]
        candidates = [[[code[0], code[1], last] for last in range(4)] for code in codes]
        _parquet(appended, [10, 11, 12], codes, candidates)
        self._append(
            appended,
            single,
            layer_sizes=(4, 4, 4),
            max_items_per_codebook=1,
            strategy=strategy,
        )
        self._torchrun(
            appended,
            distributed,
            "4,4,4",
            strategy,
            ranks=3,
            extra=("--generation", "v2", "--from_generation", "v1"),
        )
        self._assert_distributed_matches_single(single, distributed, "v2")
        self.assertEqual(
            self._manifest(distributed, "v2").since_bundle_uuid,
            self._manifest(distributed, "v1").bundle_uuid,
        )

    def test_torchrun_handles_empty_ranks_and_rate_only(self) -> None:
        inp = os.path.join(self.test_dir, "in.parquet")
        _parquet(inp, [0, 1, 2], [[0]] * 3, [[[last] for last in range(4)]] * 3)
        single = os.path.join(self.test_dir, "single")
        distributed = os.path.join(self.test_dir, "distributed")
        self._run(inp, single, layer_sizes=(4,), max_items_per_codebook=1)
        self._torchrun(inp, distributed, "4", "candidate")
        self._assert_distributed_matches_single(single, distributed, "v1")
        rate_output = os.path.join(self.test_dir, "rate")
        output = self._torchrun(
            inp, rate_output, "4", "candidate", extra=("--rate_only",)
        )
        self.assertIn("total_items=3", output)
        self.assertFalse(os.path.exists(rate_output))

    def test_append_rejects_a_different_codebook(self) -> None:
        out = self._seed_prior(
            os.path.join(self.test_dir, "out"),
            [(0, [0, 0], [0, 0], 1)],
            [([0, 0], [0])],
        )
        inp = os.path.join(self.test_dir, "v2_in.parquet")
        _parquet(inp, [10], [[0, 0]])

        with self.assertRaisesRegex(ValueError, "differs from"):
            self._runner(
                inp, out, generation="v2", from_generation="v1", layer_sizes=(8, 16)
            )

    def test_candidates_align_across_batches(self) -> None:
        inp = os.path.join(self.test_dir, "in.parquet")
        out = os.path.join(self.test_dir, "out")
        candidates = [[[0, code]] for code in [1, 2, 5, 3, 6]]
        _parquet(inp, list(range(5)), [[0, 0]] * 5, candidates)
        self._run(inp, out, batch_size=2, max_items_per_codebook=2)
        result = self._read(out, ITEM_TO_SID)
        by_id = dict(zip(result["item_id"], result["codebook"]))
        self.assertEqual(by_id[0], [0, 1])
        self.assertEqual(by_id[1], [0, 2])
        self.assertEqual(by_id[3], [0, 3])

    def test_no_overflow_reuses_buckets_with_gapped_prefixes(self) -> None:
        _, out = self._resolve(
            list(range(6)),
            [[7, 10], [7, 2], [12, 1], [7, 2], [12, 1], [7, 10]],
            layer_sizes=(13, 16),
            max_items_per_codebook=2,
        )

        resolved = self._read(out, SID_TO_ITEMS)
        self.assertEqual(resolved["codebook"], [[7, 2], [7, 10], [12, 1]])
        self.assertEqual(
            [sorted(item_group) for item_group in resolved["itemids"]],
            [[1, 3], [0, 5], [2, 4]],
        )
        self._assert_item_to_sid_matches_sid_to_items(out)

    def test_uniform_matches_core_resolver_in_a_full_prefix(self) -> None:
        item_ids = list(range(5))
        codes = [[0]] * 5
        plan = prepare_collision_plan(
            np.asarray(item_ids),
            np.asarray(codes, dtype=np.int64),
            CollisionResolutionConfig((4,), 1),
        )
        expected = UniformCollisionResolver().resolve(plan)

        stats, out = self._resolve(
            item_ids,
            codes,
            layer_sizes=(4,),
            max_items_per_codebook=1,
            strategy="uniform",
        )

        self.assertEqual(stats, expected.stats)
        item_map = self._read(out, ITEM_TO_SID)
        self.assertEqual(
            dict(zip(item_map["item_id"], item_map["codebook"])),
            {
                item_id: [int(code)]
                for item_id, code in zip(item_ids, expected.resolved_last_codes)
            },
        )
        self.assertEqual(stats.unresolved_count, 1)
        self.assertEqual(stats.max_final_bucket_size, 2)
        self._assert_item_to_sid_matches_sid_to_items(out)

    @parameterized.expand(
        [
            ("csv_to_csv", "csv", "CsvWriter"),
            ("parquet_to_csv", "parquet", "CsvWriter"),
            ("csv_to_parquet", "csv", "ParquetWriter"),
        ],
        name_func=parameterized_name_func,
    )
    def test_reader_writer_format_matrix(
        self, _name, input_format, writer_type
    ) -> None:
        inp = os.path.join(self.test_dir, f"in.{input_format}")
        out = os.path.join(self.test_dir, "out")
        item_ids = ["a", "b", "c"] if input_format == "csv" else [0, 1, 2]
        write_input = _csv if input_format == "csv" else _parquet
        write_input(inp, item_ids, [[3, 3, 0]] * 3, [[[3, 3, 1], [3, 3, 2]]] * 3)

        stats = self._run(
            inp,
            out,
            item_to_sid_writer_type=writer_type,
            layer_sizes=(4, 4, 4),
            max_items_per_codebook=1,
        )

        self.assertEqual(stats.relocated_count, 2)
        if writer_type == "CsvWriter":
            item_map = csv.read_csv(self._part(out, ITEM_TO_SID, ext="csv"))
            sids, sid_type = ["3,3,0", "3,3,1", "3,3,2"], pa.string()
        else:
            item_map = parquet.read_table(self._part(out, ITEM_TO_SID))
            sids = [[3, 3, 0], [3, 3, 1], [3, 3, 2]]
            sid_type = pa.list_(pa.int64())
        for column in ("origin_codebook", "codebook"):
            self.assertEqual(item_map.schema.field(column).type, sid_type)
        self.assertCountEqual(item_map["codebook"].to_pylist(), sids)
        self.assertEqual(item_map["origin_codebook"].to_pylist(), sids[:1] * 3)
        resolved = self._read(out, SID_TO_ITEMS)
        self.assertEqual(list(resolved), ["codebook", "offset_codebook", "itemids"])
        self.assertEqual(resolved["codebook"], [[3, 3, 0], [3, 3, 1], [3, 3, 2]])
        self.assertCountEqual(sum(resolved["itemids"], []), item_ids)

    def test_preserves_item_id_type(self) -> None:
        inp = os.path.join(self.test_dir, "in.parquet")
        out = os.path.join(self.test_dir, "out")
        _parquet(
            inp,
            ["a", "b", "c"],
            [[0, 0], [0, 1], [0, 2]],
            item_id_type=pa.large_string(),
        )
        self._run(inp, out, max_items_per_codebook=1)
        schema = parquet.read_schema(self._part(out, ITEM_TO_SID))
        self.assertEqual(schema.field("item_id").type, pa.large_string())
        bucket_schema = parquet.read_schema(self._part(out, SID_TO_ITEMS))
        self.assertEqual(
            bucket_schema.field("itemids").type, pa.list_(pa.large_string())
        )

    @parameterized.expand(
        [
            ("generation", {"generation": None}, "generation is required"),
            ("output_path", {"output_path": None}, "output_path is required"),
            (
                "progress_interval",
                {"progress_interval": 0},
                "progress_interval must be >= 1",
            ),
            ("same_generation", {"from_generation": "v1"}, "writing over the"),
            ("item_to_sid_root", {"item_to_sid_root": "input"}, "path conflict"),
        ],
        name_func=parameterized_name_func,
    )
    def test_config_rejects_invalid_arguments(self, _name, overrides, message) -> None:
        with self.assertRaisesRegex(ValueError, message):
            self._runner("input", "map", **overrides)

    @parameterized.expand(
        [
            ("default", [], "candidate"),
            ("iterative", ["--strategy", "iterative"], "iterative"),
            ("uniform", ["--strategy", "uniform"], "uniform"),
        ],
        name_func=parameterized_name_func,
    )
    def test_parser_allows_rate_only_without_outputs(
        self, _name, extra_args, expected_strategy
    ) -> None:
        args = resolve_sid_collisions.build_parser().parse_args(
            [
                "--input_path",
                "input",
                "--codebook",
                "8,8",
                "--max_items_per_codebook",
                "2",
                "--rate_only",
                *extra_args,
            ]
        )

        config = ResolveSidCollisionsConfig.from_namespace(args)

        self.assertIsNone(config.output_path)
        self.assertIsNone(config.item_to_sid_root)
        self.assertIsNone(config.generation)
        self.assertEqual(config.progress_interval, 1_000_000)
        self.assertEqual(config.strategy, expected_strategy)

    def test_serving_set_stays_parquet_under_an_odps_writer_type(self) -> None:
        inp = os.path.join(self.test_dir, "in.parquet")
        out = os.path.join(self.test_dir, "out")
        _parquet(inp, [1, 2], [[0, 0], [0, 1]])

        with mock.patch.object(
            CollisionResolutionRunner, "_make_writer"
        ) as make_writer:
            self._run(
                inp,
                out,
                item_to_sid_writer_type="OdpsWriter",
                max_items_per_codebook=1,
            )

        # item_to_sid follows the flag; sid_to_items overrides it
        self.assertCountEqual(
            make_writer.call_args_list,
            [
                mock.call(bundle.artifact_location(out, "v1", ITEM_TO_SID)),
                mock.call(
                    bundle.artifact_location(out, "v1", SID_TO_ITEMS), "ParquetWriter"
                ),
            ],
        )
        self.assertEqual(make_writer.return_value.close.call_count, 2)

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
            generation=None,
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
        self.assertFalse(os.path.exists(out))

    def test_codes_layer_count_must_match_codebook(self) -> None:
        inp = os.path.join(self.test_dir, "in.parquet")
        out = os.path.join(self.test_dir, "out")
        _parquet(inp, [0, 1], [[0, 0, 0], [0, 0, 1]])

        with self.assertRaisesRegex(ValueError, "codes have 3 layers but --codebook"):
            self._run(inp, out, layer_sizes=(8, 8))

    def test_empty_input_raises(self) -> None:
        inp = os.path.join(self.test_dir, "in.parquet")
        out = os.path.join(self.test_dir, "out")
        _parquet(inp, [], [])
        with self.assertRaisesRegex(ValueError, "SID input is empty"):
            self._run(inp, out, max_items_per_codebook=2)

    def test_candidate_tolerates_duplicate_item_ids(self) -> None:
        # Both overflow rows share ID 7, so one matched candidate list serves both.
        stats, out = self._resolve(
            [7, 7, 7],
            [[0, 0]] * 3,
            [[[0, 1], [0, 2]]] * 2 + [[[0, 2], [0, 3]]],
            max_items_per_codebook=1,
        )

        self.assertEqual(stats.total_items, 3)
        self.assertEqual(stats.relocated_count, 2)
        self.assertEqual(self._read(out, ITEM_TO_SID)["item_id"], [7, 7, 7])
        self._assert_item_to_sid_matches_sid_to_items(out)

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

    def test_offset_codebook_follows_the_resolved_sid(self) -> None:
        _, out = self._resolve([0, 1, 2, 3], [[1, 2]] * 3 + [[3, 4]], [[[1, 5]]] * 4)

        self.assertIn([1, 5], self._read(out, ITEM_TO_SID)["codebook"])
        # layer 1 is shifted by layer_sizes[0]
        for artifact in (ITEM_TO_SID, SID_TO_ITEMS):
            rows = self._read(out, artifact)
            for codebook, offset in zip(rows["codebook"], rows["offset_codebook"]):
                self.assertEqual(offset, [codebook[0], codebook[1] + 8])

    def test_append_recomputes_offset_for_a_prior_written_without_it(self) -> None:
        # offset_codebook is derived, so a generation without it still merges
        out = os.path.join(self.test_dir, "out")
        self._seed_prior(
            out,
            [(0, [0, 0], [0, 0], 1), (1, [1, 1], [1, 1], 1)],
            [([0, 0], [0]), ([1, 1], [1])],
        )
        self._append(self._prepare_single(10), out)

        item_map = self._read(out, ITEM_TO_SID, "v2")
        self.assertEqual(len(item_map["item_id"]), 3)
        for codebook, offset in zip(item_map["codebook"], item_map["offset_codebook"]):
            self.assertEqual(offset, [codebook[0], codebook[1] + 8])

    def _seed_generation(self, name="v1", item_ids=None, codes=None):
        """Run a full resolve at generation v1, the prior a later append reads."""
        inp = os.path.join(self.test_dir, f"{name}_in.parquet")
        out = os.path.join(self.test_dir, name)
        item_ids = list(range(5)) if item_ids is None else item_ids
        codes = [[0, 0], [0, 0], [0, 1], [1, 0], [1, 1]] if codes is None else codes
        _parquet(inp, item_ids, codes, [[[0, 2], [0, 3]]] * len(item_ids))
        self._run(inp, out, generation="v1")
        return out

    def _assert_manifest_rows_match_artifacts(self, out, generation):
        """Every manifest row count must equal what its artifact actually holds."""
        manifest = self._manifest(out, generation)
        for name, entry in manifest.artifacts.items():
            actual = len(self._read(out, name, generation)["codebook"])
            self.assertEqual(entry.rows, actual, f"{name} rows")
        return manifest

    def test_manifest_rows_match_the_artifacts_on_a_full_resolve(self) -> None:
        _, out = self._resolve(list(range(5)), [[0, 0], [0, 0], [0, 1], [1, 0], [1, 1]])

        manifest = self._assert_manifest_rows_match_artifacts(out, "v1")
        self.assertEqual(set(manifest.artifacts), {ITEM_TO_SID, SID_TO_ITEMS})
        for artifact in (DELTA_ITEM_TO_SID, DELTA_SID_TO_ITEMS):
            self.assertFalse(
                os.path.exists(bundle.artifact_location(out, "v1", artifact))
            )
        self.assertIsNone(manifest.since_bundle_uuid)

    def test_append_publishes_a_delta_of_exactly_the_gained_buckets(self) -> None:
        out = self._seed_generation()
        inp = os.path.join(self.test_dir, "v2_in.parquet")
        _parquet(
            inp,
            [10, 11, 12, 13],
            [[0, 0], [0, 1], [4, 4], [7, 7]],
            [[[0, 2], [0, 3]]] * 4,
        )
        self._append(inp, out)

        manifest = self._assert_manifest_rows_match_artifacts(out, "v2")
        self.assertEqual(
            set(manifest.artifacts),
            set(bundle.ITEM_TO_SID_ARTIFACTS) | set(bundle.SID_TO_ITEMS_ARTIFACTS),
        )
        self.assertEqual(
            manifest.since_bundle_uuid, self._manifest(out, "v1").bundle_uuid
        )
        delta_map = self._read(out, DELTA_ITEM_TO_SID, "v2")
        self.assertCountEqual(delta_map["item_id"], [10, 11, 12, 13])
        # Delta rows are the whole post-append bucket, so upserts are idempotent.
        delta = self._read(out, DELTA_SID_TO_ITEMS, "v2")
        self.assertEqual(delta["codebook"], [[0, 1], [0, 2], [4, 4], [7, 7]])
        self.assertEqual(delta["itemids"], [[2, 11], [10], [12], [13]])
        for artifact in (delta_map, delta):
            for codebook, offset in zip(
                artifact["codebook"], artifact["offset_codebook"]
            ):
                self.assertEqual(offset, [codebook[0], codebook[1] + 8])

    def test_append_never_moves_a_published_item(self) -> None:
        out = self._seed_generation()
        published = self._item_to_sid_rows(out, "v1")

        inp = os.path.join(self.test_dir, "v2_in.parquet")
        _parquet(inp, [10, 11, 12], [[0, 0], [0, 0], [1, 1]], [[[0, 2], [0, 3]]] * 3)
        stats = self._append(inp, out)

        self.assertEqual(stats.total_items, 3)
        merged = self._item_to_sid_rows(out, "v2")
        self.assertEqual(len(merged), 8)
        for item_id, row in published.items():
            self.assertEqual(merged[item_id], row, f"item {item_id} moved")
        self._assert_dense_indices(out, "v2")
        self._assert_item_to_sid_matches_sid_to_items(out, "v2")

    @parameterized.expand(
        [
            ("candidate", "candidate"),
            ("iterative", "iterative"),
            ("uniform", "uniform"),
        ],
        name_func=parameterized_name_func,
    )
    def test_append_preserves_over_capacity_published_bucket(
        self, _name, strategy
    ) -> None:
        out = os.path.join(self.test_dir, "out")
        self._seed_prior(
            out,
            [
                (0, [0, 0], [0, 0], 1),
                (1, [0, 0], [0, 0], 2),
                (2, [0, 0], [0, 0], 3),
            ],
            [([0, 0], [0, 1, 2])],
        )
        published = self._item_to_sid_rows(out, "v1")
        inp = os.path.join(self.test_dir, "v2_in.parquet")
        _parquet(inp, [10, 11], [[0, 0], [0, 0]], [[[0, 1], [0, 2]]] * 2)

        stats = self._append(inp, out, strategy=strategy)

        self.assertEqual(stats.relocated_count, 2)
        self.assertEqual(stats.max_final_bucket_size, 3)
        merged = self._item_to_sid_rows(out, "v2")
        for item_id, row in published.items():
            self.assertEqual(merged[item_id], row)
        self.assertNotIn((0, 0), {merged[10][1], merged[11][1]})
        self._assert_item_to_sid_matches_sid_to_items(out, "v2")

    @parameterized.expand(
        [("one_bucket_per_batch", 1), ("one_batch", 100000)],
        name_func=parameterized_name_func,
    )
    def test_append_merges_new_buckets_around_published_ones(
        self, _name, batch_size
    ) -> None:
        # New buckets land before, between, inside and after the reader batches,
        # and [4, 4] joins the last published bucket of its batch.
        out = self._seed_generation(item_ids=[0, 1], codes=[[2, 2], [4, 4]])
        inp = os.path.join(self.test_dir, "v2_in.parquet")
        _parquet(inp, [10, 11, 12, 13], [[0, 0], [3, 3], [4, 4], [7, 7]])
        self._append(inp, out, batch_size=batch_size)

        buckets = self._read(out, SID_TO_ITEMS, "v2")
        self.assertEqual(buckets["codebook"], [[0, 0], [2, 2], [3, 3], [4, 4], [7, 7]])
        self.assertEqual(buckets["itemids"], [[10], [0], [11], [1, 12], [13]])
        delta = self._read(out, DELTA_SID_TO_ITEMS, "v2")
        self.assertEqual(delta["codebook"], [[0, 0], [3, 3], [4, 4], [7, 7]])
        self.assertEqual(delta["itemids"], [[10], [11], [1, 12], [13]])
        self._assert_item_to_sid_matches_sid_to_items(out, "v2")

    def test_append_chain_stays_consistent(self) -> None:
        out = self._seed_generation()
        for generation in range(2, 5):
            inp = os.path.join(self.test_dir, f"v{generation}_in.parquet")
            previous, current = f"v{generation - 1}", f"v{generation}"
            base = 100 * generation
            _parquet(
                inp,
                [base, base + 1],
                [[0, 0], [generation, generation]],
                [[[0, 2], [0, 3]]] * 2,
            )
            published = self._item_to_sid_rows(out, previous)
            self._append(inp, out, previous=previous, generation=current)
            merged = self._item_to_sid_rows(out, current)
            for item_id, row in published.items():
                self.assertEqual(merged[item_id], row)
            self._assert_dense_indices(out, current)
            self._assert_item_to_sid_matches_sid_to_items(out, current)

    def test_append_reads_a_csv_prior_and_converts_format(self) -> None:
        out = os.path.join(self.test_dir, "out")
        self._seed_prior(
            out,
            [(0, [0, 0], [0, 0], 1), (1, [1, 1], [1, 1], 1)],
            [([0, 0], [0]), ([1, 1], [1])],
            csv=True,
        )

        inp = os.path.join(self.test_dir, "v2_in.parquet")
        _parquet(inp, [10, 11], [[0, 0], [2, 2]], [[[0, 2]]] * 2)
        self._append(inp, out)

        merged = self._item_to_sid_rows(out, "v2")
        self.assertEqual(len(merged), 4)
        self.assertEqual(merged[0], ((0, 0), (0, 0), 1))
        self.assertEqual(merged[10], ((0, 0), (0, 0), 2))
        self._assert_dense_indices(out, "v2")
        self._assert_item_to_sid_matches_sid_to_items(out, "v2")

    def test_append_rejects_already_published_item_id(self) -> None:
        out = self._seed_generation()
        inp = os.path.join(self.test_dir, "v2_in.parquet")
        _parquet(inp, [0, 99], [[0, 0], [2, 2]], [[[0, 2]]] * 2)
        with self.assertRaisesRegex(ValueError, "already published"):
            self._append(inp, out)

    @parameterized.expand(
        [
            (
                "item_to_sid_longer",
                [(0, [0, 0], [0, 0], 1), (1, [5, 5], [5, 5], 1)],
                [([0, 0], [0])],
            ),
            (
                "sid_to_items_longer",
                [(0, [0, 0], [0, 0], 1)],
                [([0, 0], [0]), ([5, 5], [1])],
            ),
        ],
        name_func=parameterized_name_func,
    )
    def test_append_rejects_prior_truncated_outside_affected_prefixes(
        self, _name, map_rows, bucket_rows
    ) -> None:
        # Prefix 5 has no new items, so only the totals catch the missing item.
        out = self._seed_prior(
            os.path.join(self.test_dir, "out"), map_rows, bucket_rows
        )
        with self.assertRaisesRegex(ValueError, "one of the two artifacts is"):
            self._append(self._prepare_single(10), out)

    def test_append_rejects_unsorted_published_sid_to_items(self) -> None:
        out = os.path.join(self.test_dir, "out")
        self._seed_prior(
            out,
            [(0, [1, 1], [1, 1], 1), (1, [0, 0], [0, 0], 1)],
            [([1, 1], [0]), ([0, 0], [1])],
        )
        with self.assertRaisesRegex(ValueError, "ascending"):
            self._append(self._prepare_single(10), out)

    def test_append_rate_only_publishes_no_generation(self) -> None:
        out = self._seed_generation()
        stats = self._run(
            self._prepare_single(10),
            out,
            rate_only=True,
            generation=None,
            from_generation="v1",
        )
        self.assertEqual(stats.total_items, 1)
        self.assertEqual(os.listdir(out), ["v1"])


if __name__ == "__main__":
    unittest.main()
