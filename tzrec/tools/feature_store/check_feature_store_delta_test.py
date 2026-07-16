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
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from tzrec.tools.feature_store.check_feature_store_delta import (
    LocalSample,
    committed_parquet_paths,
    load_committed_upload,
    parse_args,
    resolve_output_dir,
    sample_local_records,
    verify_samples,
)


class _FakeView:
    def get_online_features(self, feature_name, keys, version):
        self.feature_name = feature_name
        self.keys = keys
        self.version = version
        return [
            {"sk": str(keys[0]), "embedding": [1.0, 2.0]},
            {"sk": str(keys[1]), "embedding": [9.0, 9.0]},
        ]


class CheckFeatureStoreDeltaTest(unittest.TestCase):
    def test_parse_args_requires_command_line_access_key_pair(self):
        args = parse_args(
            [
                "--pipeline_config",
                "pipeline.config",
                "--ak",
                "access-key-id",
                "--sk",
                "access-key-secret",
                "--featuredb_username",
                "featuredb-user",
                "--featuredb_password",
                "featuredb-password",
            ]
        )
        self.assertEqual(args.access_key_id, "access-key-id")
        self.assertEqual(args.access_key_secret, "access-key-secret")
        self.assertEqual(args.featuredb_username, "featuredb-user")
        self.assertEqual(args.featuredb_password, "featuredb-password")

    def test_resolve_output_dir_uses_colocated_relocated_outbox(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = os.path.join(tmp_dir, "pipeline.config")
            colocated = os.path.join(tmp_dir, "delta_embedding_dump")
            os.mkdir(colocated)
            self.assertEqual(
                resolve_output_dir(config_path, "/missing/model", "", None),
                colocated,
            )

    def test_load_committed_upload_checks_target_and_success(self):
        settings = SimpleNamespace(
            project_name="project", feature_view_name="view", version="v1"
        )
        with tempfile.TemporaryDirectory() as output_dir:
            state_dir = os.path.join(output_dir, ".feature_store_upload", "target")
            os.makedirs(state_dir)
            target = {
                "project_name": "project",
                "feature_view_name": "view",
                "version": "v1",
            }
            with open(os.path.join(state_dir, "committed.json"), "w") as output:
                json.dump({**target, "committed_global_step": 20}, output)
            with open(
                os.path.join(state_dir, "step_20._FS_SUCCESS.json"), "w"
            ) as output:
                json.dump(
                    {
                        **target,
                        "global_step": 20,
                        "success_records": 3,
                        "total_records": 3,
                        "shards": [{}],
                    },
                    output,
                )

            actual_state_dir, committed, success = load_committed_upload(
                output_dir, settings
            )

        self.assertEqual(actual_state_dir, state_dir)
        self.assertEqual(committed["committed_global_step"], 20)
        self.assertEqual(success["global_step"], 20)

    def test_committed_parquet_paths_and_sampling(self):
        with tempfile.TemporaryDirectory() as output_dir:
            path = os.path.join(output_dir, "delta__fs_target_step_20.parquet")
            pq.write_table(
                pa.table(
                    {
                        "embedding_name": ["table_a", "table_a", "table_b"],
                        "key_id": pa.array([1, 2, 3], type=pa.int64()),
                        "embedding": pa.array(
                            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                            type=pa.list_(pa.float32()),
                        ),
                        "operation": ["UPSERT", "UPSERT", "UPSERT"],
                    }
                ),
                path,
            )
            paths = committed_parquet_paths(
                output_dir, "delta__fs_target", 20, expected_shards=1
            )
            samples = sample_local_records(paths, 2)

        self.assertEqual(paths, [path])
        self.assertEqual(
            [(sample.embedding_name, sample.key_id) for sample in samples],
            [("table_a", 1), ("table_a", 2)],
        )
        np.testing.assert_array_equal(samples[0].embedding, [1.0, 2.0])

    def test_verify_samples_separates_presence_from_value_match(self):
        samples = [
            LocalSample("table_a", 1, np.array([1.0, 2.0], np.float32), "a"),
            LocalSample("table_a", 2, np.array([3.0, 4.0], np.float32), "a"),
            LocalSample("table_a", 3, np.array([5.0, 6.0], np.float32), "a"),
        ]
        view = _FakeView()

        results, summary = verify_samples(view, "v1", samples)

        self.assertEqual(view.feature_name, "table_a")
        self.assertEqual(view.keys, [1, 2, 3])
        self.assertEqual(view.version, "v1")
        self.assertEqual(
            [result["status"] for result in results],
            ["MATCH", "PRESENT_DIFFERENT", "MISSING"],
        )
        self.assertEqual(results[0]["remote_embedding"], [1.0, 2.0])
        self.assertEqual(results[1]["remote_embedding"], [9.0, 9.0])
        self.assertIsNone(results[2]["remote_embedding"])
        self.assertEqual(
            summary,
            {
                "requested": 3,
                "found": 2,
                "matching": 1,
                "present_different": 1,
                "missing": 1,
            },
        )


if __name__ == "__main__":
    unittest.main()
