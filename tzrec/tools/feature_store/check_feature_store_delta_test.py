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
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from tzrec.tools.feature_store.check_feature_store_delta import (
    LocalSample,
    create_feature_store_view,
    parse_args,
    resolve_output_dir,
    resolve_upload_step,
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
    def test_create_feature_store_view_uses_public_endpoint_when_supported(self):
        captured_kwargs = {}
        view = SimpleNamespace(
            pk_field="embedding_name",
            sk_field="key_id",
            embedding_field="embedding",
        )
        project = SimpleNamespace(get_dynamic_embedding_feature_view=lambda name: view)

        class FakeFeatureStoreClient:
            def __init__(self, test_mode=False, **kwargs):
                captured_kwargs.update(kwargs)
                captured_kwargs["test_mode"] = test_mode

            def get_project(self, name):
                return project

        settings = SimpleNamespace(
            access_key_id="ak-id",
            access_key_secret="ak-secret",
            region="cn-test",
            endpoint="",
            security_token="",
            featuredb_username="featuredb-user",
            featuredb_password="featuredb-password",
            project_name="project",
            feature_view_name="view",
        )
        feature_store_module = SimpleNamespace(
            FeatureStoreClient=FakeFeatureStoreClient
        )

        with mock.patch.dict(sys.modules, {"feature_store_py": feature_store_module}):
            actual = create_feature_store_view(settings)

        self.assertIs(actual, view)
        self.assertTrue(captured_kwargs["test_mode"])

    def test_parse_args_does_not_accept_credentials(self):
        args = parse_args(["--pipeline_config", "pipeline.config"])

        self.assertEqual(args.pipeline_config, "pipeline.config")
        self.assertFalse(hasattr(args, "access_key_id"))
        self.assertFalse(hasattr(args, "access_key_secret"))
        self.assertFalse(hasattr(args, "featuredb_username"))
        self.assertFalse(hasattr(args, "featuredb_password"))

    def test_resolve_output_dir_uses_colocated_relocated_outbox(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = os.path.join(tmp_dir, "pipeline.config")
            colocated = os.path.join(tmp_dir, "delta_embedding_dump")
            os.mkdir(colocated)
            self.assertEqual(
                resolve_output_dir(config_path, "/missing/model", "", None),
                colocated,
            )

    def test_resolve_upload_step_finds_latest_single_rank(self):
        with tempfile.TemporaryDirectory() as output_dir:
            prefix = "delta__fs_target"
            for step in (10, 20, 5):
                path = os.path.join(output_dir, f"{prefix}_step_{step}.parquet")
                pq.write_table(
                    pa.table(
                        {
                            "embedding_name": ["a"],
                            "key_id": pa.array([1], type=pa.int64()),
                            "embedding": pa.array([[1.0]], type=pa.list_(pa.float32())),
                        }
                    ),
                    path,
                )
            step, paths = resolve_upload_step(output_dir, prefix, world_size=1)
            self.assertEqual(step, 20)
            self.assertEqual(len(paths), 1)
            self.assertIn("step_20", paths[0])

    def test_resolve_upload_step_finds_latest_multi_rank(self):
        with tempfile.TemporaryDirectory() as output_dir:
            prefix = "delta__fs_target"
            for step in (10, 20):
                step_dir = os.path.join(output_dir, f"step_{step}")
                os.makedirs(step_dir)
                for rank in range(2):
                    path = os.path.join(
                        step_dir,
                        f"{prefix}_step_{step}_rank_{rank}_of_2.parquet",
                    )
                    pq.write_table(
                        pa.table(
                            {
                                "embedding_name": ["a"],
                                "key_id": pa.array([rank + 1], type=pa.int64()),
                                "embedding": pa.array(
                                    [[1.0]], type=pa.list_(pa.float32())
                                ),
                            }
                        ),
                        path,
                    )
            step, paths = resolve_upload_step(output_dir, prefix, world_size=2)
            self.assertEqual(step, 20)
            self.assertEqual(len(paths), 2)

    def test_resolve_upload_step_explicit_step(self):
        with tempfile.TemporaryDirectory() as output_dir:
            prefix = "delta__fs_target"
            path = os.path.join(output_dir, f"{prefix}_step_15.parquet")
            pq.write_table(
                pa.table(
                    {
                        "embedding_name": ["a"],
                        "key_id": pa.array([1], type=pa.int64()),
                        "embedding": pa.array([[1.0]], type=pa.list_(pa.float32())),
                    }
                ),
                path,
            )
            step, paths = resolve_upload_step(
                output_dir, prefix, world_size=1, global_step=15
            )
            self.assertEqual(step, 15)
            self.assertEqual(paths, [path])

    def test_resolve_upload_step_raises_when_not_found(self):
        with tempfile.TemporaryDirectory() as output_dir:
            with self.assertRaises(FileNotFoundError):
                resolve_upload_step(output_dir, "delta__fs_target", world_size=1)

    def test_parquet_paths_and_sampling(self):
        with tempfile.TemporaryDirectory() as output_dir:
            prefix = "delta__fs_target"
            path = os.path.join(output_dir, f"{prefix}_step_20.parquet")
            pq.write_table(
                pa.table(
                    {
                        "embedding_name": ["table_a", "table_a", "table_b"],
                        "key_id": pa.array([1, 2, 3], type=pa.int64()),
                        "embedding": pa.array(
                            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                            type=pa.list_(pa.float32()),
                        ),
                    }
                ),
                path,
            )
            step, paths = resolve_upload_step(
                output_dir, prefix, world_size=1, global_step=20
            )
            samples = sample_local_records(paths, 2)

        self.assertEqual(step, 20)
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
