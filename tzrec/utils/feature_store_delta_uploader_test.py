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

import json
import os
import tempfile
import threading
import unittest
from unittest import mock

import pyarrow as pa
import pyarrow.parquet as pq
from google.protobuf.descriptor import FieldDescriptor

from tzrec.protos.train_pb2 import FeatureStoreConfig
from tzrec.utils.delta_embedding_dump import _DELTA_DUMP_SCHEMA
from tzrec.utils.feature_store_delta_uploader import (
    DELTA_DUMP_GENERATION_METADATA_KEY,
    FeatureStoreDeltaUploader,
    FeatureStoreUploadError,
    FeatureStoreUploadSettings,
    feature_store_delta_file_prefix,
)

_TEST_DUMP_GENERATION = "00112233445566778899aabbccddeeff"


def _schema_with_generation(generation: str = _TEST_DUMP_GENERATION) -> pa.Schema:
    metadata = dict(_DELTA_DUMP_SCHEMA.metadata or {})
    metadata[DELTA_DUMP_GENERATION_METADATA_KEY] = generation.encode("ascii")
    return _DELTA_DUMP_SCHEMA.with_metadata(metadata)


def _feature_store_config(**overrides) -> FeatureStoreConfig:
    config = FeatureStoreConfig(
        region="cn-test",
        project_name="project_a",
        feature_entity_name="embedding_entity",
        feature_view_name="shared_embeddings",
        version="model_a@export_1",
        upload_batch_size=2,
        max_retries=1,
        retry_backoff_secs=0,
        shard_wait_timeout_secs=2,
        shutdown_timeout_secs=5,
        max_pending_steps=8,
        poll_interval_secs=1,
    )
    for name, value in overrides.items():
        setattr(config, name, value)
    return config


def _row(
    step: int,
    rank: int,
    key_id: int,
    values,
    name: str = "user_emb",
    world_size: int = 1,
):
    return {
        "global_step": step,
        "rank": rank,
        "world_size": world_size,
        "embedding_name": name,
        "embedding_role": "ebc",
        "feature_name": "user_id",
        "table_fqn": f"model.ebc.embedding_bags.{name}.weight",
        "key_id": key_id,
        "embedding": values,
    }


def _write_single_shard(
    output_dir: str,
    step: int,
    rows,
    generation: str = _TEST_DUMP_GENERATION,
    file_prefix=None,
) -> str:
    if file_prefix is None:
        file_prefix = feature_store_delta_file_prefix(_feature_store_config(), "delta")
    path = os.path.join(output_dir, f"{file_prefix}_step_{step}.parquet")
    schema = _schema_with_generation(generation)
    table = pa.Table.from_pylist(rows, schema=schema) if rows else schema.empty_table()
    pq.write_table(table, path)
    return path


def _write_rank_shard(
    output_dir: str,
    step: int,
    rank: int,
    world_size: int,
    rows,
    generation: str = _TEST_DUMP_GENERATION,
    file_prefix=None,
) -> str:
    if file_prefix is None:
        file_prefix = feature_store_delta_file_prefix(_feature_store_config(), "delta")
    step_dir = os.path.join(output_dir, f"step_{step}")
    os.makedirs(step_dir, exist_ok=True)
    path = os.path.join(
        step_dir,
        f"{file_prefix}_step_{step}_rank_{rank}_of_{world_size}.parquet",
    )
    table = pa.Table.from_pylist(rows, schema=_schema_with_generation(generation))
    pq.write_table(table, path)
    return path


class _FakeView:
    pk_field = "embedding_name"
    sk_field = "key_id"
    embedding_field = "embedding"

    def __init__(self, summaries=None, close_error=None, max_workers=4):
        self.calls = []
        self.closed = []
        self.flush_calls = []
        self._summaries = list(summaries or [])
        self._close_error = close_error
        self._batch_size = 1000
        self._max_workers = max_workers
        self._pending_sizes = []

    def write_features(self, **kwargs):
        self.calls.append(kwargs)
        self._pending_sizes.append(len(kwargs["data"]))

    def write_flush(self):
        pending_sizes = self._pending_sizes
        self._pending_sizes = []
        self.flush_calls.append(pending_sizes)
        if self._summaries:
            return self._summaries.pop(0)
        total_records = sum(pending_sizes)
        return {
            "total_batches": len(pending_sizes),
            "failed_batches": 0,
            "total_records": total_records,
            "success_records": total_records,
            "failed_records": 0,
            "errors": [],
        }

    def close(self, wait=True):
        self.closed.append(wait)
        if self._close_error is not None:
            raise self._close_error


class _BlockingView(_FakeView):
    def __init__(self):
        super().__init__()
        self.flush_started = threading.Event()
        self.release_flush = threading.Event()
        self.close_finished = threading.Event()

    def write_flush(self):
        self.flush_started.set()
        self.release_flush.wait(timeout=5)
        return super().write_flush()

    def close(self, wait=True):
        super().close(wait=wait)
        self.close_finished.set()


class _FakeGenericFeatureView:
    def __init__(
        self,
        *,
        feature_view_type="DynamicEmbedding",
        entity="embedding_entity",
        fields=None,
        provisioning=None,
    ):
        self.type = feature_view_type
        self.feature_entity_name = entity
        self.fields_dict = (
            {
                "embedding_name": {
                    "Name": "embedding_name",
                    "Type": "STRING",
                    "Attributes": ["PrimaryKey"],
                },
                "key_id": {
                    "Name": "key_id",
                    "Type": "INT64",
                    "Attributes": ["SubKey"],
                },
                "embedding": {
                    "Name": "embedding",
                    "Type": "ARRAY<FLOAT>",
                    "Attributes": [],
                },
            }
            if fields is None
            else fields
        )
        if provisioning is None:
            provisioning = {
                "ttl": 1296000,
                "shard_count": 20,
                "replication_count": 1,
            }
        self.summary = {"Config": json.dumps(provisioning)}


class _FakeProject:
    def __init__(
        self,
        view,
        *,
        created_view=None,
        generic_view=None,
        create_error=None,
        view_after_create_error=None,
    ):
        self._view = view
        self._created_view = created_view
        self._generic_view = generic_view
        self._create_error = create_error
        self._view_after_create_error = view_after_create_error
        self.dynamic_get_calls = []
        self.generic_get_calls = []
        self.create_calls = []

    def get_dynamic_embedding_feature_view(self, name):
        self.dynamic_get_calls.append(name)
        return self._view

    def get_feature_view(self, name):
        self.generic_get_calls.append(name)
        if self._generic_view is None and self._view is not None:
            self._generic_view = _FakeGenericFeatureView()
        return self._generic_view

    def create_dynamic_embedding_feature_view(self, **kwargs):
        self.create_calls.append(kwargs)
        if self._create_error is not None:
            self._view = self._view_after_create_error
            raise self._create_error
        self._view = self._created_view or _FakeView()
        self._generic_view = _FakeGenericFeatureView()
        return self._view


class _FakeCredential:
    access_key_id = "fake-ak"
    access_key_secret = "fake-sk"
    security_token = "fake-sts"


class _FakeCredentialsClient:
    def get_credential(self):
        return _FakeCredential()


class _FakeClient:
    def __init__(self, project, kwargs):
        self._project = project
        self.kwargs = kwargs

    def get_project(self, name):
        return self._project


class _FakeClientFactory:
    def __init__(self, view, **project_kwargs):
        self.view = view
        self.calls = []
        self.project = _FakeProject(view, **project_kwargs)

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeClient(self.project, kwargs)


class _SequencedClientFactory:
    def __init__(self, views):
        self._projects = [_FakeProject(view) for view in views]
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeClient(self._projects.pop(0), kwargs)


class FeatureStoreDeltaUploaderTest(unittest.TestCase):
    def setUp(self):
        self._cred_patch = mock.patch.object(
            FeatureStoreDeltaUploader,
            "_create_credentials_client",
            return_value=_FakeCredentialsClient(),
        )
        self._cred_patch.start()
        self.addCleanup(self._cred_patch.stop)

    def test_proto_groups_required_fields_before_optional_fields(self):
        required_fields = [
            "region",
            "project_name",
            "feature_view_name",
            "feature_entity_name",
            "version",
        ]
        optional_fields = [
            "endpoint",
            "upload_batch_size",
            "max_retries",
            "retry_backoff_secs",
            "shard_wait_timeout_secs",
            "shutdown_timeout_secs",
            "max_pending_steps",
            "poll_interval_secs",
            "feature_view_ttl_secs",
            "feature_view_shard_count",
            "feature_view_replication_count",
            "allow_custom_endpoint",
        ]
        fields = list(FeatureStoreConfig.DESCRIPTOR.fields)

        self.assertEqual(
            [field.name for field in fields], required_fields + optional_fields
        )
        self.assertTrue(
            all(
                field.label == FieldDescriptor.LABEL_REQUIRED
                for field in fields[: len(required_fields)]
            )
        )
        self.assertTrue(
            all(
                field.label == FieldDescriptor.LABEL_OPTIONAL
                for field in fields[len(required_fields) :]
            )
        )
        self.assertEqual(
            [field.number for field in fields],
            list(range(1, 18)),
        )
        for field_name in required_fields:
            with self.subTest(field_name=field_name):
                config = _feature_store_config()
                config.ClearField(field_name)
                self.assertFalse(config.IsInitialized())
                self.assertIn(field_name, config.FindInitializationErrors())
                with self.assertRaisesRegex(ValueError, field_name):
                    FeatureStoreUploadSettings.from_proto(config)

    def test_feature_entity_is_required_for_view_creation(self):
        config = _feature_store_config()
        config.ClearField("feature_entity_name")

        self.assertFalse(config.IsInitialized())
        self.assertIn("feature_entity_name", config.FindInitializationErrors())
        with self.assertRaisesRegex(ValueError, "feature_entity_name"):
            FeatureStoreUploadSettings.from_proto(config)

    def test_version_is_required_and_must_be_explicit(self):
        config = _feature_store_config()
        config.ClearField("version")

        self.assertFalse(config.IsInitialized())
        self.assertIn("version", config.FindInitializationErrors())
        with self.assertRaisesRegex(ValueError, "required fields.*version"):
            FeatureStoreUploadSettings.from_proto(config)

        with self.assertRaisesRegex(ValueError, "explicit non-default version"):
            FeatureStoreUploadSettings.from_proto(
                _feature_store_config(version="default")
            )

    def test_region_fallback_and_config_validation(self):
        config = _feature_store_config(region="")
        with mock.patch.dict(os.environ, {"ALIBABA_CLOUD_REGION": "cn-env"}):
            settings = FeatureStoreUploadSettings.from_proto(config)
        self.assertEqual(settings.region, "cn-env")

        with self.assertRaisesRegex(ValueError, "userinfo"):
            FeatureStoreUploadSettings.from_proto(
                _feature_store_config(endpoint="https://user:secret@example.com")
            )
        with self.assertRaisesRegex(ValueError, "must be <= 1000"):
            FeatureStoreUploadSettings.from_proto(
                _feature_store_config(upload_batch_size=1001)
            )
        with self.assertRaisesRegex(ValueError, "shard_count must be in"):
            FeatureStoreUploadSettings.from_proto(
                _feature_store_config(feature_view_shard_count=21)
            )
        with self.assertRaisesRegex(ValueError, "replication_count must be in"):
            FeatureStoreUploadSettings.from_proto(
                _feature_store_config(feature_view_replication_count=4)
            )

    def test_endpoint_must_be_a_trusted_https_host(self):
        for endpoint in (
            "featurestore.cn-hangzhou.aliyuncs.com",
            "https://featurestore.cn-hangzhou.aliyuncs.com",
            "https://featurestore-vpc.cn-beijing.aliyuncs.com/",
        ):
            with self.subTest(endpoint=endpoint):
                settings = FeatureStoreUploadSettings.from_proto(
                    _feature_store_config(endpoint=endpoint)
                )
                self.assertEqual(settings.endpoint, endpoint)
                self.assertFalse(settings.allow_custom_endpoint)

        rejected_endpoints = {
            "http://featurestore.cn-hangzhou.aliyuncs.com": "HTTPS",
            "ftp://featurestore.cn-hangzhou.aliyuncs.com": "HTTPS",
            "evil.example.com": "trusted",
            "https://aliyuncs.com": "trusted",
            "https://featurestore.cn-hangzhou.aliyuncs.com.evil.com": "trusted",
            "featurestore.cn-hangzhou.aliyuncs.com:8443": "port",
            "https://featurestore.cn-hangzhou.aliyuncs.com/v1": "path, query",
            "https://featurestore.cn-hangzhou.aliyuncs.com?x=1": "path, query",
            "https://featurestore.cn-hangzhou.aliyuncs.com#frag": "fragment",
            "https://": "host",
        }
        for endpoint, message in rejected_endpoints.items():
            with self.subTest(endpoint=endpoint):
                with self.assertRaisesRegex(ValueError, message):
                    FeatureStoreUploadSettings.from_proto(
                        _feature_store_config(endpoint=endpoint)
                    )

    def test_allow_custom_endpoint_opts_into_vetted_hosts(self):
        settings = FeatureStoreUploadSettings.from_proto(
            _feature_store_config(
                endpoint="https://featurestore.internal.example.com:8443",
                allow_custom_endpoint=True,
            )
        )
        self.assertTrue(settings.allow_custom_endpoint)

        for endpoint in (
            "http://featurestore.internal.example.com",
            "https://user:secret@featurestore.internal.example.com",
            "https://featurestore.internal.example.com/v1",
        ):
            with self.subTest(endpoint=endpoint):
                with self.assertRaisesRegex(ValueError, "endpoint"):
                    FeatureStoreUploadSettings.from_proto(
                        _feature_store_config(
                            endpoint=endpoint, allow_custom_endpoint=True
                        )
                    )

    def test_start_reuses_existing_dynamic_embedding_feature_view(self):
        with tempfile.TemporaryDirectory() as output_dir:
            view = _FakeView()
            factory = _FakeClientFactory(view)
            uploader = FeatureStoreDeltaUploader(
                _feature_store_config(),
                output_dir,
                "delta",
                1,
                {"user_emb": 2},
                client_factory=factory,
            )

            uploader.start()
            uploader.close()

            self.assertEqual(factory.project.dynamic_get_calls, ["shared_embeddings"])
            self.assertEqual(factory.project.generic_get_calls, ["shared_embeddings"])
            self.assertEqual(factory.project.create_calls, [])
            self.assertNotIn("test_mode", factory.calls[0])
            self.assertEqual(view.closed, [True])

    def test_start_creates_missing_dynamic_embedding_feature_view(self):
        with tempfile.TemporaryDirectory() as output_dir:
            created_view = _FakeView()
            factory = _FakeClientFactory(None, created_view=created_view)
            uploader = FeatureStoreDeltaUploader(
                _feature_store_config(),
                output_dir,
                "delta",
                1,
                {"user_emb": 2},
                client_factory=factory,
            )

            uploader.start()
            uploader.close()

            self.assertEqual(
                factory.project.generic_get_calls,
                ["shared_embeddings", "shared_embeddings"],
            )
            self.assertEqual(
                factory.project.create_calls,
                [
                    {
                        "name": "shared_embeddings",
                        "entity": "embedding_entity",
                        "pk_field_name": "embedding_name",
                        "sk_field_name": "key_id",
                        "embedding_field_name": "embedding",
                        "pk_field_type": "STRING",
                        "sk_field_type": "INT64",
                        "ttl": 1296000,
                        "shard_count": 20,
                        "replication_count": 1,
                    }
                ],
            )
            self.assertEqual(created_view.closed, [True])

    def test_start_rejects_same_name_non_dynamic_feature_view(self):
        with tempfile.TemporaryDirectory() as output_dir:
            generic_view = mock.Mock(type="Batch")
            factory = _FakeClientFactory(None, generic_view=generic_view)
            uploader = FeatureStoreDeltaUploader(
                _feature_store_config(),
                output_dir,
                "delta",
                1,
                {"user_emb": 2},
                client_factory=factory,
            )

            with self.assertRaisesRegex(RuntimeError, "incompatible type"):
                uploader.start()

            self.assertEqual(factory.project.create_calls, [])

    def test_start_rejects_existing_feature_view_with_wrong_entity(self):
        with tempfile.TemporaryDirectory() as output_dir:
            view = _FakeView()
            generic_view = _FakeGenericFeatureView(entity="another_entity")
            factory = _FakeClientFactory(view, generic_view=generic_view)
            uploader = FeatureStoreDeltaUploader(
                _feature_store_config(),
                output_dir,
                "delta",
                1,
                {"user_emb": 2},
                client_factory=factory,
            )

            with self.assertRaisesRegex(RuntimeError, "entity mismatch"):
                uploader.start()

            self.assertEqual(factory.project.create_calls, [])
            self.assertEqual(view.closed, [True])

    def test_start_rejects_existing_feature_view_with_wrong_field_contract(self):
        with tempfile.TemporaryDirectory() as output_dir:
            view = _FakeView()
            generic_view = _FakeGenericFeatureView()
            generic_view.fields_dict["embedding"]["Type"] = "ARRAY<DOUBLE>"
            factory = _FakeClientFactory(view, generic_view=generic_view)
            uploader = FeatureStoreDeltaUploader(
                _feature_store_config(),
                output_dir,
                "delta",
                1,
                {"user_emb": 2},
                client_factory=factory,
            )

            with self.assertRaisesRegex(RuntimeError, "field contract mismatch"):
                uploader.start()

            self.assertEqual(factory.project.create_calls, [])
            self.assertEqual(view.closed, [True])

    def test_start_rejects_existing_feature_view_with_wrong_provisioning(self):
        with tempfile.TemporaryDirectory() as output_dir:
            view = _FakeView()
            generic_view = _FakeGenericFeatureView(
                provisioning={
                    "ttl": 60,
                    "shard_count": 20,
                    "replication_count": 1,
                }
            )
            factory = _FakeClientFactory(view, generic_view=generic_view)
            uploader = FeatureStoreDeltaUploader(
                _feature_store_config(),
                output_dir,
                "delta",
                1,
                {"user_emb": 2},
                client_factory=factory,
            )

            with self.assertRaisesRegex(RuntimeError, "provisioning mismatch"):
                uploader.start()

            self.assertEqual(factory.project.create_calls, [])
            self.assertEqual(view.closed, [True])

    def test_start_recovers_from_concurrent_feature_view_creation(self):
        with tempfile.TemporaryDirectory() as output_dir:
            concurrent_view = _FakeView()
            factory = _FakeClientFactory(
                None,
                create_error=RuntimeError("already exists"),
                view_after_create_error=concurrent_view,
            )
            uploader = FeatureStoreDeltaUploader(
                _feature_store_config(),
                output_dir,
                "delta",
                1,
                {"user_emb": 2},
                client_factory=factory,
            )

            uploader.start()
            uploader.close()

            self.assertEqual(len(factory.project.create_calls), 1)
            self.assertEqual(
                factory.project.dynamic_get_calls,
                ["shared_embeddings", "shared_embeddings"],
            )
            self.assertEqual(concurrent_view.closed, [True])

    def test_start_closes_new_feature_view_with_incompatible_schema(self):
        with tempfile.TemporaryDirectory() as output_dir:
            created_view = _FakeView()
            created_view.pk_field = "wrong_pk"
            factory = _FakeClientFactory(None, created_view=created_view)
            uploader = FeatureStoreDeltaUploader(
                _feature_store_config(),
                output_dir,
                "delta",
                1,
                {"user_emb": 2},
                client_factory=factory,
            )

            with self.assertRaisesRegex(RuntimeError, "schema mismatch"):
                uploader.start()

            self.assertEqual(len(factory.project.create_calls), 1)
            self.assertEqual(created_view.closed, [True])

    def test_start_creates_missing_view_without_version_precheck(self):
        with tempfile.TemporaryDirectory() as output_dir:
            created_view = _FakeView()
            factory = _FakeClientFactory(None, created_view=created_view)
            uploader = FeatureStoreDeltaUploader(
                _feature_store_config(),
                output_dir,
                "delta",
                1,
                {"user_emb": 2},
                client_factory=factory,
            )

            uploader.start()
            uploader.close()

            self.assertEqual(len(factory.project.create_calls), 1)
            self.assertEqual(created_view.closed, [True])

    def test_start_reports_missing_entity_when_view_creation_fails(self):
        with tempfile.TemporaryDirectory() as output_dir:
            factory = _FakeClientFactory(
                None, create_error=ValueError("Entity not found")
            )
            uploader = FeatureStoreDeltaUploader(
                _feature_store_config(),
                output_dir,
                "delta",
                1,
                {"user_emb": 2},
                client_factory=factory,
            )

            with self.assertRaisesRegex(RuntimeError, "feature_entity_name"):
                uploader.start()

            self.assertEqual(len(factory.project.create_calls), 1)

    def test_submit_requires_started_uploader(self):
        with tempfile.TemporaryDirectory() as output_dir:
            uploader = FeatureStoreDeltaUploader(
                _feature_store_config(),
                output_dir,
                "delta",
                1,
                {"user_emb": 2},
                client_factory=_FakeClientFactory(_FakeView()),
            )
            with self.assertRaisesRegex(RuntimeError, "start.*before submit"):
                uploader.submit(10)
            uploader.close()

    def test_complete_step_uploads_merge_with_stable_version_and_ts(self):
        with tempfile.TemporaryDirectory() as output_dir:
            _write_single_shard(
                output_dir,
                10,
                [
                    _row(10, 0, 1, [1.0, 2.0]),
                    _row(10, 0, 2, [3.0, 4.0]),
                    _row(10, 0, 3, [0.0, 0.0]),
                ],
            )
            view = _FakeView()
            factory = _FakeClientFactory(view)
            uploader = FeatureStoreDeltaUploader(
                _feature_store_config(),
                output_dir=output_dir,
                file_prefix="delta",
                world_size=1,
                embedding_dimensions={"user_emb": 2},
                client_factory=factory,
                clock_ms=lambda: 123456,
            )
            uploader.start()
            uploader.submit(10)
            uploader.close()

            self.assertEqual(len(view.calls), 2)
            self.assertEqual([len(call["data"]) for call in view.calls], [2, 1])
            self.assertEqual(view.flush_calls, [[2, 1]])
            self.assertEqual(
                {call["version"] for call in view.calls}, {"model_a@export_1"}
            )
            self.assertEqual({call["write_mode"] for call in view.calls}, {"MERGE"})
            self.assertEqual([call["ts"] for call in view.calls], [123456, 123457])
            self.assertEqual(view.calls[1]["data"][0]["embedding"].tolist(), [0.0, 0.0])
            self.assertEqual(view.closed, [True])

    def test_upload_uses_bounded_sdk_worker_windows(self):
        with tempfile.TemporaryDirectory() as output_dir:
            _write_single_shard(
                output_dir,
                10,
                [_row(10, 0, key, [1.0, 2.0]) for key in range(1, 6)],
            )
            view = _FakeView(max_workers=2)
            uploader = FeatureStoreDeltaUploader(
                _feature_store_config(upload_batch_size=1),
                output_dir,
                "delta",
                1,
                {"user_emb": 2},
                client_factory=_FakeClientFactory(view),
                clock_ms=lambda: 100,
            )

            uploader.start()
            uploader.submit(10)
            uploader.close()

            self.assertEqual(
                [call["ts"] for call in view.calls], [100, 101, 102, 103, 104]
            )
            self.assertEqual(view.flush_calls, [[1, 1], [1, 1], [1]])

    def test_first_positive_dump_step_is_not_filtered(self):
        with tempfile.TemporaryDirectory() as output_dir:
            _write_single_shard(output_dir, 1, [_row(1, 0, 1, [1.0, 2.0])])
            view = _FakeView()
            uploader = FeatureStoreDeltaUploader(
                _feature_store_config(),
                output_dir,
                "delta",
                1,
                {"user_emb": 2},
                client_factory=_FakeClientFactory(view),
                clock_ms=lambda: 100,
            )

            uploader.start()
            uploader.submit(1)
            uploader.close()

            self.assertEqual(len(view.calls), 1)
            self.assertEqual(view.calls[0]["ts"], 100)
            self.assertEqual(view.calls[0]["version"], "model_a@export_1")

    def test_submit_rejects_step_zero(self):
        with tempfile.TemporaryDirectory() as output_dir:
            uploader = FeatureStoreDeltaUploader(
                _feature_store_config(),
                output_dir,
                "delta",
                1,
                {"user_emb": 2},
                client_factory=_FakeClientFactory(_FakeView()),
            )

            uploader.start()
            try:
                with self.assertRaisesRegex(ValueError, "global_step must be > 0"):
                    uploader.submit(0)
            finally:
                uploader.close()

    def test_target_scoped_prefix_prevents_cross_version_parquet_replay(self):
        with tempfile.TemporaryDirectory() as output_dir:
            version_a = _feature_store_config(version="model_a@run_1")
            version_b = _feature_store_config(version="model_a@run_2")
            prefix_a = feature_store_delta_file_prefix(version_a, "delta")
            prefix_b = feature_store_delta_file_prefix(version_b, "delta")
            self.assertNotEqual(prefix_a, prefix_b)
            self.assertEqual(
                prefix_a, feature_store_delta_file_prefix(version_a, "delta")
            )

            _write_single_shard(
                output_dir,
                10,
                [_row(10, 0, 1, [1.0, 2.0])],
                file_prefix=prefix_a,
            )
            view = _FakeView()
            factory = _FakeClientFactory(view)
            uploader = FeatureStoreDeltaUploader(
                version_b,
                output_dir,
                "delta",
                1,
                {"user_emb": 2},
                client_factory=factory,
            )
            self.assertEqual(uploader._file_prefix, prefix_b)

            uploader.start()
            uploader.close()

            self.assertEqual(len(factory.calls), 1)
            self.assertEqual(view.calls, [])

    def test_flush_failure_raises_error(self):
        failed_summary = {
            "total_batches": 2,
            "failed_batches": 1,
            "total_records": 3,
            "success_records": 2,
            "failed_records": 1,
            "errors": ["failed future"],
        }
        with tempfile.TemporaryDirectory() as output_dir:
            _write_single_shard(
                output_dir,
                10,
                [
                    _row(10, 0, 1, [1.0, 2.0]),
                    _row(10, 0, 2, [3.0, 4.0]),
                    _row(10, 0, 3, [5.0, 6.0]),
                ],
            )
            view = _FakeView([failed_summary])
            uploader = FeatureStoreDeltaUploader(
                _feature_store_config(max_retries=1),
                output_dir,
                "delta",
                1,
                {"user_emb": 2},
                client_factory=_FakeClientFactory(view),
            )
            uploader.start()
            uploader.submit(10)
            with self.assertRaises(FeatureStoreUploadError):
                uploader.close()
            self.assertEqual(view.flush_calls, [[2, 1], []])

    def test_retry_uses_fresh_view_and_newer_timestamp_range(self):
        failed_summary = {
            "total_batches": 1,
            "failed_batches": 1,
            "total_records": 1,
            "success_records": 0,
            "failed_records": 1,
            "errors": ["failed future"],
        }
        with tempfile.TemporaryDirectory() as output_dir:
            _write_single_shard(output_dir, 10, [_row(10, 0, 1, [1.0, 2.0])])
            first_view = _FakeView([failed_summary])
            second_view = _FakeView()
            factory = _SequencedClientFactory([first_view, second_view])
            uploader = FeatureStoreDeltaUploader(
                _feature_store_config(max_retries=2),
                output_dir,
                "delta",
                1,
                {"user_emb": 2},
                client_factory=factory,
                clock_ms=lambda: 777,
            )
            uploader.start()
            uploader.submit(10)
            uploader.close()

            self.assertEqual(len(factory.calls), 2)
            self.assertEqual(first_view.closed, [True])
            self.assertEqual(second_view.closed, [True])
            all_calls = first_view.calls + second_view.calls
            self.assertEqual(
                {call["version"] for call in all_calls}, {"model_a@export_1"}
            )
            self.assertEqual([call["ts"] for call in all_calls], [777, 778])

    def test_merge_does_not_require_preprovisioned_version(self):
        with tempfile.TemporaryDirectory() as output_dir:
            _write_single_shard(output_dir, 10, [_row(10, 0, 1, [1.0, 2.0])])
            view = _FakeView()
            uploader = FeatureStoreDeltaUploader(
                _feature_store_config(),
                output_dir,
                "delta",
                1,
                {"user_emb": 2},
                client_factory=_FakeClientFactory(view),
            )
            uploader.start()
            uploader.submit(10)
            uploader.close()

            self.assertEqual(len(view.calls), 1)
            self.assertEqual(view.calls[0]["write_mode"], "MERGE")

    def test_non_draining_close_stops_without_commit(self):
        with tempfile.TemporaryDirectory() as output_dir:
            _write_single_shard(
                output_dir,
                10,
                [_row(10, 0, key, [1.0, 2.0]) for key in range(1, 10)],
            )
            view = _BlockingView()
            uploader = FeatureStoreDeltaUploader(
                _feature_store_config(upload_batch_size=1),
                output_dir,
                "delta",
                1,
                {"user_emb": 2},
                client_factory=_FakeClientFactory(view),
            )
            uploader.start()
            uploader.submit(10)
            self.assertTrue(view.flush_started.wait(timeout=5))
            uploader.close(raise_on_error=False, drain=False)
            view.release_flush.set()
            self.assertTrue(view.close_finished.wait(timeout=5))
            self.assertTrue(len(view.calls) < 9)

    def test_signed_int64_key_is_preserved(self):
        with tempfile.TemporaryDirectory() as output_dir:
            large_key = (1 << 63) - 1
            _write_single_shard(output_dir, 10, [_row(10, 0, large_key, [1.0, 2.0])])
            view = _FakeView()
            uploader = FeatureStoreDeltaUploader(
                _feature_store_config(),
                output_dir,
                "delta",
                1,
                {"user_emb": 2},
                client_factory=_FakeClientFactory(view),
            )
            uploader.start()
            uploader.submit(10)
            uploader.close()

            self.assertEqual(view.calls[0]["data"][0]["key_id"], large_key)

    def test_reserved_invalid_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as output_dir:
            _write_single_shard(output_dir, 10, [_row(10, 0, -1, [1.0, 2.0])])
            view = _FakeView()
            uploader = FeatureStoreDeltaUploader(
                _feature_store_config(max_retries=1),
                output_dir,
                "delta",
                1,
                {"user_emb": 2},
                client_factory=_FakeClientFactory(view),
            )
            uploader.start()
            uploader.submit(10)
            with self.assertRaises(FeatureStoreUploadError):
                uploader.close()
            self.assertEqual(view.calls, [])

    def test_empty_step_validates_remote_contract(self):
        with tempfile.TemporaryDirectory() as output_dir:
            _write_single_shard(output_dir, 10, [])
            view = _FakeView()
            uploader = FeatureStoreDeltaUploader(
                _feature_store_config(),
                output_dir,
                "delta",
                1,
                {"user_emb": 2},
                client_factory=_FakeClientFactory(view),
            )
            uploader.start()
            uploader.submit(10)
            uploader.close()

            self.assertEqual(view.calls, [])

    def test_multi_rank_waits_for_all_shards(self):
        with tempfile.TemporaryDirectory() as output_dir:
            _write_rank_shard(
                output_dir, 10, 0, 2, [_row(10, 0, 1, [1.0, 2.0], world_size=2)]
            )
            _write_rank_shard(
                output_dir, 10, 1, 2, [_row(10, 1, 2, [3.0, 4.0], world_size=2)]
            )
            view = _FakeView()
            uploader = FeatureStoreDeltaUploader(
                _feature_store_config(),
                output_dir,
                "delta",
                2,
                {"user_emb": 2},
                client_factory=_FakeClientFactory(view),
                clock_ms=lambda: 100,
            )
            uploader.start()
            uploader.submit(10)
            uploader.close()

            self.assertEqual(len(view.calls), 2)
            self.assertEqual(view.calls[0]["data"][0]["key_id"], 1)
            self.assertEqual(view.calls[1]["data"][0]["key_id"], 2)

    def test_dimension_and_finite_value_validation(self):
        with tempfile.TemporaryDirectory() as output_dir:
            _write_single_shard(output_dir, 10, [_row(10, 0, 1, [1.0, 2.0, 3.0])])
            view = _FakeView()
            uploader = FeatureStoreDeltaUploader(
                _feature_store_config(max_retries=1),
                output_dir,
                "delta",
                1,
                {"user_emb": 2},
                client_factory=_FakeClientFactory(view),
            )
            uploader.start()
            uploader.submit(10)
            with self.assertRaises(FeatureStoreUploadError):
                uploader.close()
            self.assertEqual(view.calls, [])

        with tempfile.TemporaryDirectory() as output_dir:
            _write_single_shard(output_dir, 10, [_row(10, 0, 1, [float("nan"), 2.0])])
            view = _FakeView()
            uploader = FeatureStoreDeltaUploader(
                _feature_store_config(max_retries=1),
                output_dir,
                "delta",
                1,
                {"user_emb": 2},
                client_factory=_FakeClientFactory(view),
            )
            uploader.start()
            uploader.submit(10)
            with self.assertRaises(FeatureStoreUploadError):
                uploader.close()
            self.assertEqual(view.calls, [])

    def test_successful_upload_deletes_shard_files(self):
        with tempfile.TemporaryDirectory() as output_dir:
            shard_path = _write_single_shard(
                output_dir, 10, [_row(10, 0, 1, [1.0, 2.0])]
            )
            self.assertTrue(os.path.isfile(shard_path))
            view = _FakeView()
            uploader = FeatureStoreDeltaUploader(
                _feature_store_config(),
                output_dir,
                "delta",
                1,
                {"user_emb": 2},
                client_factory=_FakeClientFactory(view),
            )
            uploader.start()
            uploader.submit(10)
            uploader.close()

            self.assertFalse(os.path.isfile(shard_path))

    def test_in_memory_timestamp_monotonicity_across_steps(self):
        with tempfile.TemporaryDirectory() as output_dir:
            _write_single_shard(output_dir, 10, [_row(10, 0, 1, [1.0, 2.0])])
            _write_single_shard(output_dir, 20, [_row(20, 0, 2, [3.0, 4.0])])
            view = _FakeView()
            uploader = FeatureStoreDeltaUploader(
                _feature_store_config(),
                output_dir,
                "delta",
                1,
                {"user_emb": 2},
                client_factory=_FakeClientFactory(view),
                clock_ms=lambda: 100,
            )
            uploader.start()
            uploader.submit(10)
            uploader.submit(20)
            uploader.close()

            ts_values = [call["ts"] for call in view.calls]
            self.assertEqual(ts_values, [100, 101])

    def test_in_memory_timestamp_monotonicity_across_retries(self):
        failed_summary = {
            "total_batches": 1,
            "failed_batches": 1,
            "total_records": 1,
            "success_records": 0,
            "failed_records": 1,
        }
        with tempfile.TemporaryDirectory() as output_dir:
            _write_single_shard(output_dir, 10, [_row(10, 0, 1, [1.0, 2.0])])
            first_view = _FakeView([failed_summary])
            second_view = _FakeView()
            factory = _SequencedClientFactory([first_view, second_view])
            uploader = FeatureStoreDeltaUploader(
                _feature_store_config(max_retries=2),
                output_dir,
                "delta",
                1,
                {"user_emb": 2},
                client_factory=factory,
                clock_ms=lambda: 500,
            )
            uploader.start()
            uploader.submit(10)
            uploader.close()

            self.assertEqual([call["ts"] for call in first_view.calls], [500])
            self.assertEqual([call["ts"] for call in second_view.calls], [501])

    def test_shard_wait_timeout_raises(self):
        with tempfile.TemporaryDirectory() as output_dir:
            view = _FakeView()
            uploader = FeatureStoreDeltaUploader(
                _feature_store_config(shard_wait_timeout_secs=1, poll_interval_secs=1),
                output_dir,
                "delta",
                1,
                {"user_emb": 2},
                client_factory=_FakeClientFactory(view),
            )
            uploader.start()
            uploader.submit(10)
            with self.assertRaises(FeatureStoreUploadError):
                uploader.close()
            self.assertEqual(view.calls, [])

    def test_data_parallel_duplicate_keys_uploaded_without_dedup(self):
        with tempfile.TemporaryDirectory() as output_dir:
            _write_rank_shard(
                output_dir,
                10,
                0,
                2,
                [_row(10, 0, 1, [1.0, 2.0], world_size=2)],
            )
            _write_rank_shard(
                output_dir,
                10,
                1,
                2,
                [_row(10, 1, 1, [1.0, 2.0], world_size=2)],
            )
            view = _FakeView()
            uploader = FeatureStoreDeltaUploader(
                _feature_store_config(),
                output_dir,
                "delta",
                2,
                {"user_emb": 2},
                client_factory=_FakeClientFactory(view),
                clock_ms=lambda: 100,
            )
            uploader.start()
            uploader.submit(10)
            uploader.close()

            all_keys = [item["key_id"] for call in view.calls for item in call["data"]]
            self.assertEqual(all_keys, [1, 1])

    def test_close_error_surfaces_via_check_error(self):
        with tempfile.TemporaryDirectory() as output_dir:
            _write_single_shard(output_dir, 10, [_row(10, 0, -1, [1.0, 2.0])])
            view = _FakeView()
            uploader = FeatureStoreDeltaUploader(
                _feature_store_config(max_retries=1),
                output_dir,
                "delta",
                1,
                {"user_emb": 2},
                client_factory=_FakeClientFactory(view),
            )
            uploader.start()
            uploader.submit(10)
            with self.assertRaises(FeatureStoreUploadError):
                uploader.close()
            with self.assertRaises(FeatureStoreUploadError):
                uploader.check_error()


if __name__ == "__main__":
    unittest.main()
