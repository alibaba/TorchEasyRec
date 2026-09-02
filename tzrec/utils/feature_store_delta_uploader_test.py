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

import os
import sys
import threading
import types
import unittest
from unittest import mock

import numpy as np
import pyarrow as pa
from google.protobuf.descriptor import FieldDescriptor

from tzrec.protos.train_pb2 import FeatureStoreConfig
from tzrec.utils.delta_embedding_dump import (
    _DELTA_DUMP_QUANT_SCHEMA,
    _DELTA_DUMP_SCHEMA,
)
from tzrec.utils.feature_store_delta_uploader import (
    FEATURE_STORE_DEFAULT_ENTITY_JOIN_ID,
    FEATURE_STORE_DEFAULT_ENTITY_NAME,
    FEATURE_STORE_EMBEDDING_TYPE_FLOAT,
    FEATURE_STORE_EMBEDDING_TYPE_UINT8,
    FeatureStoreDeltaUploader,
    FeatureStoreUploadError,
    FeatureStoreUploadSettings,
)


def _feature_store_config(**overrides) -> FeatureStoreConfig:
    config = FeatureStoreConfig(
        region="cn-test",
        project_name="project_a",
        feature_view_name="shared_embeddings",
        version="model_a@export_1",
        upload_batch_size=2,
        max_retries=1,
        retry_backoff_secs=0,
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
        "feature_name": "user_id",
        "table_fqn": f"model.ebc.embedding_bags.{name}",
        "key_id": key_id,
        "embedding": values,
        "source": "model_delta_tracker",
    }


def _delta_table(rows) -> pa.Table:
    if rows:
        return pa.Table.from_pylist(rows, schema=_DELTA_DUMP_SCHEMA)
    return _DELTA_DUMP_SCHEMA.empty_table()


class _FakeView:
    pk_field = "embedding_name"
    sk_field = "key_id"
    embedding_field = "embedding"

    def __init__(
        self,
        summaries=None,
        close_error=None,
        max_workers=4,
        embedding_field_type=FEATURE_STORE_EMBEDDING_TYPE_FLOAT,
    ):
        self.calls = []
        self.arrow_calls = []
        self.closed = []
        self.flush_calls = []
        self.embedding_field_type = embedding_field_type
        self._summaries = list(summaries or [])
        self._close_error = close_error
        self._batch_size = 1000
        self._max_workers = max_workers
        self._pending_sizes = []

    def write_features(self, **kwargs):
        self.calls.append(kwargs)
        self._pending_sizes.append(len(kwargs["data"]))

    def write_features_arrow(self, *, batch, version, write_mode, ts):
        # Decode the Arrow wire batch into the same {data, version, write_mode,
        # ts} call shape as write_features, so the existing JSON-path assertions
        # also exercise the default Arrow path unchanged. The raw batch is kept
        # on arrow_calls for column-type / column-name assertions.
        self.arrow_calls.append(
            {"batch": batch, "version": version, "write_mode": write_mode, "ts": ts}
        )
        data = [
            {
                self.pk_field: pk,
                self.sk_field: int(sk),
                self.embedding_field: np.asarray(emb, dtype=np.float32),
            }
            for pk, sk, emb in zip(
                batch.column(self.pk_field).to_pylist(),
                batch.column(self.sk_field).to_pylist(),
                batch.column(self.embedding_field).to_pylist(),
            )
        ]
        self.calls.append(
            {"data": data, "version": version, "write_mode": write_mode, "ts": ts}
        )
        self._pending_sizes.append(len(data))

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


class _FakeEntity:
    def __init__(self, name):
        self.feature_entity_name = name


class _FakeProject:
    def __init__(
        self,
        view,
        *,
        created_view=None,
        create_error=None,
        view_after_create_error=None,
        entity="existing_entity",
        entity_create_error=None,
    ):
        self._view = view
        self._created_view = created_view
        self._create_error = create_error
        self._view_after_create_error = view_after_create_error
        self._entity = entity
        self._entity_create_error = entity_create_error
        self.dynamic_get_calls = []
        self.create_calls = []
        self.entity_get_calls = []
        self.entity_create_calls = []

    def get_dynamic_embedding_feature_view(self, name):
        self.dynamic_get_calls.append(name)
        return self._view

    def create_dynamic_embedding_feature_view(self, **kwargs):
        self.create_calls.append(kwargs)
        if self._create_error is not None:
            self._view = self._view_after_create_error
            raise self._create_error
        self._view = self._created_view or _FakeView()
        # The created handle carries the created embedding_field_type.
        self._view.embedding_field_type = kwargs.get(
            "embedding_field_type", FEATURE_STORE_EMBEDDING_TYPE_FLOAT
        )
        return self._view

    def get_entity(self, name):
        self.entity_get_calls.append(name)
        return self._entity

    def create_entity(self, name, join_id, parent_feature_entity_name=None):
        self.entity_create_calls.append((name, join_id))
        if self._entity_create_error is not None:
            raise self._entity_create_error
        self._entity = _FakeEntity(name)
        return self._entity


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

    def _uploader(self, config=None, **kwargs):
        client_factory = kwargs.pop("client_factory", None)
        kwargs.setdefault(
            "embedding_dimensions", {"model.ebc.embedding_bags.user_emb": 2}
        )
        uploader = FeatureStoreDeltaUploader(
            config or _feature_store_config(), **kwargs
        )
        if client_factory is not None:
            # The production ctor no longer accepts a client factory; install the
            # test's fake on the _create_client instance seam so each uploader
            # keeps its own fake project/view wiring and construction counts.
            uploader._create_client = lambda *a, **k: client_factory()
        return uploader

    def test_proto_groups_required_fields_before_optional_fields(self):
        required_fields = [
            "region",
            "project_name",
            "feature_view_name",
            "version",
        ]
        optional_fields = [
            "endpoint",
            "upload_batch_size",
            "max_retries",
            "retry_backoff_secs",
            "shutdown_timeout_secs",
            "max_pending_steps",
            "poll_interval_secs",
            "feature_view_ttl_secs",
            "feature_view_shard_count",
            "feature_view_replication_count",
            "retain_local_dump",
            "upload_format",
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
            [1, 2, 3] + list(range(5, 16)) + [17, 18],
        )
        for field_name in required_fields:
            with self.subTest(field_name=field_name):
                config = _feature_store_config()
                config.ClearField(field_name)
                self.assertFalse(config.IsInitialized())
                self.assertIn(field_name, config.FindInitializationErrors())
                with self.assertRaisesRegex(ValueError, field_name):
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

    def test_start_reuses_existing_dynamic_embedding_feature_view(self):
        view = _FakeView()
        factory = _FakeClientFactory(view)
        uploader = self._uploader(client_factory=factory)

        uploader.start()
        uploader.close()

        self.assertEqual(factory.project.dynamic_get_calls, ["shared_embeddings"])
        self.assertEqual(factory.project.create_calls, [])
        self.assertEqual(view.closed, [True])

    def test_create_client_forwards_only_credential_kwargs(self):
        """_create_client forwards the fixed credential allowlist, no extras."""
        recorded = {}

        class _RecordingClient:
            def __init__(self, **kwargs):
                recorded.update(kwargs)

        fake_module = types.ModuleType("feature_store_py")
        fake_module.FeatureStoreClient = _RecordingClient
        with mock.patch.dict(sys.modules, {"feature_store_py": fake_module}):
            uploader = self._uploader()
            client = uploader._create_client()
        self.assertIsInstance(client, _RecordingClient)
        self.assertEqual(
            set(recorded),
            {
                "access_key_id",
                "access_key_secret",
                "region",
                "endpoint",
                "security_token",
                "featuredb_username",
                "featuredb_password",
            },
        )
        self.assertNotIn("test_mode", recorded)

    def test_start_creates_missing_dynamic_embedding_feature_view(self):
        created_view = _FakeView()
        factory = _FakeClientFactory(None, created_view=created_view)
        uploader = self._uploader(client_factory=factory)

        uploader.start()
        uploader.close()

        self.assertEqual(factory.project.dynamic_get_calls, ["shared_embeddings"])
        self.assertEqual(
            factory.project.create_calls,
            [
                {
                    "name": "shared_embeddings",
                    "entity": FEATURE_STORE_DEFAULT_ENTITY_NAME,
                    "pk_field_name": "embedding_name",
                    "sk_field_name": "key_id",
                    "embedding_field_name": "embedding",
                    "pk_field_type": "STRING",
                    "sk_field_type": "INT64",
                    "ttl": 1296000,
                    "shard_count": 20,
                    "replication_count": 1,
                    "embedding_field_type": FEATURE_STORE_EMBEDDING_TYPE_FLOAT,
                }
            ],
        )
        self.assertEqual(factory.project.entity_create_calls, [])
        self.assertEqual(created_view.closed, [True])

    def test_start_creates_quantized_view_with_uint8_embedding_type(self):
        created_view = _FakeView()
        factory = _FakeClientFactory(None, created_view=created_view)
        uploader = self._uploader(
            client_factory=factory,
            embedding_field_type=FEATURE_STORE_EMBEDDING_TYPE_UINT8,
        )

        uploader.start()
        uploader.close()

        self.assertEqual(
            factory.project.create_calls[0]["embedding_field_type"],
            FEATURE_STORE_EMBEDDING_TYPE_UINT8,
        )
        self.assertEqual(
            created_view.embedding_field_type, FEATURE_STORE_EMBEDDING_TYPE_UINT8
        )
        self.assertEqual(created_view.closed, [True])

    def test_quantized_uint8_batch_keeps_wire_type_on_arrow_path(self):
        view = _FakeView(embedding_field_type=FEATURE_STORE_EMBEDDING_TYPE_UINT8)
        factory = _FakeClientFactory(view)
        uploader = self._uploader(
            client_factory=factory,
            embedding_field_type=FEATURE_STORE_EMBEDDING_TYPE_UINT8,
            embedding_dimensions={"model.ebc.embedding_bags.user_emb": 6},
        )
        table = pa.Table.from_pylist(
            [_row(10, 0, 7, [1, 2, 3, 250, 251, 252])],
            schema=_DELTA_DUMP_QUANT_SCHEMA,
        )

        uploader.start()
        uploader.submit(10, table)
        uploader.close()

        wire_batch = view.arrow_calls[0]["batch"]
        self.assertEqual(
            wire_batch.schema.field("embedding").type, pa.list_(pa.uint8())
        )
        self.assertEqual(
            view.calls[0]["data"][0]["embedding"].tolist(),
            [1, 2, 3, 250, 251, 252],
        )

    def test_start_fails_when_existing_view_embedding_type_mismatches(self):
        view = _FakeView(embedding_field_type=FEATURE_STORE_EMBEDDING_TYPE_FLOAT)
        factory = _FakeClientFactory(view)
        uploader = self._uploader(
            client_factory=factory,
            embedding_field_type=FEATURE_STORE_EMBEDDING_TYPE_UINT8,
        )

        with self.assertRaisesRegex(RuntimeError, "embedding type mismatch"):
            uploader.start()

        self.assertEqual(factory.project.create_calls, [])
        self.assertEqual(view.closed, [True])

    def test_legacy_untyped_view_accepts_float_dump(self):
        # A view handle that reports no embedding_field_type (legacy view or old
        # SDK) is float by definition, so the default FLOAT dump must proceed.
        for untyped in (None, ""):
            with self.subTest(embedding_field_type=untyped):
                view = _FakeView(embedding_field_type=untyped)
                factory = _FakeClientFactory(view)
                uploader = self._uploader(client_factory=factory)

                uploader.start()
                uploader.submit(10, _delta_table([_row(10, 0, 1, [1.0, 2.0])]))
                uploader.close()

                self.assertEqual(factory.project.create_calls, [])
                self.assertEqual(len(view.calls), 1)
                self.assertEqual(view.closed, [True])

    def test_legacy_untyped_view_rejects_uint8_dump(self):
        # A UINT8 dump must not write into an untyped legacy (float) view.
        for untyped in (None, ""):
            with self.subTest(embedding_field_type=untyped):
                view = _FakeView(embedding_field_type=untyped)
                factory = _FakeClientFactory(view)
                uploader = self._uploader(
                    client_factory=factory,
                    embedding_field_type=FEATURE_STORE_EMBEDDING_TYPE_UINT8,
                )

                with self.assertRaisesRegex(RuntimeError, "embedding type mismatch"):
                    uploader.start()

                self.assertEqual(factory.project.create_calls, [])
                self.assertEqual(view.closed, [True])

    def test_non_primary_start_fails_on_existing_view_embedding_type_mismatch(self):
        view = _FakeView(embedding_field_type=FEATURE_STORE_EMBEDDING_TYPE_UINT8)
        factory = _FakeClientFactory(view)
        uploader = self._uploader(
            rank=1,
            manage_remote_view=False,
            client_factory=factory,
        )

        with self.assertRaisesRegex(RuntimeError, "embedding type mismatch"):
            uploader.start()

        self.assertEqual(view.closed, [True])

    def test_rejects_unknown_embedding_field_type(self):
        with self.assertRaisesRegex(ValueError, "embedding_field_type"):
            self._uploader(embedding_field_type="ARRAY<FP16>")

    def test_start_recovers_from_concurrent_feature_view_creation(self):
        concurrent_view = _FakeView()
        factory = _FakeClientFactory(
            None,
            create_error=RuntimeError("already exists"),
            view_after_create_error=concurrent_view,
        )
        uploader = self._uploader(client_factory=factory)

        uploader.start()
        uploader.close()

        self.assertEqual(len(factory.project.create_calls), 1)
        self.assertEqual(
            factory.project.dynamic_get_calls,
            ["shared_embeddings", "shared_embeddings"],
        )
        self.assertEqual(concurrent_view.closed, [True])

    def test_start_closes_new_feature_view_with_incompatible_schema(self):
        created_view = _FakeView()
        created_view.pk_field = "wrong_pk"
        factory = _FakeClientFactory(None, created_view=created_view)
        uploader = self._uploader(client_factory=factory)

        with self.assertRaisesRegex(RuntimeError, "schema mismatch"):
            uploader.start()

        self.assertEqual(len(factory.project.create_calls), 1)
        self.assertEqual(created_view.closed, [True])

    def test_start_creates_missing_view_without_version_precheck(self):
        created_view = _FakeView()
        factory = _FakeClientFactory(None, created_view=created_view)
        uploader = self._uploader(client_factory=factory)

        uploader.start()
        uploader.close()

        self.assertEqual(len(factory.project.create_calls), 1)
        self.assertEqual(created_view.closed, [True])

    def test_start_raises_when_view_creation_fails_and_view_never_appears(self):
        factory = _FakeClientFactory(None, create_error=ValueError("boom"))
        uploader = self._uploader(client_factory=factory)

        with self.assertRaisesRegex(
            RuntimeError, "failed to create configured DynamicEmbedding FeatureView"
        ):
            uploader.start()

        self.assertEqual(len(factory.project.create_calls), 1)
        self.assertEqual(factory.project.entity_create_calls, [])

    def test_create_type_error_reports_feature_store_py_version(self):
        # An old feature_store_py rejects the embedding_field_type kwarg with a
        # TypeError; the uploader must surface the pinned SDK version instead of
        # burning the wait/retry loop and re-raising a generic failure.
        factory = _FakeClientFactory(
            None,
            create_error=TypeError(
                "create_dynamic_embedding_feature_view() got an unexpected "
                "keyword argument 'embedding_field_type'"
            ),
        )
        uploader = self._uploader(client_factory=factory)

        with self.assertRaisesRegex(RuntimeError, "feature_store_py.*2.2.10") as ctx:
            uploader.start()

        self.assertIsInstance(ctx.exception.__cause__, TypeError)
        self.assertEqual(factory.project.dynamic_get_calls, ["shared_embeddings"])
        self.assertEqual(len(factory.project.create_calls), 1)

    def test_start_creates_default_entity_when_it_does_not_exist(self):
        created_view = _FakeView()
        factory = _FakeClientFactory(None, created_view=created_view, entity=None)
        uploader = self._uploader(client_factory=factory)

        uploader.start()
        uploader.close()

        self.assertEqual(
            factory.project.entity_get_calls, [FEATURE_STORE_DEFAULT_ENTITY_NAME]
        )
        self.assertEqual(
            factory.project.entity_create_calls,
            [
                (
                    FEATURE_STORE_DEFAULT_ENTITY_NAME,
                    FEATURE_STORE_DEFAULT_ENTITY_JOIN_ID,
                )
            ],
        )
        self.assertEqual(
            factory.project.create_calls[0]["entity"],
            FEATURE_STORE_DEFAULT_ENTITY_NAME,
        )
        self.assertEqual(created_view.closed, [True])

    def test_start_recovers_from_concurrent_default_entity_creation(self):
        created_view = _FakeView()
        factory = _FakeClientFactory(
            None,
            created_view=created_view,
            entity=None,
            entity_create_error=RuntimeError("entity already exists"),
        )
        # A concurrent creator wins the race: the first get_entity sees no entity,
        # create_entity then fails, and the entity is visible on the retry.
        entity_results = [None, _FakeEntity(FEATURE_STORE_DEFAULT_ENTITY_NAME)]

        def get_entity(name):
            factory.project.entity_get_calls.append(name)
            return entity_results.pop(0)

        factory.project.get_entity = get_entity
        uploader = self._uploader(client_factory=factory)

        uploader.start()
        uploader.close()

        self.assertEqual(
            factory.project.entity_create_calls,
            [
                (
                    FEATURE_STORE_DEFAULT_ENTITY_NAME,
                    FEATURE_STORE_DEFAULT_ENTITY_JOIN_ID,
                )
            ],
        )
        self.assertEqual(len(factory.project.create_calls), 1)
        self.assertEqual(created_view.closed, [True])

    def test_non_primary_uploader_opens_view_without_create_or_metadata_checks(self):
        view = _FakeView()
        factory = _FakeClientFactory(view)
        uploader = self._uploader(
            rank=1,
            manage_remote_view=False,
            client_factory=factory,
            clock_ms=lambda: 100,
        )

        uploader.start()
        uploader.submit(10, _delta_table([_row(10, 1, 7, [1.0, 2.0])]))
        uploader.close()

        self.assertEqual(factory.project.dynamic_get_calls, ["shared_embeddings"])
        self.assertEqual(factory.project.create_calls, [])
        self.assertEqual(len(view.calls), 1)
        self.assertEqual(view.calls[0]["data"][0]["key_id"], 7)

    def test_non_primary_uploader_fails_when_view_is_missing(self):
        factory = _FakeClientFactory(None)
        uploader = self._uploader(
            rank=1,
            manage_remote_view=False,
            client_factory=factory,
        )

        with self.assertRaisesRegex(RuntimeError, "rank-zero uploader must create"):
            uploader.start()

        self.assertEqual(factory.project.create_calls, [])

    def test_submit_requires_started_uploader(self):
        uploader = self._uploader(client_factory=_FakeClientFactory(_FakeView()))
        with self.assertRaisesRegex(RuntimeError, "start.*before submit"):
            uploader.submit(10, _delta_table([_row(10, 0, 1, [1.0, 2.0])]))
        uploader.close()

    def test_complete_step_uploads_merge_with_stable_version_and_ts(self):
        view = _FakeView()
        factory = _FakeClientFactory(view)
        uploader = self._uploader(
            client_factory=factory,
            clock_ms=lambda: 123456,
        )
        uploader.start()
        uploader.submit(
            10,
            _delta_table(
                [
                    _row(10, 0, 1, [1.0, 2.0]),
                    _row(10, 0, 2, [3.0, 4.0]),
                    _row(10, 0, 3, [0.0, 0.0]),
                ]
            ),
        )
        uploader.close()

        self.assertEqual(len(view.calls), 2)
        self.assertEqual([len(call["data"]) for call in view.calls], [2, 1])
        self.assertEqual(view.flush_calls, [[2, 1]])
        self.assertEqual({call["version"] for call in view.calls}, {"model_a@export_1"})
        self.assertEqual({call["write_mode"] for call in view.calls}, {"MERGE"})
        self.assertEqual([call["ts"] for call in view.calls], [123456, 123457])
        self.assertEqual(view.calls[1]["data"][0]["embedding"].tolist(), [0.0, 0.0])
        self.assertEqual(view.closed, [True])

    def test_upload_uses_bounded_sdk_worker_windows(self):
        view = _FakeView(max_workers=2)
        uploader = self._uploader(
            _feature_store_config(upload_batch_size=1),
            client_factory=_FakeClientFactory(view),
            clock_ms=lambda: 100,
        )

        uploader.start()
        uploader.submit(
            10, _delta_table([_row(10, 0, key, [1.0, 2.0]) for key in range(1, 6)])
        )
        uploader.close()

        self.assertEqual([call["ts"] for call in view.calls], [100, 101, 102, 103, 104])
        self.assertEqual(view.flush_calls, [[1, 1], [1, 1], [1]])

    def test_first_positive_dump_step_is_not_filtered(self):
        view = _FakeView()
        uploader = self._uploader(
            client_factory=_FakeClientFactory(view),
            clock_ms=lambda: 100,
        )

        uploader.start()
        uploader.submit(1, _delta_table([_row(1, 0, 1, [1.0, 2.0])]))
        uploader.close()

        self.assertEqual(len(view.calls), 1)
        self.assertEqual(view.calls[0]["ts"], 100)
        self.assertEqual(view.calls[0]["version"], "model_a@export_1")

    def test_submit_rejects_step_zero(self):
        uploader = self._uploader(client_factory=_FakeClientFactory(_FakeView()))

        uploader.start()
        try:
            with self.assertRaisesRegex(ValueError, "global_step must be > 0"):
                uploader.submit(0, _delta_table([]))
        finally:
            uploader.close()

    def test_flush_failure_raises_error(self):
        failed_summary = {
            "total_batches": 2,
            "failed_batches": 1,
            "total_records": 3,
            "success_records": 2,
            "failed_records": 1,
            "errors": ["failed future"],
        }
        view = _FakeView([failed_summary])
        uploader = self._uploader(
            _feature_store_config(max_retries=1),
            client_factory=_FakeClientFactory(view),
        )
        uploader.start()
        uploader.submit(
            10,
            _delta_table(
                [
                    _row(10, 0, 1, [1.0, 2.0]),
                    _row(10, 0, 2, [3.0, 4.0]),
                    _row(10, 0, 3, [5.0, 6.0]),
                ]
            ),
        )
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
        first_view = _FakeView([failed_summary])
        second_view = _FakeView()
        factory = _SequencedClientFactory([first_view, second_view])
        uploader = self._uploader(
            _feature_store_config(max_retries=2),
            client_factory=factory,
            clock_ms=lambda: 777,
        )
        uploader.start()
        uploader.submit(10, _delta_table([_row(10, 0, 1, [1.0, 2.0])]))
        uploader.close()

        self.assertEqual(len(factory.calls), 2)
        self.assertEqual(first_view.closed, [True])
        self.assertEqual(second_view.closed, [True])
        all_calls = first_view.calls + second_view.calls
        self.assertEqual({call["version"] for call in all_calls}, {"model_a@export_1"})
        self.assertEqual([call["ts"] for call in all_calls], [777, 778])

    def test_merge_does_not_require_preprovisioned_version(self):
        view = _FakeView()
        uploader = self._uploader(client_factory=_FakeClientFactory(view))
        uploader.start()
        uploader.submit(10, _delta_table([_row(10, 0, 1, [1.0, 2.0])]))
        uploader.close()

        self.assertEqual(len(view.calls), 1)
        self.assertEqual(view.calls[0]["write_mode"], "MERGE")

    def test_non_draining_close_stops_without_commit(self):
        view = _BlockingView()
        uploader = self._uploader(
            _feature_store_config(upload_batch_size=1),
            client_factory=_FakeClientFactory(view),
        )
        uploader.start()
        uploader.submit(
            10, _delta_table([_row(10, 0, key, [1.0, 2.0]) for key in range(1, 10)])
        )
        self.assertTrue(view.flush_started.wait(timeout=5))
        uploader.close(raise_on_error=False, drain=False)
        view.release_flush.set()
        self.assertTrue(view.close_finished.wait(timeout=5))
        self.assertTrue(len(view.calls) < 9)

    def test_signed_int64_key_is_preserved(self):
        large_key = (1 << 63) - 1
        view = _FakeView()
        uploader = self._uploader(client_factory=_FakeClientFactory(view))
        uploader.start()
        uploader.submit(10, _delta_table([_row(10, 0, large_key, [1.0, 2.0])]))
        uploader.close()

        self.assertEqual(view.calls[0]["data"][0]["key_id"], large_key)

    def test_reserved_invalid_key_is_rejected(self):
        view = _FakeView()
        uploader = self._uploader(
            _feature_store_config(max_retries=1),
            client_factory=_FakeClientFactory(view),
        )
        uploader.start()
        uploader.submit(10, _delta_table([_row(10, 0, -1, [1.0, 2.0])]))
        with self.assertRaises(FeatureStoreUploadError):
            uploader.close()
        self.assertEqual(view.calls, [])

    def test_empty_table_upload_writes_nothing(self):
        view = _FakeView()
        uploader = self._uploader(client_factory=_FakeClientFactory(view))
        uploader.start()
        uploader.submit(10, _delta_table([]))
        uploader.close()

        self.assertEqual(view.calls, [])

    def test_dimension_and_finite_value_validation(self):
        view = _FakeView()
        uploader = self._uploader(
            _feature_store_config(max_retries=1),
            client_factory=_FakeClientFactory(view),
        )
        uploader.start()
        uploader.submit(10, _delta_table([_row(10, 0, 1, [1.0, 2.0, 3.0])]))
        with self.assertRaises(FeatureStoreUploadError):
            uploader.close()
        self.assertEqual(view.calls, [])

        view = _FakeView()
        uploader = self._uploader(
            _feature_store_config(max_retries=1),
            client_factory=_FakeClientFactory(view),
        )
        uploader.start()
        uploader.submit(10, _delta_table([_row(10, 0, 1, [float("nan"), 2.0])]))
        with self.assertRaises(FeatureStoreUploadError):
            uploader.close()
        self.assertEqual(view.calls, [])

        # Inf is also rejected (np.isfinite covers both; a regression to
        # np.isnan would let Inf embeddings through undetected).
        view = _FakeView()
        uploader = self._uploader(
            _feature_store_config(max_retries=1),
            client_factory=_FakeClientFactory(view),
        )
        uploader.start()
        uploader.submit(10, _delta_table([_row(10, 0, 1, [float("inf"), 2.0])]))
        with self.assertRaises(FeatureStoreUploadError):
            uploader.close()
        self.assertEqual(view.calls, [])

    def test_in_memory_timestamp_monotonicity_across_steps(self):
        view = _FakeView()
        uploader = self._uploader(
            client_factory=_FakeClientFactory(view),
            clock_ms=lambda: 100,
        )
        uploader.start()
        uploader.submit(10, _delta_table([_row(10, 0, 1, [1.0, 2.0])]))
        uploader.submit(20, _delta_table([_row(20, 0, 2, [3.0, 4.0])]))
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
        first_view = _FakeView([failed_summary])
        second_view = _FakeView()
        factory = _SequencedClientFactory([first_view, second_view])
        uploader = self._uploader(
            _feature_store_config(max_retries=2),
            client_factory=factory,
            clock_ms=lambda: 500,
        )
        uploader.start()
        uploader.submit(10, _delta_table([_row(10, 0, 1, [1.0, 2.0])]))
        uploader.close()

        self.assertEqual([call["ts"] for call in first_view.calls], [500])
        self.assertEqual([call["ts"] for call in second_view.calls], [501])

    def test_data_parallel_ranks_upload_duplicate_keys_independently(self):
        rank0_view = _FakeView()
        rank1_view = _FakeView()
        rank0 = self._uploader(
            client_factory=_FakeClientFactory(rank0_view),
            clock_ms=lambda: 100,
        )
        rank1 = self._uploader(
            rank=1,
            manage_remote_view=False,
            client_factory=_FakeClientFactory(rank1_view),
            clock_ms=lambda: 100,
        )
        rank0.start()
        rank1.start()
        rank0.submit(10, _delta_table([_row(10, 0, 1, [1.0, 2.0], world_size=2)]))
        rank1.submit(10, _delta_table([_row(10, 1, 1, [1.0, 2.0], world_size=2)]))
        rank0.close()
        rank1.close()

        rank0_keys = [
            item["key_id"] for call in rank0_view.calls for item in call["data"]
        ]
        rank1_keys = [
            item["key_id"] for call in rank1_view.calls for item in call["data"]
        ]
        self.assertEqual(rank0_keys, [1])
        self.assertEqual(rank1_keys, [1])
        self.assertEqual({call["write_mode"] for call in rank0_view.calls}, {"MERGE"})
        self.assertEqual({call["write_mode"] for call in rank1_view.calls}, {"MERGE"})

    def test_close_error_surfaces_via_check_error(self):
        view = _FakeView()
        uploader = self._uploader(
            _feature_store_config(max_retries=1),
            client_factory=_FakeClientFactory(view),
        )
        uploader.start()
        uploader.submit(10, _delta_table([_row(10, 0, -1, [1.0, 2.0])]))
        with self.assertRaises(FeatureStoreUploadError):
            uploader.close()
        with self.assertRaises(FeatureStoreUploadError):
            uploader.check_error()

    def test_default_upload_format_is_arrow(self):
        # An unset upload_format inherits the proto default ARROW.
        settings = FeatureStoreUploadSettings.from_proto(_feature_store_config())
        self.assertEqual(settings.upload_format, "ARROW")

    def test_rejects_unknown_upload_format(self):
        config = _feature_store_config()
        config.upload_format = "protobuf"
        with self.assertRaisesRegex(ValueError, "upload_format must be one of"):
            FeatureStoreUploadSettings.from_proto(config)

    def test_json_path_routes_through_write_features(self):
        # Explicit JSON keeps the legacy per-row write_features payload path.
        view = _FakeView()
        factory = _FakeClientFactory(view)
        uploader = self._uploader(
            _feature_store_config(upload_format="JSON"),
            client_factory=factory,
            clock_ms=lambda: 123456,
        )
        uploader.start()
        uploader.submit(
            10,
            _delta_table(
                [
                    _row(10, 0, 1, [1.0, 2.0]),
                    _row(10, 0, 2, [3.0, 4.0]),
                    _row(10, 0, 3, [0.0, 0.0]),
                ]
            ),
        )
        uploader.close()

        self.assertEqual(view.arrow_calls, [])
        self.assertEqual(len(view.calls), 2)
        self.assertEqual([len(call["data"]) for call in view.calls], [2, 1])
        self.assertEqual(view.flush_calls, [[2, 1]])
        self.assertEqual({call["version"] for call in view.calls}, {"model_a@export_1"})
        self.assertEqual({call["write_mode"] for call in view.calls}, {"MERGE"})
        self.assertEqual([call["ts"] for call in view.calls], [123456, 123457])
        self.assertEqual(view.calls[0]["data"][0]["key_id"], 1)
        self.assertEqual(
            view.calls[0]["data"][0]["embedding_name"],
            "model.ebc.embedding_bags.user_emb",
        )
        self.assertTrue(
            np.array_equal(
                view.calls[0]["data"][0]["embedding"],
                np.array([1.0, 2.0], dtype=np.float32),
            )
        )
        # The sliced second batch exercises the offset-indexed embedding slice.
        self.assertTrue(
            np.array_equal(
                view.calls[1]["data"][0]["embedding"],
                np.array([0.0, 0.0], dtype=np.float32),
            )
        )
        self.assertEqual(view.closed, [True])

    def test_arrow_path_builds_wire_batch_columns(self):
        # Default ARROW path builds a wire RecordBatch whose configured field
        # names the SDK remaps to its pk/sk/embedding wire columns.
        view = _FakeView()
        factory = _FakeClientFactory(view)
        uploader = self._uploader(
            _feature_store_config(upload_batch_size=2),
            client_factory=factory,
            clock_ms=lambda: 100,
        )
        uploader.start()
        uploader.submit(
            10,
            _delta_table(
                [
                    _row(10, 0, 1, [1.0, 2.0]),
                    _row(10, 0, 2, [3.0, 4.0]),
                    _row(10, 0, 3, [5.0, 6.0]),
                ]
            ),
        )
        uploader.close()

        self.assertEqual(len(view.arrow_calls), 2)
        self.assertEqual([c["ts"] for c in view.arrow_calls], [100, 101])
        self.assertEqual({c["version"] for c in view.arrow_calls}, {"model_a@export_1"})
        self.assertEqual({c["write_mode"] for c in view.arrow_calls}, {"MERGE"})

        batch0 = view.arrow_calls[0]["batch"]
        self.assertEqual(batch0.schema.names, ["embedding_name", "key_id", "embedding"])
        self.assertEqual(batch0.num_rows, 2)
        # PK is the remapped table_fqn (string); SK stays int64 (the SDK casts to
        # the string wire type); embedding is list<float32> reused zero-copy.
        self.assertEqual(batch0.column("embedding_name").type, pa.string())
        self.assertEqual(batch0.column("key_id").type, pa.int64())
        self.assertEqual(batch0.column("embedding").type, pa.list_(pa.float32()))
        self.assertEqual(
            batch0.column("embedding_name").to_pylist(),
            ["model.ebc.embedding_bags.user_emb"] * 2,
        )
        self.assertEqual(batch0.column("key_id").to_pylist(), [1, 2])
        self.assertEqual(
            batch0.column("embedding").to_pylist(),
            [[1.0, 2.0], [3.0, 4.0]],
        )
        self.assertEqual(view.closed, [True])

    def test_multi_chunk_table_keeps_timestamps_monotonic_across_steps(self):
        # A multi-FQN delta table concatenates one chunk per FQN; to_batches()
        # splits each chunk independently, so the ts range must cover every
        # actual batch or a stuck clock reuses a prior step's timestamps and
        # Next-Ts incremental readers miss updates.
        view = _FakeView()
        factory = _FakeClientFactory(view)
        uploader = self._uploader(
            _feature_store_config(upload_batch_size=1000),
            client_factory=factory,
            clock_ms=lambda: 100,
            embedding_dimensions={
                "model.ebc.embedding_bags.user_emb": 2,
                "model.ebc.embedding_bags.item_emb": 2,
            },
        )
        uploader.start()
        rows_a = [_row(10, 0, k, [1.0, 2.0]) for k in range(5)]
        rows_b = [_row(10, 0, k, [3.0, 4.0], name="item_emb") for k in range(5)]
        table = pa.concat_tables([_delta_table(rows_a), _delta_table(rows_b)])
        uploader.submit(10, table)
        uploader.submit(20, table)
        uploader.close()

        ts_values = [call["ts"] for call in view.calls]
        # Two chunks -> two batches per step; the stuck clock forces step 2 to
        # start strictly after step 1's last ts (101), i.e. 102.
        self.assertEqual(ts_values, [100, 101, 102, 103])
        self.assertEqual([len(call["data"]) for call in view.calls], [5, 5, 5, 5])
        self.assertEqual(view.closed, [True])

    def test_mixed_fqn_dimensions_validated_per_row(self):
        # _validate_delta_batch keys the expected dimension per row, so a batch
        # carrying multiple FQNs with different dimensions must not be
        # mis-flagged as a dimension mismatch.
        view = _FakeView()
        factory = _FakeClientFactory(view)
        uploader = self._uploader(
            _feature_store_config(upload_batch_size=1000),
            client_factory=factory,
            clock_ms=lambda: 100,
            embedding_dimensions={
                "model.ebc.embedding_bags.user_emb": 2,
                "model.ebc.embedding_bags.item_emb": 3,
            },
        )
        uploader.start()
        uploader.submit(
            10,
            _delta_table(
                [
                    _row(10, 0, 1, [1.0, 2.0]),
                    _row(10, 0, 2, [3.0, 4.0, 5.0], name="item_emb"),
                ]
            ),
        )
        uploader.close()

        self.assertEqual(len(view.calls), 1)
        self.assertEqual(len(view.calls[0]["data"]), 2)
        self.assertEqual(
            view.calls[0]["data"][0]["embedding_name"],
            "model.ebc.embedding_bags.user_emb",
        )
        self.assertEqual(
            view.calls[0]["data"][1]["embedding_name"],
            "model.ebc.embedding_bags.item_emb",
        )
        self.assertTrue(
            np.array_equal(
                view.calls[0]["data"][0]["embedding"],
                np.array([1.0, 2.0], dtype=np.float32),
            )
        )
        self.assertTrue(
            np.array_equal(
                view.calls[0]["data"][1]["embedding"],
                np.array([3.0, 4.0, 5.0], dtype=np.float32),
            )
        )
        self.assertEqual(view.closed, [True])


if __name__ == "__main__":
    unittest.main()
