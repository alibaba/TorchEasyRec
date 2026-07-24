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

"""Ephemeral per-rank uploader for in-memory delta-embedding tables.

Best-effort upload for the current live training process only. No cross-restart
recovery, no durable state, no replay. A process crash means restart from the
latest checkpoint and pending deltas are discarded.
"""

import json
import os
import threading
import time
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Tuple,
    cast,
)
from urllib.parse import urlsplit

import numpy as np
import pyarrow as pa

from tzrec.protos.train_pb2 import FeatureStoreConfig
from tzrec.utils.logging_util import logger
from tzrec.utils.sparse_embedding_contract import (
    SPARSE_EMBEDDING_INVALID_KEY,
    SPARSE_EMBEDDING_ROLES,
)

FEATURE_STORE_PK_FIELD = "embedding_name"
FEATURE_STORE_SK_FIELD = "key_id"
FEATURE_STORE_VALUE_FIELD = "embedding"
FEATURE_STORE_WRITE_MODE = "MERGE"
FEATURE_STORE_SDK_BATCH_SIZE = 1000

_FEATURE_STORE_PROGRESS_LOG_INTERVAL_BATCHES = 100


class FeatureStoreUploadError(RuntimeError):
    """Safe, credential-free error propagated from the uploader thread."""


class _UploadAborted(RuntimeError):
    """Internal control flow for abnormal, non-draining shutdown."""


@dataclass(frozen=True)
class FeatureStoreUploadSettings:
    """Validated immutable settings copied from the runtime protobuf."""

    region: str
    endpoint: str
    project_name: str
    feature_entity_name: str
    feature_view_name: str
    feature_view_ttl_secs: int
    feature_view_shard_count: int
    feature_view_replication_count: int
    version: str
    upload_batch_size: int
    max_retries: int
    retry_backoff_secs: int
    shutdown_timeout_secs: int
    max_pending_steps: int
    poll_interval_secs: int
    allow_custom_endpoint: bool

    @classmethod
    def from_proto(cls, config: FeatureStoreConfig) -> "FeatureStoreUploadSettings":
        """Validate configuration without resolving credentials."""
        initialization_errors = config.FindInitializationErrors()
        if initialization_errors:
            raise ValueError(
                "feature_store_config is missing required fields: "
                + ", ".join(initialization_errors)
            )
        region = config.region or os.environ.get("ALIBABA_CLOUD_REGION", "")
        endpoint = config.endpoint

        if not region:
            raise ValueError(
                "feature_store_config.region must not be empty "
                "(it may come from ALIBABA_CLOUD_REGION)"
            )
        allow_custom_endpoint = bool(config.allow_custom_endpoint)
        _validate_feature_store_endpoint(endpoint, allow_custom_endpoint)
        project_name = config.project_name.strip()
        feature_entity_name = config.feature_entity_name.strip()
        feature_view_name = config.feature_view_name.strip()
        version = config.version.strip()
        if not project_name:
            raise ValueError("feature_store_config.project_name must not be empty")
        if not feature_entity_name:
            raise ValueError(
                "feature_store_config.feature_entity_name must not be empty"
            )
        if not feature_view_name:
            raise ValueError("feature_store_config.feature_view_name must not be empty")
        if not version or version == "default":
            raise ValueError(
                "feature_store_config.version must be an explicit non-default version"
            )

        positive_values = {
            "feature_view_ttl_secs": int(config.feature_view_ttl_secs),
            "upload_batch_size": int(config.upload_batch_size),
            "max_retries": int(config.max_retries),
            "shutdown_timeout_secs": int(config.shutdown_timeout_secs),
            "max_pending_steps": int(config.max_pending_steps),
            "poll_interval_secs": int(config.poll_interval_secs),
        }
        for name, value in positive_values.items():
            if value <= 0:
                raise ValueError(f"feature_store_config.{name} must be > 0")
        feature_view_shard_count = int(config.feature_view_shard_count)
        if not 1 <= feature_view_shard_count <= 20:
            raise ValueError(
                "feature_store_config.feature_view_shard_count must be in [1, 20]"
            )
        feature_view_replication_count = int(config.feature_view_replication_count)
        if not 1 <= feature_view_replication_count <= 3:
            raise ValueError(
                "feature_store_config.feature_view_replication_count must be in [1, 3]"
            )
        if positive_values["upload_batch_size"] > FEATURE_STORE_SDK_BATCH_SIZE:
            raise ValueError(
                "feature_store_config.upload_batch_size must be <= "
                f"{FEATURE_STORE_SDK_BATCH_SIZE} so one publish timestamp maps to "
                "exactly one FeatureStore SDK HTTP batch"
            )

        return cls(
            region=region,
            endpoint=endpoint,
            project_name=project_name,
            feature_entity_name=feature_entity_name,
            feature_view_name=feature_view_name,
            feature_view_ttl_secs=positive_values["feature_view_ttl_secs"],
            feature_view_shard_count=feature_view_shard_count,
            feature_view_replication_count=feature_view_replication_count,
            version=version,
            upload_batch_size=positive_values["upload_batch_size"],
            max_retries=positive_values["max_retries"],
            retry_backoff_secs=int(config.retry_backoff_secs),
            shutdown_timeout_secs=positive_values["shutdown_timeout_secs"],
            max_pending_steps=positive_values["max_pending_steps"],
            poll_interval_secs=positive_values["poll_interval_secs"],
            allow_custom_endpoint=allow_custom_endpoint,
        )


def validate_feature_store_config(config: FeatureStoreConfig) -> None:
    """Validate upload configuration without resolving credentials."""
    FeatureStoreUploadSettings.from_proto(config)


def _validate_feature_store_endpoint(
    endpoint: str, allow_custom_endpoint: bool
) -> None:
    """Reject endpoints that could divert cloud or FeatureDB credentials.

    The endpoint receives the access keys, STS token and FeatureDB
    credentials, so only trusted Alibaba Cloud hosts are accepted unless the
    operator explicitly opts in to a vetted custom deployment.
    """
    if not endpoint:
        return
    parsed_endpoint = urlsplit(endpoint if "://" in endpoint else f"//{endpoint}")
    if parsed_endpoint.scheme and parsed_endpoint.scheme != "https":
        raise ValueError(
            "feature_store_config.endpoint must use HTTPS when set; "
            f"got scheme={parsed_endpoint.scheme!r}"
        )
    if parsed_endpoint.username or parsed_endpoint.password:
        raise ValueError(
            "feature_store_config.endpoint must not contain userinfo credentials"
        )
    if parsed_endpoint.path not in ("", "/"):
        raise ValueError(
            "feature_store_config.endpoint must not contain a path, query, or fragment"
        )
    if parsed_endpoint.query or parsed_endpoint.fragment:
        raise ValueError(
            "feature_store_config.endpoint must not contain a path, query, or fragment"
        )
    if parsed_endpoint.port is not None and not allow_custom_endpoint:
        raise ValueError(
            "feature_store_config.endpoint must not contain an explicit port"
        )
    hostname = (parsed_endpoint.hostname or "").lower()
    if not hostname:
        raise ValueError("feature_store_config.endpoint must contain a hostname")
    trusted_suffixes = (
        ".aliyuncs.com",
        ".alibabacloud.com",
        ".aliyun.com",
    )
    if not hostname.endswith(trusted_suffixes):
        if not allow_custom_endpoint:
            raise ValueError(
                "feature_store_config.endpoint must be a trusted Alibaba Cloud "
                f"host (got {hostname!r}); set allow_custom_endpoint only for a "
                "vetted private deployment"
            )


class FeatureStoreDeltaUploader:
    """Ephemeral per-rank uploader for in-memory delta-embedding tables.

    Best-effort upload for the current live process only. No cross-restart
    recovery, no durable state, no replay. A process crash means restart from
    the latest checkpoint and pending deltas are discarded.

    Every rank owns one uploader and streams only its local shard rows, so no
    cross-rank aggregation or deduplication is needed: table-wise and row-wise
    sharding give each (embedding_name, key_id) a unique owner rank, while
    data-parallel replicas issue identical idempotent MERGE writes. Publish
    timestamps are allocated per rank and only need per-rank monotonicity,
    because a key's owner rank is fixed for the lifetime of the process.
    """

    def __init__(
        self,
        config: FeatureStoreConfig,
        embedding_dimensions: Mapping[str, int],
        rank: int = 0,
        manage_remote_view: bool = True,
        client_factory: Optional[Callable[..., Any]] = None,
        credentials_client: Optional[Any] = None,
        clock_ms: Optional[Callable[[], int]] = None,
    ) -> None:
        """Initialize the uploader with validated settings and in-memory state."""
        self._settings = FeatureStoreUploadSettings.from_proto(config)
        self._rank = int(rank)
        self._manage_remote_view = bool(manage_remote_view)
        self._embedding_dimensions = {
            str(name): int(dimension)
            for name, dimension in embedding_dimensions.items()
        }
        invalid_dimensions = {
            name: dimension
            for name, dimension in self._embedding_dimensions.items()
            if not name or dimension <= 0
        }
        if invalid_dimensions:
            raise ValueError(
                "invalid sparse embedding dimensions in FeatureStore contract: "
                f"{invalid_dimensions}"
            )

        self._client_factory = client_factory
        self._credentials_client = (
            credentials_client or self._create_credentials_client()
        )
        self._clock_ms = clock_ms or (lambda: time.time_ns() // 1_000_000)
        self._view = None
        self._condition = threading.Condition()
        self._pending: Dict[int, pa.Table] = {}
        self._started = False
        self._closing = False
        self._aborting = False
        self._closed = False
        self._worker: Optional[threading.Thread] = None
        self._error: Optional[FeatureStoreUploadError] = None
        self._last_publish_ts: int = 0

    def start(self) -> None:
        """Start the background upload worker after validating the remote view."""
        with self._condition:
            if self._started:
                return
            if self._closed:
                raise RuntimeError("FeatureStoreDeltaUploader is already closed")
            self._raise_if_failed_locked()
            try:
                self._get_view()
            except BaseException:
                self._reset_view(suppress_errors=True)
                raise
            self._started = True
            self._worker = threading.Thread(
                target=self._run,
                name="tzrec-feature-store-delta-uploader",
                daemon=True,
            )
            self._worker.start()
        logger.info(
            "FeatureStore delta uploader started: project=%s feature_view=%s "
            "version=%s rank=%s",
            self._settings.project_name,
            self._settings.feature_view_name,
            self._settings.version,
            self._rank,
        )

    def submit(self, global_step: int, table: pa.Table) -> None:
        """Enqueue one step's in-memory delta table with back-pressure."""
        global_step = int(global_step)
        if global_step <= 0:
            raise ValueError("FeatureStore delta global_step must be > 0")
        with self._condition:
            self._raise_if_failed_locked()
            if not self._started:
                raise RuntimeError(
                    "FeatureStoreDeltaUploader.start() must be called before submit()"
                )
            if self._closing or self._closed:
                raise RuntimeError("cannot submit to a closing FeatureStore uploader")
            while (
                global_step not in self._pending
                and len(self._pending) >= self._settings.max_pending_steps
            ):
                self._condition.wait(self._settings.poll_interval_secs)
                self._raise_if_failed_locked()
            self._pending.setdefault(global_step, table)
            self._condition.notify_all()

    def check_error(self) -> None:
        """Surface a background failure at a safe training-thread boundary."""
        with self._condition:
            self._raise_if_failed_locked()

    def close(self, raise_on_error: bool = True, drain: bool = True) -> None:
        """Close the worker, draining only during a normal training shutdown."""
        with self._condition:
            if self._closed:
                if raise_on_error:
                    self._raise_if_failed_locked()
                return
            if not self._started:
                self._closed = True
                return
            self._closing = True
            self._aborting = not drain
            self._condition.notify_all()
            worker = self._worker

        if drain and worker is not None:
            worker.join(timeout=self._settings.shutdown_timeout_secs)
            if worker.is_alive():
                timeout_error = FeatureStoreUploadError(
                    "FeatureStore uploader did not drain before shutdown timeout"
                )
                with self._condition:
                    if self._error is None:
                        self._error = timeout_error
                    self._aborting = True
                    self._condition.notify_all()

        with self._condition:
            self._closed = True
            if raise_on_error:
                self._raise_if_failed_locked()

    def _run(self) -> None:
        current_step: Optional[int] = None
        try:
            while True:
                with self._condition:
                    if self._aborting:
                        return
                    if not self._pending:
                        if self._closing:
                            return
                        self._condition.wait(self._settings.poll_interval_secs)
                        continue
                    current_step = min(self._pending)
                    table = self._pending[current_step]

                self._upload_with_retries(current_step, table)

                with self._condition:
                    self._pending.pop(current_step, None)
                    self._condition.notify_all()
                current_step = None
        except _UploadAborted:
            return
        except BaseException as exc:
            step_context = (
                f" at global_step={current_step}" if current_step is not None else ""
            )
            error = FeatureStoreUploadError(
                f"FeatureStore delta upload failed{step_context}: {exc}"
            )
            with self._condition:
                if self._error is None:
                    self._error = error
                self._condition.notify_all()
            logger.error(
                "FeatureStore delta upload failed%s: %s",
                step_context,
                exc,
                exc_info=True,
            )
        finally:
            self._reset_view(suppress_errors=True)

    def _raise_if_failed_locked(self) -> None:
        if self._error is not None:
            raise self._error

    def _raise_if_aborting(self) -> None:
        with self._condition:
            if self._aborting:
                raise _UploadAborted()

    def _upload_with_retries(self, global_step: int, table: pa.Table) -> None:
        for attempt in range(1, self._settings.max_retries + 1):
            self._raise_if_aborting()
            try:
                self._stream_upload(global_step, table)
                return
            except _UploadAborted:
                raise
            except BaseException as exc:
                self._reset_view(suppress_errors=True)
                if attempt >= self._settings.max_retries:
                    raise
                logger.warning(
                    "FeatureStore delta upload attempt %s/%s failed for step %s "
                    "(%s); retrying after backoff",
                    attempt,
                    self._settings.max_retries,
                    global_step,
                    exc,
                )
                if self._settings.retry_backoff_secs > 0:
                    time.sleep(self._settings.retry_backoff_secs * attempt)
        raise AssertionError("unreachable FeatureStore retry state")

    def _allocate_timestamp_range(self, batch_count: int) -> Tuple[int, int]:
        """Allocate a rank-locally monotonic timestamp range (in-memory only).

        Per-rank monotonicity is sufficient for Next-Ts incremental readers:
        sharding is fixed for the lifetime of the process, so every key is
        always republished by the same rank with a strictly newer timestamp.
        """
        reserved = max(batch_count, 1)
        range_start = max(int(self._clock_ms()), self._last_publish_ts + 1, 1)
        range_end = range_start + reserved - 1
        self._last_publish_ts = range_end
        return range_start, range_end

    def _stream_upload(self, global_step: int, table: pa.Table) -> None:
        """Stream the in-memory delta table directly to the FeatureStore SDK."""
        view = self._get_view()
        max_in_flight = int(getattr(view, "_max_workers", 1))

        total_batches = self._count_total_batches(table)
        ts_range = self._allocate_timestamp_range(total_batches)
        range_start = ts_range[0]

        completed_batches = 0
        window_batches = 0
        window_records = 0
        started_at = time.monotonic()
        next_progress_batch = _FEATURE_STORE_PROGRESS_LOG_INTERVAL_BATCHES
        logged_first_window = False

        logger.info(
            "FeatureStore delta upload started: step=%s rank=%s version=%s "
            "batches=%s ts_range=%s-%s",
            global_step,
            self._rank,
            self._settings.version,
            total_batches,
            ts_range[0],
            ts_range[1],
        )

        try:
            for batch in table.to_batches(
                max_chunksize=self._settings.upload_batch_size
            ):
                self._raise_if_aborting()
                payload = self._validate_and_build_payload(batch)
                if not payload:
                    continue
                view.write_features(
                    data=payload,
                    version=self._settings.version,
                    write_mode=FEATURE_STORE_WRITE_MODE,
                    ts=range_start + completed_batches,
                )
                completed_batches += 1
                window_batches += 1
                window_records += len(payload)

                if window_batches < max_in_flight and completed_batches < total_batches:
                    continue
                summary = view.write_flush()
                self._validate_flush_summary(
                    summary,
                    expected_records=window_records,
                    expected_batches=window_batches,
                )
                if (
                    not logged_first_window
                    or completed_batches >= next_progress_batch
                    or completed_batches == total_batches
                ):
                    logger.info(
                        "FeatureStore delta upload progress: step=%s "
                        "batches=%s/%s elapsed_secs=%.1f",
                        global_step,
                        completed_batches,
                        total_batches,
                        time.monotonic() - started_at,
                    )
                    logged_first_window = True
                    while next_progress_batch <= completed_batches:
                        next_progress_batch += (
                            _FEATURE_STORE_PROGRESS_LOG_INTERVAL_BATCHES
                        )
                window_batches = 0
                window_records = 0

            if window_batches > 0:
                summary = view.write_flush()
                self._validate_flush_summary(
                    summary,
                    expected_records=window_records,
                    expected_batches=window_batches,
                )
        except BaseException:
            try:
                view.write_flush()
            except BaseException:
                pass
            raise

        logger.info(
            "FeatureStore delta upload completed: step=%s batches=%s elapsed_secs=%.1f",
            global_step,
            completed_batches,
            time.monotonic() - started_at,
        )

    def _count_total_batches(self, table: pa.Table) -> int:
        """Count the number of upload batches in one step's delta table."""
        total_rows = int(table.num_rows)
        if total_rows == 0:
            return 1
        return (
            total_rows + self._settings.upload_batch_size - 1
        ) // self._settings.upload_batch_size

    def _validate_and_build_payload(
        self,
        batch: pa.RecordBatch,
    ) -> List[Dict[str, Any]]:
        """Validate one delta batch and build the SDK payload."""
        num_rows = batch.num_rows
        if num_rows == 0:
            return []
        batch_roles = set(batch.column("embedding_role").to_pylist())
        if not batch_roles <= set(SPARSE_EMBEDDING_ROLES):
            raise ValueError("delta shard has an invalid embedding role")
        embedding_names = batch.column("embedding_name").to_pylist()
        for embedding_name in set(embedding_names):
            if not embedding_name:
                raise ValueError("delta shard embedding_name must not be empty")
            if embedding_name not in self._embedding_dimensions:
                raise ValueError(
                    "delta shard embedding_name is absent from model contract: "
                    f"{embedding_name!r}"
                )

        key_ids = batch.column("key_id").to_numpy(zero_copy_only=False)
        if bool((key_ids == SPARSE_EMBEDDING_INVALID_KEY).any()):
            raise ValueError(
                "delta shard key_id=-1 is reserved as the Processor/"
                "NvEmbeddings invalid-key sentinel"
            )

        embedding_column = cast(pa.ListArray, batch.column("embedding"))
        flat_embeddings = embedding_column.values.to_numpy(zero_copy_only=False)
        if not bool(np.isfinite(flat_embeddings).all()):
            raise ValueError("delta embedding contains NaN or Inf")
        offsets = embedding_column.offsets.to_numpy()
        lengths = np.diff(offsets)
        expected_dims = np.array(
            [self._embedding_dimensions[name] for name in embedding_names],
            dtype=lengths.dtype,
        )
        bad_rows = np.flatnonzero(lengths != expected_dims)
        if bad_rows.size > 0:
            row = int(bad_rows[0])
            raise ValueError(
                f"delta embedding dimension mismatch for {embedding_names[row]!r}: "
                f"expected={int(expected_dims[row])}, "
                f"actual={int(lengths[row])}"
            )

        return [
            {
                FEATURE_STORE_PK_FIELD: embedding_names[i],
                FEATURE_STORE_SK_FIELD: int(key_ids[i]),
                FEATURE_STORE_VALUE_FIELD: flat_embeddings[
                    int(offsets[i]) : int(offsets[i + 1])
                ].copy(),
            }
            for i in range(num_rows)
        ]

    @staticmethod
    def _create_credentials_client() -> Any:
        """Create the Alibaba Cloud credential provider (default chain)."""
        try:
            from alibabacloud_credentials.client import Client as CredClient
        except ImportError as exc:
            raise RuntimeError(
                "alibabacloud_credentials is required when feature_store_config "
                "is set; install it via: pip install alibabacloud_credentials"
            ) from exc
        return CredClient()

    def _get_view(self) -> Any:
        if self._view is not None:
            return self._view
        if self._client_factory is None:
            try:
                from feature_store_py import FeatureStoreClient
            except ImportError as exc:
                raise RuntimeError(
                    "feature_store_py is required when feature_store_config is set"
                ) from exc
            client_factory = FeatureStoreClient
        else:
            client_factory = self._client_factory

        credential = self._credentials_client.get_credential()
        kwargs = {
            "access_key_id": credential.access_key_id,
            "access_key_secret": credential.access_key_secret,
            "region": self._settings.region or None,
            "endpoint": self._settings.endpoint or None,
            "security_token": credential.security_token or None,
            "featuredb_username": os.environ.get("FEATUREDB_USERNAME") or None,
            "featuredb_password": os.environ.get("FEATUREDB_PASSWORD") or None,
        }
        client = client_factory(**kwargs)
        project = client.get_project(self._settings.project_name)
        if project is None:
            raise RuntimeError("configured FeatureStore project was not found")
        view = self._get_or_create_view(project)
        self._view = view
        actual_fields = (view.pk_field, view.sk_field, view.embedding_field)
        expected_fields = (
            FEATURE_STORE_PK_FIELD,
            FEATURE_STORE_SK_FIELD,
            FEATURE_STORE_VALUE_FIELD,
        )
        if actual_fields != expected_fields:
            raise RuntimeError(
                "DynamicEmbedding FeatureView schema mismatch: "
                f"expected={expected_fields}, actual={actual_fields}"
            )
        sdk_batch_size = getattr(view, "_batch_size", FEATURE_STORE_SDK_BATCH_SIZE)
        if (
            type(sdk_batch_size) is not int
            or sdk_batch_size < self._settings.upload_batch_size
        ):
            raise RuntimeError(
                "FeatureStore SDK batch_size is smaller than the configured outer "
                "batch; one publish timestamp could span multiple HTTP requests"
            )
        sdk_max_workers = getattr(view, "_max_workers", 1)
        if type(sdk_max_workers) is not int or sdk_max_workers <= 0:
            raise RuntimeError("FeatureStore SDK max_workers must be a positive int")
        return view

    def _reset_view(self, suppress_errors: bool = False) -> None:
        view = self._view
        self._view = None
        if view is not None:
            try:
                view.close(wait=True)
            except BaseException as exc:
                close_error = FeatureStoreUploadError(
                    "FeatureStore SDK writer close failed"
                )
                if not suppress_errors:
                    raise close_error from exc
                with self._condition:
                    if self._error is None:
                        self._error = close_error
                    self._condition.notify_all()
                logger.error(
                    "Failed to close FeatureStore SDK writer cleanly (%s)",
                    type(exc).__name__,
                )

    def _get_or_create_view(self, project: Any) -> Any:
        """Return the configured DynamicEmbedding view, creating it if absent.

        Only the primary (rank-zero) uploader creates the view and validates
        control-plane metadata; other ranks open a handle to the view that the
        primary published before they started.
        """
        if not self._manage_remote_view:
            view = project.get_dynamic_embedding_feature_view(
                self._settings.feature_view_name
            )
            if view is None:
                view = self._wait_for_dynamic_embedding_view(project)
            if view is None:
                raise RuntimeError(
                    "configured DynamicEmbedding FeatureView was not found; "
                    "the rank-zero uploader must create it before other ranks "
                    "start"
                )
            self._view = view
            return view
        provisioned = False
        view = project.get_dynamic_embedding_feature_view(
            self._settings.feature_view_name
        )
        if view is not None:
            self._view = view
        metadata = self._wait_for_feature_view_metadata(project)
        if view is None and metadata is not None:
            self._validate_feature_view_metadata(metadata)
            view = self._wait_for_dynamic_embedding_view(project)
            if view is None:
                raise RuntimeError(
                    "configured DynamicEmbedding FeatureView exists but did not "
                    "become ready"
                )
            self._view = view
        elif view is None:
            create_error: Optional[Exception] = None
            try:
                view = project.create_dynamic_embedding_feature_view(
                    name=self._settings.feature_view_name,
                    entity=self._settings.feature_entity_name,
                    pk_field_name=FEATURE_STORE_PK_FIELD,
                    sk_field_name=FEATURE_STORE_SK_FIELD,
                    embedding_field_name=FEATURE_STORE_VALUE_FIELD,
                    pk_field_type="STRING",
                    sk_field_type="INT64",
                    ttl=self._settings.feature_view_ttl_secs,
                    shard_count=self._settings.feature_view_shard_count,
                    replication_count=self._settings.feature_view_replication_count,
                )
                provisioned = True
            except Exception as exc:
                create_error = exc
                view = self._wait_for_dynamic_embedding_view(project)
            if view is None:
                error = RuntimeError(
                    "failed to create configured DynamicEmbedding FeatureView; "
                    "verify that feature_entity_name already exists"
                )
                if create_error is not None:
                    raise error from create_error
                raise error
            self._view = view
            metadata = self._wait_for_feature_view_metadata(project)

        if metadata is None:
            self._view = view
            raise RuntimeError(
                "DynamicEmbedding FeatureView control-plane metadata was not found"
            )
        self._view = view
        self._validate_feature_view_metadata(metadata)
        if provisioned:
            logger.info(
                "Created DynamicEmbedding FeatureView: project=%s entity=%s view=%s",
                self._settings.project_name,
                self._settings.feature_entity_name,
                self._settings.feature_view_name,
            )
        return view

    def _wait_for_dynamic_embedding_view(self, project: Any) -> Any:
        """Bounded re-get after a concurrent or partially completed create."""
        last_error: Optional[Exception] = None
        for attempt in range(1, self._settings.max_retries + 1):
            try:
                view = project.get_dynamic_embedding_feature_view(
                    self._settings.feature_view_name
                )
            except Exception as exc:
                last_error = exc
            else:
                if view is not None:
                    return view
            if (
                attempt < self._settings.max_retries
                and self._settings.retry_backoff_secs > 0
            ):
                time.sleep(self._settings.retry_backoff_secs * attempt)
        if last_error is not None:
            raise RuntimeError(
                "DynamicEmbedding FeatureView did not become ready after creation"
            ) from last_error
        return None

    def _wait_for_feature_view_metadata(self, project: Any) -> Any:
        """Bounded control-plane lookup used to validate the created resource."""
        last_error: Optional[Exception] = None
        for attempt in range(1, self._settings.max_retries + 1):
            try:
                feature_view = project.get_feature_view(
                    self._settings.feature_view_name
                )
            except Exception as exc:
                last_error = exc
            else:
                if feature_view is not None:
                    return feature_view
            if (
                attempt < self._settings.max_retries
                and self._settings.retry_backoff_secs > 0
            ):
                time.sleep(self._settings.retry_backoff_secs * attempt)
        if last_error is not None:
            raise RuntimeError(
                "FeatureView control-plane metadata did not become ready"
            ) from last_error
        return None

    def _validate_feature_view_metadata(self, feature_view: Any) -> None:
        """Validate immutable control-plane schema and provisioning settings."""
        actual_type = getattr(feature_view, "type", None)
        if actual_type != "DynamicEmbedding":
            raise RuntimeError(
                "configured FeatureView exists with an incompatible type: "
                f"expected='DynamicEmbedding', actual={actual_type!r}"
            )
        actual_entity = getattr(feature_view, "feature_entity_name", None)
        if actual_entity != self._settings.feature_entity_name:
            raise RuntimeError(
                "DynamicEmbedding FeatureView entity mismatch: "
                f"expected={self._settings.feature_entity_name!r}, "
                f"actual={actual_entity!r}"
            )

        expected_fields = {
            FEATURE_STORE_PK_FIELD: ("STRING", {"PrimaryKey"}),
            FEATURE_STORE_SK_FIELD: ("INT64", {"SubKey"}),
            FEATURE_STORE_VALUE_FIELD: ("ARRAY<FLOAT>", set()),
        }
        fields = getattr(feature_view, "fields_dict", None)
        if not isinstance(fields, dict) or set(fields) != set(expected_fields):
            actual_names = sorted(fields) if isinstance(fields, dict) else None
            raise RuntimeError(
                "DynamicEmbedding FeatureView field set mismatch: "
                f"expected={sorted(expected_fields)}, actual={actual_names}"
            )
        for name, (expected_type, expected_attributes) in expected_fields.items():
            field_info = fields[name]
            if not isinstance(field_info, dict):
                raise RuntimeError(
                    "DynamicEmbedding FeatureView has invalid field metadata for "
                    f"{name!r}"
                )
            actual_field_type = field_info.get("Type")
            attributes = field_info.get("Attributes", [])
            if not isinstance(attributes, (list, tuple, set)):
                raise RuntimeError(
                    "DynamicEmbedding FeatureView has invalid field attributes for "
                    f"{name!r}"
                )
            actual_attributes = set(attributes)
            if (
                actual_field_type != expected_type
                or actual_attributes != expected_attributes
            ):
                raise RuntimeError(
                    "DynamicEmbedding FeatureView field contract mismatch for "
                    f"{name!r}: expected_type={expected_type!r}, "
                    f"actual_type={actual_field_type!r}, "
                    f"expected_attributes={sorted(expected_attributes)}, "
                    f"actual_attributes={sorted(actual_attributes)}"
                )

        summary = getattr(feature_view, "summary", None)
        config_value = summary.get("Config") if isinstance(summary, dict) else None
        try:
            provisioning = (
                json.loads(config_value)
                if isinstance(config_value, str)
                else dict(config_value)
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "DynamicEmbedding FeatureView has invalid provisioning config"
            ) from exc
        expected_provisioning = {
            "ttl": self._settings.feature_view_ttl_secs,
            "shard_count": self._settings.feature_view_shard_count,
            "replication_count": self._settings.feature_view_replication_count,
        }
        actual_provisioning = {
            name: provisioning.get(name) for name in expected_provisioning
        }
        if actual_provisioning != expected_provisioning:
            raise RuntimeError(
                "DynamicEmbedding FeatureView provisioning mismatch: "
                f"expected={expected_provisioning}, actual={actual_provisioning}"
            )

    @staticmethod
    def _validate_flush_summary(
        summary: Any, expected_records: int, expected_batches: int
    ) -> None:
        required = {
            "total_batches",
            "failed_batches",
            "total_records",
            "success_records",
            "failed_records",
        }
        if not isinstance(summary, dict) or not required.issubset(summary):
            raise RuntimeError("FeatureStore write_flush returned an invalid summary")
        if (
            int(summary["total_batches"]) != expected_batches
            or int(summary["failed_batches"]) != 0
            or int(summary["failed_records"]) != 0
            or int(summary["success_records"]) != int(summary["total_records"])
            or int(summary["total_records"]) != expected_records
        ):
            raise RuntimeError("FeatureStore write_flush reported incomplete writes")
