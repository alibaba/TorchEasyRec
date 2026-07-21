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

"""Durable rank-zero uploader for delta-embedding parquet outboxes."""

import fcntl
import hashlib
import json
import os
import re
import shutil
import sqlite3
import threading
import time
import uuid
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import (
    Any,
    BinaryIO,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    cast,
)
from urllib.parse import urlsplit

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

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
DELTA_OPERATION_UPSERT = "UPSERT"
DELTA_DUMP_SCHEMA_VERSION = "2"
DELTA_DUMP_GENERATION_METADATA_KEY = b"tzrec.delta_embedding.dump_generation"

_FEATURE_STORE_PROGRESS_LOG_INTERVAL_BATCHES = 100
_RECORD_STORE_DB_FILENAME = "upload_records.sqlite"
_RECORD_STORE_CACHE_SIZE_KB = 65536
_VERSION_INITIALIZATION = "AUTO_CREATE_ON_FIRST_DELTA_MERGE"
_LEGACY_VERSION_INITIALIZATION = "PREPROVISIONED_FOR_DELTA_MERGE"

_POISONED_WRITER_LOCKS: List[BinaryIO] = []

_SCHEMA_VERSION_METADATA_KEY = b"tzrec.delta_embedding.schema_version"
_REQUIRED_PARQUET_FIELDS = {
    "global_step": pa.int64(),
    "rank": pa.int32(),
    "world_size": pa.int32(),
    "embedding_name": pa.string(),
    "embedding_role": pa.string(),
    "feature_name": pa.string(),
    "table_fqn": pa.string(),
    "key_id": pa.int64(),
    "embedding": pa.list_(pa.float32()),
}


class FeatureStoreUploadError(RuntimeError):
    """Safe, credential-free error propagated from the uploader thread."""


class _UploadAborted(RuntimeError):
    """Internal control flow for abnormal, non-draining shutdown."""


class _ShardSetNotReady(RuntimeError):
    """A multi-rank canonical shard set is still being atomically replaced."""


@dataclass(frozen=True)
class FeatureStoreUploadSettings:
    """Validated immutable settings copied from the runtime protobuf."""

    region: str
    endpoint: str
    access_key_id: str = field(repr=False)
    access_key_secret: str = field(repr=False)
    security_token: str = field(repr=False)
    featuredb_username: str = field(repr=False)
    featuredb_password: str = field(repr=False)
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
    shard_wait_timeout_secs: int
    shutdown_timeout_secs: int
    max_pending_steps: int
    poll_interval_secs: int
    allow_custom_endpoint: bool

    @classmethod
    def from_proto(cls, config: FeatureStoreConfig) -> "FeatureStoreUploadSettings":
        """Resolve environment-backed credentials without logging them."""
        initialization_errors = config.FindInitializationErrors()
        if initialization_errors:
            raise ValueError(
                "feature_store_config is missing required fields: "
                + ", ".join(initialization_errors)
            )
        region = config.region or os.environ.get("ALIBABA_CLOUD_REGION", "")
        endpoint = config.endpoint
        security_token = config.security_token or os.environ.get(
            "ALIBABA_CLOUD_SECURITY_TOKEN", ""
        )
        credential_env_names = {
            "access_key_id": "ALIBABA_CLOUD_ACCESS_KEY_ID",
            "access_key_secret": "ALIBABA_CLOUD_ACCESS_KEY_SECRET",
            "featuredb_username": "FEATUREDB_USERNAME",
            "featuredb_password": "FEATUREDB_PASSWORD",
        }
        credentials = {
            field_name: os.environ.get(env_name, "")
            for field_name, env_name in credential_env_names.items()
        }
        missing_env_names = [
            credential_env_names[field_name]
            for field_name, value in credentials.items()
            if not value
        ]
        if missing_env_names:
            raise ValueError(
                "feature_store_config requires non-empty environment variables: "
                + ", ".join(missing_env_names)
                + "; set them in the training process environment"
            )
        access_key_id = credentials["access_key_id"]
        access_key_secret = credentials["access_key_secret"]
        featuredb_username = credentials["featuredb_username"]
        featuredb_password = credentials["featuredb_password"]

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
            "shard_wait_timeout_secs": int(config.shard_wait_timeout_secs),
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
            access_key_id=access_key_id,
            access_key_secret=access_key_secret,
            security_token=security_token,
            featuredb_username=featuredb_username,
            featuredb_password=featuredb_password,
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
            shard_wait_timeout_secs=positive_values["shard_wait_timeout_secs"],
            shutdown_timeout_secs=positive_values["shutdown_timeout_secs"],
            max_pending_steps=positive_values["max_pending_steps"],
            poll_interval_secs=positive_values["poll_interval_secs"],
            allow_custom_endpoint=allow_custom_endpoint,
        )


def validate_feature_store_config(config: FeatureStoreConfig) -> None:
    """Validate upload configuration and its environment-resolved credentials."""
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
            "feature_store_config.endpoint must use HTTPS when a scheme is given"
        )
    if parsed_endpoint.username is not None or parsed_endpoint.password is not None:
        raise ValueError("feature_store_config.endpoint must not contain URI userinfo")
    if (
        parsed_endpoint.path not in ("", "/")
        or parsed_endpoint.query
        or parsed_endpoint.fragment
    ):
        raise ValueError(
            "feature_store_config.endpoint must not contain path, query, or "
            "fragment components"
        )
    hostname = parsed_endpoint.hostname
    if not hostname:
        raise ValueError("feature_store_config.endpoint must name a host")
    if allow_custom_endpoint:
        return
    if parsed_endpoint.port is not None:
        raise ValueError(
            "feature_store_config.endpoint must not contain a port unless "
            "allow_custom_endpoint is set"
        )
    if not hostname.endswith(".aliyuncs.com"):
        raise ValueError(
            "feature_store_config.endpoint must be a trusted *.aliyuncs.com "
            "FeatureStore endpoint; set allow_custom_endpoint only for a "
            "vetted private deployment"
        )


def _feature_store_target_hash(settings: FeatureStoreUploadSettings) -> str:
    target_identity = {
        "region": settings.region,
        "endpoint": settings.endpoint,
        "project_name": settings.project_name,
        "feature_view_name": settings.feature_view_name,
        "version": settings.version,
    }
    return _json_digest(target_identity)


def feature_store_delta_file_prefix(
    config: FeatureStoreConfig, file_prefix: str
) -> str:
    """Scope canonical parquet names to one immutable FeatureStore target."""
    settings = FeatureStoreUploadSettings.from_proto(config)
    return _scoped_feature_store_file_prefix(settings, file_prefix)


def _scoped_feature_store_file_prefix(
    settings: FeatureStoreUploadSettings, file_prefix: str
) -> str:
    target_hash = _feature_store_target_hash(settings)
    return f"{file_prefix}__fs_{target_hash[:16]}"


def _feature_store_state_dir(
    settings: FeatureStoreUploadSettings, output_dir: str
) -> str:
    target_hash = _feature_store_target_hash(settings)
    return os.path.join(
        os.path.abspath(output_dir), ".feature_store_upload", target_hash[:16]
    )


def feature_store_upload_error_marker_path(
    config: FeatureStoreConfig, output_dir: str
) -> str:
    """Return the credential-free shared failure marker path for all ranks."""
    settings = FeatureStoreUploadSettings.from_proto(config)
    return os.path.join(
        _feature_store_state_dir(settings, output_dir), "last_error.json"
    )


def _fsync_parent_directory(path: str) -> None:
    directory_fd = os.open(os.path.dirname(path) or ".", os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _durable_makedirs(path: str) -> None:
    """Create missing directories and durably publish every new directory entry."""
    path = os.path.abspath(path)
    missing = []
    current = path
    while not os.path.exists(current):
        missing.append(current)
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    if os.path.exists(current) and not os.path.isdir(current):
        raise NotADirectoryError(current)

    for directory in reversed(missing):
        try:
            os.mkdir(directory)
        except FileExistsError:
            if not os.path.isdir(directory):
                raise
            # Another creator may have won the race but not fsynced its parent.
            _fsync_parent_directory(directory)
        else:
            _fsync_parent_directory(directory)
    if not os.path.isdir(path):
        raise NotADirectoryError(path)
    # Close the race where another rank created the final directory after our
    # initial exists() check but before it fsynced the parent directory entry.
    _fsync_parent_directory(path)


def _atomic_write_json(path: str, value: Mapping[str, Any]) -> None:
    _durable_makedirs(os.path.dirname(path) or ".")
    tmp_path = f"{path}.tmp-{os.getpid()}-{threading.get_ident()}-{uuid.uuid4().hex}"
    try:
        with open(tmp_path, "w") as output:
            json.dump(value, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(tmp_path, path)
        _fsync_parent_directory(path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def _read_json(path: str) -> Dict[str, Any]:
    with open(path) as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _json_digest(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(source: BinaryIO) -> str:
    digest = hashlib.sha256()
    source.seek(0)
    while True:
        chunk = source.read(8 * 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    source.seek(0)
    return digest.hexdigest()


class _RecordStore:
    """Deduplicated delta records spilled to a throwaway SQLite database.

    One delta dump can contain an arbitrary number of unique keys, so keeping
    the dedup state and the upload ordering in memory would make rank-zero
    peak memory scale with a single dump's size. The store keeps both on disk
    behind a bounded page cache, so memory depends only on the streaming
    Parquet read batch and one in-flight upload batch. The database is
    disposable: after a crash the whole step replays from its durable shard
    snapshot.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        if os.path.exists(db_path):
            os.remove(db_path)
        self._connection = sqlite3.connect(db_path)
        self._connection.execute("PRAGMA journal_mode = OFF")
        self._connection.execute("PRAGMA synchronous = OFF")
        self._connection.execute(f"PRAGMA cache_size = -{_RECORD_STORE_CACHE_SIZE_KB}")
        self._connection.execute(
            "CREATE TABLE dedup ("
            "embedding_name TEXT NOT NULL, "
            "key_id INTEGER NOT NULL, "
            "embedding BLOB NOT NULL, "
            "PRIMARY KEY (embedding_name, key_id))"
        )
        self._record_count: Optional[int] = None

    def __enter__(self) -> "_RecordStore":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    @property
    def record_count(self) -> int:
        """Return the number of unique deduplicated records."""
        if self._record_count is None:
            cursor = self._connection.execute("SELECT COUNT(*) FROM dedup")
            self._record_count = int(cursor.fetchone()[0])
        return self._record_count

    def add_batch(
        self,
        embedding_names: List[str],
        key_ids: np.ndarray,
        flat_embeddings: np.ndarray,
        offsets: np.ndarray,
    ) -> None:
        """Spill one streamed shard batch, deduplicating on (name, key_id)."""
        rows = [
            (
                embedding_names[index],
                int(key_ids[index]),
                flat_embeddings[
                    int(offsets[index]) : int(offsets[index + 1])
                ].tobytes(),
            )
            for index in range(len(embedding_names))
        ]
        try:
            self._connection.executemany("INSERT INTO dedup VALUES (?, ?, ?)", rows)
        except sqlite3.IntegrityError:
            # A duplicate can be exact (harmless) or conflicting (corruption);
            # replay the batch row by row to tell the two apart.
            for row in rows:
                try:
                    self._connection.execute("INSERT INTO dedup VALUES (?, ?, ?)", row)
                except sqlite3.IntegrityError:
                    existing = self._connection.execute(
                        "SELECT embedding FROM dedup "
                        "WHERE embedding_name = ? AND key_id = ?",
                        (row[0], row[1]),
                    ).fetchone()
                    if existing[0] != row[2]:
                        raise ValueError(
                            "conflicting duplicate delta row for "
                            f"embedding_name={row[0]!r}, key_id={row[1]}"
                        ) from None
        self._connection.commit()
        self._record_count = None

    def iter_upload_batches(
        self, batch_size: int
    ) -> Iterator[Tuple[List[Dict[str, Any]], int]]:
        """Yield (payload, record count) upload batches in a deterministic order."""
        cursor = self._connection.execute(
            "SELECT embedding_name, key_id, embedding FROM dedup "
            "ORDER BY embedding_name, key_id"
        )
        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                return
            payload = [
                {
                    FEATURE_STORE_PK_FIELD: embedding_name,
                    FEATURE_STORE_SK_FIELD: key_id,
                    FEATURE_STORE_VALUE_FIELD: np.frombuffer(
                        embedding, dtype=np.float32
                    ).copy(),
                }
                for embedding_name, key_id, embedding in rows
            ]
            yield payload, len(payload)

    def to_records(self) -> List[Tuple[str, int, np.ndarray]]:
        """Materialize every deduplicated record in upload order."""
        cursor = self._connection.execute(
            "SELECT embedding_name, key_id, embedding FROM dedup "
            "ORDER BY embedding_name, key_id"
        )
        return [
            (
                embedding_name,
                int(key_id),
                np.frombuffer(embedding, dtype=np.float32).copy(),
            )
            for embedding_name, key_id, embedding in cursor
        ]

    def close(self) -> None:
        """Close the connection and remove the throwaway database file."""
        try:
            self._connection.close()
        finally:
            if os.path.exists(self._db_path):
                os.remove(self._db_path)


class FeatureStoreDeltaUploader:
    """Publish complete delta-dump steps from a persistent local outbox.

    The training thread only writes/parquet-renames and enqueues a step. This
    object's single worker waits for the exact rank shard set and persists a
    monotonic timestamp range before each full-step MERGE attempt. It never
    invokes a torch.distributed collective. Restored training may replay a
    repeated or lower step; dump generation identifies the new publication.
    """

    def __init__(
        self,
        config: FeatureStoreConfig,
        output_dir: str,
        file_prefix: str,
        world_size: int,
        embedding_dimensions: Mapping[str, int],
        client_factory: Optional[Callable[..., Any]] = None,
        clock_ms: Optional[Callable[[], int]] = None,
    ) -> None:
        self._settings = FeatureStoreUploadSettings.from_proto(config)
        self._output_dir = os.path.abspath(output_dir)
        self._file_prefix = _scoped_feature_store_file_prefix(
            self._settings, file_prefix
        )
        self._world_size = int(world_size)
        if self._world_size <= 0:
            raise ValueError("world_size must be > 0")
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
        self._clock_ms = clock_ms or (lambda: time.time_ns() // 1_000_000)
        self._view = None
        self._condition = threading.Condition()
        self._pending: Dict[int, float] = {}
        # Steps already reconciled as committed with a matching generation;
        # discovery skips their JSON/Parquet re-reads so poll lock hold time
        # does not grow with the job's committed history.
        self._reconciled_steps: Set[int] = set()
        self._started = False
        self._closing = False
        self._aborting = False
        self._closed = False
        self._worker: Optional[threading.Thread] = None
        self._error: Optional[FeatureStoreUploadError] = None
        self._writer_quiescence_failed = False

        contract = {
            "schema_version": 5,
            "delta_dump_schema_version": DELTA_DUMP_SCHEMA_VERSION,
            "region": self._settings.region,
            "endpoint": self._settings.endpoint,
            "project_name": self._settings.project_name,
            "feature_entity_name": self._settings.feature_entity_name,
            "feature_view_name": self._settings.feature_view_name,
            "feature_view_ttl_secs": self._settings.feature_view_ttl_secs,
            "feature_view_shard_count": self._settings.feature_view_shard_count,
            "feature_view_replication_count": (
                self._settings.feature_view_replication_count
            ),
            "version": self._settings.version,
            "version_initialization": _VERSION_INITIALIZATION,
            "feature_view_provisioning": "CHECK_OR_CREATE_DYNAMIC_EMBEDDING",
            "writer_ownership": "SINGLE_WRITER_PER_VERSION_REQUIRED",
            "publish_semantics": "MONOTONIC_TS_RANGE_PER_ATTEMPT_FULL_REPLAY",
            "consistency": "ROW_LEVEL_EVENTUAL",
            "step_atomicity": False,
            "outbox_snapshot": "PRIVATE_READ_ONLY_CONTENT_VERIFIED",
            "canonical_outbox_scope": "FEATURE_STORE_TARGET_HASH",
            "dump_generation": "ONE_SHARED_TOKEN_PER_GLOBAL_STEP_SHARD_SET",
            "minimum_global_step": 1,
            "world_size": self._world_size,
            "file_prefix": self._file_prefix,
            "upload_batch_size": self._settings.upload_batch_size,
            "pk_field": FEATURE_STORE_PK_FIELD,
            "sk_field": FEATURE_STORE_SK_FIELD,
            "value_field": FEATURE_STORE_VALUE_FIELD,
            "key_dtype": "INT64",
            "dynamic_key_encoding": "UINT64_BIT_PATTERN_IN_SIGNED_INT64",
            "value_dtype": "FLOAT32",
            "operation": DELTA_OPERATION_UPSERT,
            "invalid_key": SPARSE_EMBEDDING_INVALID_KEY,
            "embedding_dimensions": dict(sorted(self._embedding_dimensions.items())),
        }
        contract_bytes = json.dumps(
            contract, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        self._contract = contract
        self._contract_hash = hashlib.sha256(contract_bytes).hexdigest()
        self._state_dir = _feature_store_state_dir(self._settings, self._output_dir)
        # The lock file needs its directory to exist. Directory creation itself
        # contains no remote-write state, but every new parent entry is fsynced so
        # a later durable attempt journal cannot disappear after a host crash.
        _durable_makedirs(self._state_dir)
        self._snapshot_root = os.path.join(self._state_dir, "snapshots")
        self._error_marker_path = os.path.join(self._state_dir, "last_error.json")
        self._writer_lock: Optional[BinaryIO] = None
        self._contract_path = os.path.join(self._state_dir, "contract.json")
        self._journal_initialized = False
        self._committed_global_step = -1
        self._last_publish_ts = 0

    @property
    def state_dir(self) -> str:
        """Directory containing credential-free manifests and success markers."""
        return self._state_dir

    def start(self) -> None:
        """Start the single background worker and discover restart work."""
        with self._condition:
            if self._started:
                return
            if self._closed:
                raise RuntimeError("FeatureStoreDeltaUploader is already closed")
            self._raise_if_failed_locked()
            self._acquire_writer_lock()
            try:
                self._initialize_journal()
                self._cleanup_stale_snapshot_staging()
                self._cleanup_committed_snapshots()
                self._add_discovered_steps_locked()
                # Validate or provision the remote DynamicEmbedding FeatureView
                # synchronously. Training must not start with an incompatible target.
                self._get_view()
                self._clear_error_marker()
                self._started = True
                self._worker = threading.Thread(
                    target=self._run,
                    name="tzrec-feature-store-delta-uploader",
                    daemon=True,
                )
                self._worker.start()
            except BaseException:
                self._started = False
                # A later retry must reconcile again after reacquiring the lock;
                # another process may have advanced the journal in between.
                self._journal_initialized = False
                self._reconciled_steps = set()
                self._reset_view(suppress_errors=True)
                self._release_writer_lock()
                raise
            logger.info(
                "FeatureStore delta uploader started: project=%s feature_view=%s "
                "version=%s",
                self._settings.project_name,
                self._settings.feature_view_name,
                self._settings.version,
            )

    def submit(self, global_step: int, dump_generation: Optional[str] = None) -> None:
        """Enqueue a durably written rank-zero shard with bounded back-pressure."""
        global_step = int(global_step)
        if global_step <= 0:
            raise ValueError("FeatureStore delta global_step must be > 0")
        if dump_generation is not None and not dump_generation:
            raise ValueError("FeatureStore delta dump_generation must not be empty")
        with self._condition:
            self._raise_if_failed_locked()
            if not self._started:
                raise RuntimeError(
                    "FeatureStoreDeltaUploader.start() must be called before submit()"
                )
            if self._closing or self._closed:
                raise RuntimeError("cannot submit to a closing FeatureStore uploader")
            if self._is_committed(global_step, dump_generation):
                return
            while (
                global_step not in self._pending
                and len(self._pending) >= self._settings.max_pending_steps
            ):
                self._condition.wait(self._settings.poll_interval_secs)
                self._raise_if_failed_locked()
            self._pending.setdefault(global_step, time.monotonic())
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
                    "FeatureStore uploader did not drain before shutdown timeout; "
                    "parquet outbox files were retained for restart"
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
                    self._add_discovered_steps_locked()
                    if not self._pending:
                        if self._closing:
                            return
                        self._condition.wait(self._settings.poll_interval_secs)
                        continue
                    current_step = min(self._pending)
                    pending_since = self._pending[current_step]

                canonical_paths = self._expected_shard_paths(current_step)
                manifest_exists = os.path.isfile(self._manifest_path(current_step))
                snapshot_paths = self._snapshot_paths(current_step)
                snapshot_dir_exists = os.path.isdir(self._snapshot_dir(current_step))
                if snapshot_dir_exists and self._is_committed(current_step):
                    try:
                        canonical_generation = self._canonical_dump_generation(
                            current_step
                        )
                    except _ShardSetNotReady:
                        canonical_generation = None
                        snapshot_paths = None
                    manifest_generation = _read_json(
                        self._manifest_path(current_step)
                    ).get("dump_generation")
                    if (
                        canonical_generation is not None
                        and canonical_generation != manifest_generation
                    ):
                        self._reclaim_snapshot(current_step)
                        snapshot_dir_exists = os.path.isdir(
                            self._snapshot_dir(current_step)
                        )
                        if snapshot_dir_exists:
                            raise FeatureStoreUploadError(
                                "cannot replace a committed FeatureStore shard "
                                "snapshot with a newer dump generation"
                            )
                if snapshot_paths is not None:
                    if snapshot_dir_exists:
                        if not all(os.path.isfile(path) for path in snapshot_paths):
                            raise FeatureStoreUploadError(
                                "an uncommitted FeatureStore upload journal is "
                                "missing its durable shard snapshot; recovery "
                                "cannot continue"
                            )
                    elif manifest_exists and not self._is_committed(current_step):
                        raise FeatureStoreUploadError(
                            "an uncommitted FeatureStore upload journal is missing "
                            "its durable shard snapshot; recovery cannot continue"
                        )
                    else:
                        try:
                            snapshot_paths = self._snapshot_canonical_shards(
                                current_step, canonical_paths
                            )
                        except _ShardSetNotReady:
                            snapshot_paths = None

                if snapshot_paths is None:
                    elapsed = time.monotonic() - pending_since
                    if elapsed >= self._settings.shard_wait_timeout_secs:
                        raise TimeoutError(
                            "timed out waiting for a complete same-generation "
                            "delta shard set"
                        )
                    with self._condition:
                        self._condition.wait(self._settings.poll_interval_secs)
                    continue

                with self._condition:
                    if self._aborting:
                        return
                # Hash, parse and upload only uploader-owned read-only snapshots.
                # The canonical dump files may be atomically replaced or cleaned
                # after this point without changing restart/replay semantics.
                with ExitStack() as stack:
                    shard_sources = [
                        stack.enter_context(open(path, "rb")) for path in snapshot_paths
                    ]
                    shard_descriptions = self._describe_shards(
                        snapshot_paths, shard_sources
                    )
                    self._validate_dump_generation(shard_descriptions)
                    record_store = stack.enter_context(
                        self._build_record_store(
                            current_step, snapshot_paths, shard_sources
                        )
                    )
                    manifest = self._load_or_create_manifest(
                        current_step,
                        snapshot_paths,
                        record_store.record_count,
                        shard_descriptions=shard_descriptions,
                    )
                    summary = self._upload_with_retries(
                        current_step, record_store, manifest
                    )
                    # Rehashing can scan large shards. Keep it outside the
                    # condition so close(drain=False) can publish the abort bit
                    # immediately; the small durable commit remains serialized.
                    self._validate_shard_identities(
                        snapshot_paths, manifest["shards"], shard_sources
                    )
                    with self._condition:
                        if self._aborting:
                            return
                        # Keep the condition locked through the local commit so
                        # aborting close cannot return before success-state writes.
                        self._commit_success(current_step, manifest, summary)
                        self._pending.pop(current_step, None)
                        self._condition.notify_all()
                self._reclaim_snapshot(current_step)
                current_step = None
        except _UploadAborted:
            return
        except BaseException as exc:
            step_context = (
                f" at global_step={current_step}" if current_step is not None else ""
            )
            if isinstance(exc, FeatureStoreUploadError):
                safe_error = FeatureStoreUploadError(
                    f"{exc}{step_context}; parquet outbox files were retained"
                )
            else:
                safe_error = FeatureStoreUploadError(
                    f"FeatureStore delta upload failed{step_context} "
                    f"({type(exc).__name__}); parquet outbox files were retained"
                )
            try:
                _atomic_write_json(
                    self._error_marker_path,
                    {
                        "schema_version": 1,
                        "project_name": self._settings.project_name,
                        "feature_view_name": self._settings.feature_view_name,
                        "version": self._settings.version,
                        "contract_hash": self._contract_hash,
                        "error": str(safe_error),
                    },
                )
            except BaseException as marker_error:
                logger.error(
                    "Failed to persist FeatureStore upload failure marker (%s).",
                    type(marker_error).__name__,
                )
            with self._condition:
                self._error = safe_error
                self._condition.notify_all()
            logger.error("%s", safe_error)
        finally:
            self._reset_view(suppress_errors=True)
            self._release_writer_lock()

    def _acquire_writer_lock(self) -> None:
        if self._writer_lock is not None:
            return
        lock_path = os.path.join(self._state_dir, "writer.lock")
        lock_file = open(lock_path, "a+b")
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            lock_file.close()
            raise FeatureStoreUploadError(
                "another process is already publishing this FeatureStore target "
                "from the configured output_dir"
            ) from exc
        self._writer_lock = lock_file

    def _release_writer_lock(self) -> None:
        lock_file = self._writer_lock
        self._writer_lock = None
        if lock_file is None:
            return
        if self._writer_quiescence_failed:
            # close(wait=True) is the only SDK proof that no asynchronous write
            # remains in flight. If it fails, retain the flock until process exit
            # so another local writer cannot overlap an indeterminate old request.
            _POISONED_WRITER_LOCKS.append(lock_file)
            logger.error(
                "FeatureStore writer lock retained until process exit because "
                "SDK writer quiescence could not be confirmed."
            )
            return
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        finally:
            lock_file.close()

    def _initialize_journal(self) -> None:
        """Validate and reload all timestamp fences while holding the writer lock."""
        if self._writer_lock is None:
            raise RuntimeError(
                "FeatureStore upload journal must be initialized under writer lock"
            )
        if os.path.isfile(self._contract_path):
            stored_contract = _read_json(self._contract_path)
            if stored_contract != self._contract:
                legacy_contract = dict(self._contract)
                legacy_contract["version_initialization"] = (
                    _LEGACY_VERSION_INITIALIZATION
                )
                if stored_contract == legacy_contract:
                    # Preserve the old hash so existing manifests and success
                    # markers remain valid after removing the version precheck.
                    self._contract = stored_contract
                    self._contract_hash = _json_digest(stored_contract)
                else:
                    raise ValueError(
                        "FeatureStore upload contract changed for an existing remote "
                        "target; provision and configure a new immutable version"
                    )
        else:
            _atomic_write_json(self._contract_path, self._contract)

        # A previously constructed uploader may have waited behind another
        # process. Always reconcile and reload after flock acquisition so it
        # cannot allocate timestamps from stale constructor-time state.
        self._reconcile_committed_state()
        self._committed_global_step = self._load_committed_global_step()
        self._last_publish_ts = self._load_latest_publish_ts()
        self._journal_initialized = True

    def _cleanup_stale_snapshot_staging(self) -> None:
        if not os.path.isdir(self._snapshot_root):
            return
        pattern = re.compile(r"^step_\d+\.tmp-")
        for name in os.listdir(self._snapshot_root):
            path = os.path.join(self._snapshot_root, name)
            if pattern.match(name) and os.path.isdir(path):
                shutil.rmtree(path)
                _fsync_parent_directory(path)

    def _cleanup_committed_snapshots(self) -> None:
        """Retry best-effort reclamation left behind after a committed upload."""
        if not os.path.isdir(self._snapshot_root):
            return
        pattern = re.compile(r"^step_(\d+)$")
        for name in os.listdir(self._snapshot_root):
            match = pattern.match(name)
            if match is None:
                continue
            global_step = int(match.group(1))
            if self._is_committed(global_step):
                self._reclaim_snapshot(global_step)

    def _clear_error_marker(self) -> None:
        if not os.path.exists(self._error_marker_path):
            return
        os.remove(self._error_marker_path)
        _fsync_parent_directory(self._error_marker_path)

    def _raise_if_failed_locked(self) -> None:
        if self._error is not None:
            raise self._error

    def _raise_if_aborting(self) -> None:
        with self._condition:
            if self._aborting:
                raise _UploadAborted()

    def _add_discovered_steps_locked(self) -> None:
        """Incrementally enqueue new outbox steps without rescanning history.

        Full reconciliation happens once in ``start()`` with an empty
        reconciled set. Afterwards, steps already pending or already verified
        committed with a matching generation are skipped without any JSON or
        Parquet reads, so the condition lock hold time stays independent of
        how many steps this job has committed. A committed step whose
        canonical generation no longer matches its manifest is deliberately
        never marked reconciled, keeping the replacement alarm alive.
        """
        remaining_capacity = self._settings.max_pending_steps - len(self._pending)
        if remaining_capacity <= 0:
            return
        for step in sorted(self._discover_steps()):
            if remaining_capacity <= 0:
                break
            if step <= 0:
                raise ValueError(
                    "found invalid FeatureStore delta outbox global_step; "
                    "global_step must be > 0"
                )
            if step in self._reconciled_steps or step in self._pending:
                continue
            if self._is_committed(step):
                try:
                    canonical_generation = self._canonical_dump_generation(step)
                except _ShardSetNotReady:
                    canonical_generation = ""
                if canonical_generation is None:
                    self._reconciled_steps.add(step)
                    continue
                manifest_generation = _read_json(self._manifest_path(step)).get(
                    "dump_generation"
                )
                if canonical_generation == manifest_generation:
                    self._reconciled_steps.add(step)
                    continue
            if step not in self._pending:
                self._pending[step] = time.monotonic()
                remaining_capacity -= 1

    def _discover_steps(self) -> Iterable[int]:
        if not os.path.isdir(self._output_dir):
            return []
        steps = set()
        # Already pending or reconciled steps are known to be real entries;
        # re-statting their directories/shards on every poll would make the
        # scan cost grow with the retained outbox history.
        known_steps = self._reconciled_steps | set(self._pending)
        manifest_pattern = re.compile(r"^step_(\d+)\.manifest\.json$")
        for name in os.listdir(self._state_dir):
            match = manifest_pattern.match(name)
            if match:
                steps.add(int(match.group(1)))
        if os.path.isdir(self._snapshot_root):
            snapshot_pattern = re.compile(r"^step_(\d+)$")
            for name in os.listdir(self._snapshot_root):
                match = snapshot_pattern.match(name)
                if match is None:
                    continue
                step = int(match.group(1))
                if step in known_steps or os.path.isdir(
                    os.path.join(self._snapshot_root, name)
                ):
                    steps.add(step)
        if self._world_size == 1:
            pattern = re.compile(
                rf"^{re.escape(self._file_prefix)}_step_(\d+)\.parquet$"
            )
            for name in os.listdir(self._output_dir):
                match = pattern.match(name)
                if match:
                    steps.add(int(match.group(1)))
        else:
            pattern = re.compile(r"^step_(\d+)$")
            for name in os.listdir(self._output_dir):
                match = pattern.match(name)
                if match is None:
                    continue
                step = int(match.group(1))
                if step in known_steps:
                    steps.add(step)
                    continue
                if not os.path.isdir(os.path.join(self._output_dir, name)):
                    continue
                # A partial shard set for this exact contract is real pending
                # work and must time out fail-safe. Empty dirs and shards from
                # another prefix/world-size contract are ignored.
                if any(
                    os.path.isfile(path) for path in self._expected_shard_paths(step)
                ):
                    steps.add(step)
        return steps

    def _snapshot_dir(self, global_step: int) -> str:
        return os.path.join(self._snapshot_root, f"step_{global_step}")

    def _snapshot_paths(self, global_step: int) -> List[str]:
        snapshot_dir = self._snapshot_dir(global_step)
        return [
            os.path.join(snapshot_dir, f"rank_{rank}.parquet")
            for rank in range(self._world_size)
        ]

    def _expected_shard_paths(self, global_step: int) -> List[str]:
        if self._world_size == 1:
            return [
                os.path.join(
                    self._output_dir,
                    f"{self._file_prefix}_step_{global_step}.parquet",
                )
            ]
        step_dir = os.path.join(self._output_dir, f"step_{global_step}")
        return [
            os.path.join(
                step_dir,
                f"{self._file_prefix}_step_{global_step}_rank_{rank}"
                f"_of_{self._world_size}.parquet",
            )
            for rank in range(self._world_size)
        ]

    def _canonical_dump_generation(self, global_step: int) -> Optional[str]:
        """Read one complete canonical shard set's generation without hashing it."""
        paths = self._expected_shard_paths(global_step)
        existing = [os.path.isfile(path) for path in paths]
        if not any(existing):
            return None
        if not all(existing):
            raise _ShardSetNotReady("canonical delta shard set is incomplete")

        generations = set()
        for path in paths:
            metadata = pq.read_schema(path).metadata or {}
            if metadata.get(
                _SCHEMA_VERSION_METADATA_KEY
            ) != DELTA_DUMP_SCHEMA_VERSION.encode("ascii"):
                raise ValueError(f"unsupported delta dump schema version in {path}")
            generation = metadata.get(DELTA_DUMP_GENERATION_METADATA_KEY)
            if not generation:
                raise ValueError(f"delta dump generation is missing in {path}")
            try:
                generations.add(generation.decode("ascii"))
            except UnicodeDecodeError as exc:
                raise ValueError(
                    f"delta dump generation is not ASCII in {path}"
                ) from exc
        if len(generations) != 1:
            raise _ShardSetNotReady(
                "delta shards from different dump generations are present"
            )
        return next(iter(generations))

    def _manifest_path(self, global_step: int) -> str:
        return os.path.join(self._state_dir, f"step_{global_step}.manifest.json")

    def _success_path(self, global_step: int) -> str:
        return os.path.join(self._state_dir, f"step_{global_step}._FS_SUCCESS.json")

    def _load_success_state(
        self, global_step: int
    ) -> Optional[Tuple[Dict[str, Any], Dict[str, Any], bool]]:
        """Validate one success marker and report whether its manifest is active."""
        if global_step <= 0:
            raise ValueError("FeatureStore delta global_step must be > 0")
        path = self._success_path(global_step)
        if not os.path.isfile(path):
            return None
        success = _read_json(path)
        expected = {
            "global_step": global_step,
            "version": self._settings.version,
            "contract_hash": self._contract_hash,
        }
        for name, value in expected.items():
            if success.get(name) != value:
                raise ValueError(f"FeatureStore success marker mismatch for {name}")
        publish_ts = success.get("publish_ts")
        if type(publish_ts) is not int or publish_ts <= 0:
            raise ValueError("FeatureStore success marker has invalid publish_ts")
        manifest_path = self._manifest_path(global_step)
        if not os.path.isfile(manifest_path):
            raise ValueError("FeatureStore success marker is missing its manifest")
        manifest = _read_json(manifest_path)
        shards = success.get("shards")
        if not isinstance(shards, list):
            raise ValueError("FeatureStore success marker has invalid shards")
        success_manifest_digest = success.get("manifest_digest")
        if not isinstance(success_manifest_digest, str) or not success_manifest_digest:
            raise ValueError("FeatureStore success marker has invalid manifest_digest")
        success_dump_generation = self._validate_dump_generation(shards)
        if success.get("dump_generation", success_dump_generation) != (
            success_dump_generation
        ):
            raise ValueError("FeatureStore success marker has invalid dump_generation")
        if success_manifest_digest == _json_digest(manifest):
            if (
                shards != manifest.get("shards")
                or manifest.get("dump_generation") != success_dump_generation
            ):
                raise ValueError("FeatureStore success marker has invalid shards")
            return success, manifest, True

        if (
            manifest.get("supersedes_manifest_digest") != success_manifest_digest
            or manifest.get("supersedes_publish_ts") != publish_ts
            or manifest.get("supersedes_dump_generation") != success_dump_generation
        ):
            raise ValueError("FeatureStore success marker manifest digest mismatch")
        if manifest.get("dump_generation") == manifest.get(
            "supersedes_dump_generation"
        ):
            raise ValueError("FeatureStore superseding manifest generation is invalid")
        return success, manifest, False

    def _is_committed(
        self, global_step: int, dump_generation: Optional[str] = None
    ) -> bool:
        state = self._load_success_state(global_step)
        if state is None:
            return False
        _, manifest, active = state
        if not active:
            return False
        return dump_generation is None or (
            manifest.get("dump_generation") == dump_generation
        )

    def _reconcile_committed_state(self) -> None:
        """Repair a crash between atomic success and committed-state writes."""
        pattern = re.compile(r"^step_(\d+)\._FS_SUCCESS\.json$")
        latest_success: Optional[Dict[str, Any]] = None
        for name in os.listdir(self._state_dir):
            match = pattern.match(name)
            if match is None:
                continue
            step = int(match.group(1))
            state = self._load_success_state(step)
            if state is None:
                continue
            success, _, _ = state
            if latest_success is None or int(success["publish_ts"]) > int(
                latest_success["publish_ts"]
            ):
                latest_success = success
        if latest_success is None:
            return

        committed_path = os.path.join(self._state_dir, "committed.json")
        committed_publish_ts = 0
        if os.path.isfile(committed_path):
            committed_publish_ts = int(_read_json(committed_path).get("publish_ts", 0))
        latest_step = int(latest_success["global_step"])
        if committed_publish_ts >= int(latest_success["publish_ts"]):
            return
        committed = {
            "schema_version": 1,
            "project_name": self._settings.project_name,
            "feature_view_name": self._settings.feature_view_name,
            "version": self._settings.version,
            "committed_global_step": latest_step,
            "publish_ts": int(latest_success["publish_ts"]),
            "contract_hash": self._contract_hash,
            "dump_generation": latest_success.get("dump_generation")
            or self._validate_dump_generation(latest_success["shards"]),
            "manifest_digest": latest_success["manifest_digest"],
        }
        _atomic_write_json(committed_path, committed)

    def _load_latest_publish_ts(self) -> int:
        latest = 0
        committed_path = os.path.join(self._state_dir, "committed.json")
        if os.path.isfile(committed_path):
            latest = max(latest, int(_read_json(committed_path).get("publish_ts", 0)))
        if os.path.isdir(self._state_dir):
            for name in os.listdir(self._state_dir):
                if not name.endswith(".manifest.json"):
                    continue
                manifest = _read_json(os.path.join(self._state_dir, name))
                latest = max(latest, int(manifest.get("supersedes_publish_ts", 0)))
                for attempt in manifest.get("attempts", []):
                    latest = max(latest, int(attempt.get("range_end", 0)))
        return latest

    def _load_committed_global_step(self) -> int:
        path = os.path.join(self._state_dir, "committed.json")
        if not os.path.isfile(path):
            return -1
        global_step = int(_read_json(path).get("committed_global_step", -1))
        if global_step == 0 or global_step < -1:
            raise ValueError(
                "FeatureStore committed global_step must be -1 or greater than 0"
            )
        return global_step

    @staticmethod
    def _file_identity(stat_result: os.stat_result) -> Dict[str, int]:
        return {
            "device": int(stat_result.st_dev),
            "inode": int(stat_result.st_ino),
            "size_bytes": int(stat_result.st_size),
            "mtime_ns": int(stat_result.st_mtime_ns),
            "ctime_ns": int(stat_result.st_ctime_ns),
        }

    def _snapshot_canonical_shards(
        self, global_step: int, canonical_paths: List[str]
    ) -> Optional[List[str]]:
        """Copy one stable, same-generation shard set into the recovery journal."""
        if not all(os.path.isfile(path) for path in canonical_paths):
            return None
        _durable_makedirs(self._snapshot_root)
        snapshot_dir = self._snapshot_dir(global_step)
        if os.path.isdir(snapshot_dir):
            snapshot_paths = self._snapshot_paths(global_step)
            if not all(os.path.isfile(path) for path in snapshot_paths):
                raise FeatureStoreUploadError(
                    "durable FeatureStore shard snapshot is incomplete"
                )
            return snapshot_paths

        staging_dir = f"{snapshot_dir}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
        os.mkdir(staging_dir)
        _fsync_parent_directory(staging_dir)
        staging_paths = [
            os.path.join(staging_dir, f"rank_{rank}.parquet")
            for rank in range(self._world_size)
        ]
        try:
            for source_path, snapshot_path in zip(canonical_paths, staging_paths):
                with open(source_path, "rb") as source:
                    before_identity = self._file_identity(os.fstat(source.fileno()))
                    with open(snapshot_path, "xb") as output:
                        source.seek(0)
                        shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
                        output.flush()
                        os.fsync(output.fileno())
                    after_identity = self._file_identity(os.fstat(source.fileno()))
                    if before_identity != after_identity:
                        raise _ShardSetNotReady(
                            "canonical shard changed while creating snapshot"
                        )
                os.chmod(snapshot_path, 0o400)

            # Validate the generation before atomically publishing the directory.
            descriptions = self._describe_shards(staging_paths)
            self._validate_dump_generation(descriptions)
            _fsync_parent_directory(staging_paths[0])
            os.replace(staging_dir, snapshot_dir)
            _fsync_parent_directory(snapshot_dir)
            return self._snapshot_paths(global_step)
        except BaseException:
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise

    def _describe_shards(
        self,
        shard_paths: List[str],
        shard_sources: Optional[List[BinaryIO]] = None,
    ) -> List[Dict[str, Any]]:
        if shard_sources is None:
            with ExitStack() as stack:
                opened_sources = [
                    stack.enter_context(open(path, "rb")) for path in shard_paths
                ]
                return self._describe_shards(shard_paths, opened_sources)
        if len(shard_paths) != len(shard_sources):
            raise ValueError("delta shard paths and sources must have equal length")

        descriptions = []
        for path, source in zip(shard_paths, shard_sources):
            before_identity = self._file_identity(os.fstat(source.fileno()))
            source.seek(0)
            parquet_file = pq.ParquetFile(source)
            num_rows = int(parquet_file.metadata.num_rows)
            metadata = parquet_file.schema_arrow.metadata or {}
            schema_version = metadata.get(_SCHEMA_VERSION_METADATA_KEY)
            dump_generation = metadata.get(DELTA_DUMP_GENERATION_METADATA_KEY)
            if schema_version != DELTA_DUMP_SCHEMA_VERSION.encode("ascii"):
                raise ValueError(f"unsupported delta dump schema version in {path}")
            if not dump_generation:
                raise ValueError(f"delta dump generation is missing in {path}")
            try:
                dump_generation_text = dump_generation.decode("ascii")
            except UnicodeDecodeError as exc:
                raise ValueError(
                    f"delta dump generation is not ASCII in {path}"
                ) from exc
            sha256 = _file_sha256(source)
            after_identity = self._file_identity(os.fstat(source.fileno()))
            try:
                path_identity = self._file_identity(os.stat(path))
            except OSError as exc:
                raise RuntimeError(
                    "delta shard snapshot path changed while being described"
                ) from exc
            if before_identity != after_identity or path_identity != after_identity:
                raise RuntimeError("delta shard changed while building upload manifest")
            descriptions.append(
                {
                    "path": os.path.relpath(path, self._output_dir),
                    "size_bytes": after_identity["size_bytes"],
                    "num_rows": num_rows,
                    "sha256": sha256,
                    "delta_dump_schema_version": DELTA_DUMP_SCHEMA_VERSION,
                    "dump_generation": dump_generation_text,
                }
            )
        return descriptions

    @staticmethod
    def _validate_dump_generation(descriptions: List[Dict[str, Any]]) -> str:
        generations = {
            description.get("dump_generation") for description in descriptions
        }
        if None in generations or "" in generations:
            raise ValueError("delta shard set has invalid dump generation metadata")
        if len(generations) != 1:
            raise _ShardSetNotReady(
                "delta shards from different dump generations are present"
            )
        return str(next(iter(generations)))

    def _validate_shard_identities(
        self,
        shard_paths: List[str],
        descriptions: List[Dict[str, Any]],
        shard_sources: Optional[List[BinaryIO]] = None,
    ) -> None:
        if shard_sources is None:
            with ExitStack() as stack:
                opened_sources = [
                    stack.enter_context(open(path, "rb")) for path in shard_paths
                ]
                self._validate_shard_identities(
                    shard_paths, descriptions, opened_sources
                )
                return
        if len(shard_paths) != len(descriptions) or (
            len(shard_paths) != len(shard_sources)
        ):
            raise ValueError("FeatureStore shard identity count mismatch")
        for index, (path, description) in enumerate(zip(shard_paths, descriptions)):
            expected_path = os.path.relpath(path, self._output_dir)
            if description.get("path") != expected_path:
                raise ValueError("FeatureStore shard identity metadata is invalid")
            try:
                source = shard_sources[index]
                before_identity = self._file_identity(os.fstat(source.fileno()))
                path_identity = self._file_identity(os.stat(path))
                sha256 = _file_sha256(source)
                after_identity = self._file_identity(os.fstat(source.fileno()))
            except OSError as exc:
                raise RuntimeError(
                    "delta shard changed after upload snapshot was claimed"
                ) from exc
            if (
                before_identity != after_identity
                or path_identity != after_identity
                or after_identity["size_bytes"] != description.get("size_bytes")
                or sha256 != description.get("sha256")
            ):
                raise RuntimeError(
                    "delta shard changed after upload snapshot was claimed"
                )

    def _load_or_create_manifest(
        self,
        global_step: int,
        shard_paths: List[str],
        record_count: int,
        shard_descriptions: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        if not self._journal_initialized or self._writer_lock is None:
            raise RuntimeError(
                "FeatureStore upload manifests may only be changed under the "
                "initialized writer journal lock"
            )
        path = self._manifest_path(global_step)
        expected_shards = (
            self._describe_shards(shard_paths)
            if shard_descriptions is None
            else shard_descriptions
        )
        dump_generation = self._validate_dump_generation(expected_shards)
        if os.path.isfile(path):
            manifest = _read_json(path)
            expected = {
                "schema_version": 3,
                "global_step": global_step,
                "world_size": self._world_size,
                "version": self._settings.version,
                "contract_hash": self._contract_hash,
                "record_count": record_count,
                "dump_generation": dump_generation,
                "shards": expected_shards,
            }
            mismatch = next(
                (
                    name
                    for name, value in expected.items()
                    if manifest.get(name) != value
                ),
                None,
            )
            if mismatch is None:
                attempts = manifest.get("attempts")
                if not isinstance(attempts, list):
                    raise ValueError(
                        "FeatureStore upload manifest has invalid attempts"
                    )
                for attempt in attempts:
                    if (
                        not isinstance(attempt, dict)
                        or type(attempt.get("range_start")) is not int
                        or type(attempt.get("range_end")) is not int
                        or attempt["range_start"] <= 0
                        or attempt["range_end"] < attempt["range_start"]
                    ):
                        raise ValueError(
                            "FeatureStore upload manifest has an invalid ts range"
                        )
                    self._last_publish_ts = max(
                        self._last_publish_ts, int(attempt["range_end"])
                    )
                return manifest

            if manifest.get(
                "dump_generation"
            ) == dump_generation or not self._is_committed(global_step):
                raise ValueError(
                    f"FeatureStore upload manifest mismatch for {mismatch}"
                )

            success_state = self._load_success_state(global_step)
            if success_state is None or not success_state[2]:
                raise ValueError(
                    "FeatureStore committed manifest cannot be superseded safely"
                )
            # Keep the old success verifiable across a crash during replacement.
            success, previous_manifest, _ = success_state
            manifest = {
                "schema_version": 3,
                "global_step": global_step,
                "world_size": self._world_size,
                "project_name": self._settings.project_name,
                "feature_view_name": self._settings.feature_view_name,
                "version": self._settings.version,
                "write_mode": FEATURE_STORE_WRITE_MODE,
                "contract_hash": self._contract_hash,
                "record_count": record_count,
                "dump_generation": dump_generation,
                "shards": expected_shards,
                "attempts": [],
                "supersedes_manifest_digest": success["manifest_digest"],
                "supersedes_publish_ts": int(success["publish_ts"]),
                "supersedes_dump_generation": previous_manifest["dump_generation"],
            }
            _atomic_write_json(path, manifest)
            return manifest

        manifest = {
            "schema_version": 3,
            "global_step": global_step,
            "world_size": self._world_size,
            "project_name": self._settings.project_name,
            "feature_view_name": self._settings.feature_view_name,
            "version": self._settings.version,
            "write_mode": FEATURE_STORE_WRITE_MODE,
            "contract_hash": self._contract_hash,
            "record_count": record_count,
            "dump_generation": dump_generation,
            "shards": expected_shards,
            "attempts": [],
        }
        _atomic_write_json(path, manifest)
        return manifest

    def _start_attempt(
        self,
        global_step: int,
        record_count: int,
        manifest: Dict[str, Any],
    ) -> Dict[str, int]:
        if not self._journal_initialized or self._writer_lock is None:
            raise RuntimeError(
                "FeatureStore timestamp ranges may only be reserved under the "
                "initialized writer journal lock"
            )
        batch_count = (
            record_count + self._settings.upload_batch_size - 1
        ) // self._settings.upload_batch_size
        reserved_count = max(batch_count, 1)
        range_start = max(int(self._clock_ms()), self._last_publish_ts + 1, 1)
        attempt = {
            "attempt_id": len(manifest["attempts"]) + 1,
            "record_count": record_count,
            "batch_count": batch_count,
            "range_start": range_start,
            "range_end": range_start + reserved_count - 1,
        }
        # Persist the whole range before any remote request. After a partial
        # write or process crash, restart allocates a newer range and replays
        # the entire step so a Processor Next-Ts cursor cannot miss late rows.
        manifest["attempts"].append(attempt)
        _atomic_write_json(self._manifest_path(global_step), manifest)
        self._last_publish_ts = int(attempt["range_end"])
        return attempt

    def _upload_with_retries(
        self,
        global_step: int,
        store: _RecordStore,
        manifest: Dict[str, Any],
    ) -> Dict[str, int]:
        for local_attempt in range(1, self._settings.max_retries + 1):
            self._raise_if_aborting()
            attempt = self._start_attempt(global_step, store.record_count, manifest)
            try:
                if store.record_count:
                    summary = self._upload_records(global_step, store, attempt)
                else:
                    # An empty step still validates the FeatureView schema before
                    # it can be committed.
                    self._get_view()
                    summary = {
                        "total_batches": 0,
                        "failed_batches": 0,
                        "total_records": 0,
                        "success_records": 0,
                        "failed_records": 0,
                    }
                summary.update(attempt)
                return summary
            except _UploadAborted:
                raise
            except BaseException as exc:
                self._reset_view()
                if local_attempt >= self._settings.max_retries:
                    raise
                logger.warning(
                    "FeatureStore delta upload attempt %s/%s failed at step %s "
                    "(%s); replaying the full step with a newer persisted ts range.",
                    local_attempt,
                    self._settings.max_retries,
                    global_step,
                    type(exc).__name__,
                )
                if self._settings.retry_backoff_secs > 0:
                    with self._condition:
                        if self._aborting:
                            raise _UploadAborted() from None
                        self._condition.wait(
                            self._settings.retry_backoff_secs * local_attempt
                        )
                        if self._aborting:
                            raise _UploadAborted() from None
        raise AssertionError("unreachable FeatureStore retry state")

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

    def _get_or_create_view(self, project: Any) -> Any:
        """Return the configured DynamicEmbedding view, creating it if absent."""
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
                # Another writer may have won the create race, or control-plane
                # creation may have succeeded before SDK data-plane initialization
                # failed. Re-get before declaring the operation failed.
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
            # The specialized view owns the FeatureDB writer. Retain it before
            # reporting a control-plane validation failure so start() can close
            # it through the normal reset path.
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

        kwargs = {
            "access_key_id": self._settings.access_key_id,
            "access_key_secret": self._settings.access_key_secret,
            "region": self._settings.region or None,
            "endpoint": self._settings.endpoint or None,
            "security_token": self._settings.security_token or None,
            "featuredb_username": self._settings.featuredb_username or None,
            "featuredb_password": self._settings.featuredb_password or None,
        }
        client = client_factory(**kwargs)
        project = client.get_project(self._settings.project_name)
        if project is None:
            raise RuntimeError("configured FeatureStore project was not found")
        view = self._get_or_create_view(project)
        # Own the SDK writer before validating the remote contract so every
        # failure path is closed by the retry/reset logic.
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
                self._writer_quiescence_failed = True
                close_error = FeatureStoreUploadError(
                    "FeatureStore SDK writer close failed; retry was aborted to "
                    "avoid overlapping requests"
                )
                if not suppress_errors:
                    raise close_error from None

                with self._condition:
                    new_error = self._error is None
                    if new_error:
                        self._error = close_error
                    self._condition.notify_all()
                if new_error:
                    try:
                        _atomic_write_json(
                            self._error_marker_path,
                            {
                                "schema_version": 1,
                                "project_name": self._settings.project_name,
                                "feature_view_name": self._settings.feature_view_name,
                                "version": self._settings.version,
                                "contract_hash": self._contract_hash,
                                "error": str(close_error),
                            },
                        )
                    except BaseException as marker_error:
                        logger.error(
                            "Failed to persist FeatureStore upload failure marker "
                            "after SDK close failure (%s).",
                            type(marker_error).__name__,
                        )
                logger.error(
                    "Failed to close FeatureStore SDK writer cleanly (%s); the "
                    "local writer lock will be retained until process exit.",
                    type(exc).__name__,
                )

    def _upload_records(
        self,
        global_step: int,
        store: _RecordStore,
        attempt: Mapping[str, int],
    ) -> Dict[str, int]:
        view = self._get_view()
        max_in_flight = int(getattr(view, "_max_workers", 1))
        total_records = store.record_count
        batch_count = (
            total_records + self._settings.upload_batch_size - 1
        ) // self._settings.upload_batch_size
        aggregate = {
            "total_batches": 0,
            "failed_batches": 0,
            "total_records": 0,
            "success_records": 0,
            "failed_records": 0,
        }
        started_at = time.monotonic()
        next_progress_batch = _FEATURE_STORE_PROGRESS_LOG_INTERVAL_BATCHES
        logger.info(
            "FeatureStore delta upload started: step=%s attempt=%s version=%s "
            "records=%s batches=%s max_in_flight=%s ts_range=%s-%s",
            global_step,
            attempt["attempt_id"],
            self._settings.version,
            total_records,
            batch_count,
            max_in_flight,
            attempt["range_start"],
            attempt["range_end"],
        )
        try:
            completed_batches = 0
            window_batches = 0
            window_records = 0
            logged_first_window = False
            for payload, payload_records in store.iter_upload_batches(
                self._settings.upload_batch_size
            ):
                self._raise_if_aborting()
                view.write_features(
                    data=payload,
                    version=self._settings.version,
                    write_mode=FEATURE_STORE_WRITE_MODE,
                    ts=int(attempt["range_start"]) + completed_batches,
                )
                completed_batches += 1
                window_batches += 1
                window_records += payload_records
                if window_batches < max_in_flight and completed_batches < batch_count:
                    continue
                summary = view.write_flush()
                self._validate_flush_summary(
                    summary,
                    expected_records=window_records,
                    expected_batches=window_batches,
                )
                for name in aggregate:
                    aggregate[name] += int(summary[name])
                if (
                    not logged_first_window
                    or completed_batches >= next_progress_batch
                    or completed_batches == batch_count
                ):
                    logger.info(
                        "FeatureStore delta upload progress: step=%s attempt=%s "
                        "batches=%s/%s records=%s/%s elapsed_secs=%.1f",
                        global_step,
                        attempt["attempt_id"],
                        completed_batches,
                        batch_count,
                        aggregate["success_records"],
                        total_records,
                        time.monotonic() - started_at,
                    )
                    logged_first_window = True
                    while next_progress_batch <= completed_batches:
                        next_progress_batch += (
                            _FEATURE_STORE_PROGRESS_LOG_INTERVAL_BATCHES
                        )
                window_batches = 0
                window_records = 0
                self._raise_if_aborting()
        except BaseException:
            # A write_features() call can enqueue part of its work before raising.
            # Drain it so a retry never mixes futures from two attempts.
            try:
                view.write_flush()
            except BaseException:
                pass
            raise
        return aggregate

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

    def _build_record_store(
        self,
        global_step: int,
        shard_paths: List[str],
        shard_sources: Optional[List[BinaryIO]] = None,
    ) -> _RecordStore:
        """Stream the step's shards into a disk-spilled deduplicated store.

        Shards are read batch by batch and validated with vectorized checks,
        so peak memory stays bounded by one Parquet batch and the store's
        page cache, independently of one dump's size.
        """
        if shard_sources is None:
            with ExitStack() as stack:
                opened_sources = [
                    stack.enter_context(open(path, "rb")) for path in shard_paths
                ]
                return self._build_record_store(
                    global_step, shard_paths, opened_sources
                )
        if len(shard_paths) != len(shard_sources):
            raise ValueError("delta shard paths and sources must have equal length")

        store = _RecordStore(os.path.join(self._state_dir, _RECORD_STORE_DB_FILENAME))
        try:
            for expected_rank, (path, source) in enumerate(
                zip(shard_paths, shard_sources)
            ):
                source.seek(0)
                parquet_file = pq.ParquetFile(source)
                self._validate_parquet_schema(parquet_file.schema_arrow, path)
                for batch in parquet_file.iter_batches(
                    batch_size=self._settings.upload_batch_size
                ):
                    self._ingest_record_batch(store, global_step, expected_rank, batch)
                source.seek(0)
        except BaseException:
            store.close()
            raise
        return store

    def _ingest_record_batch(
        self,
        store: _RecordStore,
        global_step: int,
        expected_rank: int,
        batch: pa.RecordBatch,
    ) -> None:
        """Validate one streamed shard batch and spill it into the store."""
        num_rows = batch.num_rows
        if num_rows == 0:
            return
        # The parquet schema check already enforces every column type, so the
        # value checks below stay vectorized instead of per-row Python loops.
        global_steps = batch.column("global_step").to_numpy(zero_copy_only=False)
        if not bool((global_steps == global_step).all()):
            raise ValueError("delta shard global_step mismatch")
        ranks = batch.column("rank").to_numpy(zero_copy_only=False)
        if not bool((ranks == expected_rank).all()):
            raise ValueError("delta shard rank mismatch")
        world_sizes = batch.column("world_size").to_numpy(zero_copy_only=False)
        if not bool((world_sizes == self._world_size).all()):
            raise ValueError("delta shard world_size mismatch")

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
                f"actual_shape=({int(lengths[row])},)"
            )

        store.add_batch(embedding_names, key_ids, flat_embeddings, offsets)

    @staticmethod
    def _validate_parquet_schema(schema: pa.Schema, path: str) -> None:
        metadata = schema.metadata or {}
        if metadata.get(
            _SCHEMA_VERSION_METADATA_KEY
        ) != DELTA_DUMP_SCHEMA_VERSION.encode("ascii"):
            raise ValueError(f"unsupported delta dump schema version in {path}")
        for field_name, field_type in _REQUIRED_PARQUET_FIELDS.items():
            index = schema.get_field_index(field_name)
            if index < 0 or schema.field(index).type != field_type:
                raise ValueError(
                    f"delta dump schema mismatch for field {field_name!r} in {path}"
                )

    def _commit_success(
        self,
        global_step: int,
        manifest: Mapping[str, Any],
        summary: Mapping[str, int],
    ) -> None:
        manifest_digest = _json_digest(manifest)
        publish_ts = int(summary["range_end"])
        success = {
            "schema_version": 2,
            "global_step": global_step,
            "project_name": self._settings.project_name,
            "feature_view_name": self._settings.feature_view_name,
            "version": self._settings.version,
            "attempt_id": int(summary["attempt_id"]),
            "range_start": int(summary["range_start"]),
            "range_end": publish_ts,
            "publish_ts": publish_ts,
            "write_mode": FEATURE_STORE_WRITE_MODE,
            "contract_hash": self._contract_hash,
            "dump_generation": manifest["dump_generation"],
            "manifest_digest": manifest_digest,
            "shards": manifest["shards"],
            "total_records": int(summary["total_records"]),
            "success_records": int(summary["success_records"]),
        }
        _atomic_write_json(self._success_path(global_step), success)
        committed = {
            "schema_version": 1,
            "project_name": self._settings.project_name,
            "feature_view_name": self._settings.feature_view_name,
            "version": self._settings.version,
            "committed_global_step": global_step,
            "publish_ts": publish_ts,
            "contract_hash": self._contract_hash,
            "dump_generation": manifest["dump_generation"],
            "manifest_digest": manifest_digest,
        }
        _atomic_write_json(os.path.join(self._state_dir, "committed.json"), committed)
        self._committed_global_step = global_step
        # Committed under the condition lock: later polls never re-read this
        # step's success marker, manifest, or canonical shard schemas again.
        self._reconciled_steps.add(global_step)

        logger.info(
            "FeatureStore delta upload committed: step=%s version=%s records=%s ts=%s",
            global_step,
            self._settings.version,
            summary["success_records"],
            publish_ts,
        )

    def _reclaim_snapshot(self, global_step: int) -> None:
        """Best-effort snapshot reclamation after durable success publication."""
        snapshot_dir = self._snapshot_dir(global_step)
        if not os.path.isdir(snapshot_dir):
            return
        try:
            shutil.rmtree(snapshot_dir)
            _fsync_parent_directory(snapshot_dir)
        except OSError as exc:
            # Success marker + manifest are already durable. Snapshot cleanup is
            # storage reclamation only and must not turn a committed write into
            # a reported failure.
            logger.warning(
                "Failed to reclaim committed FeatureStore shard snapshot (%s).",
                type(exc).__name__,
            )
