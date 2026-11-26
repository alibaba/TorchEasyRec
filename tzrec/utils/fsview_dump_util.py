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

"""Dump a FeatureDB DynamicEmbedding FeatureView and merge with baseline NPZs."""

import copy
import datetime
import json
import os
import re
import shutil
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, cast

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import torch
import torch.distributed as dist

from tzrec.protos.train_pb2 import DeltaEmbeddingQuantType
from tzrec.utils import config_util, npz_util
from tzrec.utils.feature_store_uploader import (
    FEATURE_STORE_EMBEDDING_TYPE_FLOAT,
    FEATURE_STORE_EMBEDDING_TYPE_UINT8,
    FeatureStoreUploadSettings,
    FeatureStoreWriter,
)
from tzrec.utils.logging_util import logger
from tzrec.utils.quant_util import (
    DISTRIBUTED_SPARSE_QUANT_SCALE_OFFSET_BYTES,
    DISTRIBUTED_SPARSE_QUANT_STORAGE_DTYPE,
    DISTRIBUTED_SPARSE_SUPPORTED_QUANT_FORMATS,
)
from tzrec.utils.sparse_embedding_contract import SPARSE_EMBEDDING_INVALID_KEY

SPARSE_FS_DUMP_DIR = "sparse_fs_dump"
VERSIONS_DIR = "versions"
CURRENT_JSON = "current.json"
META_JSON = "fsview_dump_meta.json"
READY_FILE = "READY"
FEATURE_STORE_SCORE_FIELD = "score"

_DEFAULT_READ_TIMEOUT_SECS = 600
_DEFAULT_READ_CHUNK_SIZE_MB = 64
_DEFAULT_KEEP_VERSIONS = 3
_VERSION_TIME_FORMAT = "%Y%m%d%H%M%S"
_STATIC_NPZ_PATTERN = re.compile(r"^sparse_embeddings-\d+-of-\d+\.npz$")
_DYNAMIC_NPZ_PATTERN = re.compile(r"^sparse_dynamic_embedding-\d+-of-\d+\.npz$")
_INT64_MAX = int(np.iinfo(np.int64).max)


@dataclass(frozen=True)
class _TableSpec:
    """Validated sparse table schema derived from baseline metadata."""

    index: int
    name: str
    is_dynamic: bool
    dimension: int
    row_cols: int
    row_bytes: int
    storage_dtype: np.dtype
    num_embeddings: int
    key_name: str
    value_name: str
    score_name: str
    default_value_name: str
    quantized: bool


@dataclass(frozen=True)
class _DynamicShardStats:
    """Per-shard, per-table surviving row count and touched keys."""

    live_rows: int
    touched_keys: np.ndarray


@dataclass(frozen=True)
class _StaticShardUpdates:
    """Per-shard, per-table static key/value updates."""

    keys: np.ndarray
    values: np.ndarray


def _utc_now() -> str:
    """Return current UTC time as an ISO-8601 string."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _atomic_write_text(path: str, content: str) -> None:
    """Atomically write text via a PID-scoped sibling temp file."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp_path, "w") as f:
            f.write(content)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def _atomic_write_json(
    path: str, data: Mapping[str, Any], *, indent: int = 2, sort_keys: bool = True
) -> None:
    """Atomically serialize a JSON object to *path*."""
    _atomic_write_text(
        path, json.dumps(data, indent=indent, sort_keys=sort_keys) + "\n"
    )


def _read_current_version(current_path: str) -> Optional[str]:
    """Return the version currently published in *current_path*, if readable."""
    try:
        with open(current_path) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    version = data.get("version")
    return version if isinstance(version, str) and version else None


def _allocate_dump_version(
    versions_root: str,
    current_path: str,
    now: Optional[datetime.datetime] = None,
) -> str:
    """Allocate a strictly monotonic timestamp version directory name."""
    now = now or datetime.datetime.now()
    dt = now
    last_version = _read_current_version(current_path)
    if last_version:
        try:
            last_dt = datetime.datetime.strptime(last_version, _VERSION_TIME_FORMAT)
            if dt <= last_dt:
                dt = last_dt + datetime.timedelta(seconds=1)
        except ValueError:
            pass
    while True:
        candidate = dt.strftime(_VERSION_TIME_FORMAT)
        if not os.path.exists(os.path.join(versions_root, candidate)):
            return candidate
        dt = dt + datetime.timedelta(seconds=1)


def _prune_old_versions(
    dump_root: str,
    versions_root: str,
    keep_versions: int,
    active_version: str,
) -> None:
    """Sweep stale temp artifacts and prune old versions beyond *keep_versions*."""
    for base in (dump_root, versions_root):
        try:
            entries = os.listdir(base)
        except FileNotFoundError:
            continue
        for name in entries:
            if ".tmp." not in name and not name.endswith(".tmp"):
                continue
            path = os.path.join(base, name)
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
            except OSError as exc:
                logger.warning("Failed to remove stale dump tmp %s: %s", path, exc)

    if keep_versions <= 0:
        return
    try:
        entries = os.listdir(versions_root)
    except FileNotFoundError:
        return
    version_dirs = sorted(
        os.path.join(versions_root, name)
        for name in entries
        if os.path.isdir(os.path.join(versions_root, name)) and ".tmp." not in name
    )
    if len(version_dirs) <= keep_versions:
        return
    for path in version_dirs[:-keep_versions]:
        if os.path.basename(path) == active_version:
            continue
        try:
            shutil.rmtree(path)
            logger.info("Pruned old sparse_fs_dump version: %s", path)
        except OSError as exc:
            logger.warning("Failed to prune old dump version %s: %s", path, exc)


def _list_sorted_npz_files(sparse_dir: str, pattern: re.Pattern[str]) -> List[str]:
    """List full paths of sorted NPZ files in *sparse_dir* matching *pattern*."""
    if not os.path.isdir(sparse_dir):
        return []
    names = sorted(
        name
        for name in os.listdir(sparse_dir)
        if pattern.match(name) and os.path.isfile(os.path.join(sparse_dir, name))
    )
    return [os.path.join(sparse_dir, name) for name in names]


def _load_baseline_contract(
    base_sparse_dir: str,
    is_quantized: bool,
) -> Tuple[
    Dict[str, Any],
    Dict[str, Any],
    Dict[str, _TableSpec],
    List[str],
    List[str],
]:
    """Load and validate baseline sparse metadata and NPZ shard files."""
    if not os.path.isdir(base_sparse_dir):
        raise FileNotFoundError(
            f"Baseline sparse directory does not exist: {base_sparse_dir}"
        )
    emb_json_path = os.path.join(base_sparse_dir, "sparse_embedding.json")
    feat_json_path = os.path.join(base_sparse_dir, "sparse_features.json")
    with open(emb_json_path) as f:
        base_emb_meta = json.load(f)
    with open(feat_json_path) as f:
        base_feat_meta = json.load(f)
    if not isinstance(base_emb_meta, dict) or not isinstance(base_feat_meta, dict):
        raise ValueError("Baseline sparse_embedding.json/sparse_features.json invalid")

    expected_quant_format = DISTRIBUTED_SPARSE_SUPPORTED_QUANT_FORMATS[0]
    table_specs: Dict[str, _TableSpec] = {}
    for idx, (name, meta) in enumerate(base_emb_meta.items()):
        dim = int(meta["dimension"])
        shape = meta["shape"]
        if len(shape) != 2 or int(shape[1]) != dim:
            raise ValueError(
                f"Baseline table {name!r} has invalid shape {shape} for dim={dim}"
            )
        dtype_str = str(meta["dtype"])
        is_dynamic = bool(meta.get("is_dynamic", False))
        if is_quantized:
            if dtype_str != expected_quant_format:
                raise ValueError(
                    f"Quantization mismatch for baseline table {name!r}: config "
                    f"uses INT8 ({expected_quant_format}) but sparse_embedding.json "
                    f"has dtype={dtype_str!r}"
                )
            if meta.get("storage_dtype") != DISTRIBUTED_SPARSE_QUANT_STORAGE_DTYPE:
                raise ValueError(
                    f"Quantized baseline table {name!r} must declare "
                    f"storage_dtype={DISTRIBUTED_SPARSE_QUANT_STORAGE_DTYPE!r}"
                )
            row_cols = dim + DISTRIBUTED_SPARSE_QUANT_SCALE_OFFSET_BYTES
            row_bytes = int(meta.get("row_bytes", row_cols))
            if row_bytes != row_cols or row_bytes % 2 != 0:
                raise ValueError(
                    f"Quantized baseline table {name!r} has invalid "
                    f"row_bytes={row_bytes}"
                )
            storage_dtype = np.dtype(np.uint8)
        else:
            if dtype_str != "float32":
                raise ValueError(
                    f"Quantization mismatch for baseline table {name!r}: config "
                    f"uses FP32 but sparse_embedding.json has dtype={dtype_str!r}"
                )
            row_cols = dim
            row_bytes = int(meta.get("row_bytes", dim * 4))
            if row_bytes != dim * 4:
                raise ValueError(
                    f"FP32 baseline table {name!r} has invalid row_bytes={row_bytes}"
                )
            storage_dtype = np.dtype(np.float32)

        key_name = str(meta.get("key_name", ""))
        value_name = str(meta.get("value_name", ""))
        score_name = str(meta.get("score_name", ""))
        default_value_name = str(meta.get("default_value_name", ""))
        if is_dynamic and (not key_name or not value_name or not score_name):
            raise ValueError(
                f"Dynamic baseline table {name!r} is missing key/value/score "
                "entry names"
            )
        table_specs[name] = _TableSpec(
            index=idx,
            name=name,
            is_dynamic=is_dynamic,
            dimension=dim,
            row_cols=row_cols,
            row_bytes=row_bytes,
            storage_dtype=storage_dtype,
            num_embeddings=int(shape[0]),
            key_name=key_name,
            value_name=value_name,
            score_name=score_name,
            default_value_name=default_value_name,
            quantized=is_quantized,
        )

    base_static_files = _list_sorted_npz_files(base_sparse_dir, _STATIC_NPZ_PATTERN)
    base_dynamic_files = _list_sorted_npz_files(base_sparse_dir, _DYNAMIC_NPZ_PATTERN)
    if (
        any(not spec.is_dynamic for spec in table_specs.values())
        and not base_static_files
    ):
        raise FileNotFoundError(
            f"Baseline static tables declared but no sparse_embeddings-*.npz in "
            f"{base_sparse_dir}"
        )
    if any(spec.is_dynamic for spec in table_specs.values()) and not base_dynamic_files:
        raise FileNotFoundError(
            f"Baseline dynamic tables declared but no sparse_dynamic_embedding-*.npz "
            f"in {base_sparse_dir}"
        )
    return (
        base_emb_meta,
        base_feat_meta,
        table_specs,
        base_static_files,
        base_dynamic_files,
    )


def _open_feature_view(
    settings: FeatureStoreUploadSettings,
    expected_field_type: str,
    read_timeout_secs: int,
) -> Tuple[Any, bool]:
    """Open the configured DynamicEmbeddingFeatureView and inspect its schema."""
    try:
        from feature_store_py import FeatureStoreClient
    except ImportError as exc:
        raise RuntimeError(
            "feature_store_py is required for tzrec.dump_fsview"
        ) from exc

    cred_client = FeatureStoreWriter._create_credentials_client()
    credential = cred_client.get_credential()
    client = FeatureStoreClient(
        access_key_id=credential.access_key_id,
        access_key_secret=credential.access_key_secret,
        region=settings.region or None,
        endpoint=settings.endpoint or None,
        security_token=credential.security_token or None,
        featuredb_username=os.environ.get("FEATUREDB_USERNAME") or None,
        featuredb_password=os.environ.get("FEATUREDB_PASSWORD") or None,
        test_mode=settings.test_mode,
    )
    project = client.get_project(settings.project_name)
    if project is None:
        raise RuntimeError(
            f"Configured FeatureStore project {settings.project_name!r} was not found"
        )
    view = project.get_dynamic_embedding_feature_view(settings.feature_view_name)
    if view is None:
        raise RuntimeError(
            f"Configured DynamicEmbedding FeatureView {settings.feature_view_name!r} "
            "was not found"
        )
    if not hasattr(view, "get_shard_count") or not hasattr(view, "iterate_features"):
        raise RuntimeError(
            "feature_store_py does not support get_shard_count()/iterate_features(); "
            "install feature_store_py >= 2.2.15.dev1"
        )

    actual_type = (
        getattr(view, "embedding_field_type", None)
        or FEATURE_STORE_EMBEDDING_TYPE_FLOAT
    )
    if actual_type != expected_field_type:
        raise RuntimeError(
            f"FeatureView {settings.feature_view_name!r} has embedding_field_type="
            f"{actual_type!r}, expected {expected_field_type!r}"
        )

    additional_fields = getattr(view, "additional_fields", None) or {}
    has_score = FEATURE_STORE_SCORE_FIELD in additional_fields
    if has_score and additional_fields[FEATURE_STORE_SCORE_FIELD] != "INT64":
        actual_type = additional_fields[FEATURE_STORE_SCORE_FIELD]
        raise RuntimeError(
            f"FeatureView {settings.feature_view_name!r} field "
            f"{FEATURE_STORE_SCORE_FIELD!r} has type {actual_type!r}, expected 'INT64'"
        )
    return _clone_view_for_worker(view, read_timeout_secs), has_score


def _clone_view_for_worker(view: Any, read_timeout_secs: int) -> Any:
    """Create a thread-isolated view handle with an enlarged HTTP read timeout."""
    fdb_client = getattr(view, "_featuredb_client", None)
    if fdb_client is None:
        return view

    from feature_store_py.fs_dynamic_embedding_feature_view import (
        DynamicEmbeddingFeatureView,
    )
    from feature_store_py.utils.featuredb_utils import FeatureDBClient

    timeout = max(int(getattr(fdb_client, "_timeout", 30)), int(read_timeout_secs))
    retry_count = int(getattr(fdb_client, "_retry_count", 3))
    worker_client = FeatureDBClient(
        endpoint=fdb_client._endpoint,
        token=getattr(fdb_client, "_token", None),
        timeout=timeout,
        retry_count=retry_count,
    )
    worker_client._auth_signature = getattr(fdb_client, "_auth_signature", None)
    additional_fields = getattr(view, "additional_fields", None)
    view_kwargs: Dict[str, Any] = {
        "feature_view_id": view.feature_view_id,
        "feature_view_name": view.feature_view_name,
        "database_name": view.database_name,
        "schema_name": view.schema_name,
        "table_name": view.table_name,
        "featuredb_client": worker_client,
        "pk_field": view.pk_field,
        "sk_field": view.sk_field,
        "embedding_field": view.embedding_field,
        "batch_size": getattr(view, "_batch_size", 1000),
        "max_workers": getattr(view, "_max_workers", 4),
        "embedding_field_type": getattr(
            view, "embedding_field_type", FEATURE_STORE_EMBEDDING_TYPE_FLOAT
        ),
    }
    if additional_fields:
        view_kwargs["additional_fields"] = additional_fields
    return DynamicEmbeddingFeatureView(**view_kwargs)


def _decode_sk_array(sk_col: pa.Array) -> np.ndarray:
    """Vectorized cast of Arrow sk column to int64 with sentinel validation."""
    if sk_col.null_count > 0:
        raise ValueError("FeatureDB batch contains null sub_key values")
    if pa.types.is_string(sk_col.type) or pa.types.is_large_string(sk_col.type):
        sk_int = pc.cast(sk_col, pa.int64(), safe=True)
    elif pa.types.is_integer(sk_col.type):
        sk_int = pc.cast(sk_col, pa.int64(), safe=True)
    else:
        raise ValueError(f"Unsupported sk Arrow column type: {sk_col.type}")
    keys = np.ascontiguousarray(sk_int.to_numpy(zero_copy_only=False), dtype=np.int64)
    if bool((keys == SPARSE_EMBEDDING_INVALID_KEY).any()) or bool(
        (keys == _INT64_MAX).any()
    ):
        raise ValueError("FeatureDB batch contains reserved key (-1 or INT64_MAX)")
    return keys


def _decode_score_array(score_col: pa.Array) -> np.ndarray:
    """Vectorized decode of nullable INT64 score column with null -> 0."""
    if not pa.types.is_integer(score_col.type):
        raise ValueError(f"Unsupported score Arrow column type: {score_col.type}")
    if score_col.type != pa.int64():
        score_col = pc.cast(score_col, pa.int64(), safe=True)
    # Fill nulls in Arrow before to_numpy() so nullable int64 never widens to float64.
    if score_col.null_count > 0:
        score_col = score_col.fill_null(0)
    return np.ascontiguousarray(
        score_col.to_numpy(zero_copy_only=False), dtype=np.int64
    )


def _validate_emb_list_column(
    emb_col: pa.Array, expected_value_dtype: np.dtype
) -> pa.ListArray:
    """Validate list embedding column structure and element dtype."""
    if emb_col.null_count > 0:
        raise ValueError("FeatureDB batch contains null embedding rows")
    if not (pa.types.is_list(emb_col.type) or pa.types.is_large_list(emb_col.type)):
        raise ValueError(f"Unsupported embedding Arrow column type: {emb_col.type}")
    list_col = cast(pa.ListArray, emb_col)
    flat_values = list_col.values
    if flat_values.null_count > 0:
        raise ValueError("FeatureDB batch contains null elements inside embedding")
    expected_pa_type = (
        pa.uint8() if expected_value_dtype == np.dtype(np.uint8) else pa.float32()
    )
    if flat_values.type != expected_pa_type:
        raise ValueError(
            f"FeatureDB batch embedding element type {flat_values.type} does not "
            f"match expected {expected_pa_type}"
        )
    return list_col


def _extract_group_values(
    list_col: pa.ListArray,
    indices: Optional[np.ndarray],
    spec: _TableSpec,
) -> np.ndarray:
    """Extract contiguous (rows, row_cols) values for one table slice."""
    sub_col = list_col if indices is None else pc.take(list_col, indices)
    if sub_col.offset != 0 or sub_col.null_count != 0:
        sub_col = cast(pa.ListArray, sub_col)
    lengths = pc.list_value_length(sub_col).to_numpy(zero_copy_only=False)
    if bool((lengths != spec.row_cols).any()):
        bad = int(lengths[lengths != spec.row_cols][0])
        raise ValueError(
            f"FeatureDB embedding width mismatch for {spec.name!r}: "
            f"expected {spec.row_cols}, got {bad}"
        )
    offsets = sub_col.offsets.to_numpy(zero_copy_only=False)
    start = int(offsets[0])
    stop = int(offsets[-1])
    flat = sub_col.values.to_numpy(zero_copy_only=False)[start:stop]
    values = np.ascontiguousarray(
        flat.reshape(len(sub_col), spec.row_cols), dtype=spec.storage_dtype
    )
    if not spec.quantized and not bool(np.isfinite(values).all()):
        raise ValueError(
            f"FeatureDB FP32 embedding for {spec.name!r} contains NaN or Inf"
        )
    return values


def _spill_batch_to_shard_dir(
    batch: pa.RecordBatch,
    shard_dir: str,
    pk_field: str,
    sk_field: str,
    embedding_field: str,
    has_score: bool,
    table_specs: Mapping[str, _TableSpec],
    expected_value_dtype: np.dtype,
    open_files: Dict[Tuple[int, str], Any],
) -> None:
    """Decode one Arrow RecordBatch and append per-table binary temp files."""
    if batch.num_rows == 0:
        return

    pk_col = batch.column(pk_field)
    if pk_col.null_count > 0:
        raise ValueError("FeatureDB batch contains null primary_key values")
    sk_arr = _decode_sk_array(batch.column(sk_field))
    emb_col = _validate_emb_list_column(
        batch.column(embedding_field), expected_value_dtype
    )
    score_arr = (
        _decode_score_array(batch.column(FEATURE_STORE_SCORE_FIELD))
        if has_score
        else None
    )

    pks = pk_col.to_pylist()
    unique_pks = list(dict.fromkeys(pks))

    def _get_fp(table_idx: int, suffix: str) -> Any:
        key = (table_idx, suffix)
        fp = open_files.get(key)
        if fp is None:
            fp = open(os.path.join(shard_dir, f"{table_idx}.{suffix}"), "ab")
            open_files[key] = fp
        return fp

    if len(unique_pks) == 1:
        pk = unique_pks[0]
        spec = table_specs.get(pk)
        if spec is None:
            raise ValueError(
                f"FeatureDB returned unknown table {pk!r} absent from "
                "baseline sparse_embedding.json"
            )
        values = _extract_group_values(emb_col, None, spec)
        _get_fp(spec.index, "keys").write(sk_arr.tobytes())
        _get_fp(spec.index, "values").write(values.tobytes())
        if spec.is_dynamic and score_arr is not None:
            _get_fp(spec.index, "scores").write(score_arr.tobytes())
        return

    pk_arr = np.asarray(pks, dtype=object)
    for pk in unique_pks:
        spec = table_specs.get(pk)
        if spec is None:
            raise ValueError(
                f"FeatureDB returned unknown table {pk!r} absent from "
                "baseline sparse_embedding.json"
            )
        idx = np.flatnonzero(pk_arr == pk)
        values = _extract_group_values(emb_col, idx, spec)
        _get_fp(spec.index, "keys").write(np.ascontiguousarray(sk_arr[idx]).tobytes())
        _get_fp(spec.index, "values").write(values.tobytes())
        if spec.is_dynamic and score_arr is not None:
            _get_fp(spec.index, "scores").write(
                np.ascontiguousarray(score_arr[idx]).tobytes()
            )


def _is_tombstone_rows(values: np.ndarray, quantized: bool) -> np.ndarray:
    """Identify deletion tombstone rows matching Processor IsDeletionTombstoneRow."""
    if quantized:
        # QUint8RowwiseF16: all uint8 codes == 0 and fp16 offset (last 2B) == 0.
        return (
            np.all(values[:, :-4] == 0, axis=1)
            & (values[:, -2] == 0)
            & (values[:, -1] == 0)
        )
    return np.all(values == 0.0, axis=1)


def _dedup_last_positions(raw_keys: np.ndarray) -> np.ndarray:
    """Return ascending row positions keeping the last occurrence of each key."""
    if len(raw_keys) == 0:
        return np.empty(0, dtype=np.int64)
    _, rev_idx = np.unique(raw_keys[::-1], return_index=True)
    return np.sort(len(raw_keys) - 1 - rev_idx)


def _compact_dynamic_shard(
    shard_dir: str,
    spec: _TableSpec,
    has_score: bool,
    chunk_rows: int,
) -> _DynamicShardStats:
    """Deduplicate and filter tombstones in one spilled dynamic shard table."""
    raw_keys_path = os.path.join(shard_dir, f"{spec.index}.keys")
    raw_values_path = os.path.join(shard_dir, f"{spec.index}.values")
    raw_scores_path = os.path.join(shard_dir, f"{spec.index}.scores")
    if not os.path.exists(raw_keys_path) or os.path.getsize(raw_keys_path) == 0:
        return _DynamicShardStats(live_rows=0, touched_keys=np.empty(0, dtype=np.int64))

    raw_keys = np.fromfile(raw_keys_path, dtype=np.int64)
    dedup_pos = _dedup_last_positions(raw_keys)
    touched_keys = np.sort(raw_keys[dedup_pos])

    live_keys_path = os.path.join(shard_dir, f"{spec.index}.live.keys")
    live_values_path = os.path.join(shard_dir, f"{spec.index}.live.values")
    live_scores_path = os.path.join(shard_dir, f"{spec.index}.live.scores")

    live_rows = 0
    num_raw = len(raw_keys)
    pos_cursor = 0

    with (
        open(raw_values_path, "rb") as vf,
        open(live_keys_path, "wb") as lkf,
        open(live_values_path, "wb") as lvf,
    ):
        sf = open(raw_scores_path, "rb") if has_score else None
        lsf = open(live_scores_path, "wb") if has_score else None
        try:
            for chunk_start in range(0, num_raw, chunk_rows):
                rows = min(chunk_rows, num_raw - chunk_start)
                chunk_end = chunk_start + rows
                raw_val_bytes = vf.read(rows * spec.row_bytes)
                if len(raw_val_bytes) != rows * spec.row_bytes:
                    raise ValueError(
                        f"Truncated raw values file {raw_values_path}: expected "
                        f"{rows * spec.row_bytes} bytes, got {len(raw_val_bytes)}"
                    )
                raw_vals = np.frombuffer(
                    raw_val_bytes, dtype=spec.storage_dtype
                ).reshape(rows, spec.row_cols)
                if sf is not None:
                    raw_score_bytes = sf.read(rows * 8)
                    if len(raw_score_bytes) != rows * 8:
                        raise ValueError(
                            f"Truncated raw scores file {raw_scores_path}: expected "
                            f"{rows * 8} bytes, got {len(raw_score_bytes)}"
                        )
                    raw_scores = np.frombuffer(raw_score_bytes, dtype=np.int64)
                else:
                    raw_scores = None
                next_cursor = int(np.searchsorted(dedup_pos, chunk_end, side="left"))
                if next_cursor == pos_cursor:
                    continue
                local_idx = dedup_pos[pos_cursor:next_cursor] - chunk_start
                pos_cursor = next_cursor

                vals = raw_vals[local_idx]
                live_mask = ~_is_tombstone_rows(vals, spec.quantized)
                if not bool(live_mask.any()):
                    continue
                kept_idx = local_idx[live_mask]
                live_rows += int(len(kept_idx))
                lkf.write(
                    np.ascontiguousarray(raw_keys[chunk_start + kept_idx]).tobytes()
                )
                lvf.write(np.ascontiguousarray(vals[live_mask]).tobytes())
                if lsf is not None and raw_scores is not None:
                    lsf.write(np.ascontiguousarray(raw_scores[kept_idx]).tobytes())
        finally:
            if sf is not None:
                sf.close()
            if lsf is not None:
                lsf.close()

    for path in (raw_keys_path, raw_values_path, raw_scores_path):
        if os.path.exists(path):
            os.remove(path)
    return _DynamicShardStats(live_rows=live_rows, touched_keys=touched_keys)


def _read_static_shard_updates(shard_dir: str, spec: _TableSpec) -> _StaticShardUpdates:
    """Load and deduplicate spilled static key/value updates for one shard."""
    raw_keys_path = os.path.join(shard_dir, f"{spec.index}.keys")
    raw_values_path = os.path.join(shard_dir, f"{spec.index}.values")
    if not os.path.exists(raw_keys_path) or os.path.getsize(raw_keys_path) == 0:
        return _StaticShardUpdates(
            keys=np.empty(0, dtype=np.int64),
            values=np.empty((0, spec.row_cols), dtype=spec.storage_dtype),
        )
    raw_keys = np.fromfile(raw_keys_path, dtype=np.int64)
    if bool((raw_keys < 0).any()) or bool((raw_keys >= spec.num_embeddings).any()):
        raise ValueError(
            f"Static table {spec.name!r} received out-of-range key_id outside "
            f"[0, {spec.num_embeddings})"
        )
    dedup_pos = _dedup_last_positions(raw_keys)
    raw_vals = np.fromfile(raw_values_path, dtype=spec.storage_dtype).reshape(
        len(raw_keys), spec.row_cols
    )
    keys = np.ascontiguousarray(raw_keys[dedup_pos], dtype=np.int64)
    values = np.ascontiguousarray(raw_vals[dedup_pos], dtype=spec.storage_dtype)
    for path in (raw_keys_path, raw_values_path):
        if os.path.exists(path):
            os.remove(path)
    return _StaticShardUpdates(keys=keys, values=values)


def _pull_and_compact_shard(
    shard_id: int,
    base_view: Any,
    version: str,
    end_ts: int,
    output_fields: List[str],
    has_score: bool,
    table_specs: Mapping[str, _TableSpec],
    expected_value_dtype: np.dtype,
    tmp_root: str,
    read_chunk_bytes: int,
    read_timeout_secs: int,
) -> Tuple[int, Dict[str, _DynamicShardStats], Dict[str, _StaticShardUpdates]]:
    """Pull one FeatureDB shard, compact dynamic rows, and collect static updates."""
    shard_dir = os.path.join(tmp_root, f"shard_{shard_id}")
    os.makedirs(shard_dir, exist_ok=True)
    worker_view = _clone_view_for_worker(base_view, read_timeout_secs)
    open_files: Dict[Tuple[int, str], Any] = {}
    try:
        with worker_view.iterate_features(
            ts=0,
            version=version,
            end_ts=end_ts,
            shard_id=shard_id,
            output_fields=output_fields,
        ) as stream:
            for batch in stream:
                _spill_batch_to_shard_dir(
                    batch=batch,
                    shard_dir=shard_dir,
                    pk_field=worker_view.pk_field,
                    sk_field=worker_view.sk_field,
                    embedding_field=worker_view.embedding_field,
                    has_score=has_score,
                    table_specs=table_specs,
                    expected_value_dtype=expected_value_dtype,
                    open_files=open_files,
                )
        if stream.next_ts != end_ts:
            raise RuntimeError(
                f"Shard {shard_id} returned next_ts={stream.next_ts}, expected {end_ts}"
            )
    finally:
        for fp in open_files.values():
            fp.close()

    dyn_stats: Dict[str, _DynamicShardStats] = {}
    static_updates: Dict[str, _StaticShardUpdates] = {}
    for spec in table_specs.values():
        if spec.is_dynamic:
            chunk_rows = max(1, read_chunk_bytes // max(1, 8 + spec.row_bytes))
            dyn_stats[spec.name] = _compact_dynamic_shard(
                shard_dir, spec, has_score, chunk_rows
            )
        else:
            static_updates[spec.name] = _read_static_shard_updates(shard_dir, spec)
    return shard_id, dyn_stats, static_updates


def _merge_sorted_unique_int64(arrays: Sequence[np.ndarray]) -> np.ndarray:
    """Merge sorted/unsorted int64 arrays into a single sorted unique array."""
    non_empty = [a for a in arrays if len(a) > 0]
    if not non_empty:
        return np.empty(0, dtype=np.int64)
    if len(non_empty) == 1:
        return np.unique(non_empty[0])
    return np.unique(np.concatenate(non_empty))


def _gather_global_delta_keys(
    local_keys: np.ndarray, world_size: int, rank: int
) -> np.ndarray:
    """Gather and deduplicate touched int64 keys across all ranks."""
    if world_size <= 1:
        return local_keys
    local_tensor = torch.from_numpy(np.ascontiguousarray(local_keys, dtype=np.int64))
    local_len = torch.tensor([local_tensor.numel()], dtype=torch.int64)
    all_lens = [torch.zeros(1, dtype=torch.int64) for _ in range(world_size)]
    dist.all_gather(all_lens, local_len)

    parts: List[np.ndarray] = []
    for src_rank in range(world_size):
        n = int(all_lens[src_rank].item())
        if n == 0:
            continue
        buf = local_tensor if rank == src_rank else torch.empty(n, dtype=torch.int64)
        dist.broadcast(buf, src=src_rank)
        parts.append(buf.numpy().copy())
    return _merge_sorted_unique_int64(parts)


def _gather_static_updates_to_rank0(
    local_updates: Mapping[str, _StaticShardUpdates],
    static_specs: Sequence[_TableSpec],
    world_size: int,
    rank: int,
) -> Dict[str, List[_StaticShardUpdates]]:
    """Transfer static table updates from all ranks to rank 0 in shard order."""
    collected: Dict[str, List[_StaticShardUpdates]] = {
        spec.name: [local_updates[spec.name]] if rank == 0 else []
        for spec in static_specs
    }
    if world_size <= 1 or not static_specs:
        return collected
    for src_rank in range(1, world_size):
        for spec in static_specs:
            upd = local_updates[spec.name]
            n = int(len(upd.keys)) if rank == src_rank else 0
            count_t = torch.tensor([n], dtype=torch.int64)
            dist.broadcast(count_t, src=src_rank)
            remote_n = int(count_t.item())
            if remote_n == 0:
                continue
            if rank == src_rank:
                k_t = torch.from_numpy(np.ascontiguousarray(upd.keys, dtype=np.int64))
                v_t = torch.from_numpy(
                    np.ascontiguousarray(upd.values, dtype=spec.storage_dtype)
                )
                dist.send(k_t, dst=0)
                dist.send(v_t, dst=0)
            elif rank == 0:
                k_t = torch.empty(remote_n, dtype=torch.int64)
                v_dtype = torch.uint8 if spec.quantized else torch.float32
                v_t = torch.empty((remote_n, spec.row_cols), dtype=v_dtype)
                dist.recv(k_t, src=src_rank)
                dist.recv(v_t, src=src_rank)
                collected[spec.name].append(
                    _StaticShardUpdates(keys=k_t.numpy(), values=v_t.numpy())
                )
    return collected


def _compute_keep_mask(base_keys: np.ndarray, delta_keys: np.ndarray) -> np.ndarray:
    """Return boolean mask of baseline keys absent from sorted *delta_keys*."""
    if len(base_keys) == 0:
        return np.empty(0, dtype=bool)
    if len(delta_keys) == 0:
        return np.ones(len(base_keys), dtype=bool)
    idx = np.searchsorted(delta_keys, base_keys)
    clamped = np.minimum(idx, len(delta_keys) - 1)
    in_delta = (idx < len(delta_keys)) & (delta_keys[clamped] == base_keys)
    return ~in_delta


def _load_default_value_arrays(
    base_dynamic_files: Sequence[str],
    dynamic_specs: Sequence[_TableSpec],
) -> Dict[str, np.ndarray]:
    """Load optional ZCH default_value arrays from baseline dynamic NPZ shards."""
    defaults: Dict[str, np.ndarray] = {}
    pending = {
        spec.default_value_name: spec
        for spec in dynamic_specs
        if spec.default_value_name
    }
    if not pending:
        return defaults
    for path in base_dynamic_files:
        if not pending:
            break
        with zipfile.ZipFile(path, "r") as zf:
            names = set(zf.namelist())
            for entry_base in list(pending):
                if f"{entry_base}.npy" not in names:
                    continue
                chunks = list(npz_util.iter_npz_entry(zf, entry_base, 1))
                if not chunks:
                    raise ValueError(
                        f"Empty default_value entry {entry_base!r} in {path}"
                    )
                defaults[entry_base] = chunks[0]
                del pending[entry_base]
    if pending:
        raise FileNotFoundError(
            "Baseline dynamic NPZs are missing default_value entries: "
            + ", ".join(sorted(pending))
        )
    return defaults


def _iter_filtered_base_and_live(
    base_zf: Optional[zipfile.ZipFile],
    entry_name: str,
    keep_mask: np.ndarray,
    live_file_path: Optional[str],
    live_rows: int,
    dtype: np.dtype,
    trailing_shape: Tuple[int, ...],
    chunk_rows: int,
    fill_zeros_when_missing: bool = False,
) -> Iterator[np.ndarray]:
    """Yield filtered baseline chunks followed by live FeatureDB chunks."""
    row_cols = trailing_shape[0] if trailing_shape else 1
    row_bytes = int(row_cols * dtype.itemsize)

    if base_zf is not None and bool(keep_mask.any()):
        offset = 0
        for chunk in npz_util.iter_npz_entry(base_zf, entry_name, chunk_rows):
            n = len(chunk)
            slice_mask = keep_mask[offset : offset + n]
            offset += n
            if bool(slice_mask.any()):
                yield np.ascontiguousarray(chunk[slice_mask], dtype=dtype)
        if offset != len(keep_mask):
            raise ValueError(
                f"Baseline entry {entry_name!r} row count {offset} != "
                f"expected {len(keep_mask)}"
            )

    if live_rows > 0:
        if live_file_path is not None and os.path.exists(live_file_path):
            with open(live_file_path, "rb") as f:
                remaining = live_rows
                while remaining > 0:
                    cur = min(chunk_rows, remaining)
                    raw = f.read(cur * row_bytes)
                    if len(raw) != cur * row_bytes:
                        raise ValueError(
                            f"Truncated live file {live_file_path}: expected "
                            f"{cur * row_bytes} bytes, got {len(raw)}"
                        )
                    arr = np.frombuffer(raw, dtype=dtype)
                    if trailing_shape:
                        arr = arr.reshape((cur,) + trailing_shape)
                    yield np.ascontiguousarray(arr)
                    remaining -= cur
        elif fill_zeros_when_missing:
            remaining = live_rows
            while remaining > 0:
                cur = min(chunk_rows, remaining)
                shape = (cur,) + trailing_shape if trailing_shape else (cur,)
                yield np.zeros(shape, dtype=dtype)
                remaining -= cur
        else:
            raise FileNotFoundError(
                f"Missing live shard data file {live_file_path} for {live_rows} rows"
            )


def _write_merged_dynamic_part(
    part_idx: int,
    num_parts: int,
    version_dir: str,
    base_dynamic_path: Optional[str],
    dynamic_specs: Sequence[_TableSpec],
    delta_keys_by_table: Mapping[str, np.ndarray],
    shard_stats: Mapping[str, _DynamicShardStats],
    shard_dir: Optional[str],
    has_score: bool,
    default_values: Mapping[str, np.ndarray],
    read_chunk_bytes: int,
) -> Dict[str, int]:
    """Merge baseline and FeatureDB shard rows into one dynamic NPZ part."""
    width = max(2, len(str(num_parts)))
    out_path = os.path.join(
        version_dir,
        f"sparse_dynamic_embedding-{part_idx:0{width}d}-of-{num_parts:0{width}d}.npz",
    )
    part_rows_by_table: Dict[str, int] = {}
    base_zf = (
        zipfile.ZipFile(base_dynamic_path, "r")
        if base_dynamic_path is not None
        else None
    )
    try:
        arrays: Dict[str, Any] = {}
        for spec in dynamic_specs:
            chunk_rows = max(1, read_chunk_bytes // max(1, 8 + spec.row_bytes))
            if base_zf is not None:
                base_keys_chunks = list(
                    npz_util.iter_npz_entry(base_zf, spec.key_name, chunk_rows)
                )
                base_keys = (
                    np.concatenate(base_keys_chunks)
                    if base_keys_chunks
                    else np.empty(0, dtype=np.int64)
                )
                keep_mask = _compute_keep_mask(
                    base_keys, delta_keys_by_table[spec.name]
                )
                kept_base_rows = int(np.count_nonzero(keep_mask))
                del base_keys_chunks, base_keys
            else:
                keep_mask = np.empty(0, dtype=bool)
                kept_base_rows = 0

            live_rows = (
                shard_stats[spec.name].live_rows if spec.name in shard_stats else 0
            )
            part_rows = kept_base_rows + live_rows
            part_rows_by_table[spec.name] = part_rows

            live_keys_path = (
                os.path.join(shard_dir, f"{spec.index}.live.keys")
                if shard_dir is not None
                else None
            )
            live_values_path = (
                os.path.join(shard_dir, f"{spec.index}.live.values")
                if shard_dir is not None
                else None
            )
            live_scores_path = (
                os.path.join(shard_dir, f"{spec.index}.live.scores")
                if shard_dir is not None and has_score
                else None
            )

            arrays[spec.key_name] = npz_util.StreamingArray(
                np.dtype(np.int64),
                (part_rows,),
                _iter_filtered_base_and_live(
                    base_zf,
                    spec.key_name,
                    keep_mask,
                    live_keys_path,
                    live_rows,
                    np.dtype(np.int64),
                    (),
                    chunk_rows,
                ),
            )
            arrays[spec.value_name] = npz_util.StreamingArray(
                spec.storage_dtype,
                (part_rows, spec.row_cols),
                _iter_filtered_base_and_live(
                    base_zf,
                    spec.value_name,
                    keep_mask,
                    live_values_path,
                    live_rows,
                    spec.storage_dtype,
                    (spec.row_cols,),
                    chunk_rows,
                ),
            )
            arrays[spec.score_name] = npz_util.StreamingArray(
                np.dtype(np.int64),
                (part_rows,),
                _iter_filtered_base_and_live(
                    base_zf,
                    spec.score_name,
                    keep_mask,
                    live_scores_path,
                    live_rows,
                    np.dtype(np.int64),
                    (),
                    chunk_rows,
                    fill_zeros_when_missing=not has_score,
                ),
            )
            if part_idx == 0 and spec.default_value_name:
                arrays[spec.default_value_name] = default_values[
                    spec.default_value_name
                ]

        npz_util.savez_streaming(out_path, arrays)
    finally:
        if base_zf is not None:
            base_zf.close()
    return part_rows_by_table


def _write_merged_static_npz(
    version_dir: str,
    base_static_files: Sequence[str],
    static_specs: Sequence[_TableSpec],
    updates_by_table: Mapping[str, Sequence[_StaticShardUpdates]],
    read_chunk_bytes: int,
) -> None:
    """Merge baseline static tables with FeatureDB updates and write NPZ."""
    out_path = os.path.join(version_dir, "sparse_embeddings-00-of-01.npz")
    if not static_specs:
        npz_util.savez_streaming(out_path, {})
        return

    static_arrays: Dict[str, np.ndarray] = {}
    for spec in static_specs:
        chunk_rows = max(1, read_chunk_bytes // max(1, spec.row_bytes))
        table_arr = np.empty(
            (spec.num_embeddings, spec.row_cols), dtype=spec.storage_dtype
        )
        offset = 0
        for path in base_static_files:
            with zipfile.ZipFile(path, "r") as zf:
                if f"{spec.name}.npy" not in zf.namelist():
                    continue
                for chunk in npz_util.iter_npz_entry(zf, spec.name, chunk_rows):
                    if chunk.ndim != 2 or chunk.shape[1] != spec.row_cols:
                        raise ValueError(
                            f"Baseline static table {spec.name!r} has unexpected chunk "
                            f"shape {chunk.shape}"
                        )
                    end = offset + len(chunk)
                    if end > spec.num_embeddings:
                        raise ValueError(
                            f"Baseline static table {spec.name!r} exceeds "
                            f"declared rows {spec.num_embeddings}"
                        )
                    table_arr[offset:end] = chunk
                    offset = end
        if offset != spec.num_embeddings:
            raise ValueError(
                f"Baseline static table {spec.name!r} loaded {offset} rows, expected "
                f"{spec.num_embeddings}"
            )
        for upd in updates_by_table.get(spec.name, ()):
            if len(upd.keys) > 0:
                table_arr[upd.keys] = upd.values
        static_arrays[spec.name] = table_arr
    npz_util.savez_streaming(out_path, static_arrays)


def _build_updated_embedding_meta(
    base_emb_meta: Mapping[str, Any],
    dynamic_total_rows: Mapping[str, int],
    table_specs: Mapping[str, _TableSpec],
) -> Dict[str, Any]:
    """Update dynamic table row counts, memory, and storage_shape in metadata."""
    new_meta = copy.deepcopy(dict(base_emb_meta))
    for name, spec in table_specs.items():
        if not spec.is_dynamic:
            continue
        total_rows = int(dynamic_total_rows[name])
        entry = new_meta[name]
        entry["shape"] = [total_rows, spec.dimension]
        entry["memory"] = total_rows * spec.row_bytes
        if "storage_shape" in entry:
            entry["storage_shape"] = [total_rows, spec.row_bytes]
    return new_meta


def dump_fsview(
    pipeline_config_path: str,
    model_dir: str,
    edit_config_json: Optional[str] = None,
    max_workers: Optional[int] = None,
    read_chunk_size_mb: int = _DEFAULT_READ_CHUNK_SIZE_MB,
    keep_versions: int = _DEFAULT_KEEP_VERSIONS,
    read_timeout_secs: int = _DEFAULT_READ_TIMEOUT_SECS,
) -> Optional[Dict[str, Any]]:
    """Dump a FeatureDB FeatureView into ``<model_dir>/sparse_fs_dump``.

    Pulls ``[0, end_ts]`` concurrently across shards from the FeatureView
    configured in ``pipeline_config_path``, merges with the baseline
    ``<model_dir>/sparse`` snapshot using zero-shuffle filtering, writes
    ``sparse_dynamic_embedding-*.npz``, ``sparse_embeddings-00-of-01.npz``,
    updated ``sparse_embedding.json``, ``fsview_dump_meta.json``, and ``READY``,
    then atomically flips ``sparse_fs_dump/current.json``.

    Args:
        pipeline_config_path: Path to the streaming training pipeline config.
        model_dir: Full-export model directory containing baseline ``sparse/``.
        edit_config_json: Optional JSON string overrides for ``pipeline_config``.
        max_workers: Concurrent shard/part worker threads per rank.
        read_chunk_size_mb: Streaming chunk budget in MiB.
        keep_versions: Maximum historical dump versions to retain (0 = keep all).
        read_timeout_secs: HTTP read timeout per shard stream in seconds.

    Returns:
        Published ``current.json`` payload on rank 0, or None on non-zero ranks.
    """
    if read_chunk_size_mb <= 0:
        raise ValueError("read_chunk_size_mb must be > 0")
    if keep_versions < 0:
        raise ValueError("keep_versions must be >= 0")
    if read_timeout_secs <= 0:
        raise ValueError("read_timeout_secs must be > 0")

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    initialized_pg = False
    if world_size > 1 and not (dist.is_available() and dist.is_initialized()):
        dist.init_process_group(backend="gloo")
        initialized_pg = True

    try:
        pipeline_config = config_util.load_pipeline_config(pipeline_config_path)
        if edit_config_json:
            config_util.edit_config(pipeline_config, json.loads(edit_config_json))

        train_config = pipeline_config.train_config
        if not train_config.HasField("delta_embedding_dump_config"):
            raise ValueError(
                "pipeline_config.train_config.delta_embedding_dump_config is required"
            )
        delta_cfg = train_config.delta_embedding_dump_config
        if not delta_cfg.HasField("feature_store_config"):
            raise ValueError(
                "delta_embedding_dump_config.feature_store_config is required"
            )
        fs_settings = FeatureStoreUploadSettings.from_proto(
            delta_cfg.feature_store_config
        )
        effective_workers = (
            int(max_workers) if max_workers is not None else fs_settings.max_workers
        )
        if effective_workers <= 0:
            raise ValueError("max_workers must be > 0")

        is_quantized = (
            delta_cfg.quant_type == DeltaEmbeddingQuantType.DELTA_EMBEDDING_QUANT_INT8
        )
        expected_field_type = (
            FEATURE_STORE_EMBEDDING_TYPE_UINT8
            if is_quantized
            else FEATURE_STORE_EMBEDDING_TYPE_FLOAT
        )
        expected_value_dtype = np.dtype(np.uint8 if is_quantized else np.float32)

        base_sparse_dir = os.path.join(model_dir, "sparse")
        (
            base_emb_meta,
            base_feat_meta,
            table_specs,
            base_static_files,
            base_dynamic_files,
        ) = _load_baseline_contract(base_sparse_dir, is_quantized)
        dynamic_specs = [spec for spec in table_specs.values() if spec.is_dynamic]
        static_specs = [spec for spec in table_specs.values() if not spec.is_dynamic]

        view, has_score = _open_feature_view(
            fs_settings, expected_field_type, read_timeout_secs
        )

        dump_root = os.path.join(model_dir, SPARSE_FS_DUMP_DIR)
        versions_root = os.path.join(dump_root, VERSIONS_DIR)
        current_path = os.path.join(dump_root, CURRENT_JSON)

        if rank == 0:
            shard_count = view.get_shard_count()
            if type(shard_count) is not int or shard_count <= 0:
                raise ValueError(f"Invalid FeatureDB shard_count: {shard_count!r}")
            end_ts = time.time_ns() // 1_000_000
            os.makedirs(versions_root, exist_ok=True)
            dump_version = _allocate_dump_version(versions_root, current_path)
            version_dir = os.path.join(versions_root, dump_version)
            os.makedirs(version_dir, exist_ok=True)
            broadcast_state: List[Any] = [
                end_ts,
                shard_count,
                dump_version,
                has_score,
            ]
        else:
            broadcast_state = [0, 0, "", False]

        if world_size > 1:
            dist.broadcast_object_list(broadcast_state, src=0)
        end_ts = int(broadcast_state[0])
        shard_count = int(broadcast_state[1])
        dump_version = str(broadcast_state[2])
        has_score = bool(broadcast_state[3])
        version_dir = os.path.join(versions_root, dump_version)

        output_fields = [view.pk_field, view.sk_field, view.embedding_field]
        if has_score:
            output_fields.append(FEATURE_STORE_SCORE_FIELD)

        num_output_parts = (
            max(len(base_dynamic_files), shard_count, 1) if dynamic_specs else 0
        )
        my_shards = [s for s in range(shard_count) if s % world_size == rank]
        my_parts = [p for p in range(num_output_parts) if p % world_size == rank]
        read_chunk_bytes = int(read_chunk_size_mb) * 1024 * 1024

        if rank == 0:
            logger.info(
                "Starting sparse_fs_dump: project=%s view=%s fs_version=%s "
                "dump_version=%s end_ts=%s shards=%s output_parts=%s has_score=%s",
                fs_settings.project_name,
                fs_settings.feature_view_name,
                fs_settings.version,
                dump_version,
                end_ts,
                shard_count,
                num_output_parts,
                has_score,
            )

        with tempfile.TemporaryDirectory(
            prefix=f"tzrec_dump_fsview_r{rank}_"
        ) as tmp_root:
            shard_dyn_stats: Dict[int, Dict[str, _DynamicShardStats]] = {}
            static_updates_by_shard: Dict[int, Dict[str, _StaticShardUpdates]] = {}
            if my_shards:
                pool_size = min(effective_workers, len(my_shards))
                with ThreadPoolExecutor(max_workers=pool_size) as pool:
                    futures = [
                        pool.submit(
                            _pull_and_compact_shard,
                            shard_id,
                            view,
                            fs_settings.version,
                            end_ts,
                            output_fields,
                            has_score,
                            table_specs,
                            expected_value_dtype,
                            tmp_root,
                            read_chunk_bytes,
                            read_timeout_secs,
                        )
                        for shard_id in my_shards
                    ]
                    for fut in futures:
                        shard_id, dyn_stats, static_upd = fut.result()
                        shard_dyn_stats[shard_id] = dyn_stats
                        static_updates_by_shard[shard_id] = static_upd

            # Sync touched dynamic keys globally so baseline filtering drops all
            # updated or tombstoned keys.
            delta_keys_by_table: Dict[str, np.ndarray] = {}
            for spec in dynamic_specs:
                local_keys = _merge_sorted_unique_int64(
                    [
                        shard_dyn_stats[s][spec.name].touched_keys
                        for s in sorted(shard_dyn_stats)
                    ]
                )
                delta_keys_by_table[spec.name] = _gather_global_delta_keys(
                    local_keys, world_size, rank
                )

            # Gather static updates per table onto rank 0 preserving shard order.
            local_static_updates: Dict[str, _StaticShardUpdates] = {}
            for spec in static_specs:
                shard_keys = [
                    static_updates_by_shard[s][spec.name].keys
                    for s in sorted(static_updates_by_shard)
                    if len(static_updates_by_shard[s][spec.name].keys) > 0
                ]
                shard_vals = [
                    static_updates_by_shard[s][spec.name].values
                    for s in sorted(static_updates_by_shard)
                    if len(static_updates_by_shard[s][spec.name].keys) > 0
                ]
                if shard_keys:
                    local_static_updates[spec.name] = _StaticShardUpdates(
                        keys=np.concatenate(shard_keys),
                        values=np.concatenate(shard_vals, axis=0),
                    )
                else:
                    local_static_updates[spec.name] = _StaticShardUpdates(
                        keys=np.empty(0, dtype=np.int64),
                        values=np.empty((0, spec.row_cols), dtype=spec.storage_dtype),
                    )
            gathered_static = _gather_static_updates_to_rank0(
                local_static_updates, static_specs, world_size, rank
            )

            default_values = (
                _load_default_value_arrays(base_dynamic_files, dynamic_specs)
                if rank == 0 and 0 in my_parts
                else {}
            )

            local_table_rows: Dict[str, int] = {spec.name: 0 for spec in dynamic_specs}
            if my_parts and dynamic_specs:
                part_pool_size = min(effective_workers, len(my_parts))
                with ThreadPoolExecutor(max_workers=part_pool_size) as pool:
                    part_futures = [
                        pool.submit(
                            _write_merged_dynamic_part,
                            part_idx,
                            num_output_parts,
                            version_dir,
                            base_dynamic_files[part_idx]
                            if part_idx < len(base_dynamic_files)
                            else None,
                            dynamic_specs,
                            delta_keys_by_table,
                            shard_dyn_stats.get(part_idx, {}),
                            os.path.join(tmp_root, f"shard_{part_idx}")
                            if part_idx in shard_dyn_stats
                            else None,
                            has_score,
                            default_values,
                            read_chunk_bytes,
                        )
                        for part_idx in my_parts
                    ]
                    for fut in part_futures:
                        part_rows = fut.result()
                        for name, cnt in part_rows.items():
                            local_table_rows[name] += cnt

            if rank == 0:
                _write_merged_static_npz(
                    version_dir,
                    base_static_files,
                    static_specs,
                    gathered_static,
                    read_chunk_bytes,
                )

            if dynamic_specs and world_size > 1:
                counts_tensor = torch.tensor(
                    [local_table_rows[spec.name] for spec in dynamic_specs],
                    dtype=torch.int64,
                )
                dist.all_reduce(counts_tensor, op=dist.ReduceOp.SUM)
                dynamic_total_rows = {
                    spec.name: int(counts_tensor[i].item())
                    for i, spec in enumerate(dynamic_specs)
                }
            else:
                dynamic_total_rows = local_table_rows

            if world_size > 1:
                dist.barrier()

            result_payload: Optional[Dict[str, Any]] = None
            if rank == 0:
                new_emb_meta = _build_updated_embedding_meta(
                    base_emb_meta, dynamic_total_rows, table_specs
                )
                for filename in (
                    "sparse_embeddings-00-of-01.json",
                    "sparse_embedding.json",
                ):
                    _atomic_write_json(
                        os.path.join(version_dir, filename),
                        new_emb_meta,
                        indent=4,
                        sort_keys=False,
                    )
                _atomic_write_json(
                    os.path.join(version_dir, "sparse_features.json"),
                    base_feat_meta,
                    indent=4,
                    sort_keys=False,
                )

                dumped_at = _utc_now()
                dump_meta = {
                    "start_ts": 0,
                    "end_ts": int(end_ts),
                    "region": fs_settings.region,
                    "project_name": fs_settings.project_name,
                    "feature_view_name": fs_settings.feature_view_name,
                    "version": fs_settings.version,
                    "dump_version": dump_version,
                    "shard_count": int(shard_count),
                    "num_dynamic_parts": int(num_output_parts),
                    "quant_type": DeltaEmbeddingQuantType.Name(delta_cfg.quant_type),
                    "has_score": bool(has_score),
                    "dumped_at": dumped_at,
                }
                _atomic_write_json(
                    os.path.join(version_dir, META_JSON), dump_meta, indent=2
                )
                _atomic_write_text(os.path.join(version_dir, READY_FILE), "")

                result_payload = {
                    "version": dump_version,
                    "end_ts": int(end_ts),
                    "feature_view_name": fs_settings.feature_view_name,
                    "fs_version": fs_settings.version,
                    "has_score": bool(has_score),
                    "created_at": dumped_at,
                }
                _atomic_write_json(current_path, result_payload, indent=2)
                _prune_old_versions(
                    dump_root, versions_root, keep_versions, dump_version
                )
                logger.info(
                    "Completed sparse_fs_dump version=%s end_ts=%s dynamic_rows=%s",
                    dump_version,
                    end_ts,
                    dynamic_total_rows,
                )

            if world_size > 1:
                dist.barrier()
            return result_payload
    finally:
        if initialized_pg and dist.is_initialized():
            dist.destroy_process_group()
