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

"""Bounded full-checkpoint upload into an unpublished FeatureDB version."""

import time
from typing import List, Mapping, Optional, Tuple

import numpy as np
import pyarrow as pa

from tzrec.protos.train_pb2 import FeatureStoreConfig
from tzrec.utils.feature_store_delta_uploader import (
    FEATURE_STORE_EMBEDDING_TYPE_FLOAT,
    FEATURE_STORE_EMBEDDING_TYPE_UINT8,
    FEATURE_STORE_UPLOAD_FORMAT_DEFAULT,
    FeatureStoreDeltaUploader,
    FeatureStoreUploadError,
)
from tzrec.utils.logging_util import logger
from tzrec.utils.quant_util import DISTRIBUTED_SPARSE_QUANT_SCALE_OFFSET_BYTES

_MAX_BATCH_BYTES = 8 * 1024 * 1024
_BATCH_OVERHEAD_BYTES = 4096


class FeatureStoreExportUploader:
    """Write bounded chunks, retrying only the current full-upload window.

    The delta uploader is used only for its existing connection, schema, and
    batch-validation helpers; its background worker is never started. Each
    ``write`` drains before returning, releasing all references to the source
    chunk. Retried windows reuse their values and timestamps within the explicit
    unpublished version. This uploader does not activate the version.
    """

    def __init__(
        self,
        config: FeatureStoreConfig,
        embedding_dimensions: Mapping[str, int],
        embedding_field_type: str = FEATURE_STORE_EMBEDDING_TYPE_FLOAT,
        max_in_flight_batches: int = 4,
    ) -> None:
        """Initialize the writer using logical, unquantized embedding dimensions."""
        if max_in_flight_batches <= 0:
            raise ValueError("upload_max_in_flight_batches must be > 0")
        dimensions = {str(name): int(dim) for name, dim in embedding_dimensions.items()}
        if any(not name or dim <= 0 for name, dim in dimensions.items()):
            raise ValueError("embedding names must be nonempty and dimensions positive")
        self._quantized = embedding_field_type == FEATURE_STORE_EMBEDDING_TYPE_UINT8
        self._dimensions = {
            name: dim
            + (DISTRIBUTED_SPARSE_QUANT_SCALE_OFFSET_BYTES if self._quantized else 0)
            for name, dim in dimensions.items()
        }
        self._writer = FeatureStoreDeltaUploader(
            config,
            self._dimensions,
            embedding_field_type=embedding_field_type,
        )
        self._max_in_flight_batches = max_in_flight_batches
        self._window: List[Tuple[pa.RecordBatch, int]] = []
        self._started = False
        self._closed = False
        self._error: Optional[FeatureStoreUploadError] = None
        self._uploaded_records = 0
        self._last_progress_time = time.monotonic()

    def start(self) -> None:
        """Create or validate the remote view before reading checkpoint data."""
        if self._closed:
            raise RuntimeError("FeatureStoreExportUploader is already closed")
        if self._started:
            return
        try:
            self._writer._get_view()
        except BaseException:
            self._writer._reset_view(suppress_errors=True)
            raise
        self._started = True

    def write(self, embedding_name: str, keys: np.ndarray, values: np.ndarray) -> None:
        """Upload one bounded chunk, waiting before its arrays can be released.

        Args:
            embedding_name: Table FQN from the sparse export contract.
            keys: One-dimensional int64 raw feature IDs.
            values: FP32 rows, or UINT8 rows including four scale/offset bytes.
        """
        if self._error is not None:
            raise self._error
        if not self._started or self._closed:
            raise RuntimeError("FeatureStoreExportUploader must be started and open")
        if embedding_name not in self._dimensions:
            raise ValueError(
                f"embedding name is absent from model contract: {embedding_name!r}"
            )
        if keys.ndim != 1 or keys.dtype != np.dtype(np.int64):
            raise ValueError("export keys must be a one-dimensional int64 array")
        dimension = self._dimensions[embedding_name]
        if values.ndim != 2 or values.shape != (len(keys), dimension):
            raise ValueError(
                f"export embedding dimension mismatch for {embedding_name!r}: "
                f"expected=({len(keys)}, {dimension}), actual={values.shape}"
            )
        expected_dtype = np.dtype(np.uint8 if self._quantized else np.float32)
        if values.dtype != expected_dtype:
            raise ValueError(f"export embedding dtype must be {expected_dtype}")

        name_bytes = len(embedding_name.encode("utf-8"))
        if self._writer._settings.upload_format == FEATURE_STORE_UPLOAD_FORMAT_DEFAULT:
            row_bytes = dimension * expected_dtype.itemsize + name_bytes + 32
        else:
            row_bytes = dimension * 32 + name_bytes * 6 + 256
        rows_per_batch = min(
            self._writer._settings.upload_batch_size,
            (_MAX_BATCH_BYTES - _BATCH_OVERHEAD_BYTES) // row_bytes,
        )
        if rows_per_batch <= 0:
            raise ValueError(
                f"one embedding row for {embedding_name!r} exceeds the "
                f"FeatureStore upload byte budget ({_MAX_BATCH_BYTES} bytes)"
            )

        arrow_type = pa.uint8() if self._quantized else pa.float32()
        try:
            for start in range(0, len(keys), rows_per_batch):
                stop = min(start + rows_per_batch, len(keys))
                count = stop - start
                embeddings = pa.ListArray.from_arrays(
                    pa.array(np.arange(count + 1, dtype=np.int32) * dimension),
                    pa.array(values[start:stop].reshape(-1), type=arrow_type),
                )
                batch = pa.RecordBatch.from_arrays(
                    [
                        pa.array([embedding_name] * count, type=pa.string()),
                        pa.array(keys[start:stop], type=pa.int64()),
                        embeddings,
                    ],
                    names=["table_fqn", "key_id", "embedding"],
                )
                self._writer._validate_delta_batch(batch)
                timestamp, _ = self._writer._allocate_timestamp_range(1)
                self._window.append((batch, timestamp))
                if len(self._window) >= self._max_in_flight_batches:
                    self.flush()
            self.flush()
        except BaseException:
            self._window.clear()
            raise

    def flush(self) -> None:
        """Drain the current window, replaying only it after a failed attempt."""
        if self._error is not None:
            raise self._error
        if not self._window:
            return
        settings = self._writer._settings
        expected_records = sum(batch.num_rows for batch, _ in self._window)
        for attempt in range(1, settings.max_retries + 1):
            try:
                view = self._writer._get_view()
                for batch, timestamp in self._window:
                    self._writer._submit_one_batch(
                        view, batch, timestamp, write_mode="OVERWRITE"
                    )
                self._writer._validate_flush_summary(
                    view.write_flush(),
                    expected_records=expected_records,
                    expected_batches=len(self._window),
                )
                self._uploaded_records += expected_records
                self._window.clear()
                now = time.monotonic()
                if now - self._last_progress_time >= 30:
                    logger.info(
                        "FeatureStore full export progress: version=%s records=%s",
                        settings.version,
                        self._uploaded_records,
                    )
                    self._last_progress_time = now
                return
            except Exception as exc:
                try:
                    self._writer._reset_view()
                except Exception as close_exc:
                    self._error = FeatureStoreUploadError(
                        "FeatureStore full export could not drain the failed writer"
                    )
                    raise self._error from close_exc
                if attempt >= settings.max_retries:
                    self._error = FeatureStoreUploadError(
                        "FeatureStore full export failed after "
                        f"{attempt} attempts for the current upload window"
                    )
                    raise self._error from exc
                logger.warning(
                    "FeatureStore full export window attempt %s/%s failed (%s); "
                    "retrying after backoff",
                    attempt,
                    settings.max_retries,
                    type(exc).__name__,
                )
                if settings.retry_backoff_secs:
                    time.sleep(settings.retry_backoff_secs * attempt)

    def close(self, raise_on_error: bool = True) -> None:
        """Drain and close, optionally preserving an existing export exception."""
        if self._closed:
            if raise_on_error and self._error is not None:
                raise self._error
            return
        try:
            if raise_on_error:
                self.flush()
        finally:
            self._window.clear()
            self._closed = True
            self._writer._reset_view(suppress_errors=not raise_on_error)
        if raise_on_error:
            logger.info(
                "FeatureStore full export completed: version=%s records=%s",
                self._writer._settings.version,
                self._uploaded_records,
            )
