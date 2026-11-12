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

"""Module-level read tasks for the sample-level shuffle read plan.

Every function is a generator called by a reader process as
``fn(stop_event=..., **kwargs)``. The parameters must be picklable because
the plan travels to the spawned coordinator; resources such as consumers,
storage clients and process groups are created inside the reader process
instead of being handed over.
"""

import os
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pyarrow as pa

# per reader process: one account and one client per table identity
_ODPS_ACCOUNT: Optional[Any] = None
_ODPS_CLIENTS: Dict[Tuple[str, Optional[str], str, str], Any] = {}


def _get_odps_client(input_path: str, quota_name: str) -> Any:
    """Create or reuse this process's storage client for a table and quota."""
    from tzrec.datasets.odps_dataset import (
        _create_odps_account,
        _create_storage_client,
        _parse_table_path,
    )

    global _ODPS_ACCOUNT
    project, table_name, _, schema = _parse_table_path(input_path)
    key = (project, schema, table_name, quota_name)
    client = _ODPS_CLIENTS.get(key)
    if client is not None:
        return client
    if _ODPS_ACCOUNT is None:
        _ODPS_ACCOUNT = _create_odps_account()
    account, endpoint = _ODPS_ACCOUNT
    _, client = _create_storage_client(input_path, account, endpoint, quota_name)
    _ODPS_CLIENTS[key] = client
    return client


def read_kafka(
    stop_event: Any,
    input_path: str,
    reader_id: int,
    num_readers: int,
    start_offsets: Dict[int, int],
    reader_schema: pa.Schema,
    full_schema: pa.Schema,
    has_embedded_schema: bool,
    batch_size: int,
    consumer_factory: Optional[Any] = None,
    round_robin_partition_order: Optional[List[int]] = None,
    kafka_log_every_n_messages: int = 0,
) -> Iterator[pa.RecordBatch]:
    """Consume Kafka with this run's fixed per-partition start offsets.

    Args:
        stop_event: coordinator stop signal observed by the consume loop.
        input_path: kafka URI.
        reader_id: global consumer slot passed to the consumer factory.
        num_readers: global consumer slot count.
        start_offsets: fixed {partition: offset} start of this run.
        reader_schema: schema to project and cast each message to.
        full_schema: full schema for schema-less messages.
        has_embedded_schema: whether messages embed an IPC stream schema.
        batch_size: target rows per consume round.
        consumer_factory: consumer builder; the default builds real consumers.
        round_robin_partition_order: parent-resolved partition order, or None
            for ordinary consumer-group subscription.
        kafka_log_every_n_messages: decoded-message log interval, 0 disables.

    Yields:
        decoded record batches with checkpoint metadata and event time.
    """
    from tzrec.datasets.kafka_dataset import _consume_kafka

    yield from _consume_kafka(
        input_path,
        consumer_factory,
        reader_id,
        num_readers,
        batch_size,
        reader_schema,
        full_schema,
        has_embedded_schema,
        start_offsets=start_offsets,
        stop_event=stop_event,
        # this process was forked by the coordinator
        orphan_pid=os.getppid(),
        round_robin_partition_order=round_robin_partition_order,
        kafka_log_every_n_messages=kafka_log_every_n_messages,
    )


def read_parquet(
    stop_event: Any,
    input_files: List[str],
    batch_size: int,
    parquet_metas: Dict[str, Any],
    columns: Optional[List[str]],
    intervals: List[Tuple[int, int]],
    source_prefix: str,
    reader_id: int,
) -> Iterator[pa.RecordBatch]:
    """Read assigned parquet row intervals from the shared file list.

    Args:
        stop_event: unused; the reader loop checks it between batches.
        input_files: all parquet files of the dataset, in a stable order.
        batch_size: rows per scanner batch.
        parquet_metas: per-file metadata used to skip whole row groups.
        columns: selected columns, or None for all.
        intervals: absolute (start, end) row intervals of this task.
        source_prefix: checkpoint source id prefix.
        reader_id: reader slot, used in debug logs.

    Yields:
        parquet record batches with checkpoint metadata.
    """
    from tzrec.datasets.parquet_dataset import _reader_iter

    for start, end in intervals:
        yield from _reader_iter(
            input_files,
            batch_size,
            parquet_metas,
            columns,
            start,
            end,
            reader_id,
            source_id=f"{source_prefix}:{start}",
        )


def read_odps(
    stop_event: Any,
    input_path: str,
    quota_name: str,
    session_id: str,
    intervals: List[Tuple[int, int]],
    source_prefix: str,
    batch_size: int,
    compression: str,
) -> Iterator[pa.RecordBatch]:
    """Read assigned row intervals of one restored ODPS session.

    The child process creates its own storage client and reuses the session
    created (or restored) by the trainer process.

    Args:
        stop_event: unused; the reader loop checks it between batches.
        input_path: odps table path owning the session.
        quota_name: storage api quota name.
        session_id: batch scan session id.
        intervals: absolute (start, end) row intervals of this task.
        source_prefix: checkpoint source id prefix.
        batch_size: rows per read request.
        compression: storage api compression name.

    Yields:
        ODPS record batches with checkpoint metadata.
    """
    from odps.apis.storage_api import SessionRequest

    from tzrec.datasets.odps_dataset import _get_compression_type, _reader_iter

    os.environ["STORAGE_API_QUOTA_NAME"] = quota_name
    client = _get_odps_client(input_path, quota_name)
    sess_req = SessionRequest(session_id=session_id)
    comp = _get_compression_type(compression)
    for start, end in intervals:
        yield from _reader_iter(
            client,
            sess_req,
            batch_size,
            comp,
            start,
            end,
            f"{source_prefix}:{start}",
        )


def read_csv(
    stop_event: Any,
    input_files: List[str],
    batch_size: int,
    columns: Optional[List[str]],
    column_names: Optional[List[str]],
    delimiter: str,
    column_types: Optional[Dict[str, pa.DataType]],
) -> Iterator[pa.RecordBatch]:
    """Read the csv files assigned to this reader slot.

    Args:
        stop_event: unused; the reader loop checks it between batches.
        input_files: files assigned to this slot.
        batch_size: scanner batch size.
        columns: selected columns in output order, or None for all.
        column_names: explicit names when the files have no header.
        delimiter: csv delimiter.
        column_types: explicit column types inferred by the dataset.

    Yields:
        csv record batches.
    """
    import pyarrow.dataset as ds

    from tzrec.datasets.csv_dataset import _make_csv_file_format

    csv_fmt = _make_csv_file_format(column_names, delimiter, column_types)
    dataset = ds.dataset(input_files, format=csv_fmt)
    yield from dataset.to_batches(batch_size=batch_size, columns=columns)
