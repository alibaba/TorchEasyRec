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


import math
import os
from typing import Any, Dict, Iterator, List, Optional, Tuple
from urllib.parse import parse_qsl, urlparse

import pyarrow as pa
from confluent_kafka import OFFSET_INVALID, Consumer, TopicPartition

from tzrec.datasets.dataset import BaseDataset, BaseReader
from tzrec.datasets.utils import (
    CKPT_ROW_IDX,
    CKPT_SOURCE_ID,
    FIELD_TYPE_TO_PA,
    get_input_fields_proto,
)
from tzrec.features.feature import BaseFeature
from tzrec.protos import data_pb2
from tzrec.utils.logging_util import logger


def _parse_kafka_uri(uri: str) -> Tuple[str, Dict[str, Any], Optional[int]]:
    """Parse kafka URI into configuration dict.

    Args:
        uri: kafka URI.
            e.g. kafka://broker:9092/topic?group.id=xxx&auto.offset.reset=earliest
            Supports an optional ``start.timestamp.ms`` query parameter to begin
            consuming from the earliest offset whose timestamp is >= the given
            value (milliseconds since epoch).  Example::

                kafka://broker:9092/topic?group.id=g1&start.timestamp.ms=1711929600000

    Returns:
        topic: kafka topic name.
        params: consumer config params (``start.timestamp.ms`` is removed from
            this dict because it is not a native ``confluent_kafka`` consumer
            configuration key).
        start_timestamp_ms: start timestamp in milliseconds, or ``None`` if not
            specified.
    """
    parsed = urlparse(uri)
    if parsed.scheme != "kafka":
        raise ValueError(f"Invalid kafka URI scheme: {parsed.scheme}")

    broker = parsed.netloc
    if not broker:
        raise ValueError("Kafka broker not specified in URI")

    topic = parsed.path.lstrip("/")
    if not topic:
        raise ValueError("Kafka topic not specified in URI")

    params = dict(parse_qsl(parsed.query))
    if "group.id" not in params:
        raise ValueError("Consumer group not specified in URI (use ?group.id=xxx)")

    # Extract custom start.timestamp.ms parameter (not a confluent_kafka config key)
    start_timestamp_ms: Optional[int] = None
    raw_ts = params.pop("start.timestamp.ms", None)
    if raw_ts is not None:
        try:
            start_timestamp_ms = int(raw_ts)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"start.timestamp.ms must be a valid integer, got: {raw_ts!r}"
            ) from e
        if start_timestamp_ms < 0:
            raise ValueError("start.timestamp.ms must be a non-negative integer")

    if "debug" not in params:
        params["debug"] = "assignor,conf"
    params["bootstrap.servers"] = broker

    return topic, params, start_timestamp_ms


class KafkaDataset(BaseDataset):
    """Dataset for reading data from Kafka (each message is a serialized Arrow batch).

    Args:
        data_config (DataConfig): an instance of DataConfig.
        features (list): list of features.
        input_path (str): kafka URI, e.g. kafka://broker:9092/topic?group.id=xxx
    """

    def __init__(
        self,
        data_config: data_pb2.DataConfig,
        features: List[BaseFeature],
        input_path: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(data_config, features, input_path, **kwargs)

        input_fields_list = get_input_fields_proto(self._data_config)
        input_fields = []
        for f in input_fields_list:
            pa_type = FIELD_TYPE_TO_PA.get(f.input_type)
            input_fields.append(pa.field(f.input_name, pa_type))

        # pyre-ignore [29]
        self._reader = KafkaReader(
            input_path,
            self._batch_size,
            list(self._selected_input_names) if self._selected_input_names else None,
            self._data_config.drop_remainder,
            input_fields=input_fields if input_fields else None,
        )


class KafkaReader(BaseReader):
    """Kafka reader class.

    Args:
        input_path (str): kafka URI, e.g. kafka://broker:9092/topic?group.id=xxx
        batch_size (int): batch size.
        selected_cols (list): selection column names.
        drop_remainder (bool): drop last batch.
        input_fields (list): list of pa.Field for schema definition.
    """

    def __init__(
        self,
        input_path: str,
        batch_size: int,
        selected_cols: Optional[List[str]] = None,
        drop_remainder: bool = False,
        input_fields: Optional[List[pa.Field]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            input_path,
            batch_size,
            selected_cols,
            drop_remainder,
        )

        self._drop_columns = []
        self._has_embedded_schema = False

        if input_fields:
            self._full_schema = pa.schema(input_fields)
        else:
            # Embedded schema mode - peek schema from first message
            self._full_schema = self._peek_schema_from_kafka()
            self._has_embedded_schema = True

        if self._selected_cols:
            fields = []
            self._ordered_cols = []
            for field in self._full_schema:
                if field.name in self._selected_cols:
                    fields.append(field)
                    self._ordered_cols.append(field.name)
                else:
                    self._drop_columns.append(field.name)
            self._schema = pa.schema(fields)
        else:
            self._schema = self._full_schema
            self._ordered_cols = [f.name for f in self._full_schema]

    @property
    def schema(self) -> pa.Schema:
        """Table schema."""
        return self._schema

    def _peek_schema_from_kafka(self) -> pa.Schema:
        """Peek at newest Kafka message to extract schema.

        Uses assign() and seek() to directly read the newest message from partition 0,
        avoiding affecting consumer group.

        Returns:
            PyArrow schema from the message.

        Raises:
            ValueError: If no messages available or message doesn't contain
                valid schema.
        """
        topic, config, _ = _parse_kafka_uri(self._input_path)
        consumer = Consumer(config)

        try:
            # Directly assign partition 0 (no subscribe, no group coordination)
            tp = TopicPartition(topic, 0)
            consumer.assign([tp])

            # Get the high watermark (newest offset) for the partition
            low, high = consumer.get_watermark_offsets(tp, timeout=60.0)
            if high <= low:
                raise ValueError(
                    "No messages available in Kafka topic to infer schema. "
                    "Either provide input_fields in config or ensure topic "
                    "has messages."
                )

            # Re-assign with specific offset to read the last message
            tp = TopicPartition(topic, 0, high - 1)
            consumer.assign([tp])

            # Poll for the message
            msg = consumer.poll(timeout=10.0)
            if msg is None:
                raise ValueError("Failed to read message from Kafka topic after seek.")
            if msg.error():
                raise ValueError(f"Error reading Kafka message: {msg.error()}")

            msg_data = msg.value()
            reader = pa.ipc.open_stream(msg_data)
            try:
                schema = reader.schema
            finally:
                reader.close()
            return schema
        except pa.ArrowInvalid as e:
            raise ValueError(
                "Message may not be in Arrow IPC stream format. Failed to read "
                "schema from Kafka message. You should provide input_fields in "
                f"config when using serialize record_batch without schema: {e}."
            ) from e
        finally:
            consumer.close()

    def _reader(
        self, worker_id: int = 0, num_workers: int = 1
    ) -> Iterator[pa.RecordBatch]:
        """Read Arrow batches from Kafka.

        Args:
            worker_id: Worker ID
            num_workers: Total number of workers

        Yields:
            PyArrow RecordBatch
        """
        topic, config, start_timestamp_ms = _parse_kafka_uri(self._input_path)
        consumer = Consumer(config)

        # Define on_assign callback to seek to checkpointed offsets
        def on_assign(consumer: Consumer, partitions: List[TopicPartition]) -> None:
            ts_partitions = []
            for tp in partitions:
                source_key = f"{tp.topic}:{tp.partition}"
                if self._checkpoint_state and source_key in self._checkpoint_state:
                    # Resume after last consumed offset
                    tp.offset = self._checkpoint_state[source_key] + 1
                elif start_timestamp_ms is not None:
                    ts_partitions.append(tp)
                else:
                    tp.offset = OFFSET_INVALID

            if ts_partitions:
                for tp in ts_partitions:
                    tp.offset = start_timestamp_ms
                resolved = consumer.offsets_for_times(ts_partitions, timeout=30.0)
                resolved_map = {(r.topic, r.partition): r.offset for r in resolved}
                for tp in ts_partitions:
                    res_offset = resolved_map.get((tp.topic, tp.partition))
                    if res_offset is not None and res_offset >= 0:
                        tp.offset = res_offset
                    else:
                        logger.warning(
                            f"No offset found for timestamp "
                            f"{start_timestamp_ms} on "
                            f"{tp.topic}:{tp.partition}, "
                            f"falling back to auto.offset.reset"
                        )
                        tp.offset = OFFSET_INVALID

            consumer.assign(partitions)
            logger.info(
                f"KafkaReader[rank-{os.environ.get('RANK', 0)}|worker-{worker_id}] "
                f"assignment: {consumer.assignment()}"
            )

        consumer.subscribe([topic], on_assign=on_assign)

        batch_size_per_msg = None
        try:
            while True:
                num_messages = (
                    int(math.ceil(self._batch_size / batch_size_per_msg))
                    if batch_size_per_msg
                    else 2
                )
                messages = consumer.consume(num_messages)

                current_batch_size = 0
                record_batchs = []
                # Track checkpoint metadata for each message
                source_ids = []
                row_indices = []

                for msg in messages:
                    msg_error = msg.error()
                    if msg_error:
                        logger.error(msg_error)
                        continue
                    msg_data = msg.value()
                    if self._has_embedded_schema:
                        # Read with embedded schema using IPC stream reader
                        # NOTE: Only the first record batch is read from each message.
                        # If a Kafka message contains multiple record batches in the IPC
                        # stream format, subsequent batches will be silently dropped.
                        # Each Kafka message should contain exactly one record batch.
                        reader = pa.ipc.open_stream(msg_data)
                        try:
                            record_batch = reader.read_next_batch()
                        finally:
                            reader.close()
                    else:
                        # Schema-less message (current behavior)
                        record_batch = pa.ipc.read_record_batch(
                            msg_data, self._full_schema
                        )
                    current_batch_size += len(record_batch)
                    record_batchs.append(record_batch)

                    # Generate checkpoint metadata for each row in this message
                    partition = msg.partition()
                    offset = msg.offset()
                    source_id = f"{topic}:{partition}"
                    batch_len = len(record_batch)
                    source_ids.extend([source_id] * batch_len)
                    # Use offset as the row index (each message has one offset)
                    row_indices.extend([offset] * batch_len)

                if not record_batchs:
                    continue

                # estimate batch_size per message
                if batch_size_per_msg is None:
                    batch_size_per_msg = current_batch_size / len(record_batchs)
                else:
                    batch_size_per_msg = (
                        0.9 * batch_size_per_msg
                        + 0.1 * current_batch_size / len(record_batchs)
                    )

                # combine into one record batch
                t = (
                    pa.Table.from_batches(record_batchs)
                    .drop_columns(self._drop_columns)
                    .combine_chunks()
                )

                # Add checkpoint metadata columns
                t = t.append_column(
                    CKPT_SOURCE_ID, pa.array(source_ids, type=pa.string())
                )
                t = t.append_column(
                    CKPT_ROW_IDX, pa.array(row_indices, type=pa.int64())
                )

                for batch in t.to_batches():
                    yield batch
        except Exception as e:
            logger.error(f"KafkaReader exception: {e}", flush=True)
            raise e
        finally:
            consumer.close()

    def to_batches(
        self, worker_id: int = 0, num_workers: int = 1
    ) -> Iterator[Dict[str, pa.Array]]:
        """Get batch iterator.

        Args:
            worker_id: Worker ID
            num_workers: Total number of workers

        Yields:
            Dict of column name to PyArrow Array
        """
        reader = self._reader(worker_id, num_workers)
        yield from self._arrow_reader_iter(reader)
