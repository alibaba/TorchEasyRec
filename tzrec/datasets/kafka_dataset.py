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


import copy
import math
import multiprocessing as mp
import os
import threading
import time
from typing import Any, Callable, Dict, Generator, Iterator, List, Optional, Set, Tuple
from urllib.parse import parse_qsl, urlparse

import pyarrow as pa
from confluent_kafka import (
    OFFSET_INVALID,
    Consumer,
    KafkaError,
    KafkaException,
    Message,
    TopicPartition,
)
from torch import distributed as dist

from tzrec.constant import Mode
from tzrec.datasets.dataset import BaseDataset, BaseReader
from tzrec.datasets.sample_shuffle import (
    ReadPlan,
    ReadTask,
    make_sample_table_schema,
)
from tzrec.datasets.utils import (
    CKPT_ROW_IDX,
    CKPT_SOURCE_ID,
    DATA_TIMESTAMP,
    FIELD_TYPE_TO_PA,
    get_input_fields_proto,
    remove_nullable,
)
from tzrec.features.feature import BaseFeature
from tzrec.protos import data_pb2
from tzrec.utils import dist_util
from tzrec.utils.logging_util import logger


def _parse_kafka_uri(
    uri: str,
) -> Tuple[str, Dict[str, Any], Optional[int], Optional[int]]:
    """Parse kafka URI into configuration dict.

    Args:
        uri: kafka URI.
            e.g. kafka://broker:9092/topic?group.id=xxx&auto.offset.reset=earliest
            Supports an optional ``start.timestamp.ms`` query parameter to begin
            consuming from the earliest offset whose timestamp is >= the given
            value (milliseconds since epoch), and an optional
            ``end.timestamp.ms`` parameter to bound the read window: at job
            start every partition resolves one exclusive end offset from it
            and the job replays only the data that already exists.  Example::

                kafka://broker:9092/topic?group.id=g1&start.timestamp.ms=1711929600000&end.timestamp.ms=1712016000000

    Returns:
        topic: kafka topic name.
        params: consumer config params (``start.timestamp.ms`` and
            ``end.timestamp.ms`` are removed from this dict because they are
            not native ``confluent_kafka`` consumer configuration keys).
        start_timestamp_ms: start timestamp in milliseconds, or ``None`` if not
            specified.
        end_timestamp_ms: end timestamp in milliseconds, or ``None`` if not
            specified.

    Raises:
        ValueError: on a malformed URI, a non-integer or negative timestamp,
            or a start timestamp greater than the end timestamp.
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

    # Extract custom timestamp parameters (not confluent_kafka config keys)
    start_timestamp_ms: Optional[int] = None
    end_timestamp_ms: Optional[int] = None
    for param_name in ("start.timestamp.ms", "end.timestamp.ms"):
        raw_ts = params.pop(param_name, None)
        if raw_ts is None:
            continue
        try:
            ts_value = int(raw_ts)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"{param_name} must be a valid integer, got: {raw_ts!r}"
            ) from e
        if ts_value < 0:
            raise ValueError(f"{param_name} must be a non-negative integer")
        if param_name == "start.timestamp.ms":
            start_timestamp_ms = ts_value
        else:
            end_timestamp_ms = ts_value
    if (
        start_timestamp_ms is not None
        and end_timestamp_ms is not None
        and start_timestamp_ms > end_timestamp_ms
    ):
        raise ValueError(
            f"start.timestamp.ms[{start_timestamp_ms}] must not be greater "
            f"than end.timestamp.ms[{end_timestamp_ms}]"
        )

    if "debug" not in params:
        params["debug"] = "assignor,conf"
    if "enable.auto.commit" not in params:
        params["enable.auto.commit"] = "false"
    if end_timestamp_ms is not None and "enable.partition.eof" not in params:
        # partition EOF events let a bounded reader finish a partition whose
        # remaining offsets are holes instead of waiting for a boundary message
        params["enable.partition.eof"] = "true"
    params["bootstrap.servers"] = broker

    return topic, params, start_timestamp_ms, end_timestamp_ms


def is_finite_kafka_input(input_path: str) -> bool:
    """Whether a kafka URI bounds its read window with end.timestamp.ms.

    Args:
        input_path: data input path from the config.

    Returns:
        True only for kafka URIs carrying ``end.timestamp.ms``.  A malformed
        value still reports True and fails when the reader parses the URI.
    """
    if not input_path.startswith("kafka://"):
        return False
    return "end.timestamp.ms" in dict(parse_qsl(urlparse(input_path).query))


def _build_projection(
    src_schema: pa.Schema, dst_schema: pa.Schema
) -> Optional[List[int]]:
    """Map every field of dst_schema onto a column index of src_schema.

    Matching is by name, so an upstream producer may append new columns or reorder
    existing ones without breaking the reader.  A column dst_schema needs but
    src_schema does not have is an error: filling it with nulls would silently turn
    a broken upstream into a model trained on default values.

    Args:
        src_schema (pa.Schema): schema of the incoming record batch.
        dst_schema (pa.Schema): schema the reader is contracted to produce.

    Returns:
        list: src_schema column index per dst_schema field, or ``None`` when the two
            schemas are already identical and no projection is needed.

    Raises:
        ValueError: if a dst_schema column is missing from src_schema or has a
            different type there.
    """
    if src_schema.equals(dst_schema):
        return None

    indices = []
    missing = []
    mismatched = []
    for field in dst_schema:
        idx = src_schema.get_field_index(field.name)
        if idx < 0:
            missing.append(field.name)
            continue
        src_type = remove_nullable(src_schema.field(idx).type)
        if src_type != remove_nullable(field.type):
            mismatched.append(f"{field.name}: {field.type} -> {src_type}")
        indices.append(idx)

    if missing or mismatched:
        reasons = []
        if missing:
            reasons.append(f"missing column(s) {missing}")
        if mismatched:
            reasons.append(f"changed column type(s) {mismatched}")
        raise ValueError(
            f"kafka message schema is incompatible with the reader schema: "
            f"{'; '.join(reasons)}. message columns: {src_schema.names}, "
            f"reader columns: {dst_schema.names}."
        )

    return indices


def _offsets_for_times_batched(
    consumer: Consumer,
    ts_partitions: List[TopicPartition],
    start_timestamp_ms: int,
    batch_size: int,
    timeout: float,
    max_retries: int,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> Dict[Tuple[str, int], int]:
    """Resolve a start timestamp to offsets in chunks, with retry.

    ``consumer.offsets_for_times`` is split into chunks of ``batch_size``
    partitions to keep each broker request small, and every chunk is retried
    up to ``max_retries`` times on ``KafkaException`` (e.g. request timeout).
    A persistent failure re-raises rather than silently falling back, because
    this reader is shared with training, where falling back to
    ``auto.offset.reset`` would replay the whole topic.

    Args:
        consumer (Consumer): the kafka consumer.
        ts_partitions (list): partitions to resolve.
        start_timestamp_ms (int): target timestamp in milliseconds.
        batch_size (int): number of partitions per ``offsets_for_times`` call.
        timeout (float): per-call request timeout in seconds.
        max_retries (int): retries per chunk after the first attempt.
        sleep_fn (callable): sleep between retries (injectable for testing).

    Returns:
        dict: map of (topic, partition) to resolved offset, including only
            partitions the broker resolved to a real (>= 0) offset.

    Raises:
        KafkaException: if a chunk still fails after exhausting retries.
    """
    resolved_map: Dict[Tuple[str, int], int] = {}
    for i in range(0, len(ts_partitions), batch_size):
        chunk = ts_partitions[i : i + batch_size]
        # offsets_for_times reads the target timestamp from the offset field.
        for tp in chunk:
            tp.offset = start_timestamp_ms
        retry_cnt = 0
        while True:
            try:
                resolved = consumer.offsets_for_times(chunk, timeout=timeout)
                for partition in resolved:
                    if partition.error is not None:
                        raise KafkaException(partition.error)
            except KafkaException as e:
                if retry_cnt >= max_retries:
                    logger.error(
                        f"offsets_for_times failed after {max_retries} retries "
                        f"for {len(chunk)} partition(s) (timeout={timeout}s); "
                        f"tune TZREC_KAFKA_OFFSETS_FOR_TIMES_* env vars"
                    )
                    raise
                retry_cnt += 1
                logger.warning(
                    f"offsets_for_times attempt {retry_cnt}/{max_retries} failed "
                    f"for {len(chunk)} partition(s), retrying: {e}"
                )
                sleep_fn(min(2.0 * retry_cnt, 10.0))
                continue
            break
        for r in resolved:
            if r.offset is not None and r.offset >= 0:
                resolved_map[(r.topic, r.partition)] = r.offset
    return resolved_map


def _offsets_for_times_settings() -> Tuple[int, float, int]:
    """Batch size, timeout and retries of ``offsets_for_times`` calls.

    Tunable via ``TZREC_KAFKA_OFFSETS_FOR_TIMES_*`` so topics with many
    partitions can avoid per-request timeouts.
    """
    return (
        max(1, int(os.environ.get("TZREC_KAFKA_OFFSETS_FOR_TIMES_BATCH_SIZE", "32"))),
        float(os.environ.get("TZREC_KAFKA_OFFSETS_FOR_TIMES_TIMEOUT", "30.0")),
        int(os.environ.get("TZREC_KAFKA_OFFSETS_FOR_TIMES_RETRIES", "2")),
    )


def _read_sample_batches(
    reader: "KafkaReader",
    worker_id: int,
    num_workers: int,
) -> Iterator[pa.RecordBatch]:
    """Consume Kafka in a reader process for the sample-level shuffle plan.

    Args:
        reader: KafkaReader that built the plan.
        worker_id: global reader id.
        num_workers: total number of global readers.

    Yields:
        decoded record batches with checkpoint metadata and event time.
    """
    yield from reader._reader(worker_id, num_workers)


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
            self._drop_remainder,
            min_batch_size=self._min_batch_size,
            shuffle=self._data_config.shuffle and self._mode == Mode.TRAIN,
            shuffle_buffer_size=self._data_config.shuffle_buffer_size,
            input_fields=input_fields if input_fields else None,
            kafka_partition_round_robin=self._data_config.kafka_partition_round_robin,
            kafka_log_every_n_messages=self._data_config.kafka_log_every_n_messages,
        )


class KafkaReader(BaseReader):
    """Kafka reader class.

    Args:
        input_path (str): kafka URI, e.g. kafka://broker:9092/topic?group.id=xxx
        batch_size (int): batch size.
        selected_cols (list): selection column names.
        drop_remainder (bool): drop the final tail batch.  A tail batch only
            exists when the URI bounds the window with ``end.timestamp.ms``;
            the unbounded consume loop never ends.
        min_batch_size (int): drop the final tail batch when it has fewer
            rows; active only with ``end.timestamp.ms`` as well.
        shuffle (bool): shuffle data or not.
        shuffle_buffer_size (int): buffer size for shuffle.
        input_fields (list): list of pa.Field for schema definition.
        kafka_partition_round_robin (bool): whether to read the partitions owned
            by each worker in round-robin order, taking enough messages per
            turn that one full rotation yields about one batch; partitions
            already caught up with the broker are skipped instead of blocking
            the rotation.
        kafka_log_every_n_messages (int): log per-worker decoded message counts
            every N messages, 0 disables.
    """

    def __init__(
        self,
        input_path: str,
        batch_size: int,
        selected_cols: Optional[List[str]] = None,
        drop_remainder: bool = False,
        shuffle: bool = False,
        shuffle_buffer_size: int = 32,
        input_fields: Optional[List[pa.Field]] = None,
        kafka_partition_round_robin: bool = False,
        kafka_log_every_n_messages: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            input_path,
            batch_size,
            selected_cols,
            drop_remainder,
            shuffle,
            shuffle_buffer_size,
            **kwargs,
        )

        self._has_embedded_schema = False
        self._kafka_partition_round_robin = kafka_partition_round_robin
        self._log_every_n_messages = kafka_log_every_n_messages
        # the fixed partition layout also serves the end-timestamp reader,
        # which assigns each worker its partitions without a consumer group
        _, _, _, end_timestamp_ms = _parse_kafka_uri(input_path)
        self._round_robin_partition_order, self._end_offsets = (
            self._get_partition_layout()
            if kafka_partition_round_robin or end_timestamp_ms is not None
            else ([], None)
        )

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
        topic, config, _, _ = _parse_kafka_uri(self._input_path)
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

    def _get_partition_layout(self) -> Tuple[List[int], Optional[Dict[int, int]]]:
        """Query the partition order and end offsets, broadcast from rank 0.

        Rank 0 lists the topic and, with an end timestamp configured,
        resolves every partition's exclusive end offset at job start, then
        broadcasts the whole layout so all ranks and dataloader workers
        agree on partition ownership and read windows.

        Returns:
            partition ids sorted by (leader, partition id); and the
            end-timestamp offsets as a partition-id to exclusive offset map,
            or ``None`` when no end timestamp is configured.

        Raises:
            ValueError: if the topic metadata or the end offsets cannot
                be read.
        """
        topic, config, _, end_timestamp_ms = _parse_kafka_uri(self._input_path)
        pg = dist_util.get_dataset_process_group(self._world_size)
        rank = dist.get_rank(pg) if pg is not None else 0
        result: List[Any] = [None, None, None]
        if rank == 0:
            try:
                consumer = Consumer(config)
                try:
                    metadata = consumer.list_topics(topic, timeout=30.0)
                    topic_metadata = metadata.topics.get(topic)
                    if topic_metadata is None or topic_metadata.error is not None:
                        error = (
                            topic_metadata.error
                            if topic_metadata is not None
                            else "no metadata"
                        )
                        raise ValueError(
                            f"Failed to get metadata for kafka topic {topic}: {error}"
                        )
                    partitions = topic_metadata.partitions
                    partition_order = sorted(
                        partitions,
                        key=lambda p: (partitions[p].leader, p),
                    )
                    end_offsets = (
                        self._snapshot_end_offsets(
                            consumer, topic, partition_order, end_timestamp_ms
                        )
                        if end_timestamp_ms is not None
                        else None
                    )
                    result = [None, partition_order, end_offsets]
                finally:
                    consumer.close()
            except Exception as e:
                result = [f"{type(e).__name__}: {e}", None, None]

        if pg is not None:
            dist.broadcast_object_list(result, group=pg)

        error, partition_order, end_offsets = result
        if error is not None:
            raise ValueError(
                f"Failed to get kafka partition layout for topic {topic}: {error}"
            )
        return partition_order, end_offsets

    def _snapshot_end_offsets(
        self,
        consumer: Consumer,
        topic: str,
        partitions: List[int],
        end_timestamp_ms: int,
    ) -> Dict[int, int]:
        """Resolve each partition's exclusive end offset at job start.

        The high watermark ``H`` is captured first so the window never grows
        past the data that exists at startup; then the end timestamp resolves
        to ``E`` through the batched offset queries.  ``min(E, H)`` pins the
        boundary even when new data arrives during the query; a partition
        without any message at or past the timestamp (``E == -1``, including
        empty partitions) simply ends at ``H``.

        Args:
            consumer (Consumer): the metadata-query consumer of the layout
                query; not assigned to any partition.
            topic (str): kafka topic name.
            partitions (list): partition ids to resolve.
            end_timestamp_ms (int): end timestamp in milliseconds.

        Returns:
            dict: partition id to exclusive end offset.

        Raises:
            KafkaException: if the end-timestamp queries fail after their
                retries; a failed query must not be mistaken for ``E == -1``.
        """
        highs: Dict[int, int] = {}
        for partition in partitions:
            _, high = consumer.get_watermark_offsets(
                TopicPartition(topic, partition), timeout=30.0
            )
            highs[partition] = high
        batch_size, timeout, retries = _offsets_for_times_settings()
        resolved = _offsets_for_times_batched(
            consumer,
            [TopicPartition(topic, partition) for partition in partitions],
            end_timestamp_ms,
            batch_size,
            timeout,
            retries,
        )
        return {
            partition: min(resolved[(topic, partition)], highs[partition])
            if (topic, partition) in resolved
            else highs[partition]
            for partition in partitions
        }

    def _set_start_offsets(
        self,
        consumer: Consumer,
        partitions: List[TopicPartition],
        start_timestamp_ms: Optional[int],
        end_offsets: Optional[Dict[int, int]] = None,
        auto_offset_reset: str = "largest",
    ) -> None:
        """Set start offsets on partitions from checkpoint or start timestamp.

        With an end-timestamp layout every partition resolves to an explicit
        offset: the checkpoint wins as before; a start-timestamp partition
        whose lookup returns nothing is an empty window of this snapshot, so
        it starts at its end offset instead of falling back to
        ``auto.offset.reset``, which could read data from before the window;
        a partition without a start timestamp resolves ``auto.offset.reset``
        locally so the reader knows the position (``earliest``/``smallest``
        reads from the low watermark, anything else leaves the snapshot
        empty for it).

        Args:
            consumer (Consumer): the kafka consumer used for offset lookups.
            partitions (list): partitions whose offsets are modified in place.
            start_timestamp_ms (int, optional): start timestamp in milliseconds.
            end_offsets (dict, optional): partition id to exclusive end offset
                of the startup snapshot.
            auto_offset_reset (str): ``auto.offset.reset`` of the consumer
                config, consulted only with ``end_offsets``.
        """
        batch_size, timeout, retries = _offsets_for_times_settings()

        ts_partitions = []
        for tp in partitions:
            source_key = f"{tp.topic}:{tp.partition}"
            if self._checkpoint_state and source_key in self._checkpoint_state:
                # Resume after last consumed offset
                tp.offset = self._checkpoint_state[source_key] + 1
            elif start_timestamp_ms is not None:
                ts_partitions.append(tp)
            elif end_offsets is not None:
                if auto_offset_reset in ("earliest", "smallest"):
                    low, _ = consumer.get_watermark_offsets(tp, timeout=30.0)
                    tp.offset = low
                else:
                    # latest (or unset) starts at the high watermark: the
                    # startup snapshot holds nothing new for this partition
                    tp.offset = end_offsets[tp.partition]
            else:
                tp.offset = OFFSET_INVALID

        if ts_partitions:
            assert start_timestamp_ms is not None
            resolved_map = _offsets_for_times_batched(
                consumer,
                ts_partitions,
                start_timestamp_ms,
                batch_size,
                timeout,
                retries,
            )
            for tp in ts_partitions:
                res_offset = resolved_map.get((tp.topic, tp.partition))
                if res_offset is not None and res_offset >= 0:
                    tp.offset = res_offset
                elif end_offsets is not None:
                    # no message at/past the start inside this snapshot: the
                    # window is empty, the partition ends before it begins
                    tp.offset = end_offsets[tp.partition]
                else:
                    logger.warning(
                        f"No offset found for timestamp "
                        f"{start_timestamp_ms} on "
                        f"{tp.topic}:{tp.partition}, "
                        f"falling back to auto.offset.reset"
                    )
                    tp.offset = OFFSET_INVALID

    def _filter_end_offsets(
        self,
        messages: List[Message],
        end_offsets: Dict[int, int],
        completed: Set[int],
    ) -> List[Message]:
        """Keep only the messages before each partition's end offset.

        A message at or past its partition's end offset is the boundary: it
        and every later message of that partition is dropped.  A partition
        also completes once its next consume position reaches the end
        offset, without waiting for the boundary message; and a partition
        EOF event completes it even when offsets have holes.  Other error
        messages carry no rows and pass through for _decode_messages to log.

        Args:
            messages (list): kafka messages fetched from the broker.
            end_offsets (dict): partition id to exclusive end offset.
            completed (set): partition ids already finished, updated in place.

        Returns:
            list: the messages to decode, in their original order.
        """
        retained = []
        for msg in messages:
            partition = msg.partition()
            msg_error = msg.error()
            if msg_error is not None:
                if (
                    msg_error.code() == KafkaError._PARTITION_EOF
                    and partition is not None
                ):
                    completed.add(partition)
                else:
                    retained.append(msg)
                continue
            if partition is None:
                # a message without a partition has no window to check
                retained.append(msg)
                continue
            if partition in completed:
                continue
            offset = msg.offset()
            if offset >= end_offsets[partition]:
                completed.add(partition)
                continue
            retained.append(msg)
            if offset + 1 >= end_offsets[partition]:
                completed.add(partition)
        return retained

    def _decode_messages(
        self,
        messages: List[Message],
        topic: str,
        projections: Dict[pa.Schema, Optional[List[int]]],
    ) -> Tuple[Optional[pa.Table], int]:
        """Decode Kafka messages into one table with checkpoint metadata.

        Args:
            messages (list): kafka messages to decode.
            topic (str): topic name, used to build the checkpoint source id.
            projections (dict): schema projection cache reused across calls.

        Returns:
            table: combined table, or ``None`` when no message was decoded.
            num_decoded: number of successfully decoded messages, including
                messages that contain zero rows.
        """
        record_batchs = []
        # Track checkpoint metadata for each message
        source_ids = []
        row_indices = []
        # Track event-time (Unix-epoch seconds) for each message
        timestamps = []

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
                record_batch = pa.ipc.read_record_batch(msg_data, self._full_schema)

            # Project onto the reader schema by column name, so upstream
            # columns may be added or reordered. Metadata is dropped as
            # a schema carrying it is unhashable and never projected on.
            msg_schema = record_batch.schema.remove_metadata()
            if msg_schema not in projections:
                if projections:
                    logger.info(
                        f"new kafka message schema {msg_schema.names}, "
                        f"reading columns {self._schema.names}."
                    )
                projections[msg_schema] = _build_projection(msg_schema, self._schema)
            projection = projections[msg_schema]
            if projection is not None:
                # cast normalizes the nested nullability differences
                # _build_projection tolerates, so that batches read from
                # either schema still combine into one table below.
                record_batch = pa.RecordBatch.from_arrays(
                    [record_batch.column(i) for i in projection],
                    names=self._schema.names,
                ).cast(self._schema)

            record_batchs.append(record_batch)

            # Generate checkpoint metadata for each row in this message
            partition = msg.partition()
            offset = msg.offset()
            source_id = f"{topic}:{partition}"
            batch_len = len(record_batch)
            source_ids.extend([source_id] * batch_len)
            # Use offset as the row index (each message has one offset)
            row_indices.extend([offset] * batch_len)
            # msg.timestamp() -> (type, ms), -1 when unavailable; use
            # Unix-epoch seconds and keep -1 as the sentinel.
            ts_ms = msg.timestamp()[1]
            ts_s = ts_ms / 1000.0 if ts_ms >= 0 else -1.0
            timestamps.extend([ts_s] * batch_len)

            if self._log_every_n_messages > 0:
                self._decoded_partition_counts[source_id] = (
                    self._decoded_partition_counts.get(source_id, 0) + 1
                )
                self._decoded_message_count += 1
                if self._decoded_message_count % self._log_every_n_messages == 0:
                    logger.info(
                        f"KafkaReader[rank-{os.environ.get('RANK', 0)}"
                        f"|worker-{self._worker_id}] "
                        f"decoded_messages={self._decoded_message_count} "
                        f"partitions={self._decoded_partition_counts}"
                    )

        if not record_batchs:
            return None, 0

        # combine into one record batch
        t = pa.Table.from_batches(record_batchs).combine_chunks()

        # Add checkpoint metadata columns
        t = t.append_column(CKPT_SOURCE_ID, pa.array(source_ids, type=pa.string()))
        t = t.append_column(CKPT_ROW_IDX, pa.array(row_indices, type=pa.int64()))
        # transient event-time column (seconds), read in _build_batch
        t = t.append_column(DATA_TIMESTAMP, pa.array(timestamps, type=pa.float64()))
        return t, len(record_batchs)

    def _reader(
        self,
        worker_id: int = 0,
        num_workers: int = 1,
    ) -> Generator[pa.RecordBatch, None, None]:
        """Read Arrow batches from Kafka.

        In a multiprocessing child, stop polling when the parent process exits.

        When partition round-robin is enabled the partitions owned by this
        worker are read in round-robin order by :meth:`_round_robin_reader`.
        When ``end.timestamp.ms`` is configured (and partition round-robin is
        not), the worker's fixed share of partitions is read up to their
        end offsets by :meth:`_finite_reader`; partitions whose window is
        empty, e.g. when the start equals the end timestamp, finish without
        yielding anything.
        Otherwise a background heartbeat thread monitors idle time (time since
        the last ``consume()`` call).  When the generator is blocked at
        ``yield`` for longer than 80% of ``max.poll.interval.ms``, the
        heartbeat thread pauses all partitions and calls ``consumer.poll(0)``
        periodically to reset the application poll timer and keep the consumer
        in its group.  This prevents ``MAX_POLL_EXCEEDED`` errors during long
        downstream pauses (checkpoint saves, evaluation, compilation warmup,
        etc.).

        Args:
            worker_id: Worker ID
            num_workers: Total number of workers

        Yields:
            PyArrow RecordBatch
        """
        self._worker_id = worker_id
        self._decoded_message_count = 0
        self._decoded_partition_counts: Dict[str, int] = {}

        topic, config, start_timestamp_ms, end_timestamp_ms = _parse_kafka_uri(
            self._input_path
        )
        log_prefix = f"KafkaReader[rank-{os.environ.get('RANK', 0)}|worker-{worker_id}]"
        if self._kafka_partition_round_robin:
            yield from self._round_robin_reader(
                topic,
                config,
                start_timestamp_ms,
                worker_id,
                num_workers,
            )
            return
        if end_timestamp_ms is not None:
            yield from self._finite_reader(
                topic,
                config,
                start_timestamp_ms,
                worker_id,
                num_workers,
            )
            return

        max_poll_interval_ms = int(config.get("max.poll.interval.ms", "300000"))
        consumer = Consumer(config)

        stop_event = threading.Event()
        consumer_lock = threading.RLock()
        is_paused = [False]
        last_consume_time = [time.monotonic()]
        idle_threshold = max_poll_interval_ms / 1000.0 * 0.8
        stolen_msgs: List = []

        # Define on_assign callback to seek to checkpointed offsets
        def on_assign(consumer: Consumer, partitions: List[TopicPartition]) -> None:
            self._set_start_offsets(consumer, partitions, start_timestamp_ms)
            with consumer_lock:
                consumer.assign(partitions)
                if is_paused[0]:
                    consumer.pause(consumer.assignment())
            logger.info(f"{log_prefix} assignment: {consumer.assignment()}")

        def _heartbeat() -> None:
            """Keep consumer alive when generator is blocked at yield."""
            while not stop_event.is_set():
                try:
                    elapsed = time.monotonic() - last_consume_time[0]
                    if elapsed > idle_threshold:
                        with consumer_lock:
                            # Re-check elapsed inside lock to avoid racing
                            # with the main thread that updates
                            # last_consume_time under the same lock.
                            elapsed = time.monotonic() - last_consume_time[0]
                            if elapsed > idle_threshold:
                                if not is_paused[0]:
                                    assignment = consumer.assignment()
                                    if assignment:
                                        consumer.pause(assignment)
                                    is_paused[0] = True
                                    logger.debug(
                                        f"{log_prefix} heartbeat: paused "
                                        f"{len(assignment)} partitions after "
                                        f"{elapsed:.1f}s idle"
                                    )
                                msg = consumer.poll(0)
                                if msg is not None and not msg.error():
                                    stolen_msgs.append(msg)
                except Exception:
                    logger.warning(f"{log_prefix} heartbeat error", exc_info=True)
                stop_event.wait(1.0)

        heartbeat_thread = threading.Thread(
            target=_heartbeat,
            name=f"{log_prefix}-hb",
            daemon=True,
        )
        batch_size_per_msg = None
        # name-based projection per seen message schema, so an upstream schema
        # change costs one recompute rather than one per message.
        projections: Dict[pa.Schema, Optional[List[int]]] = {}
        parent = mp.parent_process()
        heartbeat_thread.start()
        try:
            consumer.subscribe([topic], on_assign=on_assign)
            while True:
                if parent is not None and os.getppid() != parent.pid:
                    logger.error(f"{log_prefix}: owner process gone, exiting")
                    return
                num_messages = (
                    int(math.ceil(self._batch_size / batch_size_per_msg))
                    if batch_size_per_msg
                    else 2
                )
                # Hold consumer_lock during consume() so the heartbeat
                # thread cannot call pause() concurrently.
                # rd_kafka_consume_batch_queue() (used by consume()) is
                # not thread-safe with concurrent pause/resume/seek.
                with consumer_lock:
                    if is_paused[0]:
                        assignment = consumer.assignment()
                        if assignment:
                            consumer.resume(assignment)
                        is_paused[0] = False
                        logger.debug(f"{log_prefix} heartbeat: resumed partitions")
                    last_consume_time[0] = time.monotonic()
                    pending = list(stolen_msgs)
                    stolen_msgs.clear()
                    messages = pending
                    messages.extend(consumer.consume(num_messages, timeout=1.0))

                t, num_decoded = self._decode_messages(messages, topic, projections)
                if t is None:
                    continue

                # estimate batch_size per message
                if batch_size_per_msg is None:
                    batch_size_per_msg = len(t) / num_decoded
                else:
                    batch_size_per_msg = (
                        0.9 * batch_size_per_msg + 0.1 * len(t) / num_decoded
                    )

                for batch in t.to_batches():
                    yield batch
        except Exception as e:
            logger.error(f"{log_prefix} exception: {e}")
            raise
        finally:
            stop_event.set()
            heartbeat_thread.join(timeout=5.0)
            if heartbeat_thread.is_alive():
                logger.warning(
                    "Heartbeat thread did not exit within timeout; "
                    "skipping consumer.close() to avoid race."
                )
            else:
                try:
                    consumer.close()
                except Exception as e:
                    logger.warning(f"consumer.close() failed: {e}")

    def _finite_reader(
        self,
        topic: str,
        config: Dict[str, Any],
        start_timestamp_ms: Optional[int],
        worker_id: int,
        num_workers: int,
    ) -> Generator[pa.RecordBatch, None, None]:
        """Read this worker's fixed share of partitions up to their end offsets.

        The partitions are assigned in the same order as the round-robin
        reader but read with a single consumer: an end-timestamp job splits
        the topic itself instead of joining the shared consumer group, so a
        worker whose partitions are all done exits without a rebalance
        handing finished partitions to other workers.  A partition whose
        start position is already at or past its end offset finishes without
        consuming; when the last partition completes, the retained messages
        of the final fetch are delivered before the generator ends.

        Args:
            topic (str): kafka topic name.
            config (dict): kafka consumer configuration.
            start_timestamp_ms (int, optional): start timestamp in milliseconds.
            worker_id (int): global worker ID.
            num_workers (int): total number of global workers.

        Yields:
            PyArrow RecordBatch
        """
        end_offsets = self._end_offsets
        assert end_offsets is not None
        log_prefix = f"KafkaReader[rank-{os.environ.get('RANK', 0)}|worker-{worker_id}]"
        owned_partitions = [
            TopicPartition(topic, partition)
            for partition in sorted(
                self._round_robin_partition_order[worker_id::num_workers]
            )
        ]
        if not owned_partitions:
            logger.info(f"{log_prefix} no partitions assigned, nothing to read")
            return
        consumer = Consumer(config)
        parent = mp.parent_process()
        projections: Dict[pa.Schema, Optional[List[int]]] = {}
        completed: Set[int] = set()
        batch_size_per_msg: Optional[float] = None
        try:
            self._set_start_offsets(
                consumer,
                owned_partitions,
                start_timestamp_ms,
                end_offsets=end_offsets,
                auto_offset_reset=config.get("auto.offset.reset", "largest"),
            )
            consumer.assign(owned_partitions)
            completed = {
                tp.partition
                for tp in owned_partitions
                if tp.offset >= end_offsets[tp.partition]
            }
            owned_ends = {
                tp.partition: end_offsets[tp.partition] for tp in owned_partitions
            }
            logger.info(
                f"{log_prefix} assignment: {owned_partitions}, "
                f"end_offsets: {owned_ends}"
            )
            while len(completed) < len(owned_partitions):
                if parent is not None and os.getppid() != parent.pid:
                    logger.error(f"{log_prefix}: owner process gone, exiting")
                    return
                num_messages = (
                    int(math.ceil(self._batch_size / batch_size_per_msg))
                    if batch_size_per_msg
                    else 2
                )
                messages = self._filter_end_offsets(
                    consumer.consume(num_messages, timeout=1.0),
                    end_offsets,
                    completed,
                )
                t, num_decoded = self._decode_messages(messages, topic, projections)
                if t is not None:
                    if batch_size_per_msg is None:
                        batch_size_per_msg = len(t) / num_decoded
                    else:
                        batch_size_per_msg = (
                            0.9 * batch_size_per_msg + 0.1 * len(t) / num_decoded
                        )
                    for batch in t.to_batches():
                        yield batch
            logger.info(f"{log_prefix} every partition reached its end offset")
        except Exception as e:
            logger.error(f"{log_prefix} exception: {e}")
            raise
        finally:
            try:
                consumer.close()
            except Exception as e:
                logger.warning(f"consumer.close() failed: {e}")

    def _round_robin_reader(
        self,
        topic: str,
        config: Dict[str, Any],
        start_timestamp_ms: Optional[int],
        worker_id: int,
        num_workers: int,
    ) -> Generator[pa.RecordBatch, None, None]:
        """Read this worker's partitions in round-robin order, N messages per turn.

        Each partition has its own consumer. Estimated rows per message set
        the turn size so one rotation yields about one batch. Wait for unread
        messages below the cached high watermark and skip empty or caught-up
        partitions.
        Sleep briefly after an idle rotation to avoid busy spinning.
        With an end timestamp, a partition is removed from the rotation once
        it reaches its end offset; partitions already starting there are
        never polled, and after the last remaining partition completes, the
        retained messages of that final turn are delivered before the
        generator ends.

        Args:
            topic (str): kafka topic name.
            config (dict): kafka consumer configuration.
            start_timestamp_ms (int, optional): start timestamp in milliseconds.
            worker_id (int): global worker ID.
            num_workers (int): total number of global workers.

        Yields:
            PyArrow RecordBatch

        Raises:
            ValueError: if the topic has fewer partitions than the total number
                of workers.
        """
        log_prefix = f"KafkaReader[rank-{os.environ.get('RANK', 0)}|worker-{worker_id}]"
        consumers: List[Consumer] = []
        parent = mp.parent_process()
        try:
            if len(self._round_robin_partition_order) < num_workers:
                raise ValueError(
                    f"kafka topic {topic} has "
                    f"{len(self._round_robin_partition_order)} partition(s), "
                    f"fewer than the {num_workers} dataloader workers; "
                )
            owned_partitions = [
                TopicPartition(topic, partition)
                for partition in sorted(
                    self._round_robin_partition_order[worker_id::num_workers]
                )
            ]
            consumer = Consumer(config)
            consumers.append(consumer)
            self._set_start_offsets(
                consumer,
                owned_partitions,
                start_timestamp_ms,
                end_offsets=self._end_offsets,
                auto_offset_reset=config.get("auto.offset.reset", "largest"),
            )
            consumer.assign([owned_partitions[0]])
            for tp in owned_partitions[1:]:
                consumer = Consumer(config)
                consumers.append(consumer)
                consumer.assign([tp])
            logger.info(f"{log_prefix} assignment: {owned_partitions}")

            projections: Dict[pa.Schema, Optional[List[int]]] = {}
            next_offsets = [tp.offset for tp in owned_partitions]
            rows_per_msg: Optional[float] = None
            idle_turns = 0

            def _record_offsets(messages: List[Message], idx: int) -> int:
                """Record the next offset and return the number of data messages."""
                got = 0
                for msg in messages:
                    if not msg.error():
                        next_offsets[idx] = msg.offset() + 1
                        got += 1
                return got

            end_offsets = self._end_offsets
            completed: Set[int] = (
                {
                    tp.partition
                    for tp in owned_partitions
                    if tp.offset >= end_offsets[tp.partition]
                }
                if end_offsets is not None
                else set()
            )
            # partitions already at their end offset never join the rotation
            rotation = [
                idx
                for idx, tp in enumerate(owned_partitions)
                if tp.partition not in completed
            ]
            rot_pos = 0
            while rotation:
                if parent is not None and os.getppid() != parent.pid:
                    logger.error(f"{log_prefix}: owner process gone, exiting")
                    return
                msg_per_turn = (
                    max(
                        1,
                        math.ceil(self._batch_size / (rows_per_msg * len(rotation))),
                    )
                    if rows_per_msg
                    else 1
                )
                idx = rotation[rot_pos]
                consumer = consumers[idx]
                messages = consumer.consume(msg_per_turn, timeout=0)
                got = _record_offsets(messages, idx)
                caught_up = False
                if got < msg_per_turn:
                    low, high = consumer.get_watermark_offsets(
                        owned_partitions[idx], cached=True
                    )
                    next_offset = next_offsets[idx]
                    caught_up = high >= 0 and (low == high or next_offset >= high)
                    if not caught_up:
                        missing = (
                            1
                            if next_offset < 0 or high < 0
                            else min(msg_per_turn - got, high - next_offset)
                        )
                        more = consumer.consume(missing, timeout=1.0)
                        _record_offsets(more, idx)
                        messages.extend(more)
                if end_offsets is not None:
                    messages = self._filter_end_offsets(
                        messages, end_offsets, completed
                    )
                t, num_decoded = self._decode_messages(messages, topic, projections)
                if t is not None:
                    idle_turns = 0
                    if rows_per_msg is None:
                        rows_per_msg = len(t) / num_decoded
                    else:
                        rows_per_msg = 0.9 * rows_per_msg + 0.1 * len(t) / num_decoded
                    # NOTE: Zero-row messages still advance the rotation.
                    for batch in t.to_batches():
                        yield batch
                if (
                    end_offsets is not None
                    and owned_partitions[idx].partition in completed
                ):
                    logger.info(
                        f"{log_prefix} partition "
                        f"{owned_partitions[idx].partition} reached its end "
                        f"offset {end_offsets[owned_partitions[idx].partition]}"
                    )
                    rotation.pop(rot_pos)
                    if not rotation:
                        return
                    if rot_pos >= len(rotation):
                        rot_pos = 0
                    continue
                if t is None:
                    if caught_up:
                        rot_pos = (rot_pos + 1) % len(rotation)
                        idle_turns += 1
                        if idle_turns >= len(rotation):
                            time.sleep(0.01)
                            idle_turns = 0
                    continue
                rot_pos = (rot_pos + 1) % len(rotation)
        except Exception as e:
            logger.error(f"{log_prefix} exception: {e}")
            raise
        finally:
            for consumer in reversed(consumers):
                try:
                    consumer.close()
                except Exception as e:
                    logger.warning(f"consumer.close() failed: {e}")

    def create_read_plan(
        self, rank: int, world_size: int, num_readers: int
    ) -> ReadPlan:
        """Build a per-rank Kafka read plan.

        Each task carries a reader and a global reader id. Round-robin tasks
        reuse the parent-resolved partition order; ordinary tasks subscribe
        to the shared consumer group, except end-timestamp tasks, which read
        the worker's fixed partition share. Checkpoint state is applied
        on each assignment, like the BATCH path.

        An end-timestamp plan is finite without a planned row count, so the
        coordinator drains the pool after EOF.  Its tasks carry a reader
        snapshot whose checkpoint state is private, so the handle clearing
        the original reader's resume position for later full passes cannot
        erase this plan's first-pass start.

        Args:
            rank: this process's rank.
            world_size: number of ranks.
            num_readers: number of reader processes on this rank.

        Returns:
            a read plan with one task per reader; finite iff ``end.timestamp.ms``
            is configured, then without a planned row count.
        """
        _, _, _, end_timestamp_ms = _parse_kafka_uri(self._input_path)
        reader = self
        if end_timestamp_ms is not None:
            reader = copy.copy(self)
            reader._checkpoint_state = (
                dict(self._checkpoint_state) if self._checkpoint_state else None
            )
        global_num_readers = world_size * num_readers
        tasks = [
            ReadTask(
                fn=_read_sample_batches,
                kwargs=dict(
                    reader=reader,
                    worker_id=rank * num_readers + reader_id,
                    num_workers=global_num_readers,
                ),
                # local slot: groups tasks for this rank's reader processes
                reader_id=reader_id,
            )
            for reader_id in range(num_readers)
        ]
        return ReadPlan(
            tasks=tasks,
            # kafka batches carry the event-time column
            schema=make_sample_table_schema(self._schema, with_timestamp=True),
            is_finite=end_timestamp_ms is not None,
            planned_rows=None,
            num_readers=num_readers,
        )

    def to_batches(
        self, worker_id: int = 0, num_workers: int = 1, world_size: Optional[int] = None
    ) -> Iterator[Dict[str, pa.Array]]:
        """Get batch iterator.

        Args:
            worker_id: Worker ID
            num_workers: Total number of workers
            world_size: Unused, partitions are split by global worker id

        Yields:
            Dict of column name to PyArrow Array
        """
        reader = self._reader(worker_id, num_workers)
        try:
            yield from self._arrow_reader_iter(reader)
        finally:
            reader.close()
