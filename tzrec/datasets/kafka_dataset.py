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
import threading
import time
from typing import Any, Callable, Dict, Generator, Iterator, List, Optional, Tuple
from urllib.parse import parse_qsl, urlparse

import pyarrow as pa
from confluent_kafka import (
    OFFSET_INVALID,
    Consumer,
    KafkaException,
    Message,
    TopicPartition,
)
from torch import distributed as dist

from tzrec.constant import Mode
from tzrec.datasets.dataset import BaseDataset, BaseReader
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
    if "enable.auto.commit" not in params:
        params["enable.auto.commit"] = "false"
    params["bootstrap.servers"] = broker

    return topic, params, start_timestamp_ms


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
        drop_remainder (bool): inert, the consume loop never ends so no tail
            batch is formed; min_batch_size is inert for the same reason.
        shuffle (bool): shuffle data or not.
        shuffle_buffer_size (int): buffer size for shuffle.
        input_fields (list): list of pa.Field for schema definition.
        kafka_partition_round_robin (bool): whether to read the partitions owned
            by each worker in strict round-robin order.
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
        )

        self._has_embedded_schema = False
        self._kafka_partition_round_robin = kafka_partition_round_robin
        self._log_every_n_messages = kafka_log_every_n_messages
        self._round_robin_partition_order: List[int] = (
            self._get_round_robin_partition_order()
            if kafka_partition_round_robin
            else []
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

    def _get_round_robin_partition_order(self) -> List[int]:
        """Query partition ids ordered by leader and partition id.

        Rank 0 queries the topic metadata and broadcasts the order so that all
        ranks and dataloader workers agree on partition ownership.

        Returns:
            partition ids sorted by (leader, partition id).

        Raises:
            ValueError: if topic metadata cannot be read.
        """
        topic, config, _ = _parse_kafka_uri(self._input_path)
        distributed = dist.is_initialized() and dist.get_world_size() > 1
        rank = dist.get_rank() if distributed else 0
        result: List[Any] = [None, None]
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
                    result = [
                        None,
                        sorted(
                            partitions,
                            key=lambda p: (partitions[p].leader, p),
                        ),
                    ]
                finally:
                    consumer.close()
            except Exception as e:
                result = [f"{type(e).__name__}: {e}", None]

        if distributed:
            dist.broadcast_object_list(result)

        error, partition_order = result
        if error is not None:
            raise ValueError(
                f"Failed to get kafka partition order for topic {topic}: {error}"
            )
        return partition_order

    def _set_start_offsets(
        self,
        consumer: Consumer,
        partitions: List[TopicPartition],
        start_timestamp_ms: Optional[int],
    ) -> None:
        """Set start offsets on partitions from checkpoint or start timestamp.

        Args:
            consumer (Consumer): the kafka consumer used for offset lookups.
            partitions (list): partitions whose offsets are modified in place.
            start_timestamp_ms (int, optional): start timestamp in milliseconds.
        """
        offt_batch_size = max(
            1, int(os.environ.get("TZREC_KAFKA_OFFSETS_FOR_TIMES_BATCH_SIZE", "32"))
        )
        offt_timeout = float(
            os.environ.get("TZREC_KAFKA_OFFSETS_FOR_TIMES_TIMEOUT", "30.0")
        )
        offt_retries = int(os.environ.get("TZREC_KAFKA_OFFSETS_FOR_TIMES_RETRIES", "2"))

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
            assert start_timestamp_ms is not None
            resolved_map = _offsets_for_times_batched(
                consumer,
                ts_partitions,
                start_timestamp_ms,
                offt_batch_size,
                offt_timeout,
                offt_retries,
            )
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
        self, worker_id: int = 0, num_workers: int = 1
    ) -> Generator[pa.RecordBatch, None, None]:
        """Read Arrow batches from Kafka.

        When partition round-robin is enabled the partitions owned by this
        worker are read strictly in turn by :meth:`_round_robin_reader`.
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

        topic, config, start_timestamp_ms = _parse_kafka_uri(self._input_path)
        if self._kafka_partition_round_robin:
            yield from self._round_robin_reader(
                topic, config, start_timestamp_ms, worker_id, num_workers
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
            logger.info(
                f"KafkaReader[rank-{os.environ.get('RANK', 0)}|worker-{worker_id}] "
                f"assignment: {consumer.assignment()}"
            )

        consumer.subscribe([topic], on_assign=on_assign)

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
                                        f"heartbeat: paused {len(assignment)}"
                                        f" partitions after {elapsed:.1f}s"
                                        f" idle"
                                    )
                                msg = consumer.poll(0)
                                if msg is not None and not msg.error():
                                    stolen_msgs.append(msg)
                except Exception:
                    logger.warning("heartbeat error", exc_info=True)
                stop_event.wait(1.0)

        heartbeat_thread = threading.Thread(
            target=_heartbeat,
            name=f"kafka-hb-w{worker_id}",
            daemon=True,
        )
        heartbeat_thread.start()

        batch_size_per_msg = None
        # name-based projection per seen message schema, so an upstream schema
        # change costs one recompute rather than one per message.
        projections: Dict[pa.Schema, Optional[List[int]]] = {}
        try:
            while True:
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
                        logger.debug("heartbeat: resumed partitions")
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
            logger.error(f"KafkaReader exception: {e}")
            raise e
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

    def _round_robin_reader(
        self,
        topic: str,
        config: Dict[str, Any],
        start_timestamp_ms: Optional[int],
        worker_id: int,
        num_workers: int,
    ) -> Generator[pa.RecordBatch, None, None]:
        """Read this worker's partitions strictly in round-robin order.

        Partitions are sliced from the leader-ordered list resolved at reader
        construction, then read in ascending partition order.  Each owned
        partition gets its own consumer, assigned up front.  The rotation
        cursor does not advance while the current partition has no valid message.

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
        consumers: List[Consumer] = []
        try:
            if len(self._round_robin_partition_order) < num_workers:
                raise ValueError(
                    f"kafka topic {topic} has "
                    f"{len(self._round_robin_partition_order)} partition(s), "
                    f"fewer than the {num_workers} dataloader workers; "
                    "kafka_partition_round_robin requires at least one partition "
                    "per worker."
                )
            owned_partitions = [
                TopicPartition(topic, partition)
                for partition in sorted(
                    self._round_robin_partition_order[worker_id::num_workers]
                )
            ]
            consumer = Consumer(config)
            consumers.append(consumer)
            self._set_start_offsets(consumer, owned_partitions, start_timestamp_ms)
            consumer.assign([owned_partitions[0]])
            for tp in owned_partitions[1:]:
                consumer = Consumer(config)
                consumers.append(consumer)
                consumer.assign([tp])
            logger.info(
                f"KafkaReader[rank-{os.environ.get('RANK', 0)}|worker-{worker_id}] "
                f"assignment: {owned_partitions}"
            )

            projections: Dict[pa.Schema, Optional[List[int]]] = {}
            idx = 0
            while True:
                messages = consumers[idx].consume(1, timeout=1.0)
                if not messages:
                    continue
                t, _ = self._decode_messages(messages, topic, projections)
                if t is None:
                    continue
                # A zero-row message advances the rotation even though it
                # yields no record batch.
                for batch in t.to_batches():
                    yield batch
                idx = (idx + 1) % len(consumers)
        except Exception as e:
            logger.error(f"KafkaReader exception: {e}")
            raise e
        finally:
            for consumer in reversed(consumers):
                try:
                    consumer.close()
                except Exception as e:
                    logger.warning(f"consumer.close() failed: {e}")

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
