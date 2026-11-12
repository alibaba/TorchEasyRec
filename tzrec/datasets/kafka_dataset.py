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
import torch.distributed as dist
from confluent_kafka import (
    OFFSET_INVALID,
    Consumer,
    KafkaException,
    TopicPartition,
)

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

_EARLIEST_RESETS = ("earliest", "smallest", "beginning")
_LATEST_RESETS = ("latest", "largest", "end")


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


def _decode_kafka_messages(
    messages: List[Any],
    has_embedded_schema: bool,
    full_schema: pa.Schema,
    schema: pa.Schema,
    projections: Dict[pa.Schema, Optional[List[int]]],
    topic: str,
    on_decoded: Optional[Callable[[str], None]] = None,
) -> Tuple[Optional[pa.Table], int, int]:
    """Decode kafka messages into one table with checkpoint metadata columns.

    Args:
        messages: kafka messages, each carrying exactly one record batch.
        has_embedded_schema: whether messages embed an IPC stream schema.
        full_schema: full schema for schema-less messages.
        schema: reader schema to project and cast to.
        projections: per-message-schema projection cache, mutated in place.
        topic: topic name for checkpoint source ids.
        on_decoded: optional callback for each successfully decoded source id.

    Returns:
        table: one table with the reader columns plus CKPT_SOURCE_ID,
            CKPT_ROW_IDX and DATA_TIMESTAMP columns; None when no message
            was decoded, including when every message was an error event.
        num_msgs: successfully decoded messages, for consume sizing.
        num_rows: total rows decoded.
    """
    record_batchs = []
    # Track checkpoint metadata for each message
    source_ids = []
    row_indices = []
    # Track event-time (Unix-epoch seconds) for each message
    timestamps = []
    num_msgs = 0
    num_rows = 0
    for msg in messages:
        msg_error = msg.error()
        if msg_error:
            logger.error(msg_error)
            continue
        msg_data = msg.value()
        if has_embedded_schema:
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
            record_batch = pa.ipc.read_record_batch(msg_data, full_schema)

        # Project onto the reader schema by column name, so upstream
        # columns may be added or reordered. Metadata is dropped as
        # a schema carrying it is unhashable and never projected on.
        msg_schema = record_batch.schema.remove_metadata()
        if msg_schema not in projections:
            if projections:
                logger.info(
                    f"new kafka message schema {msg_schema.names}, "
                    f"reading columns {schema.names}."
                )
            projections[msg_schema] = _build_projection(msg_schema, schema)
        projection = projections[msg_schema]
        if projection is not None:
            # cast normalizes the nested nullability differences
            # _build_projection tolerates, so that batches read from
            # either schema still combine into one table below.
            record_batch = pa.RecordBatch.from_arrays(
                [record_batch.column(i) for i in projection],
                names=schema.names,
            ).cast(schema)

        batch_len = len(record_batch)
        record_batchs.append(record_batch)
        num_msgs += 1
        num_rows += batch_len

        # Generate checkpoint metadata for each row in this message
        source_id = f"{topic}:{msg.partition()}"
        offset = msg.offset()
        source_ids.extend([source_id] * batch_len)
        # Use offset as the row index (each message has one offset)
        row_indices.extend([offset] * batch_len)
        # msg.timestamp() -> (type, ms), -1 when unavailable; use
        # Unix-epoch seconds and keep -1 as the sentinel.
        ts_ms = msg.timestamp()[1]
        ts_s = ts_ms / 1000.0 if ts_ms >= 0 else -1.0
        timestamps.extend([ts_s] * batch_len)

        if on_decoded is not None:
            on_decoded(source_id)

    if not record_batchs:
        return None, num_msgs, num_rows

    # combine into one record batch
    t = pa.Table.from_batches(record_batchs).combine_chunks()

    # Add checkpoint metadata columns
    t = t.append_column(CKPT_SOURCE_ID, pa.array(source_ids, type=pa.string()))
    t = t.append_column(CKPT_ROW_IDX, pa.array(row_indices, type=pa.int64()))
    # transient event-time column (seconds), read in _build_batch
    t = t.append_column(DATA_TIMESTAMP, pa.array(timestamps, type=pa.float64()))
    return t, num_msgs, num_rows


def _consume_decoded_tables(
    consumer: Consumer,
    topic: str,
    on_assign: Callable[[Consumer, List[TopicPartition]], None],
    stop_event: threading.Event,
    max_poll_interval_ms: int,
    batch_size: int,
    has_embedded_schema: bool,
    full_schema: pa.Schema,
    reader_schema: pa.Schema,
    log_prefix: str,
    orphan_pid: Optional[int] = None,
    on_decoded: Optional[Callable[[str], None]] = None,
) -> Iterator[pa.Table]:
    """Shared consume loop with heartbeat pause/resume protection.

    A background heartbeat thread monitors idle time (time since the last
    ``consume()`` call). When the caller is blocked (a full queue, a blocked
    ``yield``) for longer than 80% of ``max.poll.interval.ms``, the heartbeat
    thread pauses all partitions and calls ``consumer.poll(0)`` periodically
    to reset the application poll timer and keep the consumer in its group.
    This prevents ``MAX_POLL_EXCEEDED`` errors during long downstream pauses
    (checkpoint saves, evaluation, compilation warmup, etc.).

    The caller decides assignment offsets via ``on_assign`` (set
    ``tp.offset`` per partition; may raise); the actual ``assign`` call,
    re-pause and logging are handled here under the consumer lock.

    Args:
        consumer: the kafka consumer (not yet subscribed).
        topic: topic to subscribe to.
        on_assign: offset-resolution callback, invoked with the assigned
            partitions on every (re)assignment.
        stop_event: set to end the loop.
        max_poll_interval_ms: consumer's max.poll.interval.ms setting.
        batch_size: target rows per consume round (EMA-estimated).
        has_embedded_schema: whether messages embed an IPC stream schema.
        full_schema: full schema for schema-less messages.
        reader_schema: schema to project and cast to.
        log_prefix: log line prefix identifying this reader.
        orphan_pid: when set, the loop exits once its parent pid changes
            (the owning process died).
        on_decoded: optional callback for each successfully decoded source id.

    Yields:
        pa.Table: one decoded and projected table per consume round.
    """
    # the heartbeat gets its own stop signal: the shared stop_event is only
    # read here; setting it on exit would falsely signal a clean stop to
    # workers after an abnormal reader death
    hb_stop = threading.Event()
    consumer_lock = threading.RLock()
    is_paused = [False]
    last_consume_time = [time.monotonic()]
    idle_threshold = max_poll_interval_ms / 1000.0 * 0.8
    stolen_msgs: List = []

    def _on_assign(consumer: Consumer, partitions: List[TopicPartition]) -> None:
        on_assign(consumer, partitions)
        with consumer_lock:
            consumer.assign(partitions)
            if is_paused[0]:
                consumer.pause(consumer.assignment())
        logger.info(f"{log_prefix} assignment: {consumer.assignment()}")

    def _heartbeat() -> None:
        """Keep the consumer in its group while the caller is blocked."""
        while not hb_stop.is_set():
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
            hb_stop.wait(1.0)

    heartbeat_thread = threading.Thread(
        target=_heartbeat, name=f"{log_prefix}-hb", daemon=True
    )
    batch_size_per_msg = None
    # name-based projection per seen message schema, so an upstream schema
    # change costs one recompute rather than one per message.
    projections: Dict[pa.Schema, Optional[List[int]]] = {}
    heartbeat_thread.start()
    try:
        consumer.subscribe([topic], on_assign=_on_assign)
        while not stop_event.is_set():
            if orphan_pid is not None and os.getppid() != orphan_pid:
                logger.error(f"{log_prefix}: owner process gone, exiting")
                return
            num_messages = (
                int(math.ceil(batch_size / batch_size_per_msg))
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

            t, num_msgs, num_rows = _decode_kafka_messages(
                messages,
                has_embedded_schema,
                full_schema,
                reader_schema,
                projections,
                topic,
                on_decoded,
            )
            if t is None:
                continue

            # estimate batch_size per message
            if batch_size_per_msg is None:
                batch_size_per_msg = num_rows / num_msgs
            else:
                batch_size_per_msg = (
                    0.9 * batch_size_per_msg + 0.1 * num_rows / num_msgs
                )

            yield t
    except Exception as e:
        logger.error(f"{log_prefix} exception: {e}")
        raise
    finally:
        hb_stop.set()
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


def _consume_round_robin_tables(
    consumer_factory: Callable[..., Consumer],
    topic: str,
    config: Dict[str, Any],
    partition_order: List[int],
    worker_id: int,
    num_workers: int,
    set_start_offsets: Callable[[Consumer, List[TopicPartition]], None],
    stop_event: Any,
    has_embedded_schema: bool,
    full_schema: pa.Schema,
    reader_schema: pa.Schema,
    log_prefix: str,
    orphan_pid: Optional[int] = None,
    on_decoded: Optional[Callable[[str], None]] = None,
) -> Iterator[pa.Table]:
    """Read this worker's owned partitions strictly in round-robin order.

    Partitions are sliced from the leader-ordered list by global reader id,
    then read in ascending partition order. Each partition gets its own
    assigned consumer and one message per turn. Timeouts and error events
    keep the current partition; successfully decoded zero-row messages
    advance the rotation.

    Args:
        consumer_factory: consumer builder, called once per owned partition.
        topic: kafka topic name.
        config: kafka consumer configuration.
        partition_order: partition ids ordered by leader and partition id.
        worker_id: global reader id.
        num_workers: total number of global readers.
        set_start_offsets: callback that sets offsets on all owned partitions
            before the first consumer is assigned.
        stop_event: set to end the loop.
        has_embedded_schema: whether messages embed an IPC stream schema.
        full_schema: full schema for schema-less messages.
        reader_schema: schema to project and cast to.
        log_prefix: log line prefix identifying this reader.
        orphan_pid: when set, exit once the parent pid changes.
        on_decoded: optional callback for each successfully decoded source id.

    Yields:
        One decoded and projected table per successful message.

    Raises:
        ValueError: if the topic has fewer partitions than global readers.
    """
    consumers: List[Consumer] = []
    try:
        if len(partition_order) < num_workers:
            raise ValueError(
                f"kafka topic {topic} has {len(partition_order)} partition(s), "
                f"fewer than the {num_workers} dataloader workers; "
                "kafka_partition_round_robin requires at least one partition "
                "per worker."
            )
        owned_partitions = [
            TopicPartition(topic, partition)
            for partition in sorted(partition_order[worker_id::num_workers])
        ]
        consumer = consumer_factory(config)
        consumers.append(consumer)
        set_start_offsets(consumer, owned_partitions)
        consumer.assign([owned_partitions[0]])
        for tp in owned_partitions[1:]:
            consumer = consumer_factory(config)
            consumers.append(consumer)
            consumer.assign([tp])
        logger.info(f"{log_prefix} assignment: {owned_partitions}")

        projections: Dict[pa.Schema, Optional[List[int]]] = {}
        idx = 0
        while not stop_event.is_set():
            if orphan_pid is not None and os.getppid() != orphan_pid:
                logger.error(f"{log_prefix}: owner process gone, exiting")
                return
            messages = consumers[idx].consume(1, timeout=1.0)
            if not messages:
                continue
            table, _, _ = _decode_kafka_messages(
                messages,
                has_embedded_schema,
                full_schema,
                reader_schema,
                projections,
                topic,
                on_decoded,
            )
            if table is None:
                continue
            yield table
            idx = (idx + 1) % len(consumers)
    except Exception as e:
        logger.error(f"{log_prefix} exception: {e}")
        raise
    finally:
        for consumer in reversed(consumers):
            try:
                consumer.close()
            except Exception as e:
                logger.warning(f"consumer.close() failed: {e}")


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
    batch_size: Optional[int] = None,
    timeout: Optional[float] = None,
    max_retries: Optional[int] = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> Dict[Tuple[str, int], int]:
    """Resolve a start timestamp to offsets in chunks, with retry.

    ``consumer.offsets_for_times`` is split into chunks of ``batch_size``
    partitions to keep each broker request small, and every chunk is retried
    up to ``max_retries`` times on ``KafkaException`` (e.g. request timeout).
    A persistent failure re-raises rather than silently falling back, because
    this reader is shared with training, where falling back to
    ``auto.offset.reset`` would replay the whole topic.

    ``batch_size``, ``timeout`` and ``max_retries`` default to the
    ``TZREC_KAFKA_OFFSETS_FOR_TIMES_*`` environment variables, resolved at
    call time so callers and tests can override them explicitly.

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
    if batch_size is None:
        batch_size = max(
            1, int(os.environ.get("TZREC_KAFKA_OFFSETS_FOR_TIMES_BATCH_SIZE", "32"))
        )
    if timeout is None:
        timeout = float(os.environ.get("TZREC_KAFKA_OFFSETS_FOR_TIMES_TIMEOUT", "30.0"))
    if max_retries is None:
        max_retries = int(os.environ.get("TZREC_KAFKA_OFFSETS_FOR_TIMES_RETRIES", "2"))
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


def _consume_kafka(
    input_path: str,
    consumer_factory: Optional[Any],
    reader_id: int,
    num_readers: int,
    batch_size: int,
    reader_schema: pa.Schema,
    full_schema: pa.Schema,
    has_embedded_schema: bool,
    start_offsets: Optional[Dict[int, int]] = None,
    checkpoint_state: Optional[Dict[str, Any]] = None,
    stop_event: Optional[Any] = None,
    orphan_pid: Optional[int] = None,
    round_robin_partition_order: Optional[List[int]] = None,
    kafka_log_every_n_messages: int = 0,
) -> Iterator[pa.RecordBatch]:
    """Consume and decode Kafka with one of the two assignment policies.

    When ``start_offsets`` is given every (re)assignment seeks to this run's
    fixed offsets and a partition created after startup fails loudly; without
    it the BATCH policy (checkpoint offsets, ``start.timestamp.ms``, then
    ``auto.offset.reset``) is used. Consumer creation, decoding and the
    consumption loops are shared by both paths.

    Args:
        input_path: kafka URI.
        consumer_factory: consumer builder; the default builds real consumers.
        reader_id: global consumer slot passed to the factory.
        num_readers: global consumer slot count passed to the factory.
        batch_size: target rows per consume round.
        reader_schema: schema to project and cast each message to.
        full_schema: full schema for schema-less messages.
        has_embedded_schema: whether messages embed an IPC stream schema.
        start_offsets: fixed {partition: offset} start of a sample-shuffle run.
        checkpoint_state: BATCH resume state {topic:partition: offset}.
        stop_event: consume loop stop signal; a private event by default.
        orphan_pid: when set, the loop exits once its parent pid changes.
        round_robin_partition_order: leader-ordered partition ids, or None
            for ordinary consumer-group subscription.
        kafka_log_every_n_messages: decoded-message log interval, 0 disables.

    Yields:
        PyArrow RecordBatch.
    """
    topic, config, start_timestamp_ms = _parse_kafka_uri(input_path)
    factory = consumer_factory or _default_consumer_factory
    stop_event = stop_event if stop_event is not None else threading.Event()
    log_prefix = f"KafkaReader[rank-{os.environ.get('RANK', 0)}|worker-{reader_id}]"
    on_decoded = None
    if kafka_log_every_n_messages > 0:
        # process-local decoded-message counter and periodic logger
        decoded_message_count = 0
        decoded_partition_counts: Dict[str, int] = {}

        def on_decoded(source_id: str) -> None:
            """Count one decoded message and log at the configured interval."""
            nonlocal decoded_message_count
            decoded_partition_counts[source_id] = (
                decoded_partition_counts.get(source_id, 0) + 1
            )
            decoded_message_count += 1
            if decoded_message_count % kafka_log_every_n_messages == 0:
                logger.info(
                    f"{log_prefix} "
                    f"decoded_messages={decoded_message_count} "
                    f"partitions={decoded_partition_counts}"
                )

    def on_assign(consumer: Consumer, partitions: List[TopicPartition]) -> None:
        """Resolve fixed run offsets or the ordinary checkpoint/timestamp policy."""
        if start_offsets is not None:
            for tp in partitions:
                if tp.partition not in start_offsets:
                    # a partition created after startup has no resolved start
                    # offset; fail loudly instead of silently reading latest
                    raise ValueError(
                        f"new kafka partition {tp.partition} of {tp.topic} appeared "
                        "after startup; sample-level shuffle does not support "
                        "partition expansion during a run"
                    )
                # every takeover starts from this run's fixed start offset
                tp.offset = start_offsets[tp.partition]
            return

        ts_partitions = []
        for tp in partitions:
            source_key = f"{tp.topic}:{tp.partition}"
            if checkpoint_state and source_key in checkpoint_state:
                # Resume after last consumed offset
                tp.offset = checkpoint_state[source_key] + 1
            elif start_timestamp_ms is not None:
                ts_partitions.append(tp)
            else:
                tp.offset = OFFSET_INVALID

        if ts_partitions:
            assert start_timestamp_ms is not None
            resolved_map = _offsets_for_times_batched(
                consumer, ts_partitions, start_timestamp_ms
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

    if round_robin_partition_order is not None:
        loop = _consume_round_robin_tables(
            factory,
            topic,
            config,
            round_robin_partition_order,
            reader_id,
            num_readers,
            on_assign,
            stop_event,
            has_embedded_schema,
            full_schema,
            reader_schema,
            log_prefix,
            orphan_pid=orphan_pid,
            on_decoded=on_decoded,
        )
    else:
        max_poll_interval_ms = int(config.get("max.poll.interval.ms", "300000"))
        consumer = factory(config, reader_id=reader_id, num_readers=num_readers)
        loop = _consume_decoded_tables(
            consumer,
            topic,
            on_assign,
            stop_event,
            max_poll_interval_ms,
            batch_size,
            has_embedded_schema,
            full_schema,
            reader_schema,
            log_prefix,
            orphan_pid=orphan_pid,
            on_decoded=on_decoded,
        )
    try:
        for t in loop:
            for batch in t.to_batches():
                yield batch
    finally:
        loop.close()


# ------------------------------------------------------------------------
# sample-level shuffle: fixed per-run start offsets
# ------------------------------------------------------------------------


def _default_consumer_factory(
    config: Dict[str, Any], reader_id: int = 0, num_readers: int = 1
) -> Consumer:
    """Build a real Kafka consumer; test doubles may use the reader context."""
    return Consumer(config)


def _resolve_reset_offset(reset: str, low: int, high: int, source_key: str) -> int:
    """Resolve auto.offset.reset to a concrete offset, honoring all aliases.

    librdkafka accepts earliest/smallest/beginning and latest/largest/end;
    "error" must raise rather than silently pick either end.
    """
    if reset in _EARLIEST_RESETS:
        return low
    if reset in _LATEST_RESETS:
        return high
    raise ValueError(
        f"unsupported auto.offset.reset {reset!r} for {source_key}; "
        f"supported: {', '.join(_EARLIEST_RESETS + _LATEST_RESETS)}"
    )


def resolve_start_offsets(
    input_path: str,
    checkpoint_state: Optional[Dict[str, Any]] = None,
    consumer_factory: Optional[Any] = None,
) -> Dict[int, int]:
    """Resolve this run's fixed per-partition start offsets R.

    Every assignment during this run (initial subscribe and any rebalance)
    seeks to these offsets, so a mid-run rebalance replays data read since
    startup instead of re-interpreting latest.

    Resolution order per partition: last completed checkpoint offset + 1
    (no watermark checks, matching the BATCH path; librdkafka applies
    auto.offset.reset if the offset is out of range), then
    start.timestamp.ms, then auto.offset.reset (default latest). Timestamp
    and reset starts are resolved to concrete offsets once, here, with one
    temporary consumer that is closed before returning.

    Args:
        input_path: kafka URI.
        checkpoint_state: trusted baseline from a previous checkpoint,
            {topic:partition: last completed offset}.
        consumer_factory: consumer builder; tests inject a double, the
            default builds a real consumer.

    Returns:
        {partition: start offset}.

    Raises:
        ValueError: when a timestamp-resolved start offset is no longer
            readable because of the topic's retention policy.
    """
    topic, params, start_timestamp_ms = _parse_kafka_uri(input_path)
    factory = consumer_factory or _default_consumer_factory
    consumer = factory(params)
    try:
        meta = consumer.list_topics(topic, timeout=60.0)
        if topic not in meta.topics or not meta.topics[topic].partitions:
            raise ValueError(f"kafka topic {topic} not found or has no partitions")
        partitions = sorted(meta.topics[topic].partitions.keys())

        starts: Dict[int, int] = {}
        ts_tps: List[TopicPartition] = []
        for p in partitions:
            source_key = f"{topic}:{p}"
            tp = TopicPartition(topic, p)
            if checkpoint_state and source_key in checkpoint_state:
                # same as the BATCH path: resume at offset + 1 and leave any
                # out-of-range value to librdkafka's auto.offset.reset
                starts[p] = int(checkpoint_state[source_key]) + 1
            elif start_timestamp_ms is not None:
                # watermarks are queried after resolution, where an unresolved
                # timestamp also falls back to auto.offset.reset
                ts_tps.append(tp)
            else:
                low, high = consumer.get_watermark_offsets(tp, timeout=60.0)
                reset = params.get("auto.offset.reset", "latest").lower()
                starts[p] = _resolve_reset_offset(reset, low, high, source_key)
        if ts_tps:
            # same tunables as the batch-shuffle path
            resolved = _offsets_for_times_batched(consumer, ts_tps, start_timestamp_ms)
            for tp in ts_tps:
                low, high = consumer.get_watermark_offsets(tp, timeout=60.0)
                offset = resolved.get((tp.topic, tp.partition))
                if offset is None:
                    reset = params.get("auto.offset.reset", "latest").lower()
                    offset = _resolve_reset_offset(
                        reset, low, high, f"{tp.topic}:{tp.partition}"
                    )
                if offset < low:
                    raise ValueError(
                        f"start offset {offset} for {tp.topic}:{tp.partition} "
                        f"is no longer readable (retention low watermark {low})"
                    )
                starts[tp.partition] = offset
        return starts
    finally:
        consumer.close()


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
        consumer_factory: consumer builder; tests inject a double, the
            default builds real consumers.
        kafka_partition_round_robin: read owned partitions strictly in turn.
        kafka_log_every_n_messages: decoded-message log interval, 0 disables.
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
        consumer_factory: Optional[Any] = None,
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

        self._consumer_factory = consumer_factory or _default_consumer_factory
        self._has_embedded_schema = False
        self._round_robin_partition_order: Optional[List[int]] = (
            self._get_round_robin_partition_order()
            if kafka_partition_round_robin
            else None
        )
        self._log_every_n_messages = kafka_log_every_n_messages

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
        consumer = self._consumer_factory(config)

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
        pg = dist_util.get_dataset_process_group()
        rank = dist.get_rank(pg) if pg is not None else 0
        result: List[Any] = [None, None]
        if rank == 0:
            try:
                consumer = self._consumer_factory(config)
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

        if pg is not None:
            dist.broadcast_object_list(result, group=pg)

        error, partition_order = result
        if error is not None:
            raise ValueError(
                f"Failed to get kafka partition order for topic {topic}: {error}"
            )
        return partition_order

    def _reader(
        self, worker_id: int = 0, num_workers: int = 1
    ) -> Generator[pa.RecordBatch, None, None]:
        """Read Arrow batches from Kafka.

        Strict round-robin assigns one consumer per owned partition. In the
        ordinary subscription path, a background heartbeat thread monitors
        idle time since the last ``consume()`` call. When blocked at ``yield`` for
        longer than 80% of ``max.poll.interval.ms``, the heartbeat thread
        pauses all partitions and calls ``consumer.poll(0)`` periodically to
        reset the application poll timer and keep the consumer in its group.
        This prevents ``MAX_POLL_EXCEEDED`` errors during long downstream
        pauses (checkpoint saves, evaluation, compilation warmup, etc.).

        Args:
            worker_id: Worker ID
            num_workers: Total number of workers

        Yields:
            PyArrow RecordBatch
        """
        yield from _consume_kafka(
            self._input_path,
            self._consumer_factory,
            worker_id,
            num_workers,
            self._batch_size,
            self._schema,
            self._full_schema,
            self._has_embedded_schema,
            checkpoint_state=self._checkpoint_state,
            round_robin_partition_order=self._round_robin_partition_order,
            kafka_log_every_n_messages=self._log_every_n_messages,
        )

    def create_read_plan(
        self, rank: int, world_size: int, num_readers: int
    ) -> ReadPlan:
        """Build a per-rank Kafka read plan with fixed start offsets.

        Each rank's reader slots use global ids. Round-robin tasks receive
        the parent-resolved partition order; ordinary tasks subscribe to the
        shared consumer group. Rank 0
        resolves and broadcasts the run's fixed start offsets so a rank
        failure cannot leave the others waiting at the next collective.

        Args:
            rank: this process's rank.
            world_size: number of ranks.
            num_readers: number of reader processes on this rank.

        Returns:
            a non-finite read plan with one task per reader.
        """
        from tzrec.datasets.sample_shuffle_readers import read_kafka

        start_offsets = self._resolve_start_offsets_for_run()
        global_num_readers = world_size * num_readers
        tasks = [
            ReadTask(
                fn=read_kafka,
                kwargs=dict(
                    input_path=self._input_path,
                    reader_id=rank * num_readers + reader_id,
                    num_readers=global_num_readers,
                    start_offsets=start_offsets,
                    reader_schema=self._schema,
                    full_schema=self._full_schema,
                    has_embedded_schema=self._has_embedded_schema,
                    batch_size=self._batch_size,
                    consumer_factory=self._consumer_factory,
                    round_robin_partition_order=self._round_robin_partition_order,
                    kafka_log_every_n_messages=self._log_every_n_messages,
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
            is_finite=False,
            planned_rows=None,
            num_readers=num_readers,
        )

    def _resolve_start_offsets_for_run(self) -> Dict[int, int]:
        """Resolve this run's start offsets, broadcasting rank 0's outcome."""
        if dist.is_initialized():
            error = None
            start_offsets = None
            if dist.get_rank() == 0:
                try:
                    start_offsets = resolve_start_offsets(
                        self._input_path,
                        self._checkpoint_state,
                        self._consumer_factory,
                    )
                except Exception as e:
                    error = f"{type(e).__name__}: {e}"
            obj_list = [(start_offsets, error)]
            dist.broadcast_object_list(obj_list, src=0)
            start_offsets, error = obj_list[0]
            if error is not None:
                raise RuntimeError(f"failed to resolve kafka start offsets: {error}")
            assert start_offsets is not None
            return start_offsets
        return resolve_start_offsets(
            self._input_path, self._checkpoint_state, self._consumer_factory
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
