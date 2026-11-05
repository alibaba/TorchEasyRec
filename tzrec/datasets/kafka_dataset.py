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


import atexit
import math
import multiprocessing as mp
import os
import queue
import threading
import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple
from urllib.parse import parse_qsl, urlparse

import numpy as np
import pyarrow as pa
import torch.distributed as dist
from confluent_kafka import (
    OFFSET_INVALID,
    Consumer,
    KafkaException,
    TopicPartition,
)

from tzrec.constant import Mode
from tzrec.datasets.dataset import (
    BaseDataset,
    BaseReader,
    sample_shuffle_enabled,
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
from tzrec.protos.data_pb2 import SAMPLE_ROLLING, SAMPLE_WINDOW
from tzrec.utils.logging_util import logger

# capacity of the reader -> coordinator input queue, in decoded tables
_INPUT_QUEUE_TABLES_PER_READER = 2
# capacity of the shared dispatch queue matches the DataLoader prefetch
# depth (default 2 batches) times the worker count
_DISPATCH_QUEUE_BATCHES_PER_WORKER = 2
# coordinator loop wakeups per second while idle
_LOOP_TICK_SECS = 0.1
_WATCHDOG_TICK_SECS = 5.0
# shutdown budget for reaping reader processes: a graceful phase then a
# forced phase, each shared by all readers; the handle's stop timeout must
# comfortably exceed the sum
_READER_GRACEFUL_STOP_SECS = 6.0
_READER_FORCED_STOP_SECS = 2.0
_STATS_LOG_INTERVAL_SECS = 60.0

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
) -> Tuple[Optional[pa.Table], int, int]:
    """Decode kafka messages into one table with checkpoint metadata columns.

    Args:
        messages: kafka messages, each carrying exactly one record batch.
        has_embedded_schema: whether messages embed an IPC stream schema.
        full_schema: full schema for schema-less messages.
        schema: reader schema to project and cast to.
        projections: per-message-schema projection cache, mutated in place.
        topic: topic name for checkpoint source ids.

    Returns:
        table: one table with the reader columns plus CKPT_SOURCE_ID,
            CKPT_ROW_IDX and DATA_TIMESTAMP columns; None when no message
            carried rows.
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


# ------------------------------------------------------------------------
# sample-level shuffle: shared per-rank pool, message ledger, coordination
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


class SamplePool:
    """Chunked raw-sample pool with sample-level shuffle.

    Rows stay in the chunk they arrived in; a numpy arena of (chunk, row)
    pairs is the sampling domain. SAMPLE_WINDOW freezes the pool content
    into a shuffled window once the pool reaches capacity and emits all
    full batches of it before refilling; SAMPLE_ROLLING extracts a random
    batch once the pool is refilled to capacity. Both modes wait while the
    pool is short of capacity and retain sub-batch leftovers.

    Checkpoint metadata columns (CKPT_SOURCE_ID/CKPT_ROW_IDX) and the
    event-time column ride through the pool together with the feature
    columns, so ``_build_batch`` can aggregate per-batch source offsets
    downstream.

    Args:
        capacity_rows: pool capacity in samples.
        batch_size: training batch size in samples.
        mode: SAMPLE_WINDOW or SAMPLE_ROLLING.
        rng: random source, injectable for tests.
    """

    def __init__(
        self,
        capacity_rows: int,
        batch_size: int,
        mode: int,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        if capacity_rows <= 0 or batch_size <= 0:
            raise ValueError("capacity_rows and batch_size must be positive")
        if mode not in (SAMPLE_WINDOW, SAMPLE_ROLLING):
            raise ValueError(f"unsupported sample shuffle mode: {mode}")
        self.capacity_rows = capacity_rows
        self.batch_size = batch_size
        self.mode = mode
        self.rng = rng if rng is not None else np.random.default_rng()
        # arrival chunks carry feature + checkpoint metadata + event-time
        # columns; the arena indexes live rows as (chunk, row) pairs
        self._chunks: List[pa.RecordBatch] = []
        self._arena_chunk = np.empty(0, dtype=np.int32)
        self._arena_row = np.empty(0, dtype=np.int32)
        self.total_rows = 0
        self._dead_rows = 0
        # window mode state
        self._frozen_chunk: Optional[np.ndarray] = None
        self._frozen_row: Optional[np.ndarray] = None
        self._frozen_pos = 0
        # stats
        self.num_emits = 0
        self.num_compactions = 0
        self.rows_in = 0
        self.rows_out = 0

    def add(self, batch: pa.RecordBatch) -> None:
        """Add a decoded record batch to the pool."""
        n = batch.num_rows
        if n == 0:
            return
        cid = len(self._chunks)
        self._chunks.append(batch)
        self._arena_chunk = np.concatenate(
            [self._arena_chunk, np.full(n, cid, dtype=np.int32)]
        )
        self._arena_row = np.concatenate(
            [self._arena_row, np.arange(n, dtype=np.int32)]
        )
        self.total_rows += n
        self.rows_in += n
        # frozen-window rows are outside the arena; compaction only runs
        # when no window is frozen so their chunk positions stay valid
        if self._dead_rows > self.capacity_rows and self._frozen_chunk is None:
            self._compact()

    def emit(self) -> Optional[pa.RecordBatch]:
        """Try to emit one full shuffled batch; None while the pool is short."""
        if self.mode == SAMPLE_WINDOW:
            return self._emit_window()
        return self._emit_rolling()

    def _emit_window(self) -> Optional[pa.RecordBatch]:
        """Fixed-window emit: freeze, permute, then emit all full batches."""
        if self._frozen_chunk is not None:
            # emit the frozen window's full batches; arrivals join the
            # next window instead of the frozen one
            n = len(self._frozen_chunk)
            if self._frozen_pos + self.batch_size > n - (n % self.batch_size):
                # window exhausted; leftover rows join the next fill phase
                leftover = slice(n - (n % self.batch_size), n)
                if n % self.batch_size > 0:
                    self._arena_chunk = np.concatenate(
                        [self._arena_chunk, self._frozen_chunk[leftover]]
                    )
                    self._arena_row = np.concatenate(
                        [self._arena_row, self._frozen_row[leftover]]
                    )
                self._frozen_chunk = None
                self._frozen_row = None
                # the window boundary is a safe point to reclaim dead chunk
                # buffers: leftover rows are back in the arena and the
                # frozen arrays no longer reference chunk positions
                if self._dead_rows > self.capacity_rows:
                    self._compact()
                return None
            sel = slice(self._frozen_pos, self._frozen_pos + self.batch_size)
            self._frozen_pos += self.batch_size
            batch = self._gather(self._frozen_chunk[sel], self._frozen_row[sel])
            self._remove_from_live(batch.num_rows)
            return batch
        if self.total_rows < self.capacity_rows:
            # wait until the pool is full
            return None
        n_freeze = self.total_rows
        perm = self.rng.permutation(n_freeze)
        self._frozen_chunk = self._arena_chunk[perm]
        self._frozen_row = self._arena_row[perm]
        self._arena_chunk = np.empty(0, dtype=np.int32)
        self._arena_row = np.empty(0, dtype=np.int32)
        self._frozen_pos = 0
        return self._emit_window()

    def _emit_rolling(self) -> Optional[pa.RecordBatch]:
        """Rolling-pool emit: draw a random batch once the pool is refilled."""
        if self.total_rows < self.capacity_rows:
            # wait until the pool is refilled to capacity
            return None
        sel = self.rng.choice(
            len(self._arena_chunk), size=self.batch_size, replace=False
        )
        batch = self._gather(self._arena_chunk[sel], self._arena_row[sel])
        keep = np.ones(len(self._arena_chunk), dtype=bool)
        keep[sel] = False
        self._arena_chunk = self._arena_chunk[keep]
        self._arena_row = self._arena_row[keep]
        self._remove_from_live(batch.num_rows)
        return batch

    def _gather(self, sel_chunk: np.ndarray, sel_row: np.ndarray) -> pa.RecordBatch:
        """Materialize selected rows into one batch."""
        order = np.argsort(sel_chunk, kind="stable")
        sorted_cids = sel_chunk[order]
        sorted_rows = sel_row[order]
        uniq, starts, counts = np.unique(
            sorted_cids, return_index=True, return_counts=True
        )
        out_batches = []
        for u, s, c in zip(uniq, starts, counts):
            out_batches.append(
                self._chunks[int(u)].take(
                    pa.array(sorted_rows[s : s + c].astype(np.int64))
                )
            )
        batch = pa.Table.from_batches(out_batches).combine_chunks()
        return batch.to_batches(max_chunksize=batch.num_rows)[0]

    def _remove_from_live(self, n: int) -> None:
        """Account emitted rows; they stay as dead rows in chunk buffers."""
        self.total_rows -= n
        self.rows_out += n
        self.num_emits += 1
        self._dead_rows += n

    def _compact(self) -> None:
        """Physically drop taken rows and dead chunk slots, rebuild densely.

        Only runs while no window is frozen: frozen rows reference chunk
        positions that compaction would move. Chunk lists are rebuilt with
        live chunks only, so their size stays proportional to the number of
        live chunks rather than the number of tables ever received.
        """
        # group arena entries by chunk without per-row python loops
        order = np.argsort(self._arena_chunk, kind="stable")
        arena_c = self._arena_chunk[order]
        arena_r = self._arena_row[order]
        uniq, starts, counts = np.unique(arena_c, return_index=True, return_counts=True)
        new_chunks: List[pa.RecordBatch] = []
        for u, st, c in zip(uniq, starts, counts):
            cid = int(u)
            rows = arena_r[st : st + c]
            sorted_rows = np.sort(rows)
            chunk = self._chunks[cid]
            if len(rows) != chunk.num_rows:
                chunk = chunk.take(pa.array(sorted_rows.astype(np.int64)))
            id_map = len(new_chunks)
            new_chunks.append(chunk)
            # remap entries: new chunk id, and new row id = rank in sorted order
            arena_c[st : st + c] = id_map
            arena_r[st : st + c] = np.searchsorted(sorted_rows, rows)
        # the arena feeds random sampling only, so the chunk-grouped order
        # is fine as-is
        self._chunks = new_chunks
        self._arena_chunk = arena_c
        self._arena_row = arena_r
        self._dead_rows = 0
        self.num_compactions += 1


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
        )
        # created in prepare_for_loading; forked worker copies inherit it, but
        # only the trainer process starts and stops it
        self._sample_shuffle: Optional[SampleShuffleHandle] = None

    def prepare_for_loading(self, checkpoint_state: Optional[Dict[str, Any]]) -> None:
        """In sample mode: create the handle and configure the reader.

        The reader object is kept; it serves pre-shuffled batches from the
        shared queue once configured. In batch mode this defers to the
        default checkpoint handling.
        """
        if not sample_shuffle_enabled(self._data_config, self._mode):
            super().prepare_for_loading(checkpoint_state)
            return
        kafka_reader = self._reader
        # the reader captured the module default factory at construction
        # time; pass it explicitly so spawned processes do not need the patch
        handle = SampleShuffleHandle(
            self._data_config,
            self._input_path,
            checkpoint_state,
            consumer_factory=kafka_reader._consumer_factory,
        )
        kafka_reader._configure_sample_shuffle(
            handle.shared_dispatch_queue,
            handle.stop_event,
            handle.fatal_event,
        )
        self._sample_shuffle = handle

    def is_sample_shuffled(self) -> bool:
        """Whether the sample-level shuffle path serves this dataset."""
        return self._sample_shuffle is not None

    def start_loading(self) -> None:
        """Spawn the coordinator once the DataLoader workers exist."""
        if self._sample_shuffle is not None:
            reader = self._reader
            assert isinstance(reader, KafkaReader)
            self._sample_shuffle.start(reader)

    def stop_loading(self) -> None:
        """Stop the coordinator and its reader processes; idempotent."""
        if self._sample_shuffle is not None:
            self._sample_shuffle.stop()


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
        # bound by _configure_sample_shuffle when the shared-pool path serves
        # this reader; the forked worker copies inherit the binding
        self._sample_queue: Optional[Any] = None
        self._sample_stop_event: Optional[Any] = None
        self._sample_fatal_event: Optional[Any] = None
        self._sample_table_schema: Optional[pa.Schema] = None

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

    def _reader(
        self, worker_id: int = 0, num_workers: int = 1
    ) -> Iterator[pa.RecordBatch]:
        """Read Arrow batches from Kafka.

        A background heartbeat thread monitors idle time (time since the last
        ``consume()`` call).  When the generator is blocked at ``yield`` for
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
        topic, config, start_timestamp_ms = _parse_kafka_uri(self._input_path)
        max_poll_interval_ms = int(config.get("max.poll.interval.ms", "300000"))
        consumer = self._consumer_factory(
            config, reader_id=worker_id, num_readers=num_workers
        )

        # seek to the checkpointed offsets on every (re)assignment
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

        loop = _consume_decoded_tables(
            consumer,
            topic,
            on_assign,
            threading.Event(),
            max_poll_interval_ms,
            self._batch_size,
            self._has_embedded_schema,
            self._full_schema,
            self._schema,
            f"KafkaReader[rank-{os.environ.get('RANK', 0)}|worker-{worker_id}]",
        )
        try:
            for t in loop:
                for batch in t.to_batches():
                    yield batch
        finally:
            loop.close()

    def _configure_sample_shuffle(
        self,
        shared_dispatch_queue: Any,
        stop_event: Any,
        fatal_event: Any,
    ) -> None:
        """Bind the shared-pool path's communication objects to this reader.

        After this call ``to_batches`` consumes batches produced by the
        per-rank coordinator instead of reading Kafka in the workers. The
        dispatch schema is the reader schema plus the checkpoint metadata and
        event-time columns that ride through the pool.

        Args:
            shared_dispatch_queue: the shared dispatch queue.
            stop_event: shared clean-stop signal.
            fatal_event: set when the coordinator died unexpectedly.
        """
        self._sample_queue = shared_dispatch_queue
        self._sample_stop_event = stop_event
        self._sample_fatal_event = fatal_event
        self._sample_table_schema = pa.schema(
            list(self._schema)
            + [
                pa.field(CKPT_SOURCE_ID, pa.string()),
                pa.field(CKPT_ROW_IDX, pa.int64()),
                pa.field(DATA_TIMESTAMP, pa.float64()),
            ]
        )

    def to_batches(
        self, worker_id: int = 0, num_workers: int = 1, world_size: Optional[int] = None
    ) -> Iterator[Dict[str, pa.Array]]:
        """Get batch iterator.

        In BATCH mode this reads Kafka in the worker and shuffles whole
        batches; when sample shuffle is configured it takes pre-shuffled
        batches from the shared dispatch queue instead.

        Args:
            worker_id: Worker ID
            num_workers: Total number of workers
            world_size: Unused, partitions are split by global worker id

        Yields:
            Dict of column name to PyArrow Array
        """
        if self._sample_queue is None:
            reader = self._reader(worker_id, num_workers)
            yield from self._arrow_reader_iter(reader)
            return
        yield from self._sample_batches(self._sample_queue)

    def _sample_batches(self, q: Any) -> Iterator[Dict[str, pa.Array]]:
        """Yield pre-batched raw columns from the shared dispatch queue."""
        while True:
            try:
                payload = q.get(timeout=1.0)
            except queue.Empty:
                if self._sample_fatal_event.is_set():
                    raise RuntimeError(
                        "sample-shuffle coordinator exited unexpectedly"
                    ) from None
                if self._sample_stop_event.is_set():
                    return
                continue
            if self._sample_stop_event.is_set():
                # clean stop: end the pass
                return
            rb = pa.ipc.read_record_batch(payload, self._sample_table_schema)
            yield {name: rb.column(i) for i, name in enumerate(rb.schema.names)}


def _reader_process_main(
    input_path: str,
    reader_id: int,
    num_readers: int,
    coordinator_pid: int,
    start_offsets: Dict[int, int],
    input_q: Any,
    stop_event: Any,
    reader_schema: pa.Schema,
    full_schema: pa.Schema,
    has_embedded_schema: bool,
    batch_size: int,
    consumer_factory: Optional[Any] = None,
) -> None:
    """Reader process: one consumer, decode, send tables to the coordinator.

    Shares the consume loop (heartbeat pause/resume, decode, EMA sizing) with
    KafkaReader via ``_consume_decoded_tables``; only the assignment policy
    (fixed start offsets for this run) and the output sink differ.
    ``consumer_factory`` is the test injection point; the default builds a
    real consumer and ignores the reader context.
    """
    topic, config, _ = _parse_kafka_uri(input_path)
    max_poll_interval_ms = int(config.get("max.poll.interval.ms", "300000"))
    factory = consumer_factory or _default_consumer_factory
    consumer = factory(config, reader_id=reader_id, num_readers=num_readers)

    def _put(item: Tuple) -> None:
        # bounded queue: block as backpressure, but stay stoppable
        while not stop_event.is_set():
            try:
                input_q.put(item, timeout=0.5)
                return
            except queue.Full:
                if os.getppid() != coordinator_pid:
                    return

    def on_assign(consumer: Consumer, partitions: List[TopicPartition]) -> None:
        for tp in partitions:
            if tp.partition not in start_offsets:
                # a partition created after startup has no resolved start
                # offset; fail loudly instead of silently reading from latest
                raise ValueError(
                    f"new kafka partition {tp.partition} of {tp.topic} appeared "
                    "after startup; sample-level shuffle does not support "
                    "partition expansion during a run"
                )
            # every takeover starts from this run's fixed start offset
            tp.offset = start_offsets[tp.partition]

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
        f"sample-shuffle reader {reader_id}",
        orphan_pid=coordinator_pid,
    )
    try:
        for t in loop:
            if t.num_rows == 0:
                continue
            payload = t.to_batches(max_chunksize=t.num_rows)[0].serialize().to_pybytes()
            _put(payload)
    finally:
        # the queue feeder thread may hold undelivered data; without this the
        # interpreter would wait for the flush on exit and hang on a full pipe
        input_q.cancel_join_thread()
        loop.close()


class _Coordinator:
    """Per-rank sample pool and dispatch event loop.

    Args:
        input_path: kafka URI.
        num_readers: number of reader processes to fork.
        start_offsets: resolved per-partition start offsets R.
        input_q: bounded queue from readers, items are payload bytes.
        stop_event: shared stop signal; when set, the coordinator exits.
        shared_dispatch_queue: the one bounded queue to the workers.
        mode: SAMPLE_WINDOW or SAMPLE_ROLLING.
        capacity_rows: pool capacity in samples.
        batch_size: training batch size.
        input_table_schema: schema of decoded tables on input_q.
        reader_schema: projected reader schema for decoding.
        full_schema: full message schema for schema-less decoding.
        has_embedded_schema: whether messages embed an IPC stream schema.
        trainer_pid: trainer process pid; the coordinator exits when its
            parent changes (trainer died without stopping it).
        consumer_factory: consumer builder for reader processes; tests inject
            a double, the default builds real consumers.
    """

    def __init__(
        self,
        input_path: str,
        num_readers: int,
        start_offsets: Dict[int, int],
        input_q: Any,
        stop_event: Any,
        shared_dispatch_queue: Any,
        mode: int,
        capacity_rows: int,
        batch_size: int,
        input_table_schema: pa.Schema,
        reader_schema: pa.Schema,
        full_schema: pa.Schema,
        has_embedded_schema: bool,
        trainer_pid: int,
        consumer_factory: Optional[Any] = None,
    ) -> None:
        self._trainer_pid = trainer_pid
        self._input_path = input_path
        self._num_readers = num_readers
        self._input_q = input_q
        self._stop_event = stop_event
        self._shared_dispatch_queue = shared_dispatch_queue
        self._input_table_schema = input_table_schema
        self._reader_schema = reader_schema
        self._full_schema = full_schema
        self._has_embedded_schema = has_embedded_schema
        self._consumer_factory = consumer_factory
        self._pool = SamplePool(capacity_rows, batch_size, mode)
        self._start_offsets = start_offsets
        self._batch_size = batch_size
        self._pending_dispatch: Optional[bytes] = None
        self._readers: List[Any] = []

    def _start_readers(self) -> None:
        """Fork the reader processes."""
        # fork before the event loop starts: the coordinator is still
        # single-threaded and holds no consumer, so fork is safe
        ctx = mp.get_context("fork")
        pid = os.getpid()
        for i in range(self._num_readers):
            p = ctx.Process(
                target=_reader_process_main,
                args=(
                    self._input_path,
                    i,
                    self._num_readers,
                    pid,
                    self._start_offsets,
                    self._input_q,
                    self._stop_event,
                    self._reader_schema,
                    self._full_schema,
                    self._has_embedded_schema,
                    self._batch_size,
                    self._consumer_factory,
                ),
                daemon=True,
                name=f"tzrec-sample-reader-{i}",
            )
            p.start()
            self._readers.append(p)

    def _emit_and_dispatch(self) -> int:
        """Retry the pending dispatch, then emit and send pool batches.

        At most one payload is in flight, so a blocked output never drains
        the pool: the next batch is emitted only after the pending one was
        accepted. Returns how many batches were dispatched this call,
        including a pending payload that the output queue finally accepted.
        """
        n = 0
        while True:
            if self._pending_dispatch is None:
                out = self._pool.emit()
                if out is None:
                    return n
                self._pending_dispatch = out.serialize().to_pybytes()
            try:
                self._shared_dispatch_queue.put_nowait(self._pending_dispatch)
            except queue.Full:
                # keep servicing control while the output is blocked
                return n
            self._pending_dispatch = None
            n += 1

    def _service_input(self, timeout: float) -> None:
        """Take one decoded table into the pool."""
        try:
            payload = self._input_q.get(timeout=timeout)
        except queue.Empty:
            return
        self._pool.add(pa.ipc.read_record_batch(payload, self._input_table_schema))

    def _check_readers(self) -> None:
        """Fail when any reader process died."""
        for p in self._readers:
            if not p.is_alive():
                raise RuntimeError(
                    f"sample-shuffle reader exited unexpectedly (code {p.exitcode})"
                )

    def _log_stats(self) -> None:
        """Log pool counters."""
        logger.info(
            f"sample-shuffle stats: pool {self._pool.total_rows} rows "
            f"(capacity {self._pool.capacity_rows}), rows in {self._pool.rows_in} "
            f"out {self._pool.rows_out}, emits {self._pool.num_emits}, "
            f"compactions {self._pool.num_compactions}"
        )

    def run(self) -> None:
        """Fork readers, then serve input until the shared stop event fires."""
        self._start_readers()
        last_watchdog = time.monotonic()
        last_stats = time.monotonic()
        try:
            while not self._stop_event.is_set():
                now = time.monotonic()
                took_input = False
                if self._pool.total_rows < self._pool.capacity_rows:
                    self._service_input(_LOOP_TICK_SECS)
                    took_input = True
                emitted = self._emit_and_dispatch()
                if not took_input and emitted == 0:
                    # pool full while the output is blocked: pace the loop
                    # so control stays serviced without busy spinning
                    time.sleep(_LOOP_TICK_SECS)
                if now - last_watchdog >= _WATCHDOG_TICK_SECS:
                    if os.getppid() != self._trainer_pid:
                        logger.error("sample-shuffle coordinator orphaned; exiting")
                        break
                    self._check_readers()
                    last_watchdog = now
                if now - last_stats >= _STATS_LOG_INTERVAL_SECS:
                    self._log_stats()
                    last_stats = now
        finally:
            self._log_stats()
            self._shutdown_readers()
            # drop undelivered dispatch payloads instead of hanging this
            # process on a full queue's feeder thread
            self._shared_dispatch_queue.cancel_join_thread()

    def _shutdown_readers(self) -> None:
        """Reap all readers: one shared graceful deadline, then a forced one.

        Readers watch the shared stop event (clean stop) or their parent pid
        (coordinator death), so nothing needs signaling here.
        """
        deadline = time.monotonic() + _READER_GRACEFUL_STOP_SECS
        for p in self._readers:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            p.join(timeout=remaining)
        deadline = time.monotonic() + _READER_FORCED_STOP_SECS
        for p in self._readers:
            if p.is_alive():
                p.terminate()
        for p in self._readers:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            p.join(timeout=remaining)


def _coordinator_entry(**kwargs: Any) -> None:
    """Coordinator process entry point."""
    try:
        _Coordinator(**kwargs).run()
    except Exception:
        logger.error("sample-shuffle coordinator failed", exc_info=True)
        raise


class SampleShuffleHandle:
    """Owns the per-rank sample-shuffle background resources end to end.

    Created before the DataLoader forks its workers (the queues and events
    are inherited by them); ``start()`` spawns the coordinator process after
    the fork; ``stop()`` shuts everything down. The trainer interacts with
    it only through ``KafkaDataset``'s loading lifecycle methods.

    Args:
        data_config: dataloader config.
        input_path: kafka URI.
        checkpoint_state: trusted baseline from a previous checkpoint.
        consumer_factory: consumer builder; tests inject a double, the
            default builds real consumers.
    """

    def __init__(
        self,
        data_config: data_pb2.DataConfig,
        input_path: str,
        checkpoint_state: Optional[Dict[str, Any]],
        consumer_factory: Optional[Any] = None,
    ) -> None:
        validate_sample_shuffle_config(data_config)
        if dist.is_initialized():
            # rank 0 resolves and broadcasts the outcome; a failure on rank 0
            # must fail every rank, not leave the others waiting at the next
            # collective
            error = None
            start_offsets = None
            if dist.get_rank() == 0:
                try:
                    start_offsets = resolve_start_offsets(
                        input_path, checkpoint_state, consumer_factory
                    )
                except Exception as e:
                    error = f"{type(e).__name__}: {e}"
            obj_list = [(start_offsets, error)]
            dist.broadcast_object_list(obj_list, src=0)
            start_offsets, error = obj_list[0]
            if error is not None:
                raise RuntimeError(f"failed to resolve kafka start offsets: {error}")
        else:
            start_offsets = resolve_start_offsets(
                input_path, checkpoint_state, consumer_factory
            )

        self.start_offsets = start_offsets
        num_workers = max(data_config.num_workers, 1)
        # workers need the dispatch queue and events before they fork, so
        # they are created here; input_q is created in start() instead
        self._ctx = mp.get_context("spawn")
        self.input_q: Optional[Any] = None
        # sample contents are shuffled pool-wide, so dispatch always goes
        # through one shared queue and in_order stays irrelevant
        self.shared_dispatch_queue = self._ctx.Queue(
            maxsize=_DISPATCH_QUEUE_BATCHES_PER_WORKER * num_workers
        )
        # stop_event: clean shutdown observed by coordinator, readers and
        # workers; fatal_event: abnormal coordinator death, workers raise
        self.stop_event = self._ctx.Event()
        self.fatal_event = self._ctx.Event()
        self._data_config = data_config
        self._input_path = input_path
        self._consumer_factory = consumer_factory
        self._proc: Optional[Any] = None
        self._atexit_cleanup = None

    def start(self, reader: "KafkaReader") -> None:
        """Spawn the coordinator; the DataLoader workers must already exist.

        Args:
            reader: the dataset's Kafka reader, configured for shared
                dispatch; the coordinator and its reader processes decode
                with the schemas it carries.
        """
        if self._proc is not None:
            return
        table_schema = reader._sample_table_schema
        assert table_schema is not None, "reader is not configured for sample shuffle"
        num_workers = max(self._data_config.num_workers, 1)
        capacity_rows = (
            self._data_config.batch_size * self._data_config.shuffle_buffer_size
        )
        mode_name = data_pb2.ShuffleMode.Name(self._data_config.shuffle_mode)
        logger.info(
            f"sample-shuffle starting: mode={mode_name} "
            f"pool capacity {capacity_rows} rows (batch_size "
            f"{self._data_config.batch_size} x buffer "
            f"{self._data_config.shuffle_buffer_size}), readers {num_workers}"
        )
        # created here (not in __init__) so a never-started handle holds no
        # reader output buffer; the handle keeps it alive until the
        # coordinator owns it
        input_q = self._ctx.Queue(maxsize=_INPUT_QUEUE_TABLES_PER_READER * num_workers)
        self.input_q = input_q
        try:
            proc = self._ctx.Process(
                target=_coordinator_entry,
                kwargs=dict(
                    input_path=self._input_path,
                    num_readers=num_workers,
                    start_offsets=self.start_offsets,
                    input_q=input_q,
                    stop_event=self.stop_event,
                    shared_dispatch_queue=self.shared_dispatch_queue,
                    mode=self._data_config.shuffle_mode,
                    capacity_rows=capacity_rows,
                    batch_size=self._data_config.batch_size,
                    input_table_schema=table_schema,
                    reader_schema=reader.schema,
                    full_schema=reader._full_schema,
                    has_embedded_schema=reader._has_embedded_schema,
                    trainer_pid=os.getpid(),
                    consumer_factory=self._consumer_factory,
                ),
                # not daemon: daemonic processes cannot fork reader children;
                # orphan self-exit is handled via trainer_pid checks instead
                daemon=False,
                name="tzrec-sample-shuffle-coordinator",
            )
            proc.start()
        except Exception:
            input_q.close()
            input_q.join_thread()
            self.input_q = None
            raise

        def _terminate_if_alive(p=proc):
            if p.is_alive():
                p.terminate()

        # interpreters can exit without stop_loading() on training-init
        # failures; this runs before multiprocessing joins the non-daemon
        # child, so initialization failures cannot hang the exit
        atexit.register(_terminate_if_alive)
        self._atexit_cleanup = _terminate_if_alive
        self._proc = proc
        self._start_watchdog()

    def _start_watchdog(self) -> None:
        """Watch the coordinator; set fatal_event on unexpected exit."""
        proc = self._proc

        def _watch() -> None:
            proc.join()
            if self._proc is not None:
                self.fatal_event.set()
                logger.error(
                    f"sample-shuffle coordinator exited unexpectedly "
                    f"(code {proc.exitcode})"
                )

        threading.Thread(
            target=_watch, name="sample-shuffle-watchdog", daemon=True
        ).start()

    def stop(self) -> None:
        """Stop the coordinator and its readers; idempotent."""
        if self._proc is None:
            return
        proc = self._proc
        self._proc = None
        self.stop_event.set()
        proc.join(timeout=15.0)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5.0)
        if self._atexit_cleanup is not None:
            atexit.unregister(self._atexit_cleanup)
            self._atexit_cleanup = None


def validate_sample_shuffle_config(data_config: data_pb2.DataConfig) -> None:
    """Validate Kafka-specific config for the sample-level shuffle path.

    ``enable.auto.commit`` is passed through to librdkafka unchanged, same
    as the BATCH path; only the fields the shared pool relies on are checked.

    Args:
        data_config: dataloader config.

    Raises:
        ValueError: on invalid capacity or an unsupported batch_cost_size.
    """
    if data_config.batch_size <= 0 or data_config.shuffle_buffer_size <= 0:
        raise ValueError(
            "sample-level shuffle requires batch_size > 0 and "
            f"shuffle_buffer_size > 0, got {data_config.batch_size} and "
            f"{data_config.shuffle_buffer_size}"
        )
    if data_config.batch_cost_size > 0:
        raise ValueError(
            "sample-level shuffle always groups by batch_size and does not "
            "support batch_cost_size"
        )
