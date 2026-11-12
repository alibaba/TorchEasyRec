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

"""Data-source agnostic per-rank sample-level shuffle.

A reader builds a ``ReadPlan`` of serializable tasks. The rank-local
``Coordinator`` starts once per dataloader and lives until training ends:
it forks one reader process per dataloader worker, delivers each pass's
tasks to them, decodes the incoming batches into a ``SamplePool`` and
serves shuffled batches on request. A finite pass ends when every consumer
was served END; the next GET then opens the next pass from the full read
plan. Workers talk to the coordinator with a small request/response
protocol; one bounded request queue is shared by every consumer and each
consumer owns a capacity-1 response queue, so a consumer has at most one
outstanding request.
"""

import atexit
import multiprocessing as mp
import os
import queue
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pyarrow as pa

from tzrec.constant import Mode
from tzrec.datasets.utils import CKPT_ROW_IDX, CKPT_SOURCE_ID, DATA_TIMESTAMP
from tzrec.protos import data_pb2
from tzrec.utils.logging_util import logger

# capacity of the reader -> coordinator input queue, in decoded batches
_INPUT_QUEUE_BATCHES_PER_READER = 2
# one queued GET plus one queued CANCEL per consumer fit at once
_REQUEST_QUEUE_PER_CONSUMER = 2
# coordinator loop wakeups per second while idle
_LOOP_TICK_SECS = 0.1
_WATCHDOG_TICK_SECS = 5.0
# shutdown budget for reaping reader processes: a graceful phase then a
# forced phase, each shared by all readers; the handle's stop timeout must
# comfortably exceed the sum
_READER_GRACEFUL_STOP_SECS = 6.0
_READER_FORCED_STOP_SECS = 2.0
_STATS_LOG_INTERVAL_SECS = 60.0

# reader -> coordinator messages
_DATA = "DATA"
_EOF = "EOF"
_ERROR = "ERROR"
# worker -> coordinator messages
_GET = "GET"
_CANCEL = "CANCEL"
# coordinator -> worker messages; ERROR shares the name of the reader ->
# coordinator error, the direction is unambiguous
_BATCH = "BATCH"
_END = "END"


def make_sample_table_schema(
    reader_schema: pa.Schema, with_timestamp: bool = False
) -> pa.Schema:
    """Schema of a raw batch travelling through the sample pool.

    Sources without event time only carry position metadata; Kafka declares
    ``DATA_TIMESTAMP`` explicitly.
    """
    fields = list(reader_schema) + [
        pa.field(CKPT_SOURCE_ID, pa.string()),
        pa.field(CKPT_ROW_IDX, pa.int64()),
    ]
    if with_timestamp:
        fields.append(pa.field(DATA_TIMESTAMP, pa.float64()))
    return pa.schema(fields)


def sample_shuffle_enabled(data_config: data_pb2.DataConfig, mode: Mode) -> bool:
    """Whether the sample-level shuffle path should serve this dataloader."""
    return (
        mode == Mode.TRAIN
        and data_config.shuffle
        and data_config.shuffle_mode
        in (data_pb2.ShuffleMode.SAMPLE_WINDOW, data_pb2.ShuffleMode.SAMPLE_ROLLING)
    )


def validate_sample_shuffle_config(data_config: data_pb2.DataConfig) -> None:
    """Validate the sample-level shuffle settings of a dataloader config.

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


@dataclass
class ReadTask:
    """One module-level read function and its picklable parameters.

    The reader process calls ``fn(stop_event=..., **kwargs)``; ``fn`` must be a
    module-level generator function so that the plan can be pickled to the
    spawned coordinator. It yields ``pa.RecordBatch`` objects and ends
    normally when its data is exhausted. Event objects are supplied by the
    reader process and never travel inside ``kwargs``.

    Args:
        fn: module-level read generator.
        kwargs: serializable arguments of ``fn``.
        reader_id: reader process slot that runs this task.
    """

    fn: Callable[..., Iterator[pa.RecordBatch]]
    kwargs: Dict[str, Any] = field(default_factory=dict)
    reader_id: int = 0


@dataclass
class ReadPlan:
    """The rank-local read plan handed to the coordinator.

    Args:
        tasks: read tasks, each assigned to one reader slot.
        schema: exact schema of the raw batches the tasks yield.
        is_finite: whether every task ends after a bounded amount of data.
        planned_rows: exact row count this plan allocates, already rank-aligned
            and tail-trimmed by the reader, or None when the source cannot
            know it before reading (e.g. csv).
        num_readers: number of reader processes to fork.
    """

    tasks: List[ReadTask]
    schema: pa.Schema
    is_finite: bool
    planned_rows: Optional[int]
    num_readers: int

    def tasks_by_reader(self) -> List[List[ReadTask]]:
        """Group tasks by their reader slot."""
        grouped: List[List[ReadTask]] = [[] for _ in range(self.num_readers)]
        for task in self.tasks:
            grouped[task.reader_id].append(task)
        return grouped


def _conform_batch(batch: pa.RecordBatch, schema: pa.Schema) -> pa.RecordBatch:
    """Reorder and cast a raw batch to the plan schema.

    Sources may expose the same columns in another order or with different
    nested nullability; the coordinator decodes with the plan schema, so the
    serialized batch must match it exactly.
    """
    if batch.schema.names != schema.names:
        batch = batch.select(schema.names)
    if [f.type for f in batch.schema] != [f.type for f in schema]:
        batch = batch.cast(pa.schema([pa.field(f.name, f.type) for f in schema]))
    return batch


class SamplePool:
    """Chunked raw-sample pool with sample-level shuffle.

    Rows stay in the chunk they arrived in; a numpy arena of (chunk, row)
    pairs is the sampling domain. SAMPLE_WINDOW freezes the pool content
    into a shuffled window once it reaches capacity and emits requested
    batches from it before refilling; SAMPLE_ROLLING extracts a random
    sample once the pool is refilled to capacity.

    ``emit`` normally waits until the pool reaches its capacity. With
    ``drain=True`` it only requires enough rows for the request, which lets
    a finite input end with a tail batch smaller than ``batch_size``.

    Checkpoint metadata columns (CKPT_SOURCE_ID/CKPT_ROW_IDX) and the
    event-time column ride through the pool together with the feature
    columns, so ``_build_batch`` can aggregate per-batch source offsets
    downstream.

    Args:
        capacity_rows: pool capacity in samples.
        mode: SAMPLE_WINDOW or SAMPLE_ROLLING.
        rng: random source, injectable for tests.
    """

    def __init__(
        self,
        capacity_rows: int,
        mode: int,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.capacity_rows = capacity_rows
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

    def emit(self, num_rows: int, drain: bool = False) -> Optional[pa.RecordBatch]:
        """Try to emit ``num_rows`` shuffled rows; None while the pool is short.

        Args:
            num_rows: requested rows. It may be smaller than ``batch_size``
                for a tail batch.
            drain: accept the request once the pool has ``num_rows`` rows
                instead of waiting for a full pool. Callers use it when no
                more rows can arrive to fill the pool.

        Returns:
            a batch of exactly ``num_rows`` rows, or None.
        """
        if self.mode == data_pb2.SAMPLE_WINDOW:
            return self._emit_window(num_rows, drain)
        return self._emit_rolling(num_rows, drain)

    def discard(self) -> None:
        """Drop every live row without emitting it."""
        self._chunks = []
        self._arena_chunk = np.empty(0, dtype=np.int32)
        self._arena_row = np.empty(0, dtype=np.int32)
        self._frozen_chunk = None
        self._frozen_row = None
        self._frozen_pos = 0
        self.total_rows = 0
        self._dead_rows = 0

    def _emit_window(self, num_rows: int, drain: bool) -> Optional[pa.RecordBatch]:
        """Fixed-window emit: freeze, permute, then serve the request."""
        while True:
            if self._frozen_chunk is not None:
                n = len(self._frozen_chunk)
                if self._frozen_pos + num_rows <= n:
                    sel = slice(self._frozen_pos, self._frozen_pos + num_rows)
                    self._frozen_pos += num_rows
                    batch = self._gather(self._frozen_chunk[sel], self._frozen_row[sel])
                    self._remove_from_live(batch.num_rows)
                    return batch
                # the rest of the window joins the next one
                leftover = slice(self._frozen_pos, n)
                if self._frozen_pos < n:
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
                continue
            if drain:
                if self.total_rows < num_rows:
                    return None
            elif self.total_rows < self.capacity_rows:
                # wait until the pool is full
                return None
            n_freeze = self.total_rows
            perm = self.rng.permutation(n_freeze)
            self._frozen_chunk = self._arena_chunk[perm]
            self._frozen_row = self._arena_row[perm]
            self._arena_chunk = np.empty(0, dtype=np.int32)
            self._arena_row = np.empty(0, dtype=np.int32)
            self._frozen_pos = 0

    def _emit_rolling(self, num_rows: int, drain: bool) -> Optional[pa.RecordBatch]:
        """Rolling-pool emit: draw a random sample from the refilled pool."""
        if drain:
            if self.total_rows < num_rows:
                return None
        elif self.total_rows < self.capacity_rows:
            # wait until the pool is refilled to capacity
            return None
        sel = self.rng.choice(len(self._arena_chunk), size=num_rows, replace=False)
        batch = self._gather(self._arena_chunk[sel], self._arena_row[sel])
        keep = np.ones(len(self._arena_chunk), dtype=bool)
        keep[sel] = False
        self._arena_chunk = self._arena_chunk[keep]
        self._arena_row = self._arena_row[keep]
        self._remove_from_live(batch.num_rows)
        return batch

    def _gather(self, sel_chunk: np.ndarray, sel_row: np.ndarray) -> pa.RecordBatch:
        """Materialize selected rows into one batch, keeping the sampled order.

        Rows are read chunk by chunk to batch the ``take`` calls; the
        concatenated result is then permuted back to the order of ``sel``.
        """
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
        if not np.array_equal(order, np.arange(batch.num_rows)):
            batch = batch.take(pa.array(np.argsort(order).astype(np.int64)))
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


def _reader_process_main(
    reader_id: int,
    task_q: Any,
    input_q: Any,
    stop_event: Any,
    cancel_event: Any,
    reader_stop: Any,
    coordinator_pid: int,
) -> None:
    """Run one reader slot for the whole lifetime of the service.

    Waits for a (tasks, schema) delivery on its capacity-1 task queue,
    streams the tasks' batches to the coordinator and ends each delivery
    with one EOF marker, then waits for the next one. A task failure
    flushes an ERROR marker and ends the process: it is fatal for the
    whole service. Only stop/cancel or a dead coordinator may drop data.
    """

    def _stopped() -> bool:
        return (
            stop_event.is_set()
            or cancel_event.is_set()
            or reader_stop.is_set()
            or os.getppid() != coordinator_pid
        )

    def _put(item: Tuple) -> bool:
        while not _stopped():
            try:
                input_q.put(item, timeout=0.5)
                return True
            except queue.Full:
                continue
        return False

    error_flushed = False
    try:
        while True:
            while True:
                if _stopped():
                    return
                try:
                    tasks, schema = task_q.get(timeout=0.5)
                    break
                except queue.Empty:
                    continue
            for task in tasks:
                if _stopped():
                    return
                for batch in task.fn(stop_event=reader_stop, **task.kwargs):
                    if _stopped():
                        return
                    if batch.num_rows == 0:
                        continue
                    batch = _conform_batch(batch, schema)
                    if not _put((_DATA, reader_id, batch.serialize().to_pybytes())):
                        return
            if not _put((_EOF, reader_id)):
                return
    except BaseException:
        logger.error(f"sample-shuffle reader {reader_id} failed", exc_info=True)
        error_flushed = _put((_ERROR, reader_id, traceback.format_exc()))
    finally:
        if error_flushed:
            input_q.close()
            input_q.join_thread()
        else:
            input_q.cancel_join_thread()


class Coordinator:
    """Per-rank sample pool and request-serving event loop.

    Starts once per dataloader and serves every training pass: the first
    from ``first_plan``, every later one from ``full_plan`` when the next
    GET arrives after a finite pass ended.

    Args:
        first_plan: read plan of the first pass, built over the reader's
            restored resume position.
        full_plan: read plan of every later pass; the same object as
            ``first_plan`` when the source cannot resume or had no resume
            position.
        request_q: shared bounded request queue from the consumers.
        response_queues: one capacity-1 response queue per consumer.
        stop_event: shared clean-stop signal; the coordinator exits when set.
        cancel_event: shared early-stop signal; set by a CANCEL message.
        mode: SAMPLE_WINDOW or SAMPLE_ROLLING.
        capacity_rows: pool capacity in samples.
        batch_size: training batch size.
        min_batch_size: tail batches smaller than this are dropped.
        num_consumers: number of dataloader consumers on this rank.
        trainer_pid: trainer process pid; the coordinator exits when its
            parent changes (trainer died without stopping it).
    """

    def __init__(
        self,
        first_plan: ReadPlan,
        full_plan: ReadPlan,
        request_q: Any,
        response_queues: List[Any],
        stop_event: Any,
        cancel_event: Any,
        mode: int,
        capacity_rows: int,
        batch_size: int,
        min_batch_size: int,
        num_consumers: int,
        trainer_pid: int,
    ) -> None:
        self._first_plan = first_plan
        self._full_plan = full_plan
        self._request_q = request_q
        self._response_queues = response_queues
        self._stop_event = stop_event
        self._cancel_event = cancel_event
        self._batch_size = batch_size
        self._min_batch_size = min_batch_size
        self._num_consumers = num_consumers
        self._trainer_pid = trainer_pid
        self._capacity_rows = capacity_rows
        self._pool = SamplePool(capacity_rows, mode)
        # per-pass state, fully initialized by _start_pass
        self._plan = first_plan
        self._remaining: Optional[int] = None
        self._rows_read = 0
        self._eof_readers = set()
        self._verified = False
        # GETs in arrival order; each consumer has at most one entry
        self._request_order: deque[int] = deque()
        self._closed = [False] * num_consumers
        self._error: Optional[str] = None
        self._readers: List[Any] = []
        self._ctx = mp.get_context("fork")
        self._reader_stop = self._ctx.Event()
        num_readers = first_plan.num_readers
        # created before the readers are forked, so no queue feeder thread
        # is ever forked into a reader
        self._input_q = self._ctx.Queue(
            maxsize=_INPUT_QUEUE_BATCHES_PER_READER * num_readers
        )
        self._task_queues = [self._ctx.Queue(maxsize=1) for _ in range(num_readers)]

    def _start_readers(self) -> None:
        """Fork the reader processes."""
        # fork before the event loop starts: the coordinator is still
        # single-threaded and holds no consumer, so fork is safe; task
        # deliveries happen only after every reader exists
        pid = os.getpid()
        for i, task_q in enumerate(self._task_queues):
            p = self._ctx.Process(
                target=_reader_process_main,
                args=(
                    i,
                    task_q,
                    self._input_q,
                    self._stop_event,
                    self._cancel_event,
                    self._reader_stop,
                    pid,
                ),
                daemon=True,
                name=f"tzrec-sample-reader-{i}",
            )
            p.start()
            self._readers.append(p)

    def _start_pass(self, plan: ReadPlan) -> None:
        """Begin one training pass: reset its state, deliver reader tasks.

        Called before the first request of the pass is served, so the
        triggering GET is registered only afterwards.
        """
        self._plan = plan
        # the read plan already applied the rank-step alignment and tail
        # trimming; the coordinator only allocates its planned rows
        self._remaining = plan.planned_rows
        self._rows_read = 0
        self._eof_readers = set()
        self._verified = False
        self._request_order = deque()
        self._closed = [False] * self._num_consumers
        grouped = plan.tasks_by_reader()
        for i, task_q in enumerate(self._task_queues):
            task_q.put((grouped[i], plan.schema))

    def _fail(self, message: str) -> None:
        """Record a fatal failure and make every consumer see it."""
        if self._error is None:
            self._error = message
            logger.error(f"sample-shuffle coordinator failed: {message}")

    def _service_requests(self, timeout: float = 0.0) -> None:
        """Accept queued GET/CANCEL messages.

        With ``timeout > 0`` the first message is awaited for at most
        ``timeout`` seconds, so an idle coordinator blocks on the request
        queue instead of polling and adding latency to every batch.
        """
        blocking = timeout > 0
        while True:
            try:
                if blocking:
                    msg = self._request_q.get(timeout=timeout)
                else:
                    msg = self._request_q.get_nowait()
            except queue.Empty:
                return
            blocking = False
            kind, _, consumer_id = msg
            if kind == _CANCEL:
                self._cancel_event.set()
                return
            if self._is_complete():
                # every consumer finished the finite pass; this GET opens
                # the next one, and re-reading starts only now
                self._start_pass(self._full_plan)
            # one response queue per consumer: it cannot send a second GET
            # before receiving the response of the first
            self._request_order.append(consumer_id)

    def _check_eof(self) -> None:
        """Fail on any EOF for infinite sources; verify finite row counts."""
        if not self._plan.is_finite:
            self._fail("readers ended for a non-finite sample shuffle source")
            return
        if len(self._eof_readers) < self._plan.num_readers:
            return
        expected = self._plan.planned_rows
        if expected is not None and self._rows_read != expected:
            self._fail(
                f"read {self._rows_read} rows but the read plan expected {expected}"
            )
            return
        self._verified = True

    def _service_input(self, timeout: float) -> bool:
        """Take one reader message and update the pool; return whether it took one."""
        try:
            msg = self._input_q.get(timeout=timeout)
        except queue.Empty:
            return False
        kind = msg[0]
        if kind == _DATA:
            rb = pa.ipc.read_record_batch(msg[2], self._plan.schema)
            self._rows_read += rb.num_rows
            expected = self._plan.planned_rows
            if expected is not None and self._rows_read > expected:
                self._fail(
                    f"read more than the planned {expected} rows; "
                    "the input changed during the run"
                )
                return True
            self._pool.add(rb)
        elif kind == _EOF:
            self._eof_readers.add(msg[1])
            self._check_eof()
        elif kind == _ERROR:
            self._fail(f"reader {msg[1]} failed: {msg[2]}")
        return True

    def _serve_waiting(self) -> None:
        """Answer outstanding GETs in arrival order.

        One response queue per consumer and at most one outstanding GET
        mean every response goes into a queue its consumer has already
        drained, so the put cannot block.
        """
        if self._error is not None:
            while self._request_order:
                consumer_id = self._request_order.popleft()
                self._response_queues[consumer_id].put_nowait((_ERROR, self._error))
                self._closed[consumer_id] = True
            return

        while self._request_order:
            consumer_id = self._request_order[0]
            num_rows: Optional[int] = None
            drain = False
            end = False
            if self._plan.planned_rows is not None:
                remaining = self._remaining
                assert remaining is not None
                if remaining == 0:
                    if self._verified:
                        end = True
                    else:
                        # quota used up; the oldest request waits for EOF and
                        # the row-count check, later requests must not jump in
                        break
                else:
                    # all rows read (verified) or still arriving: the request
                    # size is the same, only capacity and drain differ
                    num_rows = min(self._batch_size, remaining)
                    drain = self._verified or remaining < self._capacity_rows
            elif self._verified:
                # unknown row count: drain what is left, applying the tail rule
                available = self._pool.total_rows
                if available == 0:
                    end = True
                else:
                    num_rows = min(self._batch_size, available)
                    if num_rows < self._batch_size and num_rows < self._min_batch_size:
                        self._pool.discard()
                        end = True
                    else:
                        drain = True
            else:
                num_rows = self._batch_size

            if end:
                self._response_queues[consumer_id].put_nowait((_END, None))
                self._closed[consumer_id] = True
                if all(self._closed):
                    # every consumer finished the pass; release the consumed
                    # chunk buffers now instead of at the next pass's start
                    self._pool.discard()
                self._request_order.popleft()
                continue
            assert num_rows is not None
            batch = self._pool.emit(num_rows, drain)
            if batch is None:
                break
            if self._remaining is not None:
                self._remaining -= batch.num_rows
            self._response_queues[consumer_id].put_nowait(
                (_BATCH, batch.serialize().to_pybytes())
            )
            self._request_order.popleft()

    @property
    def _input_done(self) -> bool:
        """Whether every reader has reported EOF."""
        return len(self._eof_readers) >= self._plan.num_readers

    def _is_complete(self) -> bool:
        """Whether a finite pass served END to every consumer."""
        return self._verified and all(self._closed)

    def _pool_has_room(self) -> bool:
        """Whether the coordinator should keep pulling reader data."""
        if self._input_done:
            # no reader can add rows any more; only requests matter now
            return False
        return self._pool.total_rows < self._capacity_rows

    def _check_readers(self) -> None:
        """Fail when a reader died; the error propagates to the consumers.

        Readers wait for their next delivery after a pass instead of
        exiting, so a dead reader is always abnormal: one killed mid-write
        can even leave a truncated frame behind, whose input must never
        be read again. The service fails instead.
        """
        for p in self._readers:
            if p.is_alive():
                continue
            self._fail(f"sample-shuffle reader exited unexpectedly (code {p.exitcode})")
            return

    def _log_stats(self) -> None:
        """Log pool counters."""
        logger.info(
            f"sample-shuffle stats: pool {self._pool.total_rows} rows "
            f"(capacity {self._pool.capacity_rows}), rows in {self._pool.rows_in} "
            f"out {self._pool.rows_out}, emits {self._pool.num_emits}, "
            f"compactions {self._pool.num_compactions}"
        )

    def run(self) -> None:
        """Serve passes until stop, cancel or failure."""
        self._start_readers()
        self._start_pass(self._first_plan)
        last_watchdog = time.monotonic()
        last_stats = time.monotonic()
        try:
            while not self._stop_event.is_set() and not self._cancel_event.is_set():
                now = time.monotonic()
                self._service_requests()
                self._serve_waiting()
                if self._error is not None:
                    break
                if self._pool_has_room():
                    if all(p.is_alive() for p in self._readers):
                        self._service_input(_LOOP_TICK_SECS)
                    else:
                        # a dead reader's input may end mid-frame; never read
                        # it again, fail the service instead
                        self._check_readers()
                elif not self._request_order:
                    # consumers are busy processing their batches; block until
                    # the next request instead of polling and adding latency
                    self._service_requests(_LOOP_TICK_SECS)
                else:
                    time.sleep(0.001)
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
            self._reader_stop.set()
            for response_q in self._response_queues:
                # a consumer that is already gone must not block this
                # process on a full queue's feeder thread at exit
                response_q.cancel_join_thread()
            self._input_q.cancel_join_thread()
            for task_q in self._task_queues:
                task_q.cancel_join_thread()
            self._shutdown_readers()
        if self._error is not None:
            raise RuntimeError(self._error)

    def _shutdown_readers(self) -> None:
        """Reap all readers: one shared graceful deadline, then a forced one.

        Readers watch the stop/cancel/reader-stop events (clean shutdown) or
        their parent pid (coordinator death), so nothing needs signaling here.
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
    Coordinator(**kwargs).run()


class SampleShuffleHandle:
    """Owns one rank's sample-shuffle background resources end to end.

    Created before the DataLoader forks its workers, so the request queue,
    response queues and events are inherited by them. ``start`` spawns the
    coordinator once, after the eager ``iter()`` forked the workers; the
    coordinator serves every training pass from the two plans built here
    and exits only on ``stop``. ``stop`` tears everything down and is
    idempotent.

    Args:
        reader: the dataset's reader; it builds the first and full plans.
        data_config: dataloader config.
        num_workers: actual number of dataloader workers on this rank.
        rank: this process's rank.
        world_size: number of ranks.
    """

    def __init__(
        self,
        reader: Any,
        data_config: data_pb2.DataConfig,
        num_workers: int,
        rank: int,
        world_size: int,
    ) -> None:
        validate_sample_shuffle_config(data_config)
        self._rank = rank
        self._num_consumers = max(num_workers, 1)
        self._batch_size = data_config.batch_size
        self._min_batch_size = reader._min_batch_size
        self._mode = data_config.shuffle_mode
        self._capacity_rows = data_config.batch_size * data_config.shuffle_buffer_size

        # the first pass continues the checkpoint's resume position; every
        # later pass reads the full source. Only the reader's internal
        # resume position is dropped; the training loop's own checkpoint
        # state is not touched.
        self._first_plan = reader.create_read_plan(
            rank, world_size, self._num_consumers
        )
        if self._first_plan.is_finite and reader._checkpoint_state:
            reader.load_state_dict(None)
            self._full_plan = reader.create_read_plan(
                rank, world_size, self._num_consumers
            )
        else:
            self._full_plan = self._first_plan
        self._schema = self._first_plan.schema

        self._ctx = mp.get_context("spawn")
        self._request_q = self._ctx.Queue(
            maxsize=_REQUEST_QUEUE_PER_CONSUMER * self._num_consumers
        )
        self._response_queues = [
            self._ctx.Queue(maxsize=1) for _ in range(self._num_consumers)
        ]
        self._stop_event = self._ctx.Event()
        self._fatal_event = self._ctx.Event()
        self._cancel_event = self._ctx.Event()
        self._proc: Optional[Any] = None
        self._stopped = False

    def _put_request(self, msg: Tuple) -> bool:
        """Queue a protocol message unless the service is stopping.

        The bounded queue holds every consumer's GET plus CANCEL at once,
        so the put cannot block.
        """
        if self._stop_event.is_set() or self._cancel_event.is_set():
            return False
        self._request_q.put_nowait(msg)
        return True

    def batches(self, consumer_id: int) -> Iterator[Dict[str, pa.Array]]:
        """Yield this consumer's raw shuffled batches of the current pass.

        Sends one GET at a time on the shared request queue and reads this
        consumer's own response queue; a finite pass ends on END.

        Args:
            consumer_id: index of this consumer in the rank.

        Yields:
            dict of column name to PyArrow Array.
        """
        response_q = self._response_queues[consumer_id]
        finished = False
        try:
            while True:
                if not self._put_request((_GET, self._rank, consumer_id)):
                    return
                while True:
                    try:
                        kind, payload = response_q.get(timeout=1.0)
                    except queue.Empty:
                        if self._fatal_event.is_set():
                            raise RuntimeError(
                                "sample-shuffle coordinator exited unexpectedly"
                            ) from None
                        if self._stop_event.is_set() or self._cancel_event.is_set():
                            return
                        continue
                    if kind == _BATCH:
                        rb = pa.ipc.read_record_batch(payload, self._schema)
                        yield {
                            name: rb.column(i) for i, name in enumerate(rb.schema.names)
                        }
                        break
                    if kind == _END:
                        finished = True
                        return
                    # kind == _ERROR
                    raise RuntimeError(payload)
        finally:
            if (
                not finished
                and not self._stop_event.is_set()
                and not self._cancel_event.is_set()
            ):
                # the pass was abandoned; stop the whole rank's service
                self._put_request((_CANCEL, self._rank, consumer_id))

    def start(self) -> None:
        """Spawn the coordinator; idempotent."""
        if self._proc is not None or self._stopped:
            return
        logger.info(
            f"sample-shuffle service starting: "
            f"mode={data_pb2.ShuffleMode.Name(self._mode)} "
            f"capacity={self._capacity_rows} rows, "
            f"consumers={self._num_consumers}, "
            f"finite={self._first_plan.is_finite}, "
            f"first_rows={self._first_plan.planned_rows}, "
            f"full_rows={self._full_plan.planned_rows}"
        )
        proc = self._ctx.Process(
            target=_coordinator_entry,
            kwargs=dict(
                first_plan=self._first_plan,
                full_plan=self._full_plan,
                request_q=self._request_q,
                response_queues=self._response_queues,
                stop_event=self._stop_event,
                cancel_event=self._cancel_event,
                mode=self._mode,
                capacity_rows=self._capacity_rows,
                batch_size=self._batch_size,
                min_batch_size=self._min_batch_size,
                num_consumers=self._num_consumers,
                trainer_pid=os.getpid(),
            ),
            # not daemon: daemonic processes cannot fork reader children;
            # orphan self-exit is handled via trainer_pid checks instead
            daemon=False,
            name="tzrec-sample-shuffle-coordinator",
        )
        proc.start()
        # interpreters can exit without close() on training-init failures;
        # this runs before multiprocessing joins the non-daemon child, so
        # initialization failures cannot hang the exit
        atexit.register(self._terminate_coordinator)
        self._proc = proc
        self._start_watchdog()

    def _start_watchdog(self) -> None:
        """Watch the coordinator; set fatal_event on an exit we did not order."""
        proc = self._proc

        def _watch() -> None:
            proc.join()
            if not self._stop_event.is_set() and not self._cancel_event.is_set():
                self._fatal_event.set()
                logger.error(
                    f"sample-shuffle coordinator exited unexpectedly "
                    f"(code {proc.exitcode})"
                )

        threading.Thread(
            target=_watch, name="sample-shuffle-watchdog", daemon=True
        ).start()

    def _terminate_coordinator(self) -> None:
        """Terminate the current coordinator at interpreter exit."""
        proc = self._proc
        if proc is not None and proc.is_alive():
            proc.terminate()

    def stop(self) -> None:
        """Stop the coordinator and its readers; idempotent."""
        if self._stopped:
            return
        self._stopped = True
        # wake consumers even when the coordinator was never started (e.g. a
        # dataloader creation failure between the eager iter and start)
        self._stop_event.set()
        proc = self._proc
        self._proc = None
        if proc is not None:
            proc.join(timeout=15.0)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5.0)
        # do not wait for a full request queue's feeder thread at exit
        self._request_q.cancel_join_thread()
        atexit.unregister(self._terminate_coordinator)
