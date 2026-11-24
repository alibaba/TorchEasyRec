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


import multiprocessing as mp
import os
import queue
import struct
import threading
import time
import unittest
from unittest import mock

import numpy as np
import pyarrow as pa
from parameterized import param, parameterized

from tzrec.datasets import sample_shuffle
from tzrec.datasets.sample_shuffle import (
    Coordinator,
    ReadPlan,
    ReadTask,
    SamplePool,
    SampleShuffleHandle,
    _reader_process_main,
)
from tzrec.datasets.utils import CKPT_ROW_IDX, CKPT_SOURCE_ID, DATA_TIMESTAMP
from tzrec.protos import data_pb2
from tzrec.utils.test_util import parameterized_name_func

_WINDOW = "SAMPLE_WINDOW"
_ROLLING = "SAMPLE_ROLLING"


def make_batch(n, start_id=0):
    """Rich batch: nulls and nested columns must survive the pool unchanged."""
    ids = np.arange(start_id, start_id + n, dtype=np.int64)
    return pa.RecordBatch.from_arrays(
        [
            pa.array(ids),
            pa.array([None if i % 3 == 0 else f"v{i}" for i in ids]),
            pa.array(
                [None if i % 4 == 0 else [int(i), int(i + 1)] for i in ids],
                type=pa.list_(pa.int64()),
            ),
            pa.array(ids % 2),
            pa.array(["topic:0"] * n),
            pa.array(ids // 16),
            pa.array(ids.astype(np.float64) + 0.5),
        ],
        names=[
            "id",
            "str_f",
            "list_f",
            "label",
            CKPT_SOURCE_ID,
            CKPT_ROW_IDX,
            DATA_TIMESTAMP,
        ],
    )


def check_batch(tc, batch, sources):
    """Whole-row correspondence: every output row matches its raw source row."""
    expected = {row["id"]: row for src in sources for row in src.to_pylist()}
    for row in batch.to_pylist():
        tc.assertEqual(row, expected[row["id"]])


_PLAN_SCHEMA = pa.schema(
    [
        pa.field("id", pa.int64()),
        pa.field(CKPT_SOURCE_ID, pa.string()),
        pa.field(CKPT_ROW_IDX, pa.int64()),
        pa.field(DATA_TIMESTAMP, pa.float64()),
    ]
)


def _plan(planned_rows, is_finite=True, num_readers=1, tasks=None):
    return ReadPlan(
        tasks=tasks or [],
        schema=_PLAN_SCHEMA,
        is_finite=is_finite,
        planned_rows=planned_rows,
        num_readers=num_readers,
    )


def _plan_batch(n, start_id=0, reorder=False):
    """Four metadata columns in the plan schema, optionally column-reversed."""
    ids = np.arange(start_id, start_id + n, dtype=np.int64)
    arrays = [
        pa.array(ids),
        pa.array(["topic:0"] * n, type=pa.string()),
        pa.array([0] * n, type=pa.int64()),
        pa.array([-1.0] * n, type=pa.float64()),
    ]
    names = ["id", CKPT_SOURCE_ID, CKPT_ROW_IDX, DATA_TIMESTAMP]
    if reorder:
        arrays, names = arrays[::-1], names[::-1]
    return pa.RecordBatch.from_arrays(arrays, names=names)


def _feed_batch(coord, n, start_id=0, reader_id=0):
    rb = _plan_batch(n, start_id)
    coord._input_q.put((sample_shuffle._DATA, reader_id, rb.serialize().to_pybytes()))


def _feed_eof(coord, reader_id=0):
    coord._input_q.put((sample_shuffle._EOF, reader_id))


class SamplePoolTest(unittest.TestCase):
    def _pool(self, capacity, mode, seed=0, batch_size=None):
        return SamplePool(
            capacity, mode, rng=np.random.default_rng(seed), batch_size=batch_size
        )

    @parameterized.expand(
        [param("window", mode=_WINDOW), param("rolling", mode=_ROLLING)],
        name_func=parameterized_name_func,
    )
    def test_fill_drain_and_discard(self, _name, mode):
        pool = self._pool(16, mode)
        source = make_batch(8)
        self.assertIsNone(pool.emit(8))
        pool.add(source)
        self.assertIsNone(pool.emit(8))
        self.assertIsNone(pool.emit(12, drain=True))
        check_batch(self, pool.emit(8, drain=True), [source])
        pool.add(make_batch(8, start_id=8))
        pool.add(make_batch(8, start_id=16))
        self.assertIsNotNone(pool.emit(8))
        pool.discard()
        self.assertEqual(pool.total_rows, 0)
        self.assertIsNone(pool.emit(8, drain=True))
        one = self._pool(8, mode)
        one.add(make_batch(8))
        self.assertEqual(one.emit(8).num_rows, 8)

    def test_window_sequence(self):
        pool = self._pool(64, _WINDOW, seed=3, batch_size=16)
        source = make_batch(80)
        for start in range(0, 40, 8):
            pool.add(source.slice(start, 8))
        large = source.slice(40)
        pool.add(large)
        order = []
        for _ in range(5):
            out = pool.emit(16)
            check_batch(self, out, [source])
            order.extend(out.column("id").to_pylist())
        self.assertEqual([c.num_rows for c in pool._chunks], [32, 8, 40])
        self.assertIs(pool._chunks[-1], large)
        self.assertEqual(order, np.random.default_rng(3).permutation(80).tolist())
        # New arrivals must not enter the frozen window.
        pool.add(make_batch(32, start_id=1000))
        pool.add(make_batch(32, start_id=2000))
        first = pool.emit(16)
        pool.add(make_batch(16, start_id=3000))
        rest = []
        while (out := pool.emit(16)) is not None:
            rest.append(out)
        frozen = set(first.column("id").to_pylist())
        for b in rest:
            frozen.update(b.column("id").to_pylist())
        self.assertEqual(frozen, set(range(1000, 1032)) | set(range(2000, 2032)))
        self.assertEqual(pool.total_rows, 16)
        tail = self._pool(16, _WINDOW, batch_size=8)
        tail_source = make_batch(12)
        tail.add(tail_source)
        check_batch(self, tail.emit(8, drain=True), [tail_source])
        tail.add(make_batch(5, start_id=100))
        second = tail.emit(8, drain=True)
        self.assertEqual([c.num_rows for c in tail._chunks], [9])
        ids = set(second.column("id").to_pylist())
        self.assertTrue(any(i < 12 for i in ids) and any(i >= 100 for i in ids))
        self.assertEqual(tail.total_rows, 1)
        self.assertEqual(tail.emit(1, drain=True).num_rows, 1)

    def test_rolling_sequence(self):
        pool = self._pool(32, _ROLLING, seed=7, batch_size=16)
        source = make_batch(32)
        for start in range(0, 32, 8):
            pool.add(source.slice(start, 8))
        first = pool.emit(16)
        self.assertEqual([c.num_rows for c in pool._chunks], [8] * 4)
        check_batch(self, first, [source])
        first_ids = set(first.column("id").to_pylist())
        self.assertEqual(len(first_ids), 16)
        self.assertIsNone(pool.emit(16))
        pool.add(make_batch(16, start_id=32))
        second = pool.emit(16)
        second_ids = set(second.column("id").to_pylist())
        self.assertEqual(len(second_ids & first_ids), 0)
        self.assertTrue(any(i < 32 for i in second_ids))
        self.assertTrue(any(i >= 32 for i in second_ids))
        self.assertEqual(pool.total_rows, 16)
        pool.discard()
        self.assertEqual(pool.total_rows, 0)

    @parameterized.expand(
        [param("window", mode=_WINDOW), param("rolling", mode=_ROLLING)],
        name_func=parameterized_name_func,
    )
    def test_compaction_bounds_and_outputs_stay_valid(self, _name, mode):
        pool = self._pool(32, mode)
        next_id = 0
        produced = []
        sources = []
        for _ in range(40):
            source = make_batch(16, start_id=next_id)
            sources.append(source)
            pool.add(source)
            next_id += 16
            for _ in range(2):
                out = pool.emit(8)
                if out is None:
                    break
                produced.append(out)
                check_batch(self, out, sources)
        self.assertGreater(pool.num_compactions, 0)
        self.assertLessEqual(len(pool._chunks), 10)
        self.assertEqual(pool.rows_in, next_id)
        self.assertEqual(pool.rows_in - pool.rows_out, pool.total_rows)
        ids = [i for b in produced for i in b.column("id").to_pylist()]
        self.assertEqual(len(ids), len(produced) * 8)
        self.assertEqual(len(set(ids)), len(ids))


class CoordinatorTest(unittest.TestCase):
    """In-process coordinator tests over the request/response protocol."""

    def _make_coordinator(
        self, first_plan, full_plan, min_batch_size=0, mode=_ROLLING, num_consumers=1
    ):
        """Create the queues and coordinator; callers drive passes and threads."""
        ctx = mp.get_context("spawn")
        request_q = ctx.Queue()
        response_queues = [ctx.Queue(maxsize=1) for _ in range(num_consumers)]
        self.addCleanup(request_q.cancel_join_thread)
        for response_q in response_queues:
            self.addCleanup(response_q.cancel_join_thread)
        coord = Coordinator(
            first_plan=first_plan,
            full_plan=full_plan,
            request_q=request_q,
            response_queues=response_queues,
            stop_event=ctx.Event(),
            cancel_event=ctx.Event(),
            mode=mode,
            capacity_rows=16,
            batch_size=8,
            min_batch_size=min_batch_size,
            num_consumers=num_consumers,
            # NOTE: The coordinator thread shares this process's parent PID.
            trainer_pid=os.getppid(),
        )
        self.addCleanup(coord._input_q.cancel_join_thread)
        for task_q in coord._task_queues:
            self.addCleanup(task_q.cancel_join_thread)
        return coord

    def _coordinator(
        self, planned_rows, is_finite=True, min_batch_size=0, num_readers=1
    ):
        plan = _plan(planned_rows, is_finite=is_finite, num_readers=num_readers)
        coord = self._make_coordinator(plan, plan, min_batch_size=min_batch_size)
        coord._start_pass(coord._first_plan)
        for task_q in coord._task_queues:
            task_q.get(timeout=1.0)
        return coord

    def _request(self, coord):
        coord._request_q.put((sample_shuffle._GET, 0))

    def _settle_input(self, coord):
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if coord._verified or coord._error is not None:
                return
            coord._service_input(0.05)
        raise AssertionError("input did not settle")

    def _next_response(self, coord):
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            coord._service_requests()
            coord._serve_waiting()
            try:
                return coord._response_queues[0].get_nowait()
            except queue.Empty:
                time.sleep(0.005)
        raise queue.Empty

    def _collect_batches(self, coord, timeout=None):
        """Send GETs until the pass ends; return its batches and end message.

        With ``timeout`` set the coordinator runs in another thread and
        responses are awaited on its queue; otherwise this thread drives it.
        """
        batches = []
        self._request(coord)
        while True:
            if timeout is None:
                msg = self._next_response(coord)
            else:
                msg = coord._response_queues[0].get(timeout=timeout)
            if msg[0] != sample_shuffle._BATCH:
                return batches, msg
            batches.append(pa.ipc.read_record_batch(msg[1], _PLAN_SCHEMA))
            self._request(coord)

    def _run_with_readers(self, coord):
        """Run the coordinator in a thread and capture errors."""
        errors = queue.Queue()

        def run():
            try:
                coord.run()
            except BaseException as error:
                errors.put(error)

        runner = threading.Thread(target=run, daemon=True)
        runner.start()
        self.addCleanup(coord._stop_event.set)
        self.addCleanup(runner.join, 10.0)
        return runner, errors

    def _stop_runner(self, coord, runner, errors):
        """Stop the coordinator and verify reader cleanup."""
        coord._stop_event.set()
        runner.join(timeout=10.0)
        self.assertFalse(runner.is_alive())
        self.assertTrue(errors.empty())
        self.assertFalse(any(p.is_alive() for p in coord._readers))

    def _wait_queued(self, coord, consumer_id, timeout=5.0):
        """Wait for an unclaimed response."""
        deadline = time.monotonic() + timeout
        while coord._response_queues[consumer_id].empty():
            if time.monotonic() > deadline:
                self.fail(f"consumer {consumer_id} was never served")
            time.sleep(0.005)

    @parameterized.expand(
        [
            ("full", 20, 0, [8, 8, 4]),
            ("no_tail", 16, 5, [8, 8]),
            ("tail_at_min", 18, 2, [8, 8, 2]),
            ("small_plan", 5, 0, [5]),
            ("empty_plan", 0, 0, []),
            ("unknown_drain", None, 0, [8, 8, 4]),
            ("unknown_tail_below_min", None, 5, [8, 8]),
        ],
        name_func=parameterized_name_func,
    )
    def test_quota(self, _name, planned_rows, min_batch_size, expected):
        coord = self._coordinator(
            planned_rows=planned_rows, min_batch_size=min_batch_size
        )
        _feed_batch(coord, 20 if planned_rows is None else planned_rows)
        _feed_eof(coord)
        self._settle_input(coord)
        self.assertTrue(coord._verified)
        batches, msg = self._collect_batches(coord)
        self.assertEqual([b.num_rows for b in batches], expected)
        self.assertEqual(msg[0], sample_shuffle._END)
        self.assertEqual(coord._pool.total_rows, 0)
        self.assertEqual(coord._pool._chunks, [])
        self.assertEqual(coord._pool._arena_chunk.size, 0)

    def test_end_waits_for_eof_after_quota(self):
        coord = self._coordinator(planned_rows=16)
        _feed_batch(coord, 16)
        coord._service_input(1.0)
        for _ in range(2):
            self._request(coord)
            self.assertEqual(self._next_response(coord)[0], sample_shuffle._BATCH)
        self._request(coord)
        coord._service_requests()
        coord._serve_waiting()
        self.assertTrue(coord._response_queues[0].empty())
        _feed_eof(coord)
        self._settle_input(coord)
        self.assertEqual(self._next_response(coord)[0], sample_shuffle._END)

    def test_window_refill_between_batches(self):
        """Refills between two batches only use reads that cannot block."""
        plan = _plan(None, is_finite=False)
        coord = self._make_coordinator(plan, plan, mode=_WINDOW, num_consumers=2)
        # NOTE: Synchronous queues make the order of responses and reads observable.
        coord._input_q = queue.Queue()
        coord._response_queues = [queue.Queue(), queue.Queue()]
        coord._pool.add(_plan_batch(16))
        _feed_batch(coord, 4, start_id=16)
        _feed_batch(coord, 4, start_id=20)
        coord._request_order.extend([0, 1])
        reads = []
        service_input = coord._service_input

        def record_read(timeout):
            reads.append((timeout, sum(q.qsize() for q in coord._response_queues)))
            return service_input(timeout)

        with mock.patch.object(coord, "_service_input", side_effect=record_read):
            coord._serve_waiting()
        self.assertEqual(reads, [(0.0, 1), (0.0, 1), (0.0, 2)])
        self.assertEqual(coord._pool.rows_out, 16)
        self.assertEqual(coord._pool.total_rows, 8)

    def test_nonblocking_input_read_sees_flushed_message(self):
        """A flushed message needs get_nowait; get(timeout=0) never polls."""
        coord = self._coordinator(planned_rows=None, is_finite=False)
        _feed_batch(coord, 4)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and coord._service_input(0.0) <= 0:
            time.sleep(0.005)
        self.assertEqual(coord._pool.total_rows, 4)

    @parameterized.expand(
        [
            ("more_rows", 12, False, None, "more than the planned"),
            ("fewer_rows", 8, True, None, "expected 10"),
            ("reader_error", 0, False, "boom", "reader 0 failed: boom"),
        ],
        name_func=parameterized_name_func,
    )
    def test_error_reaches_consumer(self, _name, rows, eof, reader_error, error):
        coord = self._coordinator(planned_rows=10)
        if reader_error is None:
            _feed_batch(coord, rows)
            if eof:
                _feed_eof(coord)
        else:
            coord._input_q.put((sample_shuffle._ERROR, 0, reader_error))
        self._settle_input(coord)
        self._request(coord)
        msg = self._next_response(coord)
        self.assertEqual(msg[0], sample_shuffle._ERROR)
        self.assertIn(error, msg[1])

    def test_eof_verification_rules(self):
        finite = self._coordinator(planned_rows=10, num_readers=2)
        _feed_batch(finite, 5, reader_id=0)
        _feed_eof(finite, reader_id=0)
        for _ in range(2):
            finite._service_input(1.0)
        self.assertEqual(finite._eof_readers, {0})
        self.assertIsNone(finite._error)
        self.assertFalse(finite._verified)
        _feed_batch(finite, 5, start_id=5, reader_id=1)
        _feed_eof(finite, reader_id=1)
        self._settle_input(finite)
        self.assertTrue(finite._verified)
        infinite = self._coordinator(planned_rows=None, is_finite=False, num_readers=2)
        _feed_eof(infinite, reader_id=0)
        infinite._service_input(1.0)
        self.assertIn("non-finite", infinite._error)

    def test_dead_reader_fails_without_reading_its_input(self):
        coord = self._coordinator(planned_rows=8)
        # A truncated frame must not block reader-death detection.
        os.write(coord._input_q._writer.fileno(), struct.pack("!i", 1 << 20))
        dead = mock.Mock()
        dead.is_alive.return_value = False
        dead.exitcode = -9
        coord._readers = [dead]
        errors = queue.Queue()

        def run():
            try:
                coord.run()
            except BaseException as error:
                errors.put(error)

        with mock.patch.object(coord, "_start_readers"):
            runner = threading.Thread(target=run, daemon=True)
            runner.start()
            runner.join(timeout=5.0)
        self.assertFalse(runner.is_alive())
        error = errors.get_nowait()
        self.assertIsInstance(error, RuntimeError)
        self.assertEqual(
            str(error), "sample-shuffle reader exited unexpectedly (code -9)"
        )
        self.assertEqual(coord._error, str(error))

    @parameterized.expand(
        [("window", _WINDOW), ("rolling", _ROLLING)],
        name_func=parameterized_name_func,
    )
    def test_readers_merge_and_reuse_across_passes(self, _name, mode):
        """Merge reader tails below and above capacity across passes."""
        first_tasks = [
            ReadTask(
                fn=_yield_rows,
                kwargs=dict(
                    n=2, chunk=2, start_id=100 * reader_id, reorder=reader_id == 0
                ),
                reader_id=reader_id,
            )
            for reader_id in range(4)
        ]
        full_tasks = [
            ReadTask(
                fn=_yield_rows,
                kwargs=dict(n=5, chunk=5, start_id=100 * reader_id),
                reader_id=reader_id,
            )
            for reader_id in range(4)
        ]
        coord = self._make_coordinator(
            _plan(8, tasks=first_tasks, num_readers=4),
            _plan(20, tasks=full_tasks, num_readers=4),
            mode=mode,
        )
        runner, errors = self._run_with_readers(coord)

        batches, msg = self._collect_batches(coord, timeout=5.0)
        self.assertEqual(msg[0], sample_shuffle._END)
        self.assertEqual([b.num_rows for b in batches], [8])
        ids = [i for b in batches for i in b.column("id").to_pylist()]
        self.assertEqual(
            sorted(ids), sorted(100 * r + i for r in range(4) for i in range(2))
        )
        self.assertTrue(runner.is_alive())
        pids = [p.pid for p in coord._readers]
        self.assertEqual(coord._pool.total_rows, 0)

        batches, msg = self._collect_batches(coord, timeout=5.0)
        self.assertEqual(msg[0], sample_shuffle._END)
        self.assertEqual([b.num_rows for b in batches], [8, 8, 4])
        ids = [i for b in batches for i in b.column("id").to_pylist()]
        self.assertEqual(
            sorted(ids), sorted(100 * r + i for r in range(4) for i in range(5))
        )
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual([p.pid for p in coord._readers], pids)

        self._stop_runner(coord, runner, errors)

    def test_slow_consumer_shares_quota_with_fast(self):
        """A consumer holding its response must not stall the other one."""
        task = ReadTask(fn=_yield_rows, kwargs=dict(n=32, chunk=16))
        plan = _plan(32, tasks=[task])
        coord = self._make_coordinator(plan, plan, num_consumers=2)
        runner, errors = self._run_with_readers(coord)
        coord._request_q.put((sample_shuffle._GET, 0))
        self._wait_queued(coord, 0)
        batches = []
        for _ in range(3):
            coord._request_q.put((sample_shuffle._GET, 1))
            kind, payload = coord._response_queues[1].get(timeout=5.0)
            self.assertEqual(kind, sample_shuffle._BATCH)
            batches.append(pa.ipc.read_record_batch(payload, _PLAN_SCHEMA))
        kind, payload = coord._response_queues[0].get(timeout=5.0)
        self.assertEqual(kind, sample_shuffle._BATCH)
        batches.append(pa.ipc.read_record_batch(payload, _PLAN_SCHEMA))
        for consumer_id in (0, 1):
            coord._request_q.put((sample_shuffle._GET, consumer_id))
            msg = coord._response_queues[consumer_id].get(timeout=5.0)
            self.assertEqual(msg[0], sample_shuffle._END)
        self.assertEqual(sorted(b.num_rows for b in batches), [8, 8, 8, 8])
        ids = [i for b in batches for i in b.column("id").to_pylist()]
        self.assertEqual(sorted(ids), list(range(32)))
        self.assertEqual(len(ids), len(set(ids)))
        self._stop_runner(coord, runner, errors)

    def test_stop_with_unclaimed_response(self):
        """Stop the serving thread with unclaimed responses."""
        task = ReadTask(fn=_yield_rows, kwargs=dict(n=32, chunk=16))
        plan = _plan(32, tasks=[task])
        coord = self._make_coordinator(plan, plan, num_consumers=2)
        runner, errors = self._run_with_readers(coord)
        coord._request_q.put((sample_shuffle._GET, 0))
        self._wait_queued(coord, 0)
        coord._request_q.put((sample_shuffle._GET, 1))
        self._wait_queued(coord, 1)
        self._stop_runner(coord, runner, errors)

    def _wait_input_backpressure(self, coord):
        """Wait for the pool to exceed capacity and back up the input queue."""
        deadline = time.monotonic() + 10.0
        while not (coord._pool.total_rows >= 20 and coord._input_q.full()):
            if time.monotonic() > deadline:
                self.fail("reader never filled the pool and the input queue")
            time.sleep(0.01)
        time.sleep(1.0)
        self.assertEqual(coord._pool.total_rows, 20)
        self.assertTrue(coord._input_q.full())

    def _backpressure_coord(self, mode=_ROLLING):
        """Coordinator whose single reader outgrows a capacity-1 input queue."""
        task = ReadTask(fn=_yield_rows, kwargs=dict(n=30, chunk=5))
        plan = _plan(30, tasks=[task])
        with mock.patch.object(sample_shuffle, "_INPUT_QUEUE_BATCHES_PER_READER", 1):
            coord = self._make_coordinator(plan, plan, mode=mode)
        runner, errors = self._run_with_readers(coord)
        return coord, runner, errors

    @parameterized.expand(
        [("window", _WINDOW), ("rolling", _ROLLING)],
        name_func=parameterized_name_func,
    )
    def test_input_queue_backpressure_resumes(self, _name, mode):
        """Consumption after the reader waited on the input queue loses nothing."""
        coord, runner, errors = self._backpressure_coord(mode)
        self._wait_input_backpressure(coord)
        batches, msg = self._collect_batches(coord, timeout=5.0)
        self.assertEqual(msg[0], sample_shuffle._END)
        self.assertEqual([b.num_rows for b in batches], [8, 8, 8, 6])
        ids = [i for b in batches for i in b.column("id").to_pylist()]
        self.assertEqual(sorted(ids), list(range(30)))
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(coord._pool.total_rows, 0)
        self._stop_runner(coord, runner, errors)

    def test_stop_while_reader_waits_on_input_queue(self):
        """Stopping while the reader is blocked on input must exit cleanly."""
        coord, runner, errors = self._backpressure_coord()
        self._wait_input_backpressure(coord)
        self._stop_runner(coord, runner, errors)


def _yield_rows(n=10, chunk=4, start_id=0, reorder=False):
    """Module-level task; reorders columns to exercise conforming."""
    for start in range(start_id, start_id + n, chunk):
        k = min(chunk, start_id + n - start)
        yield _plan_batch(k, start, reorder)


def _yield_then_raise():
    yield next(iter(_yield_rows(n=2, chunk=2)))
    raise ValueError("task boom")


class ReaderProcessTest(unittest.TestCase):
    def test_exception_flushes_error_and_exits(self):
        ctx = mp.get_context("fork")
        task_q, input_q = ctx.Queue(maxsize=1), ctx.Queue()
        self.addCleanup(task_q.cancel_join_thread)
        self.addCleanup(input_q.cancel_join_thread)
        stop_event = ctx.Event()
        p = ctx.Process(
            target=_reader_process_main,
            args=(
                0,
                task_q,
                input_q,
                stop_event,
                ctx.Event(),
                ctx.Event(),
                os.getpid(),
            ),
        )
        p.start()
        self.addCleanup(stop_event.set)
        self.addCleanup(p.join, 10.0)
        task_q.put(([ReadTask(fn=_yield_then_raise, reader_id=0)], _PLAN_SCHEMA))
        data_msg = input_q.get(timeout=10.0)
        self.assertEqual(data_msg[0], sample_shuffle._DATA)
        error_msg = input_q.get(timeout=10.0)
        self.assertEqual(error_msg[0], sample_shuffle._ERROR)
        self.assertIn("task boom", error_msg[2])
        p.join(timeout=10.0)
        self.assertFalse(p.is_alive())


class HandleTest(unittest.TestCase):
    """The handle's protocol client, plans and config validation."""

    def _mock_reader(self, *plans, checkpoint_state=None):
        reader = mock.Mock()
        reader._min_batch_size = 0
        reader._checkpoint_state = checkpoint_state
        reader.create_read_plan.side_effect = list(plans)
        return reader

    def _handle(self, reader=None):
        if reader is None:
            reader = self._mock_reader(_plan(8))
        cfg = data_pb2.DataConfig(
            batch_size=8,
            shuffle_buffer_size=2,
            shuffle=True,
            shuffle_mode=_ROLLING,
            num_workers=2,
        )
        handle = SampleShuffleHandle(reader, cfg, 2, 0, 1)
        self.addCleanup(handle.stop)
        return reader, handle

    def test_protocol_client(self):
        _, handle = self._handle()
        payload = _plan_batch(2).serialize().to_pybytes()
        handle._response_queues[0].put((sample_shuffle._BATCH, payload))
        it = handle.batches(0)
        self.assertEqual(next(it)["id"].to_pylist(), [0, 1])
        it.close()
        self.assertEqual(handle._request_q.get(timeout=1.0), (sample_shuffle._GET, 0))
        self.assertEqual(
            handle._request_q.get(timeout=1.0), (sample_shuffle._CANCEL, 0)
        )
        handle._response_queues[0].put((sample_shuffle._END, None))
        with self.assertRaises(StopIteration):
            next(handle.batches(0))
        handle._response_queues[0].put((sample_shuffle._ERROR, "boom"))
        with self.assertRaisesRegex(RuntimeError, "boom"):
            next(handle.batches(0))

    def test_plans_and_constructor_validation(self):
        plan = _plan(8)
        reader, handle = self._handle(self._mock_reader(plan))
        self.assertIs(handle._full_plan, handle._first_plan)
        self.assertIs(handle._first_plan, plan)
        reader.load_state_dict.assert_not_called()

        first, full = _plan(3), _plan(8)
        reader = self._mock_reader(first, full, checkpoint_state={"prefix:0": 3})
        _, handle = self._handle(reader)
        self.assertIs(handle._first_plan, first)
        self.assertIs(handle._full_plan, full)
        reader.load_state_dict.assert_called_once_with(None)

        plan = _plan(None, is_finite=False)
        reader = self._mock_reader(plan, checkpoint_state={"topic:0": 5})
        _, handle = self._handle(reader)
        self.assertIs(handle._full_plan, handle._first_plan)
        self.assertIs(handle._first_plan, plan)
        reader.load_state_dict.assert_not_called()

        for overrides in (
            dict(batch_size=0),
            dict(shuffle_buffer_size=0),
            dict(batch_cost_size=1024),
        ):
            with self.assertRaises(ValueError):
                SampleShuffleHandle(
                    self._mock_reader(_plan(8)),
                    data_pb2.DataConfig(**overrides),
                    1,
                    0,
                    1,
                )
