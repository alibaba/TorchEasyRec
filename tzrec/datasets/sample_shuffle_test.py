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

_WINDOW = data_pb2.SAMPLE_WINDOW
_ROLLING = data_pb2.SAMPLE_ROLLING


def make_batch(n, start_id=0, partition=0, rows_per_msg=16, offset_start=0):
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
            pa.array([f"topic:{partition}"] * n),
            pa.array(offset_start + ids // rows_per_msg),
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


def check_batch(tc, batch, partition=0, offset_start=0, rows_per_msg=16):
    """Whole-row correspondence: every column follows the row id permutation."""
    for row in batch.to_pylist():
        i = row["id"]
        tc.assertEqual(
            row,
            {
                "id": int(i),
                "str_f": None if i % 3 == 0 else f"v{i}",
                "list_f": None if i % 4 == 0 else [int(i), int(i + 1)],
                "label": int(i % 2),
                CKPT_SOURCE_ID: f"topic:{partition}",
                CKPT_ROW_IDX: offset_start + int(i) // rows_per_msg,
                DATA_TIMESTAMP: float(i) + 0.5,
            },
        )


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


def _feed_batch(coord, n, start_id=0, reader_id=0):
    ids = np.arange(start_id, start_id + n, dtype=np.int64)
    rb = pa.RecordBatch.from_arrays(
        [
            pa.array(ids),
            pa.array(["topic:0"] * n),
            pa.array(ids // 16),
            pa.array(ids.astype(np.float64)),
        ],
        schema=_PLAN_SCHEMA,
    )
    coord._input_q.put((sample_shuffle._DATA, reader_id, rb.serialize().to_pybytes()))


def _feed_eof(coord, reader_id=0):
    coord._input_q.put((sample_shuffle._EOF, reader_id))


class SamplePoolTest(unittest.TestCase):
    def _pool(self, capacity, mode, seed=0):
        return SamplePool(capacity, mode, rng=np.random.default_rng(seed))

    @parameterized.expand(
        [param("window", mode=_WINDOW), param("rolling", mode=_ROLLING)],
        name_func=parameterized_name_func,
    )
    def test_fill_drain_and_discard(self, _name, mode):
        pool = self._pool(16, mode)
        self.assertIsNone(pool.emit(8))
        pool.add(make_batch(8))
        self.assertIsNone(pool.emit(8))
        self.assertIsNone(pool.emit(12, drain=True))
        check_batch(self, pool.emit(8, drain=True))
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
        pool = self._pool(64, _WINDOW, seed=3)
        # freeze: the whole pool is permuted before emission
        pool.add(make_batch(64))
        order = []
        for _ in range(4):
            out = pool.emit(16)
            check_batch(self, out)
            order.extend(out.column("id").to_pylist())
        self.assertEqual(order, np.random.default_rng(3).permutation(64).tolist())
        # arrivals during the frozen phase wait for the next window
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
        # drain leaves frozen rows that join later arrivals
        tail = self._pool(16, _WINDOW)
        tail.add(make_batch(12))
        check_batch(self, tail.emit(8, drain=True))
        tail.add(make_batch(5, start_id=100))
        ids = set(tail.emit(8, drain=True).column("id").to_pylist())
        self.assertTrue(any(i < 12 for i in ids) and any(i >= 100 for i in ids))
        self.assertEqual(tail.total_rows, 1)
        self.assertEqual(tail.emit(1, drain=True).num_rows, 1)

    def test_rolling_sequence(self):
        pool = self._pool(32, _ROLLING, seed=7)
        pool.add(make_batch(32))
        first = pool.emit(16)
        check_batch(self, first)
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
        for _ in range(40):
            pool.add(make_batch(16, start_id=next_id))
            next_id += 16
            for _ in range(2):
                out = pool.emit(8)
                if out is None:
                    break
                produced.append(out)
                check_batch(self, out)
        self.assertGreater(pool.num_compactions, 0)
        self.assertLessEqual(len(pool._chunks), 10)
        self.assertEqual(pool.rows_in, next_id)
        self.assertEqual(pool.rows_in - pool.rows_out, pool.total_rows)
        ids = [i for b in produced for i in b.column("id").to_pylist()]
        self.assertEqual(len(ids), len(produced) * 8)
        self.assertEqual(len(set(ids)), len(ids))


class CoordinatorTest(unittest.TestCase):
    """In-process coordinator tests over the request/response protocol."""

    def _coordinator(
        self, planned_rows, is_finite=True, min_batch_size=0, num_readers=1
    ):
        ctx = mp.get_context("spawn")
        request_q = ctx.Queue()
        response_queues = [ctx.Queue(maxsize=1)]
        self.addCleanup(request_q.cancel_join_thread)
        self.addCleanup(response_queues[0].cancel_join_thread)
        coord = Coordinator(
            first_plan=_plan(
                planned_rows, is_finite=is_finite, num_readers=num_readers
            ),
            full_plan=_plan(planned_rows, is_finite=is_finite, num_readers=num_readers),
            request_q=request_q,
            response_queues=response_queues,
            stop_event=ctx.Event(),
            cancel_event=ctx.Event(),
            mode=_ROLLING,
            capacity_rows=16,
            batch_size=8,
            min_batch_size=min_batch_size,
            num_consumers=1,
            trainer_pid=os.getpid(),
        )
        self.addCleanup(coord._input_q.cancel_join_thread)
        for task_q in coord._task_queues:
            self.addCleanup(task_q.cancel_join_thread)
        coord._start_pass(coord._first_plan)
        for task_q in coord._task_queues:
            task_q.get(timeout=1.0)
        return coord

    def _request(self, coord, consumer_id=0):
        coord._request_q.put((sample_shuffle._GET, consumer_id))

    def _settle_input(self, coord, timeout=2.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if coord._verified or coord._error is not None:
                return
            coord._service_input(0.05)
        raise AssertionError("input did not settle")

    def _next_response(self, coord, consumer_id=0, timeout=1.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            coord._service_requests()
            coord._serve_waiting()
            try:
                return coord._response_queues[consumer_id].get_nowait()
            except queue.Empty:
                time.sleep(0.005)
        raise queue.Empty

    def _drain_batches(self, coord, consumer_id=0):
        sizes = []
        while True:
            self._request(coord, consumer_id)
            msg = self._next_response(coord, consumer_id)
            if msg[0] == sample_shuffle._BATCH:
                sizes.append(len(pa.ipc.read_record_batch(msg[1], _PLAN_SCHEMA)))
            else:
                return sizes, msg

    @parameterized.expand(
        [
            # planned_rows is already rank-aligned and tail-trimmed
            ("full", 20, 0, [8, 8, 4]),
            ("no_tail", 16, 5, [8, 8]),
            ("tail_at_min", 18, 2, [8, 8, 2]),
            ("small_plan", 5, 0, [5]),
            ("empty_plan", 0, 0, []),
            # unknown row counts drain after EOF, applying the tail rule
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
        sizes, msg = self._drain_batches(coord)
        self.assertEqual(sizes, expected)
        self.assertEqual(msg[0], sample_shuffle._END)
        # every consumed chunk buffer is released at the pass end
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
        # the quota is used up but EOF has not arrived: no END yet
        self._request(coord)
        coord._service_requests()
        coord._serve_waiting()
        self.assertTrue(coord._response_queues[0].empty())
        _feed_eof(coord)
        self._settle_input(coord)
        self.assertEqual(self._next_response(coord)[0], sample_shuffle._END)

    @parameterized.expand(
        [
            ("more_rows", 12, False, "more than the planned"),
            ("fewer_rows", 8, True, "expected 10"),
        ],
        name_func=parameterized_name_func,
    )
    def test_row_count_mismatch_fails(self, _name, rows, eof, error):
        coord = self._coordinator(planned_rows=10)
        _feed_batch(coord, rows)
        if eof:
            _feed_eof(coord)
        self._settle_input(coord)
        self.assertIn(error, coord._error)

    def test_reader_error_reaches_consumer(self):
        coord = self._coordinator(planned_rows=10)
        coord._input_q.put((sample_shuffle._ERROR, 0, "boom"))
        self._settle_input(coord)
        self._request(coord)
        msg = self._next_response(coord)
        self.assertEqual(msg[0], sample_shuffle._ERROR)
        self.assertIn("reader 0 failed", msg[1])
        self.assertIn("boom", msg[1])

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
        # a reader killed mid-write leaves a truncated frame behind: any read
        # of the input queue blocks forever on the incomplete message
        os.write(coord._input_q._writer.fileno(), struct.pack("!i", 1 << 20))
        dead = mock.Mock()
        dead.is_alive.return_value = False
        dead.exitcode = -9
        coord._readers = [dead]
        with mock.patch.object(coord, "_start_readers"):
            runner = threading.Thread(target=coord.run, daemon=True)
            runner.start()
            runner.join(timeout=5.0)
        self.assertFalse(runner.is_alive())
        self.assertIn("exited unexpectedly", coord._error)

    def test_two_passes_reuse_processes_with_distinct_plans(self):
        ctx = mp.get_context("spawn")
        request_q, response_q = ctx.Queue(), ctx.Queue(maxsize=1)
        self.addCleanup(request_q.cancel_join_thread)
        self.addCleanup(response_q.cancel_join_thread)
        coord = Coordinator(
            first_plan=_plan(
                6,
                tasks=[
                    ReadTask(fn=_yield_rows, kwargs=dict(n=6, chunk=4, reorder=True))
                ],
            ),
            full_plan=_plan(
                10,
                tasks=[
                    ReadTask(fn=_yield_rows, kwargs=dict(n=10, chunk=4, start_id=100))
                ],
            ),
            request_q=request_q,
            response_queues=[response_q],
            stop_event=ctx.Event(),
            cancel_event=ctx.Event(),
            mode=_ROLLING,
            capacity_rows=16,
            batch_size=8,
            min_batch_size=0,
            num_consumers=1,
            trainer_pid=os.getpid(),
        )
        self.addCleanup(coord._input_q.cancel_join_thread)
        for task_q in coord._task_queues:
            self.addCleanup(task_q.cancel_join_thread)
        runner = threading.Thread(target=coord.run, daemon=True)
        runner.start()

        def _drain_pass():
            ids = []
            request_q.put((sample_shuffle._GET, 0))
            while True:
                kind, payload = response_q.get(timeout=5.0)
                if kind == sample_shuffle._END:
                    return ids
                ids.extend(
                    pa.ipc.read_record_batch(payload, _PLAN_SCHEMA)
                    .column("id")
                    .to_pylist()
                )
                request_q.put((sample_shuffle._GET, 0))

        self.assertEqual(sorted(_drain_pass()), list(range(6)))
        self.assertTrue(runner.is_alive())
        pids = [p.pid for p in coord._readers]
        self.assertEqual(coord._pool.total_rows, 0)
        self.assertEqual(sorted(_drain_pass()), list(range(100, 110)))
        self.assertEqual([p.pid for p in coord._readers], pids)
        coord._stop_event.set()
        runner.join(timeout=10.0)
        self.assertFalse(runner.is_alive())
        self.assertFalse(any(p.is_alive() for p in coord._readers))

    def test_two_readers_merge_into_one_batch(self):
        ctx = mp.get_context("spawn")
        request_q, response_q = ctx.Queue(), ctx.Queue(maxsize=1)
        self.addCleanup(request_q.cancel_join_thread)
        self.addCleanup(response_q.cancel_join_thread)
        tasks = [
            ReadTask(fn=_yield_rows, kwargs=dict(n=4, chunk=4), reader_id=0),
            ReadTask(
                fn=_yield_rows,
                kwargs=dict(n=4, chunk=4, start_id=100),
                reader_id=1,
            ),
        ]
        coord = Coordinator(
            first_plan=_plan(8, tasks=tasks, num_readers=2),
            full_plan=_plan(8, tasks=tasks, num_readers=2),
            request_q=request_q,
            response_queues=[response_q],
            stop_event=ctx.Event(),
            cancel_event=ctx.Event(),
            mode=_ROLLING,
            capacity_rows=16,
            batch_size=8,
            min_batch_size=0,
            num_consumers=1,
            trainer_pid=os.getpid(),
        )
        self.addCleanup(coord._input_q.cancel_join_thread)
        for task_q in coord._task_queues:
            self.addCleanup(task_q.cancel_join_thread)
        runner = threading.Thread(target=coord.run, daemon=True)
        runner.start()

        request_q.put((sample_shuffle._GET, 0))
        kind, payload = response_q.get(timeout=10.0)
        self.assertEqual(kind, sample_shuffle._BATCH)
        ids = pa.ipc.read_record_batch(payload, _PLAN_SCHEMA).column("id").to_pylist()
        # finite input from both readers completes exactly one batch
        self.assertEqual(sorted(ids), [0, 1, 2, 3, 100, 101, 102, 103])
        request_q.put((sample_shuffle._GET, 0))
        self.assertEqual(response_q.get(timeout=5.0)[0], sample_shuffle._END)

        coord._stop_event.set()
        runner.join(timeout=10.0)
        self.assertFalse(runner.is_alive())
        self.assertFalse(any(p.is_alive() for p in coord._readers))


def _yield_rows(stop_event, n=10, chunk=4, start_id=0, reorder=False):
    """Module-level task; reorders columns to exercise conforming."""
    for start in range(start_id, start_id + n, chunk):
        k = min(chunk, start_id + n - start)
        arrays = [
            pa.array(np.arange(start, start + k, dtype=np.int64)),
            pa.array(["topic:0"] * k),
            pa.array([0] * k),
            pa.array([-1.0] * k),
        ]
        names = ["id", CKPT_SOURCE_ID, CKPT_ROW_IDX, DATA_TIMESTAMP]
        if reorder:
            arrays, names = arrays[::-1], names[::-1]
        yield pa.RecordBatch.from_arrays(arrays, names=names)


def _yield_then_raise(stop_event):
    yield next(iter(_yield_rows(stop_event, n=2, chunk=2)))
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
        msgs = []
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            try:
                msg = input_q.get(timeout=0.5)
            except queue.Empty:
                break
            msgs.append(msg)
            if msg[0] in (sample_shuffle._EOF, sample_shuffle._ERROR):
                break
        self.assertEqual(msgs[0][0], sample_shuffle._DATA)
        self.assertEqual(msgs[-1][0], sample_shuffle._ERROR)
        self.assertIn("task boom", msgs[-1][2])
        p.join(timeout=10.0)
        self.assertFalse(p.is_alive())


class _FakeReader:
    """Minimal reader used by the handle tests."""

    _min_batch_size = 0

    def __init__(self):
        self.states = []
        self.plan = _plan(8)
        self.plans = []
        self._checkpoint_state = None

    def create_read_plan(self, rank, world_size, num_readers):
        if self.plans:
            return self.plans.pop(0)
        return self.plan

    def load_state_dict(self, state):
        self.states.append(state)
        self._checkpoint_state = state


class HandleTest(unittest.TestCase):
    """The handle's protocol client, plans and config validation."""

    def _handle(self, reader=None, num_workers=2):
        if reader is None:
            reader = _FakeReader()
        cfg = data_pb2.DataConfig(
            batch_size=8,
            shuffle_buffer_size=2,
            shuffle=True,
            shuffle_mode=_ROLLING,
            num_workers=num_workers,
        )
        handle = SampleShuffleHandle(reader, cfg, num_workers, 0, 1)
        self.addCleanup(handle.stop)
        return reader, handle

    def test_protocol_client(self):
        _, handle = self._handle()
        payload = (
            pa.RecordBatch.from_arrays(
                [
                    pa.array(np.arange(2, dtype=np.int64)),
                    pa.array(["topic:0"] * 2),
                    pa.array([0] * 2),
                    pa.array([-1.0] * 2),
                ],
                schema=_PLAN_SCHEMA,
            )
            .serialize()
            .to_pybytes()
        )
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
        reader, handle = self._handle()
        self.assertIs(handle._full_plan, handle._first_plan)
        self.assertIs(handle._first_plan, reader.plan)
        reader = _FakeReader()
        reader.load_state_dict({"prefix:0": 3})
        reader.plans = [_plan(3), _plan(8)]
        _, handle = self._handle(reader)
        self.assertEqual(handle._first_plan.planned_rows, 3)
        self.assertEqual(handle._full_plan.planned_rows, 8)
        self.assertEqual(reader.states, [{"prefix:0": 3}, None])
        reader = _FakeReader()
        reader.load_state_dict({"topic:0": 5})
        reader.plans = [_plan(None, is_finite=False)]
        _, handle = self._handle(reader)
        self.assertIs(handle._full_plan, handle._first_plan)
        self.assertEqual(reader.states, [{"topic:0": 5}])
        for overrides in (
            dict(batch_size=0),
            dict(shuffle_buffer_size=0),
            dict(batch_cost_size=1024),
        ):
            with self.assertRaises(ValueError):
                SampleShuffleHandle(
                    _FakeReader(), data_pb2.DataConfig(**overrides), 1, 0, 1
                )
