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

from tzrec.constant import Mode
from tzrec.datasets import sample_shuffle
from tzrec.datasets.sample_shuffle import (
    Coordinator,
    ReadPlan,
    ReadTask,
    SamplePool,
    SampleShuffleHandle,
    _conform_batch,
    _reader_process_main,
    sample_shuffle_enabled,
    validate_sample_shuffle_config,
)
from tzrec.datasets.utils import CKPT_ROW_IDX, CKPT_SOURCE_ID, DATA_TIMESTAMP
from tzrec.protos import data_pb2
from tzrec.utils.test_util import parameterized_name_func


def make_batch(n, start_id=0, partition=0, rows_per_msg=16, offset_start=0):
    """Build a record batch whose every column derives from a unique row id."""
    ids = np.arange(start_id, start_id + n, dtype=np.int64)
    offsets = offset_start + ids // rows_per_msg
    return pa.RecordBatch.from_arrays(
        [
            pa.array(ids),
            pa.array(ids.astype(np.float64) * 1.5),
            pa.array([f"str_{i}" for i in ids]),
            pa.array([[int(i), int(i + 1)] for i in ids]),
            pa.array(ids % 2),
            pa.array(ids.astype(np.float32) * 0.25),
            pa.array([f"topic:{partition}"] * n),
            pa.array(offsets),
            pa.array(ids.astype(np.float64) + 0.5),
        ],
        names=[
            "id",
            "feat",
            "str_f",
            "list_f",
            "label",
            "weight",
            CKPT_SOURCE_ID,
            CKPT_ROW_IDX,
            DATA_TIMESTAMP,
        ],
    )


def check_row_correspondence(tc, batch, partition=0, offset_start=0, rows_per_msg=16):
    """All columns of an emitted batch must follow the row id permutation."""
    for row in batch.to_pylist():
        i = row["id"]
        tc.assertAlmostEqual(row["feat"], i * 1.5)
        tc.assertEqual(row["str_f"], f"str_{i}")
        tc.assertEqual(row["list_f"], [i, i + 1])
        tc.assertEqual(row["label"], i % 2)
        tc.assertAlmostEqual(row["weight"], i * 0.25, places=5)
        tc.assertEqual(row[CKPT_SOURCE_ID], f"topic:{partition}")
        tc.assertEqual(row[CKPT_ROW_IDX], offset_start + i // rows_per_msg)
        tc.assertAlmostEqual(row[DATA_TIMESTAMP], i + 0.5)


class ConfigTest(unittest.TestCase):
    @parameterized.expand(
        [
            param("zero_batch", batch_size=0),
            param("zero_buffer", shuffle_buffer_size=0),
            param("batch_cost", batch_cost_size=1024),
        ],
        name_func=parameterized_name_func,
    )
    def test_invalid_configs(self, _name, **overrides):
        cfg = data_pb2.DataConfig(**overrides)
        with self.assertRaises(ValueError):
            validate_sample_shuffle_config(cfg)

    def test_valid_config(self):
        validate_sample_shuffle_config(
            data_pb2.DataConfig(batch_size=8, shuffle_buffer_size=2)
        )

    def test_enabled_only_for_train_shuffle_sample_mode(self):
        cfg = data_pb2.DataConfig(
            shuffle=True,
            shuffle_mode=data_pb2.ShuffleMode.SAMPLE_ROLLING,
        )
        self.assertTrue(sample_shuffle_enabled(cfg, Mode.TRAIN))
        self.assertFalse(sample_shuffle_enabled(cfg, Mode.EVAL))
        self.assertFalse(sample_shuffle_enabled(cfg, Mode.PREDICT))
        self.assertFalse(
            sample_shuffle_enabled(
                data_pb2.DataConfig(
                    shuffle=False,
                    shuffle_mode=data_pb2.ShuffleMode.SAMPLE_ROLLING,
                ),
                Mode.TRAIN,
            )
        )
        self.assertFalse(
            sample_shuffle_enabled(
                data_pb2.DataConfig(
                    shuffle=True, shuffle_mode=data_pb2.ShuffleMode.BATCH
                ),
                Mode.TRAIN,
            )
        )


class ConformBatchTest(unittest.TestCase):
    def test_reorder_and_cast(self):
        schema = pa.schema(
            [
                pa.field("a", pa.int64()),
                pa.field("b", pa.string()),
            ]
        )
        batch = pa.RecordBatch.from_arrays(
            [pa.array(["x"]), pa.array([1], type=pa.int32())], names=["b", "a"]
        )
        out = _conform_batch(batch, schema)
        self.assertEqual(out.schema.names, ["a", "b"])
        self.assertEqual(out.column("a").type, pa.int64())
        self.assertEqual(out.to_pydict(), {"a": [1], "b": ["x"]})

    def test_missing_column_raises(self):
        schema = pa.schema([pa.field("a", pa.int64()), pa.field("b", pa.string())])
        batch = pa.RecordBatch.from_arrays([pa.array([1])], names=["a"])
        with self.assertRaisesRegex(KeyError, "does not exist"):
            _conform_batch(batch, schema)


class SamplePoolTest(unittest.TestCase):
    def _pool(self, capacity, mode, seed=0):
        rng = np.random.default_rng(seed)
        return SamplePool(capacity, mode, rng=rng)

    @parameterized.expand(
        [
            (data_pb2.SAMPLE_WINDOW,),
            (data_pb2.SAMPLE_ROLLING,),
        ],
        name_func=parameterized_name_func,
    )
    def test_no_emit_until_full(self, mode):
        pool = self._pool(128, mode)
        self.assertIsNone(pool.emit(16))
        # fewer rows than one batch: retained, nothing emitted
        pool.add(make_batch(8))
        self.assertIsNone(pool.emit(16))
        self.assertEqual(pool.total_rows, 8)
        # still short of capacity
        pool.add(make_batch(64, start_id=8))
        self.assertIsNone(pool.emit(16))
        # fill past capacity: emit one full batch
        pool.add(make_batch(64, start_id=72))
        out = pool.emit(16)
        self.assertIsNotNone(out)
        self.assertEqual(out.num_rows, 16)

    def test_gather_restores_selection_order(self):
        pool = self._pool(16, data_pb2.SAMPLE_ROLLING)
        pool.add(make_batch(4, start_id=0))
        pool.add(make_batch(4, start_id=100))
        out = pool._gather(
            np.array([1, 0, 1, 0], dtype=np.int32),
            np.array([2, 1, 0, 3], dtype=np.int32),
        )
        # rows stay in the sampled order, not the chunk-grouped read order
        self.assertEqual(out.column("id").to_pylist(), [102, 1, 100, 3])

    @parameterized.expand([(0,), (1,), (2,)], name_func=parameterized_name_func)
    def test_window_emits_the_sampled_permutation(self, seed):
        pool = self._pool(8, data_pb2.SAMPLE_WINDOW, seed=seed)
        for i in range(8):
            pool.add(make_batch(1, start_id=i))
        expected = np.random.default_rng(seed).permutation(8).tolist()
        out = pool.emit(8)
        self.assertIsNotNone(out)
        self.assertEqual(out.column("id").to_pylist(), expected)

    def test_capacity_one_batch(self):
        batch_size = 8
        pool = self._pool(batch_size, data_pb2.SAMPLE_ROLLING)
        pool.add(make_batch(batch_size))
        out = pool.emit(batch_size)
        self.assertIsNotNone(out)
        self.assertEqual(out.num_rows, batch_size)

    @parameterized.expand(
        [
            (data_pb2.SAMPLE_WINDOW,),
            (data_pb2.SAMPLE_ROLLING,),
        ],
        name_func=parameterized_name_func,
    )
    def test_drain_waits_for_request_rows_only(self, mode):
        pool = self._pool(64, mode)
        pool.add(make_batch(5))
        self.assertIsNone(pool.emit(8, drain=True))
        out = pool.emit(5, drain=True)
        self.assertIsNotNone(out)
        self.assertEqual(out.num_rows, 5)
        self.assertEqual(pool.total_rows, 0)

    def test_window_drain_tail_and_leftover(self):
        pool = self._pool(16, data_pb2.SAMPLE_WINDOW)
        pool.add(make_batch(12))
        out = pool.emit(8, drain=True)
        self.assertIsNotNone(out)
        out = pool.emit(4, drain=True)
        self.assertIsNotNone(out)
        self.assertEqual(pool.total_rows, 0)

    def test_window_leftover_joins_next_window(self):
        pool = self._pool(16, data_pb2.SAMPLE_WINDOW)
        pool.add(make_batch(12))
        first = pool.emit(8, drain=True)
        self.assertIsNotNone(first)
        # three rows still in the frozen window
        pool.add(make_batch(5, start_id=100))
        out = pool.emit(8, drain=True)
        self.assertIsNotNone(out)
        ids = set(out.column("id").to_pylist())
        self.assertEqual(len(ids), 8)
        self.assertTrue(any(i < 12 for i in ids))
        self.assertTrue(any(i >= 100 for i in ids))

    def test_window_emits_all_full_batches_without_new_data(self):
        pool = self._pool(96, data_pb2.SAMPLE_WINDOW)
        pool.add(make_batch(100))
        seen = []
        while True:
            out = pool.emit(16)
            if out is None:
                break
            seen.append(out)
        self.assertEqual(sum(b.num_rows for b in seen), 96)  # 6 full batches
        self.assertEqual(pool.total_rows, 4)  # leftover retained

    def test_window_arrivals_not_mixed_into_frozen_window(self):
        pool = self._pool(64, data_pb2.SAMPLE_WINDOW)
        pool.add(make_batch(64, start_id=0))
        first = pool.emit(16)
        self.assertIsNotNone(first)
        first_ids = set(first.column("id").to_pylist())
        # arrivals during the frozen phase
        pool.add(make_batch(32, start_id=1000))
        rest = []
        while True:
            out = pool.emit(16)
            if out is None:
                break
            rest.append(out)
        frozen_ids = set(first_ids)
        for b in rest:
            frozen_ids.update(b.column("id").to_pylist())
        self.assertTrue(all(i < 64 for i in frozen_ids))
        # the 32 new rows wait for the next full window
        self.assertEqual(pool.total_rows, 32)
        self.assertIsNone(pool.emit(16))
        pool.add(make_batch(32, start_id=2000))
        out = pool.emit(16)
        self.assertIsNotNone(out)
        self.assertTrue(all(i >= 1000 for i in out.column("id").to_pylist()))

    def test_rolling_extract_then_refill(self):
        # deterministic rng: the first batch is a fixed subset
        pool = self._pool(32, data_pb2.SAMPLE_ROLLING, seed=7)
        pool.add(make_batch(32, start_id=0))
        first = pool.emit(16)
        self.assertIsNotNone(first)
        first_ids = set(first.column("id").to_pylist())
        self.assertEqual(len(first_ids), 16)
        self.assertEqual(pool.total_rows, 16)
        # no emit before the refill reaches capacity
        self.assertIsNone(pool.emit(16))
        pool.add(make_batch(16, start_id=32))
        second = pool.emit(16)
        self.assertIsNotNone(second)
        second_ids = set(second.column("id").to_pylist())
        # the second batch mixes old and new samples and repeats nothing
        self.assertEqual(len(second_ids & first_ids), 0)
        self.assertTrue(any(i < 32 for i in second_ids))
        self.assertTrue(any(i >= 32 for i in second_ids))
        self.assertEqual(pool.total_rows, 16)

    def test_row_correspondence_and_conservation(self):
        for mode in (data_pb2.SAMPLE_WINDOW, data_pb2.SAMPLE_ROLLING):
            pool = self._pool(96, mode)
            total_in = 0
            for i in range(6):
                pool.add(
                    make_batch(
                        32, start_id=32 * i, partition=i % 3, offset_start=100 * (i % 3)
                    )
                )
                total_in += 32
            emitted = []
            while True:
                out = pool.emit(16)
                if out is None:
                    break
                emitted.append(out)
            for batch in emitted:
                # per-row correspondence is per source partition
                for part in range(3):
                    cols = batch.filter(
                        pa.compute.equal(
                            batch.column(CKPT_SOURCE_ID), pa.scalar(f"topic:{part}")
                        )
                    )
                    check_row_correspondence(
                        self, cols, partition=part, offset_start=100 * part
                    )
            rows_out = sum(b.num_rows for b in emitted)
            self.assertEqual(rows_out + pool.total_rows, total_in)
            self.assertEqual(pool.rows_in, total_in)
            self.assertEqual(pool.rows_out, rows_out)

    def test_msg_split_and_batch_spanning_offsets(self):
        pool = self._pool(40, data_pb2.SAMPLE_WINDOW)
        # one message of 40 rows (all at offset 7) spans multiple batches
        pool.add(make_batch(40, rows_per_msg=40, offset_start=7))
        offsets_seen = set()
        while True:
            out = pool.emit(16)
            if out is None:
                break
            offsets_seen.update(out.column(CKPT_ROW_IDX).to_pylist())
        # 40 rows -> 2 full batches, 8 retained; all carry offset 7
        self.assertEqual(offsets_seen, {7})
        self.assertEqual(pool.total_rows, 8)

    def test_window_boundary_compaction_bounds_memory(self):
        # reviewer scenario: capacity 32, batch 8, arrivals of 16 rows while
        # a window is frozen; the next window is full right when the previous
        # one ends, so add()-time compaction never fires
        pool = self._pool(32, data_pb2.SAMPLE_WINDOW)
        next_id = 0
        for _ in range(25):
            pool.add(make_batch(16, start_id=next_id))
            next_id += 16
            pool.add(make_batch(16, start_id=next_id))
            next_id += 16
            # freeze + drain the window, with arrivals mid-drain
            for i in range(10):
                out = pool.emit(8)
                if out is None:
                    break
                if i == 1:
                    pool.add(make_batch(16, start_id=next_id))
                    next_id += 16
        self.assertGreater(pool.num_compactions, 0)
        self.assertLessEqual(len(pool._chunks), 10)
        self.assertEqual(pool.rows_in, next_id)
        self.assertEqual(pool.rows_in - pool.rows_out, pool.total_rows)

    def test_output_batches_survive_compaction(self):
        pool = self._pool(32, data_pb2.SAMPLE_ROLLING)
        next_id = 0
        produced = []
        for _ in range(40):
            pool.add(make_batch(16, start_id=next_id))
            next_id += 16
            while True:
                out = pool.emit(8)
                if out is None:
                    break
                produced.append(out)
        self.assertGreater(pool.num_compactions, 0)
        # batches emitted before a compaction stay valid and row-correct
        seen = set()
        for b in produced:
            check_row_correspondence(self, b)
            seen.update(b.column("id").to_pylist())
        self.assertEqual(len(seen), len(produced) * 8)

    def test_null_and_nested_columns(self):
        pool = self._pool(32, data_pb2.SAMPLE_WINDOW)
        ids = np.arange(32, dtype=np.int64)
        rb = pa.RecordBatch.from_arrays(
            [
                pa.array(ids),
                pa.array([None if i % 3 == 0 else f"v{i}" for i in ids]),
                pa.array(
                    [None if i % 4 == 0 else [int(i)] * (i % 3) for i in ids],
                    type=pa.list_(pa.int64()),
                ),
                pa.array(["topic:0"] * 32),
                pa.array(ids // 16),
                pa.array(ids.astype(np.float64)),
            ],
            names=[
                "id",
                "str_f",
                "list_f",
                CKPT_SOURCE_ID,
                CKPT_ROW_IDX,
                DATA_TIMESTAMP,
            ],
        )
        pool.add(rb)
        out = pool.emit(16)
        self.assertIsNotNone(out)
        for row in out.to_pylist():
            i = row["id"]
            self.assertEqual(row["str_f"], None if i % 3 == 0 else f"v{i}")
            self.assertEqual(row["list_f"], None if i % 4 == 0 else [i] * (i % 3))
            self.assertEqual(row[CKPT_ROW_IDX], i // 16)

    def test_discard(self):
        pool = self._pool(16, data_pb2.SAMPLE_ROLLING)
        pool.add(make_batch(12))
        pool.discard()
        self.assertEqual(pool.total_rows, 0)
        self.assertIsNone(pool.emit(8, drain=True))
        pool.add(make_batch(8))
        self.assertIsNotNone(pool.emit(8, drain=True))


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


class CoordinatorTest(unittest.TestCase):
    """In-process coordinator tests over the request/response protocol."""

    def _coordinator(
        self,
        planned_rows,
        full_planned_rows=None,
        is_finite=True,
        batch_size=8,
        min_batch_size=0,
        capacity_rows=16,
        mode=data_pb2.SAMPLE_ROLLING,
        num_consumers=1,
        num_readers=1,
    ):
        ctx = mp.get_context("spawn")
        request_q = ctx.Queue()
        response_queues = [ctx.Queue(maxsize=1) for _ in range(num_consumers)]
        self.addCleanup(request_q.cancel_join_thread)
        for q in response_queues:
            self.addCleanup(q.cancel_join_thread)
        first_plan = _plan(planned_rows, is_finite=is_finite, num_readers=num_readers)
        if full_planned_rows is None:
            full_plan = first_plan
        else:
            full_plan = _plan(
                full_planned_rows, is_finite=is_finite, num_readers=num_readers
            )
        coord = Coordinator(
            first_plan=first_plan,
            full_plan=full_plan,
            request_q=request_q,
            response_queues=response_queues,
            stop_event=ctx.Event(),
            cancel_event=ctx.Event(),
            mode=mode,
            capacity_rows=capacity_rows,
            batch_size=batch_size,
            min_batch_size=min_batch_size,
            num_consumers=num_consumers,
            trainer_pid=os.getpid(),
        )
        self.addCleanup(coord._input_q.cancel_join_thread)
        for task_q in coord._task_queues:
            self.addCleanup(task_q.cancel_join_thread)
        # initialize the first pass like run() would, then empty the task
        # queues so tests that also call run() can deliver tasks again
        coord._start_pass(coord._first_plan)
        for task_q in coord._task_queues:
            task_q.get(timeout=1.0)
        return coord

    def _request(self, coord, consumer_id=0):
        coord._request_q.put((sample_shuffle._GET, 0, consumer_id))

    def _settle_input(self, coord, timeout=2.0):
        """Read reader messages until EOF verification or an error."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if coord._verified or coord._error is not None:
                return
            coord._service_input(0.05)
        raise AssertionError("input did not settle")

    def _wait_waiting(self, coord, consumer_id=0, timeout=2.0):
        """Wait for the coordinator to accept a queued GET."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            coord._service_requests()
            if (
                consumer_id in coord._request_order
                or not coord._response_queues[consumer_id].empty()
            ):
                return
            time.sleep(0.005)
        raise AssertionError("request was not accepted")

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
                rb = pa.ipc.read_record_batch(msg[1], _PLAN_SCHEMA)
                sizes.append(rb.num_rows)
            else:
                return sizes, msg

    @parameterized.expand(
        [
            # planned_rows is already rank-aligned and tail-trimmed
            ("full", 20, 0, [8, 8, 4]),
            ("tail_dropped", 16, 5, [8, 8]),
            ("tail_below_min", 16, 2, [8, 8]),
            ("tail_at_min", 18, 2, [8, 8, 2]),
            ("small_plan", 5, 0, [5]),
        ],
        name_func=parameterized_name_func,
    )
    def test_quota(self, _name, planned_rows, min_batch_size, expected):
        coord = self._coordinator(
            planned_rows=planned_rows, min_batch_size=min_batch_size
        )
        _feed_batch(coord, planned_rows)
        _feed_eof(coord)
        self._settle_input(coord)
        self.assertTrue(coord._verified)
        sizes, msg = self._drain_batches(coord)
        self.assertEqual(sizes, expected)
        self.assertEqual(msg[0], sample_shuffle._END)
        self.assertEqual(coord._remaining, 0)
        self.assertEqual(coord._rows_read, planned_rows)

    def test_end_waits_for_eof_after_quota(self):
        coord = self._coordinator(planned_rows=16)
        _feed_batch(coord, 16)
        coord._service_input(1.0)
        for _ in range(2):
            self._request(coord)
            msg = self._next_response(coord)
            self.assertEqual(msg[0], sample_shuffle._BATCH)
        self.assertEqual(coord._remaining, 0)
        # the quota is used up but EOF has not arrived: no END yet
        self._request(coord)
        self._wait_waiting(coord)
        coord._serve_waiting()
        self.assertTrue(coord._response_queues[0].empty())
        _feed_eof(coord)
        self._settle_input(coord)
        msg = self._next_response(coord)
        self.assertEqual(msg[0], sample_shuffle._END)

    def test_more_rows_than_plan_fails(self):
        coord = self._coordinator(planned_rows=10)
        _feed_batch(coord, 12)
        self._settle_input(coord)
        self.assertIsNotNone(coord._error)
        self.assertIn("more than the planned", coord._error)

    def test_fewer_rows_than_plan_fails_after_eof(self):
        coord = self._coordinator(planned_rows=10)
        _feed_batch(coord, 8)
        _feed_eof(coord)
        self._settle_input(coord)
        self.assertIsNotNone(coord._error)
        self.assertIn("expected 10", coord._error)

    def test_reader_error_fails(self):
        coord = self._coordinator(planned_rows=10)
        coord._input_q.put((sample_shuffle._ERROR, 0, "boom"))
        self._settle_input(coord)
        self.assertIn("reader 0 failed", coord._error)

    def test_unknown_rows_drain_after_eof(self):
        coord = self._coordinator(planned_rows=None)
        _feed_batch(coord, 20)
        coord._service_input(1.0)
        # not EOF yet: the pool is full, so one full batch is available
        self._request(coord)
        msg = self._next_response(coord)
        self.assertEqual(msg[0], sample_shuffle._BATCH)
        self.assertEqual(len(pa.ipc.read_record_batch(msg[1], _PLAN_SCHEMA)), 8)
        _feed_eof(coord)
        self._settle_input(coord)
        # the remaining 12 rows drain as a full batch plus a tail
        sizes = []
        while True:
            self._request(coord)
            msg = self._next_response(coord)
            if msg[0] == sample_shuffle._BATCH:
                sizes.append(len(pa.ipc.read_record_batch(msg[1], _PLAN_SCHEMA)))
            else:
                break
        self.assertEqual(sizes, [8, 4])
        self.assertEqual(msg[0], sample_shuffle._END)

    def test_unknown_rows_tail_below_min_is_dropped(self):
        coord = self._coordinator(planned_rows=None, min_batch_size=5)
        _feed_batch(coord, 20)
        _feed_eof(coord)
        self._settle_input(coord)
        sizes = []
        while True:
            self._request(coord)
            msg = self._next_response(coord)
            if msg[0] == sample_shuffle._BATCH:
                sizes.append(len(pa.ipc.read_record_batch(msg[1], _PLAN_SCHEMA)))
            else:
                break
        self.assertEqual(sizes, [8, 8])
        self.assertEqual(msg[0], sample_shuffle._END)

    def test_cancel_sets_event(self):
        coord = self._coordinator(planned_rows=8)
        coord._request_q.put((sample_shuffle._CANCEL, 0, 0))
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and not coord._cancel_event.is_set():
            coord._service_requests()
            time.sleep(0.005)
        self.assertTrue(coord._cancel_event.is_set())

    def test_error_binds_fail_response(self):
        coord = self._coordinator(planned_rows=8)
        self._request(coord)
        self._wait_waiting(coord)
        coord._fail("boom")
        coord._serve_waiting()
        msg = coord._response_queues[0].get(timeout=1.0)
        self.assertEqual(msg[0], sample_shuffle._ERROR)
        self.assertIn("boom", msg[1])

    def test_run_propagates_reader_error(self):
        coord = self._coordinator(planned_rows=16)
        coord._input_q.put((sample_shuffle._ERROR, 0, "boom"))
        with mock.patch.object(coord, "_start_readers"):
            with self.assertRaisesRegex(RuntimeError, "reader 0 failed"):
                coord.run()

    def test_dead_reader_fails_without_reading_its_input(self):
        coord = self._coordinator(planned_rows=8)
        # a reader killed mid-write leaves a truncated frame behind: any
        # read of the input queue blocks forever on the incomplete message
        os.write(coord._input_q._writer.fileno(), struct.pack("!i", 1 << 20))
        dead = mock.Mock()
        dead.is_alive.return_value = False
        dead.exitcode = -9
        coord._readers = [dead]
        with mock.patch.object(coord, "_start_readers"):
            runner = threading.Thread(target=coord.run, daemon=True)
            runner.start()
            runner.join(timeout=5.0)
        # the coordinator failed the service instead of hanging on the frame
        self.assertFalse(runner.is_alive())
        self.assertIsNotNone(coord._error)
        self.assertIn("exited unexpectedly", coord._error)

    def test_requests_are_served_in_arrival_order(self):
        coord = self._coordinator(planned_rows=8, num_consumers=2)
        _feed_batch(coord, 8)
        _feed_eof(coord)
        self._settle_input(coord)
        self._request(coord, consumer_id=1)
        self._wait_waiting(coord, consumer_id=1)
        self._request(coord, consumer_id=0)
        self._wait_waiting(coord, consumer_id=0)
        coord._serve_waiting()
        # the earlier GET of consumer 1 takes the only batch; consumer 0 ends
        msg = coord._response_queues[1].get(timeout=1.0)
        self.assertEqual(msg[0], sample_shuffle._BATCH)
        msg = coord._response_queues[0].get(timeout=1.0)
        self.assertEqual(msg[0], sample_shuffle._END)

    def test_input_done_stops_pulling(self):
        coord = self._coordinator(planned_rows=10, capacity_rows=16)
        _feed_batch(coord, 10)
        self.assertTrue(coord._pool_has_room())
        _feed_eof(coord)
        self._settle_input(coord)
        self.assertTrue(coord._input_done)
        self.assertFalse(coord._pool_has_room())

    def test_is_complete_only_after_end_served(self):
        coord = self._coordinator(planned_rows=8)
        _feed_batch(coord, 8)
        _feed_eof(coord)
        self._settle_input(coord)
        self._request(coord)
        msg = self._next_response(coord)
        self.assertEqual(msg[0], sample_shuffle._BATCH)
        self.assertFalse(coord._is_complete())
        self._request(coord)
        msg = self._next_response(coord)
        self.assertEqual(msg[0], sample_shuffle._END)
        self.assertTrue(coord._is_complete())

    @parameterized.expand(
        [
            (data_pb2.SAMPLE_WINDOW,),
            (data_pb2.SAMPLE_ROLLING,),
        ],
        name_func=parameterized_name_func,
    )
    def test_pass_end_releases_the_pool_chunks(self, mode):
        coord = self._coordinator(planned_rows=16, capacity_rows=16, mode=mode)
        _feed_batch(coord, 16)
        _feed_eof(coord)
        self._settle_input(coord)
        sizes, msg = self._drain_batches(coord)
        self.assertEqual(sizes, [8, 8])
        self.assertEqual(msg[0], sample_shuffle._END)
        # every row left the pool as dead chunk buffers; the idle coordinator
        # between passes must not hold on to them
        self.assertEqual(coord._pool.total_rows, 0)
        self.assertEqual(coord._pool._chunks, [])
        self.assertEqual(len(coord._pool._arena_chunk), 0)

    def test_next_get_after_end_starts_full_pass(self):
        coord = self._coordinator(planned_rows=8, full_planned_rows=10)
        _feed_batch(coord, 8)
        _feed_eof(coord)
        self._settle_input(coord)
        sizes, msg = self._drain_batches(coord)
        self.assertEqual(sizes, [8])
        self.assertEqual(msg[0], sample_shuffle._END)
        self.assertTrue(coord._is_complete())
        # the next GET opens the next pass from the full plan
        self._request(coord)
        self._wait_waiting(coord)
        self.assertEqual(coord._remaining, 10)
        self.assertEqual(coord._rows_read, 0)
        self.assertFalse(coord._verified)
        self.assertFalse(any(coord._closed))
        # exactly one task delivery per reader, and no more until asked
        # exactly one task delivery per reader, and no more until asked
        for task_q in coord._task_queues:
            tasks, schema = task_q.get(timeout=1.0)
            self.assertEqual(schema, _PLAN_SCHEMA)
            with self.assertRaises(queue.Empty):
                task_q.get(timeout=0.2)
        _feed_batch(coord, 10, start_id=100)
        _feed_eof(coord)
        self._settle_input(coord)
        # the trigger GET is still outstanding: answer it like a consumer
        # would, instead of issuing a second one for the same pass
        msg = self._next_response(coord)
        self.assertEqual(msg[0], sample_shuffle._BATCH)
        self.assertEqual(len(pa.ipc.read_record_batch(msg[1], _PLAN_SCHEMA)), 8)
        sizes, msg = self._drain_batches(coord)
        self.assertEqual(sizes, [2])
        self.assertEqual(msg[0], sample_shuffle._END)
        for task_q in coord._task_queues:
            with self.assertRaises(queue.Empty):
                task_q.get(timeout=0.2)

    def test_run_stays_alive_and_serves_the_next_pass(self):
        coord = self._coordinator(planned_rows=8)
        _feed_batch(coord, 8)
        _feed_eof(coord)
        kinds = []

        def _consume_pass() -> None:
            coord._request_q.put((sample_shuffle._GET, 0, 0))
            while True:
                kind, _ = coord._response_queues[0].get(timeout=5.0)
                kinds.append(kind)
                if kind in (sample_shuffle._END, sample_shuffle._ERROR):
                    return
                coord._request_q.put((sample_shuffle._GET, 0, 0))

        consumer = threading.Thread(target=_consume_pass, daemon=True)
        consumer.start()
        with mock.patch.object(coord, "_start_readers"):
            runner = threading.Thread(target=coord.run, daemon=True)
            runner.start()
            consumer.join(timeout=5.0)
        # run()'s initial delivery has no reader to consume it; empty the
        # task queues so the next pass's delivery can proceed
        for task_q in coord._task_queues:
            task_q.get(timeout=1.0)
        # END was served; the service stays alive waiting for the next pass
        self.assertEqual(kinds, [sample_shuffle._BATCH, sample_shuffle._END])
        self.assertTrue(runner.is_alive())

        # the queued GET opens the next pass; only then may data arrive
        coord._request_q.put((sample_shuffle._GET, 0, 0))
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if coord._rows_read == 0 and not coord._verified:
                break
            time.sleep(0.005)
        _feed_batch(coord, 8, start_id=100)
        _feed_eof(coord)

        def _consume_open_pass() -> None:
            while True:
                kind, _ = coord._response_queues[0].get(timeout=5.0)
                kinds.append(kind)
                if kind in (sample_shuffle._END, sample_shuffle._ERROR):
                    return
                coord._request_q.put((sample_shuffle._GET, 0, 0))

        consumer = threading.Thread(target=_consume_open_pass, daemon=True)
        consumer.start()
        consumer.join(timeout=5.0)
        self.assertEqual(
            kinds,
            [sample_shuffle._BATCH, sample_shuffle._END] * 2,
        )
        self.assertFalse(consumer.is_alive())
        coord._stop_event.set()
        runner.join(timeout=10.0)
        self.assertFalse(runner.is_alive())

    def test_two_passes_reuse_the_same_processes(self):
        ctx = mp.get_context("spawn")
        request_q = ctx.Queue()
        response_q = ctx.Queue(maxsize=1)
        self.addCleanup(request_q.cancel_join_thread)
        self.addCleanup(response_q.cancel_join_thread)
        plan = _plan(
            8,
            tasks=[ReadTask(fn=_yield_rows, kwargs=dict(n=8, chunk=4), reader_id=0)],
        )
        coord = Coordinator(
            first_plan=plan,
            full_plan=plan,
            request_q=request_q,
            response_queues=[response_q],
            stop_event=ctx.Event(),
            cancel_event=ctx.Event(),
            mode=data_pb2.SAMPLE_ROLLING,
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
            request_q.put((sample_shuffle._GET, 0, 0))
            while True:
                kind, payload = response_q.get(timeout=5.0)
                if kind == sample_shuffle._END:
                    return ids
                rb = pa.ipc.read_record_batch(payload, _PLAN_SCHEMA)
                ids.extend(rb.column("id").to_pylist())
                request_q.put((sample_shuffle._GET, 0, 0))

        first = _drain_pass()
        self.assertEqual(sorted(first), list(range(8)))
        reader_pids = [p.pid for p in coord._readers]
        self.assertTrue(all(p.is_alive() for p in coord._readers))
        second = _drain_pass()
        self.assertEqual(sorted(second), list(range(8)))
        # both passes came from the same, still-alive reader processes
        self.assertEqual([p.pid for p in coord._readers], reader_pids)
        coord._stop_event.set()
        runner.join(timeout=10.0)
        self.assertFalse(runner.is_alive())
        self.assertFalse(any(p.is_alive() for p in coord._readers))

    def test_empty_plan_ends_after_eof(self):
        coord = self._coordinator(planned_rows=0)
        _feed_eof(coord)
        self._settle_input(coord)
        self._request(coord)
        msg = self._next_response(coord)
        self.assertEqual(msg[0], sample_shuffle._END)

    def test_any_eof_fails_infinite_plan(self):
        coord = self._coordinator(planned_rows=None, is_finite=False, num_readers=2)
        _feed_eof(coord, reader_id=0)
        coord._service_input(1.0)
        self.assertIsNotNone(coord._error)
        self.assertIn("non-finite", coord._error)

    def test_finite_waits_for_every_reader_eof(self):
        coord = self._coordinator(planned_rows=10, num_readers=2)
        _feed_batch(coord, 5, reader_id=0)
        _feed_eof(coord, reader_id=0)
        coord._service_input(1.0)
        coord._service_input(1.0)
        self.assertIsNone(coord._error)
        self.assertFalse(coord._verified)
        _feed_batch(coord, 5, start_id=5, reader_id=1)
        _feed_eof(coord, reader_id=1)
        self._settle_input(coord)
        self.assertTrue(coord._verified)

    def test_infinite_plan_never_ends(self):
        coord = self._coordinator(planned_rows=None, is_finite=False)
        _feed_batch(coord, 16)
        coord._service_input(1.0)
        self._request(coord)
        msg = self._next_response(coord)
        self.assertEqual(msg[0], sample_shuffle._BATCH)
        self._request(coord)
        self._wait_waiting(coord)
        coord._serve_waiting()
        # the second request waits for the pool to refill
        self.assertIn(0, coord._request_order)
        self.assertTrue(coord._response_queues[0].empty())


def _yield_rows(stop_event, n=10, chunk=4):
    """Module-level task yielding simple batches; used across processes."""
    for start in range(0, n, chunk):
        yield pa.RecordBatch.from_arrays(
            [
                pa.array(np.arange(start, min(start + chunk, n), dtype=np.int64)),
                pa.array(["topic:0"] * min(chunk, n - start)),
                pa.array([0] * min(chunk, n - start)),
                pa.array([-1.0] * min(chunk, n - start)),
            ],
            schema=_PLAN_SCHEMA,
        )


def _yield_then_raise(stop_event):
    yield next(iter(_yield_rows(stop_event, n=2, chunk=2)))
    raise ValueError("task boom")


def _yield_reordered(stop_event):
    """Yield with columns in another order to exercise conforming."""
    yield pa.RecordBatch.from_arrays(
        [
            pa.array([0.0]),
            pa.array([0]),
            pa.array(["topic:0"]),
            pa.array([7]),
        ],
        names=[DATA_TIMESTAMP, CKPT_ROW_IDX, CKPT_SOURCE_ID, "id"],
    )


class ReaderProcessTest(unittest.TestCase):
    def _start_reader(self, stop_set=False):
        # reader processes are forked by the coordinator, so tests fork too
        ctx = mp.get_context("fork")
        task_q = ctx.Queue(maxsize=1)
        input_q = ctx.Queue()
        self.addCleanup(task_q.cancel_join_thread)
        self.addCleanup(input_q.cancel_join_thread)
        stop_event = ctx.Event()
        if stop_set:
            stop_event.set()
        cancel_event = ctx.Event()
        reader_stop = ctx.Event()
        p = ctx.Process(
            target=_reader_process_main,
            args=(
                0,
                task_q,
                input_q,
                stop_event,
                cancel_event,
                reader_stop,
                os.getpid(),
            ),
        )
        p.start()

        def _stop_reader():
            stop_event.set()
            p.join(timeout=10.0)

        self.addCleanup(_stop_reader)
        return p, task_q, input_q, stop_event

    def _collect_until_eof(self, input_q, timeout=10.0):
        msgs = []
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                msg = input_q.get(timeout=0.5)
            except queue.Empty:
                break
            msgs.append(msg)
            if msg[0] in (sample_shuffle._EOF, sample_shuffle._ERROR):
                break
        return msgs

    def test_deliveries_stream_data_and_wait_for_more(self):
        p, task_q, input_q, _ = self._start_reader()
        for _ in range(2):
            task_q.put(([ReadTask(fn=_yield_rows, reader_id=0)], _PLAN_SCHEMA))
            msgs = self._collect_until_eof(input_q)
            kinds = [m[0] for m in msgs]
            self.assertEqual(kinds, [sample_shuffle._DATA] * 3 + [sample_shuffle._EOF])
            rows = sum(
                pa.ipc.read_record_batch(m[2], _PLAN_SCHEMA).num_rows
                for m in msgs
                if m[0] == sample_shuffle._DATA
            )
            self.assertEqual(rows, 10)
            # the reader waits for its next delivery instead of exiting
            self.assertTrue(p.is_alive())

    def test_exception_flushes_error_and_exits(self):
        p, task_q, input_q, _ = self._start_reader()
        task_q.put(([ReadTask(fn=_yield_then_raise, reader_id=0)], _PLAN_SCHEMA))
        msgs = self._collect_until_eof(input_q)
        self.assertEqual(msgs[0][0], sample_shuffle._DATA)
        self.assertEqual(msgs[-1][0], sample_shuffle._ERROR)
        self.assertIn("task boom", msgs[-1][2])
        p.join(timeout=10.0)
        self.assertFalse(p.is_alive())

    def test_stop_drops_data_and_eof(self):
        p, task_q, input_q, _ = self._start_reader(stop_set=True)
        task_q.put(([ReadTask(fn=_yield_rows, reader_id=0)], _PLAN_SCHEMA))
        p.join(timeout=10.0)
        self.assertFalse(p.is_alive())
        msgs = self._collect_until_eof(input_q)
        self.assertEqual(msgs, [])

    def test_batch_columns_are_conformed(self):
        p, task_q, input_q, _ = self._start_reader()
        task_q.put(([ReadTask(fn=_yield_reordered, reader_id=0)], _PLAN_SCHEMA))
        msgs = self._collect_until_eof(input_q)
        self.assertEqual(msgs[0][0], sample_shuffle._DATA)
        rb = pa.ipc.read_record_batch(msgs[0][2], _PLAN_SCHEMA)
        self.assertEqual(rb.schema.names, _PLAN_SCHEMA.names)
        self.assertEqual(rb.column("id").to_pylist(), [7])


class HandleBatchesTest(unittest.TestCase):
    """The handle's merged protocol client, exercised without a coordinator."""

    def _handle(self):
        reader = _FakeReader()
        cfg = data_pb2.DataConfig(
            batch_size=8,
            shuffle_buffer_size=2,
            shuffle=True,
            shuffle_mode=data_pb2.ShuffleMode.SAMPLE_ROLLING,
            num_workers=1,
        )
        handle = SampleShuffleHandle(reader, cfg, 1, 0, 1)
        self.addCleanup(handle.stop)
        return handle

    def _payload(self, n=2):
        rb = pa.RecordBatch.from_arrays(
            [
                pa.array(np.arange(n, dtype=np.int64)),
                pa.array(["topic:0"] * n),
                pa.array([0] * n),
                pa.array([-1.0] * n),
            ],
            schema=_PLAN_SCHEMA,
        )
        return rb.serialize().to_pybytes()

    def test_batch_then_end(self):
        handle = self._handle()
        handle._response_queues[0].put((sample_shuffle._BATCH, self._payload()))
        it = handle.batches(0)
        data = next(it)
        self.assertEqual(data["id"].to_pylist(), [0, 1])
        handle._response_queues[0].put((sample_shuffle._END, None))
        with self.assertRaises(StopIteration):
            next(it)

    def test_fail_raises(self):
        handle = self._handle()
        handle._response_queues[0].put((sample_shuffle._ERROR, "boom"))
        it = handle.batches(0)
        with self.assertRaisesRegex(RuntimeError, "boom"):
            next(it)

    def test_stop_event_ends(self):
        handle = self._handle()
        handle._stop_event.set()
        with self.assertRaises(StopIteration):
            next(handle.batches(0))

    def test_abandoned_pass_sends_cancel(self):
        handle = self._handle()
        handle._response_queues[0].put((sample_shuffle._BATCH, self._payload()))
        it = handle.batches(0)
        next(it)
        it.close()
        # the GET of the pass, then the CANCEL
        get_msg = handle._request_q.get(timeout=1.0)
        self.assertEqual(get_msg, (sample_shuffle._GET, 0, 0))
        cancel = handle._request_q.get(timeout=1.0)
        self.assertEqual(cancel, (sample_shuffle._CANCEL, 0, 0))


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


class HandleLifecycleTest(unittest.TestCase):
    def _handle(self, reader=None):
        if reader is None:
            reader = _FakeReader()
        cfg = data_pb2.DataConfig(
            batch_size=8,
            shuffle_buffer_size=2,
            shuffle=True,
            shuffle_mode=data_pb2.ShuffleMode.SAMPLE_ROLLING,
            num_workers=2,
        )
        handle = SampleShuffleHandle(reader, cfg, 2, 0, 1)
        self.addCleanup(handle.stop)
        return reader, handle

    def test_without_resume_state_the_first_plan_is_reused(self):
        reader, handle = self._handle()
        self.assertIs(handle._full_plan, handle._first_plan)
        self.assertIs(handle._first_plan, reader.plan)
        self.assertEqual(reader.states, [])

    def test_resume_position_builds_a_full_plan_for_later_passes(self):
        reader = _FakeReader()
        reader.load_state_dict({"prefix:0": 3})
        reader.plans = [_plan(3), _plan(8)]
        _, handle = self._handle(reader)
        self.assertEqual(handle._first_plan.planned_rows, 3)
        self.assertEqual(handle._full_plan.planned_rows, 8)
        # the reader's internal resume position is dropped after planning
        self.assertEqual(reader.states, [{"prefix:0": 3}, None])

    def test_infinite_source_keeps_its_first_plan(self):
        reader = _FakeReader()
        reader.load_state_dict({"topic:0": 5})
        reader.plans = [_plan(None, is_finite=False)]
        _, handle = self._handle(reader)
        self.assertIs(handle._full_plan, handle._first_plan)
        # kafka keeps consuming from the first pass's fixed start offsets
        self.assertEqual(reader.states, [{"topic:0": 5}])

    def test_start_spawns_one_coordinator(self):
        reader, handle = self._handle()
        procs = []

        def _process(**kwargs):
            proc = mock.Mock()
            proc.is_alive.return_value = False
            procs.append((kwargs, proc))
            return proc

        with (
            mock.patch.object(handle._ctx, "Process", side_effect=_process),
            mock.patch.object(handle, "_start_watchdog"),
        ):
            handle.start()
            self.assertEqual(len(procs), 1)
            self.assertIs(procs[0][0]["kwargs"]["first_plan"], handle._first_plan)
            self.assertIs(procs[0][0]["kwargs"]["full_plan"], handle._full_plan)
            procs[0][1].start.assert_called_once()
            handle.start()
            self.assertEqual(len(procs), 1)

    def test_watchdog_ignores_a_stopped_exit(self):
        reader, handle = self._handle()
        proc = mock.Mock()
        handle._proc = proc
        handle._stop_event.set()
        handle._start_watchdog()
        time.sleep(0.05)
        self.assertFalse(handle._fatal_event.is_set())

    def test_stop_is_idempotent_and_blocks_restart(self):
        reader, handle = self._handle()
        with (
            mock.patch.object(handle._ctx, "Process", return_value=mock.Mock()),
            mock.patch.object(handle, "_start_watchdog"),
        ):
            handle.start()
        handle.stop()
        handle.stop()
        handle.start()

    def test_validate_config_runs_on_creation(self):
        reader = _FakeReader()
        cfg = data_pb2.DataConfig(
            batch_size=0,
            shuffle_buffer_size=2,
            shuffle=True,
            shuffle_mode=data_pb2.ShuffleMode.SAMPLE_ROLLING,
        )
        with self.assertRaises(ValueError):
            SampleShuffleHandle(reader, cfg, 1, 0, 1)


if __name__ == "__main__":
    unittest.main()
