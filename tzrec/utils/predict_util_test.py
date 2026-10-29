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

import io
import json
import os
import queue
import shutil
import threading
import time
import unittest
import zlib
from typing import Dict, Tuple
from unittest import mock

import numpy as np
import torch
from parameterized import param, parameterized
from torch import nn
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor, KeyedTensor

from tzrec.datasets.utils import Batch
from tzrec.models.model import CombinedModelWrapper
from tzrec.utils import npz_util, predict_util, quant_util
from tzrec.utils.logging_util import logger
from tzrec.utils.predict_util import (
    DistributedEmbeddingLookup,
    DistributedEmbeddingModel,
    _encode_keys,
    _gpu_cache_ratio,
    _key_tag,
    _wrap_batch_predict_model,
    load_predict_model,
)
from tzrec.utils.test_util import (
    make_test_dir,
    mark_ci_scope,
    nv_gpu_unavailable,
    parameterized_name_func,
)


class _RecordingEvent:
    def __init__(self, events):
        self._events = events
        self._set = False

    def set(self):
        self._events.append("cancel")
        self._set = True

    def is_set(self):
        return self._set


class _StubbornProcess:
    def __init__(self, events):
        self.pid = 123
        self.exitcode = None
        self._events = events
        self._alive = True

    def is_alive(self):
        return self._alive

    def terminate(self):
        self._events.append("terminate")

    def kill(self):
        self._events.append("kill")
        self._alive = False
        self.exitcode = -9

    def join(self, timeout=None):
        self._events.append(("join", timeout))


class PredictUtilTest(unittest.TestCase):
    def test_blocked_queue_operations_are_cancelled(self):
        with mock.patch.object(predict_util, "_PREDICT_PIPELINE_POLL_INTERVAL", 0.01):
            cancel_event = threading.Event()
            timer = threading.Timer(0.02, cancel_event.set)
            timer.start()
            with self.assertRaises(predict_util.PredictPipelineCancelled):
                predict_util.queue_get_interruptibly(
                    queue.Queue(), cancel_event, "blocked get", timeout=1
                )
            timer.join(timeout=1)

            full_queue = queue.Queue(maxsize=1)
            full_queue.put("full")
            cancel_event = threading.Event()
            timer = threading.Timer(0.02, cancel_event.set)
            timer.start()
            with self.assertRaises(predict_util.PredictPipelineCancelled):
                predict_util.queue_put_interruptibly(
                    full_queue,
                    "blocked",
                    cancel_event,
                    "blocked put",
                    timeout=1,
                )
            timer.join(timeout=1)

    def test_empty_queue_uses_one_total_timeout(self):
        started = time.monotonic()
        with mock.patch.object(predict_util, "_PREDICT_PIPELINE_POLL_INTERVAL", 0.01):
            with self.assertRaisesRegex(TimeoutError, "input queue.*queue input.*0.03"):
                predict_util.queue_get_interruptibly(
                    queue.Queue(),
                    threading.Event(),
                    "input queue",
                    timeout=0.03,
                )
        elapsed = time.monotonic() - started
        self.assertGreaterEqual(elapsed, 0.025)
        self.assertLess(elapsed, 0.2)

    def test_full_queue_uses_one_total_timeout(self):
        full_queue = queue.Queue(maxsize=1)
        full_queue.put("full")
        started = time.monotonic()
        with mock.patch.object(predict_util, "_PREDICT_PIPELINE_POLL_INTERVAL", 0.01):
            with self.assertRaisesRegex(
                TimeoutError, "output queue.*queue capacity.*0.03"
            ):
                predict_util.queue_put_interruptibly(
                    full_queue,
                    "blocked",
                    threading.Event(),
                    "output queue",
                    timeout=0.03,
                )
        elapsed = time.monotonic() - started
        self.assertGreaterEqual(elapsed, 0.025)
        self.assertLess(elapsed, 0.2)

    def _drain_thread(self, data_queue, cancel_event, item_count, item_seconds):
        """Consume item_count items, spending item_seconds on each."""

        def drain():
            for _ in range(item_count):
                predict_util.queue_get_interruptibly(
                    data_queue, cancel_event, "writer input", timeout=1
                )
                time.sleep(item_seconds)

        thread = threading.Thread(target=drain, name="slow-writer", daemon=True)
        thread.start()
        return thread

    def test_slow_drain_is_not_a_stall(self):
        # total drain far exceeds the timeout, but no single gap approaches it.
        data_queue = queue.Queue()
        cancel_event = threading.Event()
        for value in range(10):
            data_queue.put(value)
        thread = self._drain_thread(data_queue, cancel_event, 10, 0.05)

        started = time.monotonic()
        with mock.patch.object(predict_util, "_PREDICT_PIPELINE_POLL_INTERVAL", 0.01):
            predict_util.wait_for_pipeline(
                [], [thread], queue.Queue(), cancel_event, stall_timeout=0.2
            )
        elapsed = time.monotonic() - started

        self.assertGreater(elapsed, 0.4)
        self.assertFalse(thread.is_alive())
        self.assertTrue(data_queue.empty())

    def test_saturated_drain_is_not_a_stall(self):
        # a queue pinned at its maxsize still flows; depth alone cannot see it.
        data_queue = queue.Queue(maxsize=1)
        cancel_event = threading.Event()
        data_queue.put("first")
        thread = self._drain_thread(data_queue, cancel_event, 10, 0.05)

        def refill():
            for _ in range(9):
                predict_util.queue_put_interruptibly(
                    data_queue, "item", cancel_event, "producer", timeout=1
                )

        producer = threading.Thread(target=refill, name="producer", daemon=True)
        producer.start()

        with mock.patch.object(predict_util, "_PREDICT_PIPELINE_POLL_INTERVAL", 0.01):
            predict_util.wait_for_pipeline(
                [], [thread, producer], queue.Queue(), cancel_event, stall_timeout=0.2
            )

        self.assertFalse(thread.is_alive())
        self.assertFalse(producer.is_alive())

    def test_stalled_pipeline_times_out(self):
        cancel_event = threading.Event()
        idle = threading.Thread(
            target=cancel_event.wait, name="wedged-writer", daemon=True
        )
        idle.start()

        started = time.monotonic()
        with mock.patch.object(predict_util, "_PREDICT_PIPELINE_POLL_INTERVAL", 0.01):
            with self.assertRaisesRegex(TimeoutError, "no progress.*wedged-writer"):
                predict_util.wait_for_pipeline(
                    [], [idle], queue.Queue(), cancel_event, stall_timeout=0.2
                )
        elapsed = time.monotonic() - started
        cancel_event.set()

        self.assertGreaterEqual(elapsed, 0.2)
        self.assertLess(elapsed, 1)

    @parameterized.expand(
        [["forward", 8], ["writer", None]],
        name_func=parameterized_name_func,
    )
    def test_background_failure_preserves_context(self, stage, worker_id):
        failure_queue = queue.Queue()
        cancel_event = threading.Event()
        try:
            raise ValueError(f"{stage} boom")
        except ValueError as error:
            predict_util.report_failure(
                failure_queue, cancel_event, stage, worker_id, error
            )

        self.assertTrue(cancel_event.is_set())
        with self.assertRaises(predict_util.PredictPipelineStageError) as raised:
            predict_util.check_pipeline_health([], failure_queue, cancel_event)
        failure = raised.exception.failure
        self.assertEqual(failure.stage, stage)
        self.assertEqual(failure.worker_id, worker_id)
        self.assertEqual(failure.exception_type, "ValueError")
        self.assertEqual(failure.message, f"{stage} boom")
        self.assertIn('raise ValueError(f"{stage} boom")', failure.traceback)
        self.assertIn(f"ValueError: {stage} boom", failure.traceback)

    def test_cancellation_resolves_to_the_background_failure(self):
        failure_queue = queue.Queue()
        cancelled = predict_util.PredictPipelineCancelled("input producer cancelled")

        with mock.patch.object(predict_util, "_PREDICT_PIPELINE_POLL_INTERVAL", 0.01):
            # cancellation without a report stays as-is: it names the stalled stage.
            self.assertIs(
                predict_util.resolve_pipeline_error(cancelled, failure_queue), cancelled
            )

            try:
                raise ValueError("forward boom")
            except ValueError as error:
                predict_util.report_failure(
                    failure_queue, threading.Event(), "forward", 0, error
                )
            resolved = predict_util.resolve_pipeline_error(cancelled, failure_queue)
            self.assertIsInstance(resolved, predict_util.PredictPipelineStageError)
            self.assertEqual(resolved.failure.stage, "forward")
            self.assertEqual(resolved.failure.message, "forward boom")

            # a real pipeline error is never replaced.
            timeout = TimeoutError("writer stalled")
            self.assertIs(
                predict_util.resolve_pipeline_error(timeout, failure_queue), timeout
            )

    def test_failed_child_process_is_detected(self):
        process = _StubbornProcess([])
        process._alive = False
        process.exitcode = -9
        cancel_event = threading.Event()
        with mock.patch.object(predict_util, "_PREDICT_PIPELINE_POLL_INTERVAL", 0.01):
            with self.assertRaisesRegex(RuntimeError, "exitcode=-9"):
                predict_util.check_pipeline_health(
                    [process], queue.Queue(), cancel_event
                )
        self.assertTrue(cancel_event.is_set())

    @parameterized.expand([[1], [4]], name_func=parameterized_name_func)
    def test_multi_worker_completion_drains_before_commit(self, worker_count):
        input_queue = queue.Queue()
        output_queue = queue.Queue(maxsize=2)
        failure_queue = queue.Queue()
        cancel_event = threading.Event()
        submitted = list(range(6))
        written = []

        for value in submitted:
            input_queue.put(value)
        for _ in range(worker_count):
            input_queue.put(None)

        def forward(worker_id):
            try:
                while True:
                    value = predict_util.queue_get_interruptibly(
                        input_queue, cancel_event, f"forward[{worker_id}] input"
                    )
                    if value is None:
                        return
                    predict_util.queue_put_interruptibly(
                        output_queue,
                        value,
                        cancel_event,
                        f"forward[{worker_id}] output",
                    )
            except predict_util.PredictPipelineCancelled:
                return
            except BaseException as error:
                predict_util.report_failure(
                    failure_queue, cancel_event, "forward", worker_id, error
                )

        def write():
            try:
                while True:
                    value = predict_util.queue_get_interruptibly(
                        output_queue, cancel_event, "writer input"
                    )
                    if value is None:
                        return
                    written.append(value)
            except predict_util.PredictPipelineCancelled:
                return
            except BaseException as error:
                predict_util.report_failure(
                    failure_queue, cancel_event, "writer", None, error
                )

        forward_threads = [
            threading.Thread(
                target=forward,
                args=(worker_id,),
                name=f"forward-{worker_id}",
                daemon=True,
            )
            for worker_id in range(worker_count)
        ]
        writer_thread = threading.Thread(target=write, name="writer", daemon=True)
        threads = [*forward_threads, writer_thread]
        for thread in threads:
            thread.start()
        with mock.patch.object(predict_util, "_PREDICT_PIPELINE_POLL_INTERVAL", 0.01):
            predict_util.wait_for_pipeline(
                [], forward_threads, failure_queue, cancel_event, stall_timeout=1
            )
            predict_util.queue_put_interruptibly(
                output_queue,
                None,
                cancel_event,
                "writer completion",
                timeout=1,
            )
            predict_util.wait_for_pipeline(
                [], [writer_thread], failure_queue, cancel_event, stall_timeout=1
            )

        writer = mock.Mock()
        predict_util.commit_prediction_output(
            writer, None, len(submitted), torch.device("cpu")
        )
        self.assertEqual(sorted(written), submitted)
        self.assertTrue(all(not thread.is_alive() for thread in threads))
        writer.close.assert_called_once_with()

    def test_local_failure_reraises_original_error(self):
        writer = mock.Mock()
        error = ValueError("forward boom")
        with self.assertRaises(ValueError) as raised:
            predict_util.commit_prediction_output(writer, error, 2, torch.device("cpu"))
        self.assertIs(raised.exception, error)
        writer.close.assert_not_called()

    def test_empty_input_never_commits(self):
        writer = mock.Mock()
        with self.assertRaisesRegex(RuntimeError, "empty"):
            predict_util.commit_prediction_output(writer, None, 0, torch.device("cpu"))
        writer.close.assert_not_called()

    @parameterized.expand(
        [[0, False], [1, True]],
        name_func=parameterized_name_func,
    )
    def test_commit_follows_the_other_rank(self, peer_succeeded, expect_commit):
        writer = mock.Mock()
        with mock.patch.object(predict_util, "dist") as dist:
            dist.is_initialized.return_value = True
            dist.get_world_size.return_value = 2
            dist.all_reduce.side_effect = lambda outcome, op: outcome.add_(
                torch.tensor([peer_succeeded, 3], dtype=torch.int64)
            )
            if expect_commit:
                # an empty local shard is normal when input files are uneven.
                predict_util.commit_prediction_output(
                    writer, None, 0, torch.device("cpu")
                )
                writer.close.assert_called_once_with()
            else:
                with self.assertRaisesRegex(RuntimeError, "another rank"):
                    predict_util.commit_prediction_output(
                        writer, None, 0, torch.device("cpu")
                    )
                writer.close.assert_not_called()

    def test_cleanup_cancels_terminates_kills_and_reaps(self):
        events = []
        cancel_event = _RecordingEvent(events)
        process = _StubbornProcess(events)
        thread = mock.Mock()
        thread.name = "stuck-writer"
        thread.is_alive.return_value = True
        pipeline_queue = mock.Mock()
        pipeline_queue.close.side_effect = RuntimeError("close boom")

        with (
            mock.patch.object(predict_util, "_PREDICT_PIPELINE_CLEANUP_TIMEOUT", 0),
            mock.patch.object(predict_util, "_PREDICT_PIPELINE_TERMINATE_TIMEOUT", 0),
            mock.patch.object(predict_util, "_PREDICT_PIPELINE_KILL_TIMEOUT", 0),
            mock.patch.object(predict_util.logger, "exception") as log_exception,
        ):
            predict_util.cleanup_pipeline(
                [process], [thread], [pipeline_queue], cancel_event
            )

        self.assertEqual(events[:3], ["cancel", "terminate", "kill"])
        self.assertIn(("join", 0), events)
        thread.join.assert_called_once_with(timeout=0)
        pipeline_queue.cancel_join_thread.assert_called_once_with()
        pipeline_queue.close.assert_called_once_with()
        log_exception.assert_called_once()


_QUANT = quant_util.DISTRIBUTED_SPARSE_SUPPORTED_QUANT_FORMATS[0]
_MISS_KEY = 987654321


class _Table:
    """One exported table and the float rows a lookup should return."""

    def __init__(self, name, num_rows, dim, quant, dynamic, default=False, keys=None):
        rng = np.random.default_rng(zlib.crc32(name.encode()))
        self.name = name
        self.dim = dim
        values = rng.standard_normal((num_rows, dim)).astype(np.float32)
        if keys is not None:
            self.keys = np.asarray(keys, dtype=np.int64)
        elif dynamic:
            # FG hashes dynamic ids into the full 63-bit range.
            self.keys = rng.integers(0, (1 << 63) - 1, num_rows, dtype=np.int64)
        else:
            self.keys = np.arange(num_rows, dtype=np.int64)
        self.meta = {"dense": False, "is_dynamic": dynamic, "dimension": dim}
        self.meta["shape"] = [num_rows, dim]
        if quant:
            self.stored = quant_util.distributed_quantize_embeddings(
                values, dim, name, _QUANT
            )
            # lookups dequantize to fp16 and are cast to fp32
            self.rows = (
                quant_util.dequantize_quint8_rowwise_f16(self.stored, dim)
                .astype(np.float16)
                .astype(np.float32)
            )
            self.meta.update(dtype=_QUANT, row_bytes=dim + 4)
        else:
            self.stored = values
            self.rows = values
            self.meta["dtype"] = "float32"
        self.miss = np.zeros(dim, dtype=np.float32)
        self.stored_default = None
        if default:
            self.stored_default = self.stored[:1] + 1 if not quant else self.stored[:1]
            self.miss = (
                self.rows[0] if quant else self.stored_default[0].astype(np.float32)
            )
        if dynamic:
            self.meta.update(
                key_dtype="int64",
                score_dtype="int64",
                key_name=f"{name}.keys",
                value_name=f"{name}.values",
                score_name=f"{name}.scores",
            )
            if default:
                self.meta["default_value_name"] = f"{name}.default_value"

    def lookup(self, key):
        """Expected embedding of one key."""
        idx = np.flatnonzero(self.keys == key)
        return self.rows[idx[0]] if len(idx) else self.miss


class _DenseSum(torch.nn.Module):
    def forward(
        self, data: Dict[str, torch.Tensor], device: torch.device
    ) -> Dict[str, torch.Tensor]:
        p = data["deep"].sum(1) + data["x.values"].sum(1) + data["sd.lengths"]
        return {"p": p * data["batch_size"]}


def _write_export(export_dir, tables, features, dense_meta):
    sparse_dir = os.path.join(export_dir, "sparse")
    os.makedirs(sparse_dir)
    static, dynamic = {}, {}
    for t in tables:
        if t.meta["is_dynamic"]:
            dynamic[f"{t.name}.keys"] = t.keys
            dynamic[f"{t.name}.values"] = t.stored
            dynamic[f"{t.name}.scores"] = np.zeros_like(t.keys)
            if t.stored_default is not None:
                dynamic[f"{t.name}.default_value"] = t.stored_default
        else:
            static[t.name] = t.stored
    npz_util.savez_streaming(
        os.path.join(sparse_dir, "sparse_embeddings-00-of-01.npz"), static
    )
    npz_util.savez_streaming(
        os.path.join(sparse_dir, "sparse_dynamic_embedding-00-of-01.npz"), dynamic
    )
    with open(os.path.join(sparse_dir, "sparse_embedding.json"), "w") as f:
        json.dump({t.name: t.meta for t in tables}, f)
    with open(os.path.join(sparse_dir, "sparse_features.json"), "w") as f:
        json.dump(features, f)
    with open(os.path.join(export_dir, "dense_meta.json"), "w") as f:
        json.dump(dense_meta, f)


def _pooled(table, rows, mean, weights=None):
    res = []
    for i, r in enumerate(rows):
        w = weights[i] if weights else [1.0] * len(r)
        v = np.sum([wi * table.lookup(k) for k, wi in zip(r, w)], 0) if r else 0
        if mean and r:
            # weighted MEAN divides by the weight sum; all-zero weights give zeros.
            v = v / ((sum(w) or 1.0) if weights else len(r))
        res.append(np.broadcast_to(v, (table.dim,)))
    return np.stack(res).astype(np.float32)


class FusedKeyTest(unittest.TestCase):
    def test_encode_golden_keys(self) -> None:
        # encoded keys with 11 member bits, i.e. 52 payload bits.
        keys = torch.tensor([0, 1, 42, (1 << 52) + 5, (1 << 63) - 2, -1, (1 << 63) - 1])
        missing = (1 << 63) - 1
        expected = {
            0: [0x779B97F4A7C15, 0x779B97F4A7C14, 0x779B97F4A7C3F, 0x779B97F4A7C11],
            1: [0x1EF372FE94F82A, 0x1EF372FE94F82B, 0x1EF372FE94F800, 0x1EF372FE94F82E],
            2046: [
                0x7FE65240D4962BEB,
                0x7FE65240D4962BEA,
                0x7FE65240D4962BC1,
                0x7FE65240D4962BEF,
            ],
        }
        last = {0: 0x8864680B58414, 1: 0x110C8D016B002B, 2046: 0x7FE9ADBF2B69D3EA}
        for member, values in expected.items():
            tags = torch.full_like(keys, _key_tag(member, 52))
            self.assertEqual(
                _encode_keys(keys, tags, 52).tolist(),
                values + [last[member], missing, missing],
            )

    def test_member_keys_never_hit_miss_key(self) -> None:
        for n in (1, 3, 4, 127, 2047):
            payload_bits = 63 - n.bit_length()
            largest = _key_tag(n - 1, payload_bits) | ((1 << payload_bits) - 1)
            self.assertLess(largest, (1 << 63) - 1, n)


_GiB = 1 << 30
# a layer's overhead plus its minimum cache, as the auto budget counts them
_LAYER = (96 << 20) + (1 << 20)


def _mem_info(free_total):
    """Mocked torch.cuda.mem_get_info returning (free, total) per device index."""
    return mock.patch(
        "torch.cuda.mem_get_info", side_effect=lambda d: free_total[d.index]
    )


class GpuCacheRatioTest(unittest.TestCase):
    @parameterized.expand(
        [
            param("default", env={}, mem=[(60 * _GiB, 72 * _GiB)], expected=1.0),
            param("auto", env={"DIST_EMB_GPU_CACHE_RATIO": "auto"},
                  mem=[(60 * _GiB, 72 * _GiB)], expected=1.0),
            # 10 GiB GPU: 2 GiB reserve and 2 layers leave 2 GiB for 8 GiB.
            param("constrained", env={},
                  mem=[(4 * _GiB + 2 * _LAYER, 10 * _GiB)], expected=0.25),
            # the GPU with less free memory decides.
            param("two_gpus", env={},
                  mem=[(9 * _GiB, 10 * _GiB), (4 * _GiB + 2 * _LAYER, 10 * _GiB)],
                  expected=0.25),
            param("rounds_down", env={},
                  mem=[(4 * _GiB + 2 * _LAYER + _GiB // 3, 10 * _GiB)],
                  expected=0.2916),
            param("no_room", env={}, mem=[(_GiB, 10 * _GiB)], expected=0.0),
        ],
        name_func=parameterized_name_func,
    )  # fmt: skip
    def test_auto(self, _name, env, mem, expected) -> None:
        devices = [torch.device("cuda", i) for i in range(len(mem))]
        with (
            mock.patch.dict(os.environ, env, clear=True),
            _mem_info(mem),
            self.assertLogs(logger, "INFO") as logs,
        ):
            ratio = _gpu_cache_ratio(devices, [4 * _GiB, 4 * _GiB])
        self.assertEqual(ratio, expected)
        self.assertTrue(
            any(
                f"DIST_EMB_GPU_CACHE_RATIO=auto resolved to {expected} " in line
                for line in logs.output
            ),
            logs.output,
        )
        self.assertEqual(
            any("no GPU memory left" in line for line in logs.output), expected == 0
        )

    def test_numeric_override(self) -> None:
        with (
            mock.patch.dict(os.environ, {"DIST_EMB_GPU_CACHE_RATIO": "0.35"}),
            mock.patch("torch.cuda.mem_get_info") as mem_get_info,
        ):
            ratio = _gpu_cache_ratio([torch.device("cuda", 0)], [4 * _GiB])
        self.assertEqual(ratio, 0.35)
        mem_get_info.assert_not_called()

    @parameterized.expand(
        [param("zero", value="0"), param("above_one", value="1.5"),
         param("text", value="half")],
        name_func=parameterized_name_func,
    )  # fmt: skip
    def test_invalid_value_raises(self, _name, value) -> None:
        with mock.patch.dict(os.environ, {"DIST_EMB_GPU_CACHE_RATIO": value}):
            with self.assertRaises(ValueError):
                _gpu_cache_ratio([torch.device("cuda", 0)], [4 * _GiB])


def _lookup_batch(bags, weights, kjts=None):
    """Batch of ``bags``; ``kjts`` splits the features into KJTs."""
    sparse = {}
    for g, names in enumerate(kjts or [list(bags)]):
        values, lengths, kjt_weights = [], [], []
        for name in names:
            rows = bags[name]
            values += [k for r in rows for k in r]
            lengths += [len(r) for r in rows]
            w_rows = weights.get(name, [[1.0] * len(r) for r in rows])
            kjt_weights += [w for r in w_rows for w in r]
        sparse[f"g{g}"] = KeyedJaggedTensor(
            keys=names,
            values=torch.tensor(values, dtype=torch.int64),
            lengths=torch.tensor(lengths),
            weights=torch.tensor(kjt_weights)
            if any(name in weights for name in names)
            else None,
        )
    return Batch(sparse_features=sparse)


@mark_ci_scope("gpu")
@unittest.skipIf(*nv_gpu_unavailable)
class DistributedEmbeddingLookupTest(unittest.TestCase):
    def setUp(self) -> None:
        self.export_dir = make_test_dir()

    def tearDown(self) -> None:
        shutil.rmtree(self.export_dir, ignore_errors=True)

    @parameterized.expand(
        [
            param("quant", quant=True),
            param("fp32", quant=False),
        ],
        name_func=parameterized_name_func,
    )
    def test_lookup_matches_reference(self, _name, quant) -> None:
        # dyn and dyn2 fuse (same dtype, dim and default row); zch keeps its own
        # group for its default row; static tables never fuse with dynamic ones.
        dyn = _Table("model.ebc.dyn", 50, 8, quant, dynamic=True)
        dyn2 = _Table("model.ebc.dyn2", 40, 8, quant, dynamic=True)
        # the same raw key in two fused tables must keep both rows.
        dyn2.keys[0] = dyn.keys[5]
        zch = _Table("model.ebc.zch", 30, 8, quant, dynamic=True, default=True)
        static = _Table("model.ebc.static", 20, 8, quant, dynamic=False)
        small = _Table("model.ebc.small", 10, 4, quant, dynamic=False)
        tables = [dyn, dyn2, zch, static, small]

        def meta(table, pooling, use_weight=False):
            return {
                "embedding_name": table.name,
                "pooling": pooling,
                "use_weight": use_weight,
            }

        features = {
            "a__ebc": meta(dyn, "SUM"),
            "b__ebc": meta(zch, "MEAN", use_weight=True),
            "c__ebc": meta(static, "SUM", use_weight=True),
            "d__ebc": meta(dyn2, "MEAN"),
            "e@shared__ebc": meta(dyn, "SUM"),
            "f__ebc": meta(small, "SUM"),
            "g__ebc": meta(dyn2, "MEAN", use_weight=True),
            "seq__ec": meta(dyn, "NONE"),
            "seq2__ec": meta(dyn2, "NONE"),
            # reads input a again, from another table of the same group.
            "a@other__ebc": meta(dyn2, "SUM"),
        }
        deep = [
            "a__ebc",
            "b__ebc",
            "c__ebc",
            "d__ebc",
            "e@shared__ebc",
            "f__ebc",
            "g__ebc",
            "a@other__ebc",
        ]
        dense_meta = {
            "sequence__ec": ["seq__ec", "seq__lengths", "seq2__ec", "seq2__lengths"],
            "deep": deep,
        }
        _write_export(self.export_dir, tables, features, dense_meta)

        bags = {
            "a": ([dyn.keys[0], -1, dyn.keys[3]], [], [_MISS_KEY, dyn.keys[7]]),
            "b": ([zch.keys[1], _MISS_KEY], [zch.keys[2]], []),
            "c": ([0, 19], [5], [3, 3, 3]),
            # a key of dyn looked up in dyn2 must miss.
            "d": ([dyn2.keys[4], dyn.keys[4]], [dyn2.keys[0]], [-1]),
            "e": ([dyn.keys[9]], [dyn.keys[5], dyn.keys[1]], []),
            "f": ([], [], []),
            "g": ([dyn2.keys[2], dyn2.keys[3]], [dyn2.keys[0], dyn2.keys[1]], []),
            "seq": ([dyn.keys[1], _MISS_KEY], [], [dyn.keys[5]]),
            "seq2": ([dyn2.keys[1]], [dyn.keys[1], -1], [dyn2.keys[0]]),
        }
        weights = {
            "a": ([9.0, 9.0, 9.0], [], [9.0, 9.0]),  # use_weight=false: ignored
            "b": ([0.5, 2.0], [3.0], []),
            "c": ([1.0, 0.5], [0.25], [2.0, 1.0, 0.5]),
            "g": ([1.0, 3.0], [0.0, 0.0], []),  # all-zero weights pool to zeros
        }
        # a and e share a lookup across two KJTs; only the first is weighted.
        data = _lookup_batch(
            bags, weights, [["a", "b", "c", "d", "g"], ["e", "f", "seq", "seq2"]]
        )

        device = torch.device("cuda:0")
        lookup = DistributedEmbeddingLookup(self.export_dir, [device])
        # groups: dyn+dyn2, zch, static, small; sequences share their group layer.
        self.assertEqual(len(lookup._layers[device]), 4)
        expected_deep = np.concatenate(
            [
                _pooled(dyn, bags["a"], False),
                _pooled(zch, bags["b"], True, weights["b"]),
                _pooled(static, bags["c"], False, weights["c"]),
                _pooled(dyn2, bags["d"], True),
                _pooled(dyn, bags["e"], False),
                _pooled(small, bags["f"], False),
                _pooled(dyn2, bags["g"], True, weights["g"]),
                _pooled(dyn2, bags["a"], False),
            ],
            axis=1,
        )
        tol = 2e-3 if quant else 1e-5
        for _ in range(2):
            out = lookup(data, device)
            self.assertEqual(out["deep"].device, device)
            self.assertEqual(out["deep"].dtype, torch.float32)
            np.testing.assert_allclose(
                out["deep"].cpu().numpy(), expected_deep, rtol=tol, atol=tol
            )
            for name, table in [("seq", dyn), ("seq2", dyn2)]:
                expected = np.stack([table.lookup(k) for r in bags[name] for k in r])
                np.testing.assert_allclose(
                    out[name].cpu().numpy(), expected, rtol=tol, atol=tol
                )
            self.assertEqual(out["seq__lengths"].dtype, torch.int32)
            self.assertEqual(out["seq__lengths"].tolist(), [2, 0, 1])
            # only the embedding inputs; the dense model adds the batch's others.
            self.assertEqual(
                sorted(out), ["deep", "seq", "seq2", "seq2__lengths", "seq__lengths"]
            )

    @parameterized.expand(
        [
            # 4 members fold to 60 bits; a retry of 2 folds to 61 bits. Groups
            # are (members, member ids, payload bits); None serves raw keys.
            param(
                "first_round",
                n=4,
                collide={},
                groups=[((0, 1, 2, 3), (0, 1, 2, 3), 60)],
            ),
            param(
                "retry_success",
                n=4,
                collide={1: (60,), 2: (60,)},
                groups=[((0, 3), (0, 3), 60), ((1, 2), (0, 1), 61)],
            ),
            param(
                "second_round_failure",
                n=4,
                collide={1: (60, 61), 2: (60, 61)},
                groups=[((0, 3), (0, 3), 60), ((1,), (0,), None), ((2,), (0,), None)],
            ),
            param(
                "mixed",
                n=4,
                collide={1: (60,), 2: (60, 61)},
                groups=[((0, 3), (0, 3), 60), ((1,), (0,), 61), ((2,), (0,), None)],
            ),
            param(
                "singleton",
                n=4,
                collide={1: (60,)},
                groups=[((0, 2, 3), (0, 2, 3), 60), ((1,), (0,), None)],
            ),
            # 3 members fold to 61 bits, and so would a retry of 2: skipped.
            param(
                "unchanged_width",
                n=3,
                collide={0: (61,), 1: (61,)},
                groups=[((2,), (2,), 61), ((0,), (0,), None), ((1,), (0,), None)],
            ),
            # a lone table keeps its raw keys unchecked, though they collide folded.
            param("alone", n=1, collide={0: (62,)}, groups=[((0,), (0,), None)]),
        ],
        name_func=parameterized_name_func,
    )
    def test_fold_collision_retry(self, _name, n, collide, groups):
        # k and k ^ (1 << w) ^ 1, with bit w of k set, fold to the same value at
        # w payload bits but not at w + 1.
        tables = []
        for t in range(n):
            keys = []
            for j, w in enumerate(collide.get(t, ())):
                k = (1 << w) + 1000 * (j + 1)
                keys += [k, k ^ (1 << w) ^ 1]
            keys += [7 + 100 * t + i for i in range(4 - len(keys))]
            tables.append(
                _Table(f"model.ebc.t{t}", 4, 8, False, dynamic=True, keys=keys)
            )
        features = {
            f"f{t}__ebc": {
                "embedding_name": table.name,
                "pooling": "SUM",
                "use_weight": False,
            }
            for t, table in enumerate(tables)
        }
        _write_export(
            self.export_dir,
            tables,
            features,
            {"sequence__ec": [], "deep": list(features)},
        )
        device = torch.device("cuda:0")
        with mock.patch.object(
            DistributedEmbeddingLookup,
            "_has_fold_collision",
            wraps=DistributedEmbeddingLookup._has_fold_collision,
        ) as spy:
            lookup = DistributedEmbeddingLookup(self.export_dir, [device])
        checks = [c.args[0]["key_name"] for c in spy.call_args_list]
        widths = 63 - n.bit_length(), 63 - len(collide).bit_length()
        retried = len(collide) > 1 and widths[1] > widths[0]
        for t, table in enumerate(tables):
            expected_checks = (n > 1) + (retried and t in collide)
            self.assertEqual(checks.count(table.meta["key_name"]), expected_checks)

        def layout(group):
            members = tuple(int(m.rsplit("t", 1)[1]) for m in group.members)
            return members, tuple(group.member_ids), group.payload_bits

        self.assertCountEqual([layout(g) for g in lookup._groups], groups)
        # every key resolves to its own row.
        bags = {f"f{t}": tuple([k] for k in tb.keys) for t, tb in enumerate(tables)}
        out = lookup(_lookup_batch(bags, {}), device)
        expected = np.concatenate(
            [_pooled(tb, bags[f"f{t}"], False) for t, tb in enumerate(tables)], axis=1
        )
        np.testing.assert_array_equal(out["deep"].cpu().numpy(), expected)

    @parameterized.expand(
        [
            param("dynamic", dynamic=True),
            param("static", dynamic=False),
        ],
        name_func=parameterized_name_func,
    )
    def test_row_count_mismatch_raises(self, _name, dynamic) -> None:
        table = _Table("model.ebc.t", 20, 8, True, dynamic=dynamic)
        # sparse_embedding.json declares more rows than the shards hold.
        table.meta["shape"] = [25, 8]
        features = {
            "a__ebc": {
                "embedding_name": table.name,
                "pooling": "SUM",
                "use_weight": False,
            }
        }
        _write_export(
            self.export_dir, [table], features, {"sequence__ec": [], "deep": ["a__ebc"]}
        )
        with self.assertRaisesRegex(ValueError, "20 rows in the npz shards"):
            DistributedEmbeddingLookup(self.export_dir, [torch.device("cuda:0")])

    def test_no_devices_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires CUDA devices"):
            DistributedEmbeddingLookup(self.export_dir, [])

    def test_auto_cache_ratio_is_the_allocated_one(self) -> None:
        # one fp32 dim-8 table of 6.4 MB; the mocked GPU leaves room for half.
        table = _Table("model.ebc.big", 200000, 8, False, dynamic=True)
        features = {
            "a__ebc": {
                "embedding_name": table.name,
                "pooling": "SUM",
                "use_weight": False,
            }
        }
        _write_export(
            self.export_dir, [table], features, {"sequence__ec": [], "deep": ["a__ebc"]}
        )
        storage = 200000 * 8 * 4
        total = 10 * _GiB
        mem = [(int(0.2 * total) + _LAYER + storage // 2, total)]
        device = torch.device("cuda:0")
        with (
            mock.patch.dict(os.environ, {"DIST_EMB_GPU_CACHE_RATIO": "auto"}),
            _mem_info(mem),
            self.assertLogs(logger, "INFO") as logs,
        ):
            lookup = DistributedEmbeddingLookup(self.export_dir, [device])
        text = "\n".join(logs.output)
        self.assertIn("DIST_EMB_GPU_CACHE_RATIO=auto resolved to 0.5 ", text)
        self.assertIn(f"gpu_cache_bytes={storage // 2}.", text)
        out = lookup(_lookup_batch({"a": ([*table.keys[:3]],)}, {}), device)
        np.testing.assert_allclose(
            out["deep"].cpu().numpy()[0], table.rows[:3].sum(0), rtol=1e-5, atol=1e-5
        )

    @unittest.skipIf(torch.cuda.device_count() < 2, "requires 2 GPUs")
    def test_devices_share_tables(self) -> None:
        dyn = _Table("model.ebc.dyn", 50, 8, True, dynamic=True)
        features = {
            "a__ebc": {
                "embedding_name": dyn.name,
                "pooling": "SUM",
                "use_weight": False,
            }
        }
        _write_export(
            self.export_dir, [dyn], features, {"sequence__ec": [], "deep": ["a__ebc"]}
        )
        data = _lookup_batch({"a": ([*dyn.keys[:3]], [dyn.keys[3]])}, {})
        devices = [torch.device("cuda:0"), torch.device("cuda:1")]
        lookup = DistributedEmbeddingLookup(self.export_dir, devices)
        outs = [lookup(data, d)["deep"] for d in devices]
        # one plan per KJT layout, with index tensors on every device.
        self.assertEqual(len(lookup._plans), 1)
        self.assertEqual(outs[1].device, devices[1])
        torch.testing.assert_close(outs[0].cpu(), outs[1].cpu())


@mark_ci_scope("gpu")
@unittest.skipIf(*nv_gpu_unavailable)
class DistributedEmbeddingModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.export_dir = make_test_dir()

    def tearDown(self) -> None:
        shutil.rmtree(self.export_dir, ignore_errors=True)

    def test_runs_dense_on_lookup(self) -> None:
        dyn = _Table("model.ebc.dyn", 10, 8, False, dynamic=True)
        features = {
            "a__ebc": {
                "embedding_name": dyn.name,
                "pooling": "SUM",
                "use_weight": False,
            }
        }
        _write_export(
            self.export_dir, [dyn], features, {"sequence__ec": [], "deep": ["a__ebc"]}
        )
        torch.jit.script(_DenseSum()).save(
            os.path.join(self.export_dir, "scripted_model.pt")
        )
        bags = {"a": ([dyn.keys[0], dyn.keys[1]], [dyn.keys[2]], [])}
        batch = _lookup_batch(bags, {})
        dense = torch.rand(3, 2)
        batch.dense_features = {
            "d": KeyedTensor(keys=["x"], length_per_key=[2], values=dense)
        }
        sd_lengths = torch.tensor([2, 0, 1])
        batch.sequence_dense_features = {
            "sd": JaggedTensor(values=torch.rand(3, 4), lengths=sd_lengths)
        }
        device = torch.device("cuda:0")
        model = DistributedEmbeddingModel(self.export_dir, [device])
        out = model(batch, torch.tensor(2), device)
        np.testing.assert_allclose(
            out["p"].cpu().numpy(),
            (
                _pooled(dyn, bags["a"], False).sum(1)
                + dense.sum(1).numpy()
                + sd_lengths.numpy()
            )
            * 2,
            rtol=1e-5,
            atol=1e-5,
        )


class _Sparse(nn.Module):
    def forward(
        self, data: Dict[str, torch.Tensor], device: torch.device
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        out: Dict[str, torch.Tensor] = {}
        for k, v in data.items():
            out[k] = v.float().sum().reshape(1) * float(len(k))
        return out, out


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sparse = _Sparse()

    def forward(
        self, data: Dict[str, torch.Tensor], device: torch.device
    ) -> Dict[str, torch.Tensor]:
        return self.sparse(data, device)[0]


class _Dense(nn.Module):
    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {"p": torch.cat(list(data.values()))}


def _reload(module: nn.Module) -> torch.jit.ScriptModule:
    buf = io.BytesIO()
    torch.jit.save(torch.jit.script(module), buf)
    buf.seek(0)
    return torch.jit.load(buf)


def _predict_batch() -> Batch:
    # a predict batch: no labels or sample weights.
    torch.manual_seed(0)
    lengths = torch.randint(0, 4, (3 * 5,), dtype=torch.int32)
    n = int(lengths.sum())
    return Batch(
        dense_features={
            "deep": KeyedTensor(
                keys=["x", "y"], length_per_key=[1, 3], values=torch.rand(5, 4)
            )
        },
        sparse_features={
            "deep": KeyedJaggedTensor(
                keys=["a", "b", "c"],
                values=torch.randint(0, 100, (n,), dtype=torch.int32),
                lengths=lengths,
                weights=torch.rand(n),
            )
        },
        additional_infos={"info": torch.rand(5)},
    )


class WrapBatchPredictModelTest(unittest.TestCase):
    def test_scripted_model(self):
        batch, device = _predict_batch(), torch.device("cpu")
        model = _reload(_Model())
        expected = model(batch.to_dict(sparse_dtype=torch.int64), device)
        actual = _wrap_batch_predict_model(model)(batch, None, device)
        self.assertEqual(list(actual), list(expected))
        for k, v in expected.items():
            torch.testing.assert_close(actual[k], v, rtol=0, atol=0, msg=k)

    def test_two_stage_model(self):
        batch, device = _predict_batch(), torch.device("cpu")
        model = CombinedModelWrapper(_reload(_Sparse()), _Dense())
        expected = model(batch.to_dict(sparse_dtype=torch.int64), device)
        actual = _wrap_batch_predict_model(model)(batch, None, device)
        torch.testing.assert_close(actual["p"], expected["p"], rtol=0, atol=0)


class LoadPredictModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model_dir = make_test_dir()

    def tearDown(self) -> None:
        shutil.rmtree(self.model_dir, ignore_errors=True)

    def test_scripted_export(self):
        torch.jit.script(_Model()).save(
            os.path.join(self.model_dir, "scripted_model.pt")
        )
        with open(os.path.join(self.model_dir, "model_acc.json"), "w") as f:
            json.dump({}, f)
        batch, device = _predict_batch(), torch.device("cpu")
        expected = _reload(_Model())(batch.to_dict(sparse_dtype=torch.int64), device)
        actual = load_predict_model(self.model_dir, [device])(batch, None, device)
        self.assertEqual(list(actual), list(expected))
        for k, v in expected.items():
            torch.testing.assert_close(actual[k], v, rtol=0, atol=0, msg=k)


if __name__ == "__main__":
    unittest.main()
