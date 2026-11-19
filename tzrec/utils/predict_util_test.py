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
import io
import json
import os
import queue
import shutil
import threading
import time
import unittest
import zlib
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, Tuple
from unittest import mock

import numpy as np
import torch
from parameterized import param, parameterized
from safetensors.torch import load_file, save_file
from torch import nn
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor, KeyedTensor

from tzrec.acc.aot_utils import export_distributed_dense_model_aot
from tzrec.datasets.utils import Batch
from tzrec.models.genrec_causal_lm_model import generate_codes, resolve_beam_config
from tzrec.models.genrec_model import SLOT_EMBEDS, GenRecFrontEnd
from tzrec.models.model import CombinedModelWrapper, ScriptWrapper
from tzrec.prompt.assembler import (
    CU_SEQLENS,
    HOLE_POSITIONS,
    HOLE_SLOT_COUNTS,
    INPUT_IDS,
)
from tzrec.prompt.compile import sid_bands
from tzrec.prompt.hole_keys import HOLE_KEYS
from tzrec.protos import feature_pb2
from tzrec.protos.pipeline_pb2 import EasyRecConfig
from tzrec.protos.prompt_pb2 import PromptSlot
from tzrec.utils import config_util, npz_util, predict_util, quant_util
from tzrec.utils.export_util import (
    _run_dense_graph_sanity_check,
    build_dense_graph_module,
    init_parameters,
)
from tzrec.utils.hf_export_util import SERVING_MODEL_TYPE
from tzrec.utils.logging_util import logger
from tzrec.utils.predict_util import (
    DistributedEmbeddingLookup,
    DistributedEmbeddingModel,
    GenRecExportModel,
    _encode_keys,
    _gpu_cache_ratio,
    _key_tag,
    _load_scripted_model,
    _wrap_batch_predict_model,
    load_predict_model,
)
from tzrec.utils.test_util import (
    GENREC_LONG_HIST_CODES,
    ci_resources,
    create_genrec_test_model,
    create_tiny_causal_lm,
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


class _DenseAOTIRich(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(16, 4)

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        del device
        deep_out = self.proj(data["deep"]).sum(1)
        seq_sum = data["seq"].sum() + data["sd.values"].sum()
        mv_kl_sum = data["mv.key_lengths"].to(torch.float32).sum()
        row_terms = (
            data["x.values"].sum(1)
            + data["seq__lengths"].to(torch.float32)
            + data["sd.lengths"].to(torch.float32)
            + data["mv.lengths"].to(torch.float32)
        )
        p = (deep_out + row_terms + (seq_sum + mv_kl_sum) * 0.01) * data[
            "batch_size"
        ].to(torch.float32)
        return {"p": p}


class _DenseAOTISingleArg(torch.nn.Module):
    def __init__(self, inner: torch.nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.inner(data)


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
        first = [["a", "b", "c", "d", "g"], ["e", "f", "seq", "seq2"]]
        # another layout gets its own plan; unweighted d and weighted g now share
        # a lookup from different KJTs, the unweighted one first.
        second = [["e", "f", "seq", "seq2", "d"], ["g", "c", "b", "a"]]

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
        for kjts in (first, first, second):
            out = lookup(_lookup_batch(bags, weights, kjts), device)
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
        self.assertEqual(len(lookup._plans), 2)

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
    @ci_resources(slots=2)
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
        with open(os.path.join(self.export_dir, "model_acc.json"), "w") as f:
            json.dump({"DISTRIBUTED_EMBEDDING": "1"}, f)
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
        model = load_predict_model(self.export_dir, [device], log_model=True)
        self.assertIsInstance(model, DistributedEmbeddingModel)
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

    def _build_rich_aoti_export(
        self,
        export_dir: str,
        *,
        include_enable_aot: bool = True,
        ts_dir: Optional[str] = None,
    ) -> Tuple[_Table, _Table, torch.nn.Module]:
        dyn = _Table("model.ebc.dyn", 10, 8, False, dynamic=True)
        static = _Table("model.ebc.static", 20, 8, False, dynamic=False)
        features = {
            "a__ebc": {
                "embedding_name": dyn.name,
                "pooling": "SUM",
                "use_weight": False,
            },
            "b__ebc": {
                "embedding_name": static.name,
                "pooling": "SUM",
                "use_weight": False,
            },
            "seq__ec": {
                "embedding_name": dyn.name,
                "pooling": "NONE",
                "use_weight": False,
            },
        }
        dense_meta = {
            "sequence__ec": ["seq__ec", "seq__lengths"],
            "deep": ["a__ebc", "b__ebc"],
        }
        _write_export(export_dir, [dyn, static], features, dense_meta)

        torch.manual_seed(42)
        dense_mod = _DenseAOTIRich().cuda().eval()
        single_arg = _DenseAOTISingleArg(dense_mod).cuda().eval()
        device = torch.device("cuda:0")
        sample = {
            "deep": torch.randn(2, 16, device=device),
            "seq": torch.randn(3, 8, device=device),
            "seq__lengths": torch.tensor([2, 1], dtype=torch.int32, device=device),
            "x.values": torch.randn(2, 2, device=device),
            "sd.values": torch.randn(3, 4, device=device),
            "sd.lengths": torch.tensor([2, 1], dtype=torch.int64, device=device),
            "mv.lengths": torch.tensor([2, 1], dtype=torch.int64, device=device),
            "mv.key_lengths": torch.tensor([1, 2, 1], dtype=torch.int64, device=device),
            "batch_size": torch.tensor(1, dtype=torch.int64, device=device),
        }
        batch_dim = torch.export.Dim("batch", min=1, max=256)
        seq_dim = torch.export.Dim("seq", min=0, max=1024)
        sd_dim = torch.export.Dim("sd", min=0, max=1024)
        mv_dim = torch.export.Dim("mv", min=0, max=1024)
        dyn_shapes = {
            "deep": {0: batch_dim},
            "seq": {0: seq_dim},
            "seq__lengths": {0: batch_dim},
            "x.values": {0: batch_dim},
            "sd.values": {0: sd_dim},
            "sd.lengths": {0: batch_dim},
            "mv.lengths": {0: batch_dim},
            "mv.key_lengths": {0: mv_dim},
            "batch_size": {},
        }
        ep = torch.export.export(
            single_arg, args=(sample,), dynamic_shapes=(dyn_shapes,)
        )
        aoti_dir = os.path.join(export_dir, "aoti")
        os.makedirs(aoti_dir, exist_ok=True)
        torch._inductor.aoti_compile_and_package(
            ep, package_path=os.path.join(aoti_dir, "aoti_model.pt2")
        )
        acc_cfg = {"DISTRIBUTED_EMBEDDING": "1"}
        if include_enable_aot:
            acc_cfg["ENABLE_AOT"] = "2"
        with open(os.path.join(export_dir, "model_acc.json"), "w") as f:
            json.dump(acc_cfg, f)

        if ts_dir is not None:
            _write_export(ts_dir, [dyn, static], features, dense_meta)
            torch.jit.script(dense_mod).save(os.path.join(ts_dir, "scripted_model.pt"))
            with open(os.path.join(ts_dir, "model_acc.json"), "w") as f:
                json.dump({"DISTRIBUTED_EMBEDDING": "1"}, f)

        return dyn, static, dense_mod

    @staticmethod
    def _make_rich_batch(
        dyn: _Table, bs: int, *, empty_seq: bool = False, seed: int = 0
    ) -> Batch:
        rng = torch.Generator().manual_seed(seed + bs + (100 if empty_seq else 0))
        seq_bags = tuple(
            [] if empty_seq else [dyn.keys[(i + j) % 5] for j in range((i % 2) + 1)]
            for i in range(bs)
        )
        mv_bags = tuple(
            [] if empty_seq else ([0, 1] if i % 2 == 0 else [2]) for i in range(bs)
        )
        bags = {
            "a": tuple([dyn.keys[i % 5], -1] for i in range(bs)),
            "b": tuple([i % 20] for i in range(bs)),
            "seq": seq_bags,
            "mv": mv_bags,
        }
        batch = _lookup_batch(bags, {})
        batch.dense_features = {
            "d": KeyedTensor(
                keys=["x", "y"],
                length_per_key=[2, 3],
                values=torch.randn(bs, 5, generator=rng),
            )
        }
        sd_lengths = (
            torch.zeros(bs, dtype=torch.int32)
            if empty_seq
            else torch.tensor([(i % 2) + 1 for i in range(bs)], dtype=torch.int32)
        )
        batch.sequence_dense_features = {
            "sd": JaggedTensor(
                values=torch.randn(int(sd_lengths.sum()), 4, generator=rng),
                lengths=sd_lengths,
            )
        }
        mv_seq_list = (
            [0] * bs if empty_seq else [1 if i % 2 == 0 else 0 for i in range(bs)]
        )
        mv_seq_lengths = torch.tensor(mv_seq_list, dtype=torch.int32)
        mv_key_lengths = (
            torch.zeros(0, dtype=torch.int32)
            if empty_seq
            else torch.full((int(mv_seq_lengths.sum()),), 2, dtype=torch.int32)
        )
        batch.sequence_mulval_lengths = {
            "g_mv": KeyedJaggedTensor(
                keys=["mv"],
                values=mv_key_lengths,
                lengths=mv_seq_lengths,
            )
        }
        return batch

    @parameterized.expand(
        [
            param("with_enable_aot", include_enable_aot=True),
            param("fallback_without_enable_aot", include_enable_aot=False),
        ],
        name_func=parameterized_name_func,
    )
    def test_runs_aoti_dense_on_lookup(self, _name, include_enable_aot: bool) -> None:
        prev_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        prev_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        ts_dir = make_test_dir()
        try:
            dyn, _static, _dense_mod = self._build_rich_aoti_export(
                self.export_dir,
                include_enable_aot=include_enable_aot,
                ts_dir=ts_dir,
            )
            device = torch.device("cuda:0")
            aoti_model = load_predict_model(self.export_dir, [device], log_model=True)
            ts_model = load_predict_model(ts_dir, [device])
            self.assertIsInstance(aoti_model, DistributedEmbeddingModel)
            self.assertTrue(aoti_model._is_aot)

            cases = [
                (1, False),
                (3, False),
                (7, False),
                (3, True),
                (1, True),
            ]
            for bs, empty_seq in cases:
                batch = self._make_rich_batch(dyn, bs, empty_seq=empty_seq)
                tile_bs = torch.tensor(2, dtype=torch.int64)
                with mock.patch.object(
                    predict_util, "batch_to_dict", wraps=predict_util.batch_to_dict
                ) as spy_b2d:
                    actual = aoti_model(batch, tile_bs, device)
                self.assertEqual(spy_b2d.call_args.args[1], {})
                expected = ts_model(batch, tile_bs, device)
                self.assertEqual(actual["p"].device, device)
                self.assertEqual(actual["p"].shape, (bs,))
                torch.testing.assert_close(
                    actual["p"], expected["p"], rtol=1e-5, atol=1e-5
                )
        finally:
            shutil.rmtree(ts_dir, ignore_errors=True)
            torch.backends.cuda.matmul.allow_tf32 = prev_matmul_tf32
            torch.backends.cudnn.allow_tf32 = prev_cudnn_tf32

    def test_aoti_multithread_concurrent_inference(self) -> None:
        prev_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        prev_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            dyn, _static, _dense_mod = self._build_rich_aoti_export(self.export_dir)
            device = torch.device("cuda:0")
            model = load_predict_model(self.export_dir, [device])
            batches = [
                self._make_rich_batch(dyn, bs, empty_seq=empty_seq, seed=i)
                for i, (bs, empty_seq) in enumerate(
                    [(1, False), (3, False), (7, False), (4, True)]
                )
            ]
            tile_bs = torch.tensor(2, dtype=torch.int64)
            refs = [model(b, tile_bs, device)["p"].cpu() for b in batches]

            num_workers = 4
            streams = [torch.cuda.Stream(device) for _ in range(num_workers)]

            def _worker(worker_id: int) -> None:
                stream = streams[worker_id]
                with torch.cuda.stream(stream):
                    for step in range(8):
                        idx = (worker_id + step) % len(batches)
                        out = model(batches[idx], tile_bs, device)["p"]
                        stream.synchronize()
                        torch.testing.assert_close(
                            out.cpu(), refs[idx], rtol=1e-5, atol=1e-5
                        )

            with ThreadPoolExecutor(max_workers=num_workers) as pool:
                futures = [pool.submit(_worker, wid) for wid in range(num_workers)]
                for fut in futures:
                    fut.result()
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev_matmul_tf32
            torch.backends.cudnn.allow_tf32 = prev_cudnn_tf32

    @unittest.skipIf(torch.cuda.device_count() < 2, "requires 2 GPUs")
    @ci_resources(slots=2)
    def test_aoti_multi_gpu_concurrent_inference(self) -> None:
        prev_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        prev_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            dyn, _static, _dense_mod = self._build_rich_aoti_export(self.export_dir)
            devices = [torch.device("cuda:0"), torch.device("cuda:1")]
            model = load_predict_model(self.export_dir, devices)
            self.assertEqual(len(model._dense_aoti), 2)

            batches = [
                self._make_rich_batch(dyn, bs, empty_seq=empty_seq, seed=i)
                for i, (bs, empty_seq) in enumerate(
                    [(1, False), (3, False), (7, False), (3, True)]
                )
            ]
            tile_bs = torch.tensor(2, dtype=torch.int64)
            refs = [model(b, tile_bs, devices[0])["p"].cpu() for b in batches]

            num_workers = 4
            streams = [
                torch.cuda.Stream(devices[wid % len(devices)])
                for wid in range(num_workers)
            ]

            def _worker(worker_id: int) -> None:
                dev = devices[worker_id % len(devices)]
                stream = streams[worker_id]
                with torch.cuda.stream(stream):
                    for step in range(6):
                        idx = (worker_id + step) % len(batches)
                        out = model(batches[idx], tile_bs, dev)["p"]
                        self.assertEqual(out.device, dev)
                        stream.synchronize()
                        torch.testing.assert_close(
                            out.cpu(), refs[idx], rtol=1e-5, atol=1e-5
                        )

            with ThreadPoolExecutor(max_workers=num_workers) as pool:
                futures = [pool.submit(_worker, wid) for wid in range(num_workers)]
                for fut in futures:
                    fut.result()
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev_matmul_tf32
            torch.backends.cudnn.allow_tf32 = prev_cudnn_tf32

    def test_genrec_export_model_aoti_distributed(self) -> None:
        device = torch.device("cuda:0")
        prev_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        prev_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            with mock.patch.dict(
                os.environ,
                {
                    "USE_DISTRIBUTED_EMBEDDING": "1",
                    "INPUT_TILE": "3",
                    "ENABLE_AOT": "2",
                },
            ):
                hist_cfg = feature_pb2.FeatureConfig(
                    sequence_raw_feature=feature_pb2.RawFeature(
                        feature_name="hist", expression="user:hist"
                    )
                )
                beh_cfg = feature_pb2.FeatureConfig(
                    sequence_id_feature=feature_pb2.IdFeature(
                        feature_name="beh",
                        expression="user:beh",
                        num_buckets=32,
                        embedding_dim=8,
                        sequence_length=2,
                    )
                )
                age_cfg = feature_pb2.FeatureConfig(
                    id_feature=feature_pb2.IdFeature(
                        feature_name="age",
                        expression="user:age",
                        num_buckets=16,
                        embedding_dim=4,
                    )
                )
                genrec_model, _ = create_genrec_test_model(
                    os.path.join(self.export_dir, "src"),
                    feature_configs=[hist_cfg, beh_cfg, age_cfg],
                    prompt=(
                        "User : {{user_slot}} History : {{hist}} . {{beh}} Predict :"
                    ),
                    slots=[PromptSlot(name="user_slot", feature_names=["age"])],
                )
                init_parameters(genrec_model, device=device)
                genrec_model.eval()
                # build_dense_graph_module shrinks sparse tables on genrec_model
                # in place, so snapshot a full-table reference copy beforehand.
                ref_model = copy.deepcopy(genrec_model)
                ref_wrapped = ScriptWrapper(GenRecFrontEnd(ref_model)).to(device).eval()
                genrec_model.lm.save_pretrained(self.export_dir)
                with open(os.path.join(self.export_dir, "config.json"), "w") as f:
                    json.dump(
                        {
                            "model_type": SERVING_MODEL_TYPE,
                            "text_config": json.loads(
                                genrec_model.lm.config.to_json_string()
                            ),
                            "sid_space": {
                                "codebook": [4, 4, 4],
                                "base_vocab_size": 40,
                            },
                        },
                        f,
                    )
                pipeline_cfg = EasyRecConfig()
                pipeline_cfg.model_config.genrec_causal_lm_model.common.beam_widths.extend(
                    [2, 2, 2]
                )
                config_util.save_message(
                    pipeline_cfg, os.path.join(self.export_dir, "pipeline.config")
                )

                wrapped = ScriptWrapper(GenRecFrontEnd(genrec_model)).to(device).eval()
                export_data = {
                    "batch_size": torch.tensor(1, dtype=torch.int64, device=device),
                    "hist.values": torch.tensor(
                        GENREC_LONG_HIST_CODES * 2,
                        dtype=torch.float32,
                        device=device,
                    ).reshape(-1, 1),
                    "hist.lengths": torch.tensor(
                        [6, 6], dtype=torch.int64, device=device
                    ),
                    "beh.values": torch.zeros(4, dtype=torch.int64, device=device),
                    "beh.lengths": torch.tensor(
                        [2, 2], dtype=torch.int64, device=device
                    ),
                    "age.values": torch.zeros(2, dtype=torch.int64, device=device),
                    "age.lengths": torch.tensor(
                        [1, 1], dtype=torch.int64, device=device
                    ),
                }
                gm, full_graph, dense_meta = build_dense_graph_module(
                    wrapped, export_data, device
                )
                sparse_kts, seq_jts = _run_dense_graph_sanity_check(
                    wrapped, full_graph, gm, export_data, device
                )
                export_distributed_dense_model_aot(
                    model=wrapped,
                    gm=gm,
                    data=export_data,
                    sparse_kts=sparse_kts,
                    seq_jts=seq_jts,
                    dense_graph_config=dense_meta,
                    save_dir=self.export_dir,
                )

                emb_sd = ref_model.embedding_group.state_dict()
                age_w = (
                    emb_sd["emb_impls.__BASE__.ebc_user.embedding_bags.age_emb.weight"]
                    .detach()
                    .cpu()
                    .numpy()
                )
                beh_w = (
                    emb_sd[
                        "seq_emb_impls.__BASE__.ec_dict_user.8.embeddings.beh_emb.weight"
                    ]
                    .detach()
                    .cpu()
                    .numpy()
                )
                sparse_dir = os.path.join(self.export_dir, "sparse")
                os.makedirs(sparse_dir, exist_ok=True)
                npz_util.savez_streaming(
                    os.path.join(sparse_dir, "sparse_embeddings-00-of-01.npz"),
                    {"age_emb": age_w, "beh_emb": beh_w},
                )
                with open(os.path.join(sparse_dir, "sparse_embedding.json"), "w") as f:
                    json.dump(
                        {
                            "age_emb": {
                                "dense": False,
                                "is_dynamic": False,
                                "dimension": 4,
                                "shape": list(age_w.shape),
                                "dtype": "float32",
                            },
                            "beh_emb": {
                                "dense": False,
                                "is_dynamic": False,
                                "dimension": 8,
                                "shape": list(beh_w.shape),
                                "dtype": "float32",
                            },
                        },
                        f,
                    )
                with open(os.path.join(sparse_dir, "sparse_features.json"), "w") as f:
                    json.dump(
                        {
                            "age__ebc": {
                                "embedding_name": "age_emb",
                                "pooling": "SUM",
                                "use_weight": False,
                            },
                            "beh__ec": {
                                "embedding_name": "beh_emb",
                                "pooling": "NONE",
                                "use_weight": False,
                            },
                        },
                        f,
                    )
                with open(os.path.join(self.export_dir, "dense_meta.json"), "w") as f:
                    json.dump(dict(dense_meta), f)
                with open(os.path.join(self.export_dir, "model_acc.json"), "w") as f:
                    json.dump(
                        {
                            "DISTRIBUTED_EMBEDDING": "1",
                            "INPUT_TILE": "3",
                            "ENABLE_AOT": "2",
                        },
                        f,
                    )

            predict_model = load_predict_model(self.export_dir, [device])
            self.assertIsInstance(predict_model, GenRecExportModel)
            self.assertIsInstance(predict_model._frontend, DistributedEmbeddingModel)
            self.assertTrue(predict_model._frontend._is_aot)

            for bs in (1, 3):
                batch = Batch(
                    sparse_features={
                        "g0": KeyedJaggedTensor(
                            keys=["age", "beh"],
                            values=torch.cat(
                                [
                                    torch.arange(1, bs + 1, dtype=torch.int64),
                                    torch.arange(2, 2 + 2 * bs, dtype=torch.int64),
                                ]
                            ),
                            lengths=torch.cat(
                                [
                                    torch.ones(bs, dtype=torch.int32),
                                    torch.full((bs,), 2, dtype=torch.int32),
                                ]
                            ),
                        )
                    },
                    sequence_dense_features={
                        "hist": JaggedTensor(
                            values=torch.tensor(
                                GENREC_LONG_HIST_CODES * bs, dtype=torch.float32
                            ).reshape(-1, 1),
                            lengths=torch.full((bs,), 6, dtype=torch.int32),
                        )
                    },
                )
                tile_bs = torch.tensor(1, dtype=torch.int64)
                with (
                    torch.no_grad(),
                    mock.patch.dict(os.environ, {"INPUT_TILE": "3"}),
                ):
                    raw_inputs = {
                        k: v.to(device)
                        for k, v in batch.to_dict(sparse_dtype=torch.int64).items()
                    }
                    raw_inputs["batch_size"] = tile_bs.to(device)
                    expected_frontend = ref_wrapped(raw_inputs, device)
                with torch.no_grad():
                    actual_frontend = predict_model._frontend(batch, tile_bs, device)
                for key in (
                    INPUT_IDS,
                    CU_SEQLENS,
                    HOLE_POSITIONS,
                    HOLE_KEYS,
                    HOLE_SLOT_COUNTS,
                    SLOT_EMBEDS,
                ):
                    self.assertIn(key, actual_frontend)
                    self.assertEqual(actual_frontend[key].device, device)
                    torch.testing.assert_close(
                        actual_frontend[key],
                        expected_frontend[key],
                        rtol=1e-5,
                        atol=1e-5,
                    )

                with torch.no_grad():
                    actual = predict_model(batch, tile_bs, device)
                    ref_lm = ref_model.lm.to(device)
                    embeds = ref_lm.get_input_embeddings()(expected_frontend[INPUT_IDS])
                    embeds.index_copy_(
                        0,
                        expected_frontend[HOLE_POSITIONS],
                        expected_frontend[SLOT_EMBEDS].to(embeds.dtype),
                    )
                    _, band_lo, band_hi = sid_bands([4, 4, 4], 40)
                    expected_codes = generate_codes(
                        ref_lm,
                        embeds,
                        expected_frontend[CU_SEQLENS],
                        int(torch.diff(expected_frontend[CU_SEQLENS]).max()),
                        [2, 2, 2],
                        list(zip(band_lo, band_hi)),
                    )
                self.assertEqual(actual["generated_sids"].device, device)
                self.assertTrue(torch.equal(actual["generated_sids"], expected_codes))
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev_matmul_tf32
            torch.backends.cudnn.allow_tf32 = prev_cudnn_tf32


class GenRecExportModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.export_dir = make_test_dir()
        # a [4, 4, 4] codebook above 40 base tokens: SID tokens 40..51 of 64
        self.lm = create_tiny_causal_lm(64, tie_word_embeddings=True)
        self.lm.save_pretrained(self.export_dir)
        self._write_config(sid_space={"codebook": [4, 4, 4], "base_vocab_size": 40})
        self.pipeline_config = EasyRecConfig()
        self.common = self.pipeline_config.model_config.genrec_causal_lm_model.common
        self.common.beam_widths.extend([2, 2, 2])

    def tearDown(self) -> None:
        shutil.rmtree(self.export_dir, ignore_errors=True)

    def _write_config(self, **extra) -> None:
        text_config = json.loads(self.lm.config.to_json_string())
        with open(os.path.join(self.export_dir, "config.json"), "w") as f:
            json.dump(
                {"model_type": SERVING_MODEL_TYPE, "text_config": text_config, **extra},
                f,
            )

    def _model(self, streams: Dict[str, torch.Tensor]) -> GenRecExportModel:
        with mock.patch.object(
            predict_util, "_load_batch_model", return_value=lambda *_: streams
        ):
            model = load_predict_model(
                self.export_dir,
                [torch.device("cpu")],
                pipeline_config=self.pipeline_config,
            )
        self.assertIsInstance(model, GenRecExportModel)
        return model

    def _streams(self, with_slots: bool) -> Dict[str, torch.Tensor]:
        if with_slots:
            hole_positions = torch.tensor([1, 5])
            slot_embeds = torch.randn(2, self.lm.config.hidden_size)
        else:
            hole_positions = torch.zeros(0, dtype=torch.int64)
            slot_embeds = torch.zeros(0, 0)
        return {
            INPUT_IDS: torch.tensor([3, 7, 1, 5, 9, 2, 8, 4]),
            CU_SEQLENS: torch.tensor([0, 3, 8]),
            HOLE_POSITIONS: hole_positions,
            SLOT_EMBEDS: slot_embeds,
        }

    @parameterized.expand([[False, 2], [True, 6]], name_func=parameterized_name_func)
    def test_decodes_like_generate_codes(self, with_slots, final_width) -> None:
        self.common.beam_widths[-1] = final_width
        streams = self._streams(with_slots)
        model = self._model(streams)
        _, band_lo, band_hi = sid_bands([4, 4, 4], 40)
        bands = list(zip(band_lo, band_hi))
        widths = resolve_beam_config(self.common, [4, 4, 4])
        with torch.no_grad():
            out = model(Batch(), None, torch.device("cpu"))
            embeds = self.lm.get_input_embeddings()(streams[INPUT_IDS])
            if with_slots:
                embeds = embeds.index_copy(
                    0, streams[HOLE_POSITIONS], streams[SLOT_EMBEDS]
                )
            expected = generate_codes(
                self.lm,
                embeds,
                streams[CU_SEQLENS],
                5,
                widths,
                bands,
            )
        self.assertEqual(out["generated_sids"].shape, (2, widths[-1], 3))
        self.assertTrue(torch.equal(out["generated_sids"], expected))

    def test_reads_the_exports_pipeline_config(self) -> None:
        self.common.beam_widths[-1] = 1
        config_util.save_message(
            self.pipeline_config, os.path.join(self.export_dir, "pipeline.config")
        )
        streams = self._streams(False)
        with mock.patch.object(
            predict_util, "_load_batch_model", return_value=lambda *_: streams
        ):
            model = load_predict_model(self.export_dir, [torch.device("cpu")])
        with torch.no_grad():
            out = model(Batch(), None, torch.device("cpu"))
        self.assertEqual(out["generated_sids"].shape, (2, 1, 3))

    def test_needs_the_sid_space(self) -> None:
        self._write_config()
        with self.assertRaisesRegex(ValueError, "no sid_space"):
            self._model(self._streams(False))

    def test_refuses_an_lm_that_does_not_load_cleanly(self) -> None:
        path = os.path.join(self.export_dir, "model.safetensors")
        tensors = load_file(path)
        del tensors["model.norm.weight"]
        save_file(tensors, path, metadata={"format": "pt"})
        with self.assertRaisesRegex(RuntimeError, "does not load cleanly"):
            self._model(self._streams(False))


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


class DistributedScriptGenRecTest(unittest.TestCase):
    """Non-AOTI TorchScript distributed prediction and env isolation tests."""

    def setUp(self) -> None:
        self.model_dir = make_test_dir()
        self.addCleanup(shutil.rmtree, self.model_dir, ignore_errors=True)

    def test_load_scripted_model_does_not_mutate_texpr_env(self) -> None:
        torch.jit.script(_Model()).save(
            os.path.join(self.model_dir, "scripted_model.pt")
        )
        with mock.patch.dict(os.environ, {}, clear=True):
            loaded = _load_scripted_model(self.model_dir, torch.device("cpu"))
            self.assertIsNotNone(loaded)
            self.assertNotIn("PYTORCH_TENSOREXPR_FALLBACK", os.environ)

    def test_distributed_embedding_script_model_genrec_raw_sparse_passthrough(
        self,
    ) -> None:
        from tzrec.protos import feature_pb2
        from tzrec.protos.prompt_pb2 import PromptSlot
        from tzrec.utils import export_util
        from tzrec.utils.state_dict_util import init_parameters

        hist_cfg = feature_pb2.FeatureConfig(
            sequence_raw_feature=feature_pb2.RawFeature(
                feature_name="hist", expression="user:hist"
            )
        )
        beh_cfg = feature_pb2.FeatureConfig(
            sequence_id_feature=feature_pb2.IdFeature(
                feature_name="beh",
                expression="user:beh",
                num_buckets=16,
                embedding_dim=8,
                sequence_length=4,
            )
        )
        age_cfg = feature_pb2.FeatureConfig(
            id_feature=feature_pb2.IdFeature(
                feature_name="age",
                expression="user:age",
                num_buckets=16,
                embedding_dim=8,
            )
        )
        ref_model, _ = create_genrec_test_model(
            self.model_dir,
            feature_configs=[hist_cfg, beh_cfg, age_cfg],
            prompt="U {{user_slot}} H {{hist}} B {{beh}} P :",
            slots=[PromptSlot(name="user_slot", feature_names=["age"])],
        )
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        init_parameters(ref_model, device=device)
        ref_model.eval()
        eval_model = copy.deepcopy(ref_model)
        eval_frontend = ScriptWrapper(GenRecFrontEnd(eval_model)).to(device).eval()
        # build_dense_graph_module shrinks sparse tables to 1 row in place.
        frontend = ScriptWrapper(GenRecFrontEnd(ref_model)).to(device).eval()
        warmup_data = {
            "hist.values": torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32),
            "hist.lengths": torch.tensor([2, 1], dtype=torch.int64),
            "beh.values": torch.zeros(3, dtype=torch.int64),
            "beh.lengths": torch.tensor([2, 1], dtype=torch.int64),
            "age.values": torch.zeros(2, dtype=torch.int64),
            "age.lengths": torch.tensor([1, 1], dtype=torch.int64),
        }
        with export_util.acc_utils.export_env_scope(
            use_distributed_embedding=True, input_tile="0", input_tile_3_online=False
        ):
            gm, _, dense_cfg = export_util.build_dense_graph_module(
                frontend,
                warmup_data,
                device=device,
            )
            traced_dense = export_util.symbolic_trace(gm)
            export_util.eliminate_dense_dead_code(traced_dense.graph)
            traced_dense.recompile()
            torch.jit.script(traced_dense).save(
                os.path.join(self.model_dir, "scripted_model.pt")
            )

        with open(os.path.join(self.model_dir, "dense_meta.json"), "w") as f:
            json.dump(dict(dense_cfg), f)
        with open(os.path.join(self.model_dir, "model_acc.json"), "w") as f:
            json.dump(
                {
                    "DISTRIBUTED_EMBEDDING": "1",
                    "INPUT_TILE": "0",
                    "ENABLE_AOT": "0",
                },
                f,
            )

        emb_sd = eval_model.embedding_group.state_dict()
        age_w = next(v for k, v in emb_sd.items() if k.endswith(".age_emb.weight")).to(
            device
        )
        beh_w = next(v for k, v in emb_sd.items() if k.endswith(".beh_emb.weight")).to(
            device
        )

        class _FakeLookup:
            def __call__(self, batch: Batch, target_device: torch.device):
                kjt = batch.sparse_features["kjt"]
                age_jt = kjt["age"]
                beh_jt = kjt["beh"]
                age_emb = age_w[age_jt.values().to(target_device)].to(torch.float32)
                return {
                    "user_slot__ebc": age_emb,
                    "beh": beh_w[beh_jt.values().to(target_device)].to(torch.float32),
                    "beh__lengths": beh_jt.lengths().to(target_device, torch.int64),
                }

        with mock.patch(
            "tzrec.utils.predict_util.DistributedEmbeddingLookup",
            return_value=_FakeLookup(),
        ):
            dist_model = DistributedEmbeddingModel(
                self.model_dir, [device], log_model=False
            )

        # bs > 1 with one 0-length cold-start row and int32 CPU inputs.
        batch = Batch(
            sparse_features={
                "kjt": KeyedJaggedTensor.from_lengths_sync(
                    keys=["age", "beh"],
                    values=torch.tensor([1, 2, 3, 2, 5, 7], dtype=torch.int32),
                    lengths=torch.tensor([1, 1, 1, 0, 2, 1], dtype=torch.int32),
                )
            },
            sequence_dense_features={
                "hist": JaggedTensor(
                    values=torch.tensor([[1.0], [3.0], [4.0]], dtype=torch.float32),
                    lengths=torch.tensor([0, 2, 1], dtype=torch.int64),
                )
            },
            labels={},
        )
        with torch.no_grad():
            out = dist_model(batch, torch.tensor(3, dtype=torch.int64), device)
            raw_inputs = {
                k: v.to(device)
                for k, v in batch.to_dict(sparse_dtype=torch.int64).items()
            }
            expected = eval_frontend(raw_inputs, device)

        self.assertIn(SLOT_EMBEDS, out)
        self.assertIn(HOLE_POSITIONS, out)
        self.assertEqual(out[CU_SEQLENS].numel(), 4)
        for key in expected:
            torch.testing.assert_close(out[key], expected[key], rtol=1e-5, atol=1e-5)

    def test_needed_sparse_kjt_names_memoizes_steady_state_layout(self) -> None:
        model = object.__new__(DistributedEmbeddingModel)
        model._required_keys = {"a.values", "mv.lengths", "mv.key_lengths"}
        model._has_raw_sparse_candidates = True
        model._sparse_copy_keys_cache = {}
        model._fast_sparse_copy_entry = None

        batch1 = Batch(
            sparse_features={
                "g1": KeyedJaggedTensor.from_lengths_sync(
                    keys=["a", "b"],
                    values=torch.tensor([1, 2]),
                    lengths=torch.tensor([1, 1]),
                ),
                "g2": KeyedJaggedTensor.from_lengths_sync(
                    keys=["c"],
                    values=torch.tensor([3]),
                    lengths=torch.tensor([1]),
                ),
                "g3": KeyedJaggedTensor.from_lengths_sync(
                    keys=["mv"],
                    values=torch.tensor([4]),
                    lengths=torch.tensor([1]),
                ),
            },
            sequence_mulval_lengths={
                "g_mv": KeyedJaggedTensor.from_lengths_sync(
                    keys=["mv"],
                    values=torch.tensor([1]),
                    lengths=torch.tensor([1]),
                )
            },
        )
        self.assertEqual(model._needed_sparse_kjt_names(batch1), ("g1",))
        self.assertIsNotNone(model._fast_sparse_copy_entry)
        self.assertEqual(len(model._sparse_copy_keys_cache), 1)

        # Same layout hits the zero-hash fast entry without re-populating cache.
        batch1_repeat = Batch(
            sparse_features={
                "g1": KeyedJaggedTensor.from_lengths_sync(
                    keys=["a", "b"],
                    values=torch.tensor([5, 6, 7]),
                    lengths=torch.tensor([2, 1]),
                ),
                "g2": KeyedJaggedTensor.from_lengths_sync(
                    keys=["c"],
                    values=torch.tensor([8]),
                    lengths=torch.tensor([1]),
                ),
                "g3": KeyedJaggedTensor.from_lengths_sync(
                    keys=["mv"],
                    values=torch.tensor([9]),
                    lengths=torch.tensor([1]),
                ),
            },
            sequence_mulval_lengths={
                "g_mv": KeyedJaggedTensor.from_lengths_sync(
                    keys=["mv"],
                    values=torch.tensor([1]),
                    lengths=torch.tensor([1]),
                )
            },
        )
        with mock.patch.object(
            model,
            "_sparse_copy_keys_cache",
            side_effect=AssertionError("steady-state layout should skip dict lookup"),
        ):
            self.assertEqual(model._needed_sparse_kjt_names(batch1_repeat), ("g1",))

        # Different layout falls back and recomputes cleanly.
        batch2 = Batch(
            sparse_features={
                "g2": KeyedJaggedTensor.from_lengths_sync(
                    keys=["c", "a"],
                    values=torch.tensor([1, 2]),
                    lengths=torch.tensor([1, 1]),
                )
            }
        )
        self.assertEqual(model._needed_sparse_kjt_names(batch2), ("g2",))
        self.assertEqual(len(model._sparse_copy_keys_cache), 2)


if __name__ == "__main__":
    unittest.main()
