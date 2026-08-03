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

import multiprocessing
import queue
import threading
import time
import unittest
from unittest import mock

import pyarrow as pa
from parameterized import parameterized

from tzrec.datasets.utils import Batch, RecordBatchTensor
from tzrec.tools.tdm import retrieval
from tzrec.utils import predict_util
from tzrec.utils.test_util import parameterized_name_func


class _FailingSampler:
    _item_id_field = "item_id"

    def init(self, worker_id):
        raise ValueError("worker boom")


class _PassingSampler:
    _item_id_field = "item_id"

    def init(self, worker_id):
        return None

    def init_sampler(self, n_cluster):
        return None

    def get(self, input_data):
        return {"item_id": pa.array([2])}


class _PassingParser:
    def parse(self, input_data):
        return input_data

    def to_batch(self, output_data, force_no_tile=False):
        return Batch()


class _StubbornProcess:
    def __init__(self):
        self.pid = 123
        self.exitcode = None
        self.terminate_called = False
        self.kill_called = False
        self._alive = True

    def is_alive(self):
        return self._alive

    def terminate(self):
        self.terminate_called = True

    def kill(self):
        self.kill_called = True
        self._alive = False
        self.exitcode = -9

    def join(self, timeout=None):
        return None


class TDMRetrievalLifecycleTest(unittest.TestCase):
    @parameterized.expand([[1], [2], [4], [8]], name_func=parameterized_name_func)
    def test_normal_sentinel_fanout(self, worker_count):
        for downstream_count in (worker_count, 1):
            data_queue = queue.Queue()
            pred_queue = queue.Queue(maxsize=2 if downstream_count == 1 else 0)
            cancel_event = threading.Event()
            failure_queue = queue.Queue()
            input_drained_event = threading.Event()
            for _ in range(worker_count):
                data_queue.put((None, None, None))

            retrieval._forward_loop(
                data_queue,
                pred_queue,
                3,
                worker_count,
                downstream_count,
                mock.Mock(),
                input_drained_event,
                cancel_event,
                failure_queue,
            )

            self.assertEqual(pred_queue.qsize(), downstream_count)
            for _ in range(downstream_count):
                self.assertEqual(pred_queue.get_nowait(), (None, None))
            self.assertFalse(cancel_event.is_set())
            self.assertTrue(failure_queue.empty())
            self.assertTrue(input_drained_event.is_set())

    def test_data_worker_keeps_shared_storage_alive_until_drained(self):
        in_queue = multiprocessing.Queue()
        out_queue = multiprocessing.Queue()
        output_drained_event = multiprocessing.Event()
        cancel_event = multiprocessing.Event()
        failure_queue = multiprocessing.Queue()
        record_batch_t = RecordBatchTensor(pa.record_batch({"item_id": [1]}))
        in_queue.put((record_batch_t, pa.array([1])))
        in_queue.put((None, None))
        worker = multiprocessing.Process(
            target=retrieval._tdm_predict_data_worker,
            args=(
                _PassingSampler(),
                _PassingParser(),
                1,
                2,
                in_queue,
                out_queue,
                False,
                0,
                output_drained_event,
                cancel_event,
                failure_queue,
            ),
        )
        worker.start()

        batch, output_record_batch_t, node_ids = out_queue.get(timeout=2)
        self.assertIsInstance(batch, Batch)
        self.assertEqual(output_record_batch_t.get()["item_id"].to_pylist(), [1])
        self.assertEqual(node_ids.to_pylist(), [2])
        self.assertEqual(out_queue.get(timeout=1), (None, None, None))
        self.assertTrue(worker.is_alive())
        output_drained_event.set()
        worker.join(timeout=2)

        self.assertFalse(worker.is_alive())
        self.assertEqual(worker.exitcode, 0)
        self.assertFalse(cancel_event.is_set())
        self.assertTrue(failure_queue.empty())
        for data_queue in (in_queue, out_queue, failure_queue):
            data_queue.close()
            data_queue.join_thread()

    @parameterized.expand(
        [["worker"], ["forward"], ["writer"]],
        name_func=parameterized_name_func,
    )
    def test_stage_failure_is_propagated_without_commit(self, stage):
        cancel_event = threading.Event()
        failure_queue = queue.Queue()
        writer = mock.Mock()

        if stage == "worker":
            with self.assertRaisesRegex(ValueError, "worker boom"):
                retrieval._tdm_predict_data_worker(
                    _FailingSampler(),
                    mock.Mock(),
                    1,
                    2,
                    queue.Queue(),
                    queue.Queue(),
                    True,
                    7,
                    threading.Event(),
                    cancel_event,
                    failure_queue,
                )
        elif stage == "forward":
            data_queue = queue.Queue()
            data_queue.put((object(), object(), object()))
            retrieval._forward_loop(
                data_queue,
                queue.Queue(),
                3,
                1,
                1,
                mock.Mock(side_effect=ValueError("forward boom")),
                threading.Event(),
                cancel_event,
                failure_queue,
            )
        else:
            pred_queue = queue.Queue()
            pred_queue.put((object(), object()))
            retrieval._write_loop(
                pred_queue,
                mock.Mock(side_effect=ValueError("writer boom")),
                cancel_event,
                failure_queue,
            )

        self.assertTrue(cancel_event.is_set())
        with self.assertRaises(predict_util.PredictPipelineStageError) as error:
            predict_util.raise_background_failure(failure_queue)
        self.assertIn(f"{stage} boom", str(error.exception))
        self.assertIn("ValueError", str(error.exception))
        with self.assertRaisesRegex(RuntimeError, "not committed"):
            predict_util.validate_and_commit_writer(writer, False, 1, 1, 1, 1)
        writer.close.assert_not_called()

    def test_full_queue_is_interruptible_and_cleanup_is_bounded(self):
        data_queue = queue.Queue(maxsize=1)
        data_queue.put("full")
        cancel_event = threading.Event()

        def cancel():
            time.sleep(0.02)
            cancel_event.set()

        cancel_thread = threading.Thread(target=cancel)
        cancel_thread.start()
        with mock.patch.object(predict_util, "_PREDICT_PIPELINE_POLL_INTERVAL", 0.01):
            with self.assertRaises(predict_util.PredictPipelineCancelled):
                predict_util.queue_put_interruptibly(
                    data_queue, "blocked", cancel_event, "test queue", timeout=1
                )
            with self.assertRaisesRegex(TimeoutError, "test stall"):
                predict_util.queue_put_interruptibly(
                    data_queue,
                    "blocked",
                    threading.Event(),
                    "test stall",
                    timeout=0.02,
                )
            failed_process = _StubbornProcess()
            failed_process._alive = False
            failed_process.exitcode = -9
            failed_event = threading.Event()
            with self.assertRaisesRegex(RuntimeError, "exitcode=-9"):
                predict_util.check_pipeline_health(
                    [failed_process], queue.Queue(), failed_event
                )
            self.assertTrue(failed_event.is_set())
        cancel_thread.join(timeout=1)

        process = _StubbornProcess()
        pipeline_queue = mock.Mock()
        with (
            mock.patch.object(predict_util, "_PREDICT_PIPELINE_CLEANUP_TIMEOUT", 0),
            mock.patch.object(predict_util, "_PREDICT_PIPELINE_TERMINATE_TIMEOUT", 0),
            mock.patch.object(predict_util, "_PREDICT_PIPELINE_KILL_TIMEOUT", 0),
        ):
            predict_util.cleanup_pipeline(
                [process], [], [pipeline_queue], threading.Event()
            )
        self.assertTrue(process.terminate_called)
        self.assertTrue(process.kill_called)
        pipeline_queue.cancel_join_thread.assert_called_once_with()
        pipeline_queue.close.assert_called_once_with()

    def test_commit_requires_complete_nonempty_success(self):
        writer = mock.Mock()
        predict_util.validate_and_commit_writer(writer, True, 2, 8, 2, 8)
        writer.close.assert_called_once_with()

        invalid_cases = [
            (False, 2, 8, 2, 8),
            (True, 0, 0, 0, 0),
            (True, 2, 8, 1, 4),
        ]
        for args in invalid_cases:
            failed_writer = mock.Mock()
            with self.assertRaises(RuntimeError):
                predict_util.validate_and_commit_writer(failed_writer, *args)
            failed_writer.close.assert_not_called()


if __name__ == "__main__":
    unittest.main()
