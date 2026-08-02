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

import queue
import threading
import time
import unittest
from unittest import mock

from parameterized import parameterized

from tzrec.utils import predict_util
from tzrec.utils.test_util import parameterized_name_func


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
                [], forward_threads, failure_queue, cancel_event, timeout=1
            )
            predict_util.queue_put_interruptibly(
                output_queue,
                None,
                cancel_event,
                "writer completion",
                timeout=1,
            )
            predict_util.wait_for_pipeline(
                [], [writer_thread], failure_queue, cancel_event, timeout=1
            )

        writer = mock.Mock()
        predict_util.validate_and_commit_writer(
            writer, True, len(submitted), len(submitted), len(written), len(written)
        )
        self.assertEqual(sorted(written), submitted)
        self.assertTrue(all(not thread.is_alive() for thread in threads))
        writer.close.assert_called_once_with()

    def test_invalid_output_never_commits(self):
        invalid_cases = [
            (False, 1, 1, 1, 1),
            (True, 0, 0, 0, 0),
            (True, 2, 2, 1, 2),
            (True, 2, 2, 2, 1),
        ]
        for args in invalid_cases:
            with self.subTest(args=args):
                writer = mock.Mock()
                with self.assertRaises(RuntimeError):
                    predict_util.validate_and_commit_writer(writer, *args)
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


if __name__ == "__main__":
    unittest.main()
