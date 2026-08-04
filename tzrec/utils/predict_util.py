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

import queue as queue_lib
import time
import traceback
from dataclasses import dataclass
from multiprocessing import Process
from threading import Thread
from typing import Any, Callable, Optional, Sequence

import torch
from torch import distributed as dist
from torch.distributed import ReduceOp

from tzrec.constant import PREDICT_QUEUE_TIMEOUT
from tzrec.utils.logging_util import logger

_PREDICT_PIPELINE_POLL_INTERVAL = 0.1
_PREDICT_PIPELINE_CLEANUP_TIMEOUT = 10.0
_PREDICT_PIPELINE_TERMINATE_TIMEOUT = 5.0
_PREDICT_PIPELINE_KILL_TIMEOUT = 5.0


@dataclass(frozen=True)
class PredictPipelineFailure:
    """Serializable failure raised by a background prediction stage."""

    stage: str
    worker_id: Optional[int]
    exception_type: str
    message: str
    traceback: str


class PredictPipelineCancelled(RuntimeError):
    """Signal cooperative prediction-pipeline cancellation."""


class PredictPipelineStageError(RuntimeError):
    """Failure propagated from a background prediction stage."""

    def __init__(self, failure: PredictPipelineFailure) -> None:
        worker = "" if failure.worker_id is None else f"[{failure.worker_id}]"
        self.failure = failure
        super().__init__(
            f"Prediction pipeline {failure.stage}{worker} failed with "
            f"{failure.exception_type}: {failure.message}\n{failure.traceback}"
        )


def report_failure(
    failure_queue: Any,
    cancel_event: Any,
    stage: str,
    worker_id: Optional[int],
    error: BaseException,
) -> None:
    """Publish the active exception and cancel the prediction pipeline."""
    failure = PredictPipelineFailure(
        stage=stage,
        worker_id=worker_id,
        exception_type=type(error).__name__,
        message=str(error),
        traceback=traceback.format_exc(),
    )
    try:
        failure_queue.put_nowait(failure)
    except Exception:
        logger.exception("Failed to report a prediction pipeline failure.")
    finally:
        cancel_event.set()


def raise_background_failure(failure_queue: Any, wait: bool = False) -> None:
    """Raise the oldest reported prediction failure, if present."""
    try:
        failure = failure_queue.get(
            timeout=_PREDICT_PIPELINE_POLL_INTERVAL if wait else 0
        )
    except queue_lib.Empty:
        return
    raise PredictPipelineStageError(failure)


def check_pipeline_health(
    processes: Sequence[Process], failure_queue: Any, cancel_event: Any
) -> None:
    """Raise reported failures, cancellation, or failed child exits."""
    raise_background_failure(failure_queue)
    if cancel_event.is_set():
        raise_background_failure(failure_queue, wait=True)
        raise RuntimeError("Prediction pipeline was cancelled without an error.")

    failed_processes = [
        process
        for process in processes
        if process.exitcode is not None and process.exitcode != 0
    ]
    if failed_processes:
        cancel_event.set()
        raise_background_failure(failure_queue, wait=True)
        details = ", ".join(
            f"pid={process.pid}, exitcode={process.exitcode}"
            for process in failed_processes
        )
        raise RuntimeError(f"Prediction pipeline child process failed: {details}.")


def queue_get_interruptibly(
    data_queue: Any,
    cancel_event: Any,
    stage: str,
    timeout: float = PREDICT_QUEUE_TIMEOUT,
    health_check: Optional[Callable[[], None]] = None,
) -> Any:
    """Get a queue item within one deadline while checking pipeline health."""
    deadline = time.monotonic() + timeout
    while not cancel_event.is_set():
        if health_check is not None:
            health_check()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"Prediction pipeline {stage} stalled waiting for queue input "
                f"for {timeout} seconds."
            )
        try:
            return data_queue.get(
                timeout=min(_PREDICT_PIPELINE_POLL_INTERVAL, remaining)
            )
        except queue_lib.Empty:
            continue
    raise PredictPipelineCancelled(f"Prediction pipeline {stage} was cancelled.")


def queue_put_interruptibly(
    data_queue: Any,
    item: Any,
    cancel_event: Any,
    stage: str,
    timeout: float = PREDICT_QUEUE_TIMEOUT,
    health_check: Optional[Callable[[], None]] = None,
) -> None:
    """Put a queue item within one deadline while checking pipeline health."""
    deadline = time.monotonic() + timeout
    while not cancel_event.is_set():
        if health_check is not None:
            health_check()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"Prediction pipeline {stage} stalled waiting for queue capacity "
                f"for {timeout} seconds."
            )
        try:
            data_queue.put(
                item, timeout=min(_PREDICT_PIPELINE_POLL_INTERVAL, remaining)
            )
            return
        except queue_lib.Full:
            continue
    raise PredictPipelineCancelled(f"Prediction pipeline {stage} was cancelled.")


def wait_for_pipeline(
    processes: Sequence[Process],
    threads: Sequence[Thread],
    failure_queue: Any,
    cancel_event: Any,
    timeout: float = PREDICT_QUEUE_TIMEOUT,
) -> None:
    """Wait boundedly for normal completion while monitoring failures."""
    deadline = time.monotonic() + timeout
    while True:
        check_pipeline_health(processes, failure_queue, cancel_event)
        alive_processes = [process for process in processes if process.is_alive()]
        alive_threads = [thread for thread in threads if thread.is_alive()]
        if not alive_processes and not alive_threads:
            break

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            process_ids = [process.pid for process in alive_processes]
            thread_names = [thread.name for thread in alive_threads]
            raise TimeoutError(
                "Prediction pipeline stalled during completion; "
                f"processes={process_ids}, threads={thread_names}."
            )
        time.sleep(min(_PREDICT_PIPELINE_POLL_INTERVAL, remaining))

    for process in processes:
        process.join(timeout=0)
    for thread in threads:
        thread.join(timeout=0)
    check_pipeline_health(processes, failure_queue, cancel_event)
    raise_background_failure(failure_queue, wait=bool(processes))


def cleanup_pipeline(
    processes: Sequence[Process],
    threads: Sequence[Thread],
    queues: Sequence[Any],
    cancel_event: Any,
) -> None:
    """Cancel and reap prediction pipeline components within bounded deadlines."""
    cancel_event.set()
    graceful_deadline = time.monotonic() + _PREDICT_PIPELINE_CLEANUP_TIMEOUT
    while time.monotonic() < graceful_deadline:
        if not any(process.is_alive() for process in processes) and not any(
            thread.is_alive() for thread in threads
        ):
            break
        time.sleep(
            min(
                _PREDICT_PIPELINE_POLL_INTERVAL,
                max(0, graceful_deadline - time.monotonic()),
            )
        )

    surviving_processes = [process for process in processes if process.is_alive()]
    for process in surviving_processes:
        process.terminate()

    terminate_deadline = time.monotonic() + _PREDICT_PIPELINE_TERMINATE_TIMEOUT
    while time.monotonic() < terminate_deadline and any(
        process.is_alive() for process in surviving_processes
    ):
        time.sleep(
            min(
                _PREDICT_PIPELINE_POLL_INTERVAL,
                max(0, terminate_deadline - time.monotonic()),
            )
        )

    surviving_processes = [
        process for process in surviving_processes if process.is_alive()
    ]
    for process in surviving_processes:
        process.kill()

    kill_deadline = time.monotonic() + _PREDICT_PIPELINE_KILL_TIMEOUT
    while time.monotonic() < kill_deadline and any(
        process.is_alive() for process in surviving_processes
    ):
        time.sleep(
            min(
                _PREDICT_PIPELINE_POLL_INTERVAL,
                max(0, kill_deadline - time.monotonic()),
            )
        )
    for process in surviving_processes:
        if process.is_alive():
            logger.warning("Prediction process %s survived kill().", process.pid)

    for process in processes:
        process.join(timeout=0)
    for thread in threads:
        thread.join(timeout=0)
        if thread.is_alive():
            logger.warning(
                "Daemon prediction thread %s did not terminate.", thread.name
            )
    for data_queue in queues:
        try:
            cancel_join_thread = getattr(data_queue, "cancel_join_thread", None)
            if cancel_join_thread is not None:
                cancel_join_thread()
            close = getattr(data_queue, "close", None)
            if close is not None:
                close()
        except Exception:
            logger.exception("Failed to close a prediction pipeline queue.")


def commit_prediction_output(
    writer: Any,
    pipeline_error: Optional[BaseException],
    expected_batches: int,
    written_batches: int,
    device: torch.device,
) -> None:
    """Commit prediction output only when every rank produced complete output.

    Every rank must call this, including ranks whose pipeline failed, because
    ``writer.close()`` participates in a collective for distributed writers.
    A rank without input is normal for uneven sharding, but an input that is
    empty on every rank is an error. Withholding ``close()`` prevents a commit
    but does not necessarily remove files already written by a local file
    writer.

    Args:
        writer (BaseWriter): output writer to commit.
        pipeline_error (BaseException, optional): failure of the local pipeline.
        expected_batches (int): batches submitted to the local pipeline.
        written_batches (int): batches written by the local writer.
        device (torch.device): device used to reduce the outcome across ranks.
    """
    succeeded = pipeline_error is None and expected_batches == written_batches
    global_succeeded = succeeded
    global_batches = expected_batches
    if dist.is_initialized() and dist.get_world_size() > 1:
        outcome = torch.tensor(
            [int(succeeded), expected_batches], dtype=torch.int64, device=device
        )
        dist.all_reduce(outcome, op=ReduceOp.SUM)
        global_succeeded = int(outcome[0].item()) == dist.get_world_size()
        global_batches = int(outcome[1].item())

    if pipeline_error is not None:
        raise pipeline_error
    if expected_batches != written_batches:
        raise RuntimeError(
            f"Prediction output is incomplete; submitted={expected_batches} "
            f"batches, written={written_batches} batches."
        )
    if not global_succeeded:
        raise RuntimeError(
            "Prediction pipeline failed or wrote incomplete output on another "
            "rank; output was not committed."
        )
    if global_batches == 0:
        raise RuntimeError("Prediction input is empty; output was not committed.")
    writer.close()
