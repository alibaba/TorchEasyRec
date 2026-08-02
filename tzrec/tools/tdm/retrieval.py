# Copyright (c) 2024, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import copy
import math
import os
import queue as queue_lib
import time
import traceback
from collections import OrderedDict
from dataclasses import dataclass
from multiprocessing import Event, Process, Queue
from threading import Thread
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow as pa
import torch
from torch import distributed as dist
from torch.distributed import ReduceOp

from tzrec.constant import PREDICT_QUEUE_TIMEOUT, Mode
from tzrec.datasets.data_parser import DataParser
from tzrec.datasets.dataset import BaseWriter, create_writer
from tzrec.datasets.sampler import TDMPredictSampler
from tzrec.datasets.utils import Batch, RecordBatchTensor
from tzrec.main import _create_features, create_dataloader
from tzrec.protos.data_pb2 import DatasetType
from tzrec.utils import config_util
from tzrec.utils.dist_util import init_process_group
from tzrec.utils.logging_util import ProgressLogger, logger

_PIPELINE_POLL_INTERVAL = 1.0
_PIPELINE_CLEANUP_TIMEOUT = 10.0
_PIPELINE_TERMINATE_TIMEOUT = 5.0
_PIPELINE_KILL_TIMEOUT = 5.0


@dataclass(frozen=True)
class _PipelineFailure:
    """Serializable failure raised by a background pipeline stage."""

    stage: str
    worker_id: Optional[int]
    exception_type: str
    message: str
    traceback: str


class _PipelineCancelled(RuntimeError):
    """Signal that a pipeline queue operation was cancelled."""


class _PipelineStageError(RuntimeError):
    """Failure propagated from a background pipeline stage."""

    def __init__(self, failure: _PipelineFailure) -> None:
        worker = "" if failure.worker_id is None else f"[{failure.worker_id}]"
        super().__init__(
            f"TDM retrieval {failure.stage}{worker} failed with "
            f"{failure.exception_type}: {failure.message}\n{failure.traceback}"
        )


def _report_failure(
    failure_queue: Queue,
    cancel_event: Any,
    stage: str,
    worker_id: Optional[int],
    error: BaseException,
) -> None:
    """Report the active exception and cancel the pipeline."""
    failure = _PipelineFailure(
        stage=stage,
        worker_id=worker_id,
        exception_type=type(error).__name__,
        message=str(error),
        traceback=traceback.format_exc(),
    )
    try:
        failure_queue.put_nowait(failure)
    except Exception:
        logger.exception("Failed to report TDM retrieval pipeline failure.")
    finally:
        cancel_event.set()


def _raise_background_failure(failure_queue: Queue, wait: bool = False) -> None:
    """Raise the oldest reported pipeline failure, if present."""
    try:
        failure = failure_queue.get(timeout=_PIPELINE_POLL_INTERVAL if wait else 0)
    except queue_lib.Empty:
        return
    raise _PipelineStageError(failure)


def _check_pipeline_health(
    processes: Sequence[Process], failure_queue: Queue, cancel_event: Any
) -> None:
    """Raise reported failures or unexpected child process exits."""
    _raise_background_failure(failure_queue)
    if cancel_event.is_set():
        _raise_background_failure(failure_queue, wait=True)
        raise RuntimeError("TDM retrieval pipeline was cancelled without an error.")
    failed_processes = [
        p for p in processes if p.exitcode is not None and p.exitcode != 0
    ]
    if failed_processes:
        cancel_event.set()
        _raise_background_failure(failure_queue, wait=True)
        details = ", ".join(
            f"pid={p.pid}, exitcode={p.exitcode}" for p in failed_processes
        )
        raise RuntimeError(f"TDM retrieval data worker failed: {details}.")


def _queue_get(
    data_queue: Queue,
    cancel_event: Any,
    stage: str,
    timeout: float = PREDICT_QUEUE_TIMEOUT,
    health_check: Optional[Callable[[], None]] = None,
) -> Any:
    """Get a queue item while remaining responsive to cancellation."""
    deadline = time.monotonic() + timeout
    while not cancel_event.is_set():
        if health_check is not None:
            health_check()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"TDM retrieval {stage} stalled waiting for queue input "
                f"for {timeout} seconds."
            )
        try:
            return data_queue.get(timeout=min(_PIPELINE_POLL_INTERVAL, remaining))
        except queue_lib.Empty:
            continue
    raise _PipelineCancelled(f"TDM retrieval {stage} was cancelled.")


def _queue_put(
    data_queue: Queue,
    item: Any,
    cancel_event: Any,
    stage: str,
    timeout: float = PREDICT_QUEUE_TIMEOUT,
    health_check: Optional[Callable[[], None]] = None,
) -> None:
    """Put a queue item while remaining responsive to cancellation."""
    deadline = time.monotonic() + timeout
    while not cancel_event.is_set():
        if health_check is not None:
            health_check()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"TDM retrieval {stage} stalled waiting for queue capacity "
                f"for {timeout} seconds."
            )
        try:
            data_queue.put(item, timeout=min(_PIPELINE_POLL_INTERVAL, remaining))
            return
        except queue_lib.Full:
            continue
    raise _PipelineCancelled(f"TDM retrieval {stage} was cancelled.")


def update_data(
    input_data: pa.RecordBatch, sampled_data: Dict[str, pa.Array]
) -> Dict[str, pa.Array]:
    """Update input data based on sampled data.

    Args:
        input_data (pa.RecordBatch): raw input data.
        sampled_data (dict): sampled data.

    Returns:
        updated data.
    """
    item_fea_fields = sampled_data.keys()
    all_fea_fields = set(input_data.column_names)
    user_fea_fields = all_fea_fields - item_fea_fields

    updated_data = {}
    for item_fea in item_fea_fields:
        updated_data[item_fea] = sampled_data[item_fea]

    item_field_0 = list(item_fea_fields)[0]
    expand_num = len(sampled_data[item_field_0]) // len(input_data[item_field_0])
    for user_fea in user_fea_fields:
        _user_fea_array = input_data[user_fea]
        index = np.repeat(np.arange(len(_user_fea_array)), expand_num)

        expand_user_fea = _user_fea_array.take(index)
        updated_data[user_fea] = expand_user_fea

    return updated_data


def _tdm_predict_data_worker(
    sampler: TDMPredictSampler,
    data_parser: DataParser,
    first_recall_layer: int,
    n_cluster: int,
    in_queue: Queue,
    out_queue: Queue,
    is_first_layer: bool,
    worker_id: int,
    cancel_event: Any,
    failure_queue: Queue,
) -> None:
    stage = "data worker"
    try:
        item_id_field = sampler._item_id_field
        sampler.init(worker_id)
        sampler.init_sampler(n_cluster)

        while True:
            record_batch_t, node_ids = _queue_get(
                in_queue, cancel_event, f"{stage}[{worker_id}] input"
            )

            if record_batch_t is None:
                _queue_put(
                    out_queue,
                    (None, None, None),
                    cancel_event,
                    f"{stage}[{worker_id}] completion",
                )
                break

            record_batch = record_batch_t.get()
            if is_first_layer:
                sampler.init_sampler(1)

                gt_node_ids = record_batch[item_id_field]
                cur_batch_size = len(gt_node_ids)
                node_ids = sampler.get(
                    {item_id_field: pa.array([-1] * cur_batch_size)}
                )[item_id_field]

                # skip layers before first_recall_layer
                sampler.init_sampler(n_cluster)
                for _ in range(1, first_recall_layer):
                    sampled_result_dict = sampler.get({item_id_field: node_ids})
                    node_ids = sampled_result_dict[item_id_field]

            sampled_result_dict = sampler.get({item_id_field: node_ids})
            updated_inputs = update_data(record_batch, sampled_result_dict)
            output_data = data_parser.parse(updated_inputs)
            batch = data_parser.to_batch(output_data, force_no_tile=True)

            _queue_put(
                out_queue,
                (batch, record_batch_t, updated_inputs[item_id_field]),
                cancel_event,
                f"{stage}[{worker_id}] output",
            )
    except _PipelineCancelled:
        return
    except BaseException as error:
        _report_failure(failure_queue, cancel_event, stage, worker_id, error)
        raise


def _forward_loop(
    data_queue: Queue,
    pred_queue: Queue,
    layer_id: int,
    producer_count: int,
    downstream_consumer_count: int,
    forward_fn: Callable[
        [Batch, RecordBatchTensor, pa.Array, int],
        Tuple[RecordBatchTensor, pa.Array],
    ],
    cancel_event: Any,
    failure_queue: Queue,
) -> None:
    """Forward one tree layer and propagate normal completion downstream."""
    stage = "forward"
    try:
        completed_producers = 0
        while completed_producers < producer_count:
            batch, record_batch_t, node_ids = _queue_get(
                data_queue, cancel_event, f"{stage}[{layer_id}] input"
            )
            if batch is None:
                completed_producers += 1
                continue
            pred = forward_fn(batch, record_batch_t, node_ids, layer_id)
            _queue_put(
                pred_queue,
                pred,
                cancel_event,
                f"{stage}[{layer_id}] output",
            )
        for _ in range(downstream_consumer_count):
            _queue_put(
                pred_queue,
                (None, None),
                cancel_event,
                f"{stage}[{layer_id}] completion",
            )
    except _PipelineCancelled:
        return
    except BaseException as error:
        _report_failure(failure_queue, cancel_event, stage, layer_id, error)


def _write_loop(
    pred_queue: Queue,
    write_fn: Callable[[RecordBatchTensor, pa.Array], None],
    cancel_event: Any,
    failure_queue: Queue,
) -> None:
    """Write completed retrieval batches until normal completion."""
    stage = "writer"
    try:
        while True:
            record_batch_t, node_ids = _queue_get(
                pred_queue, cancel_event, f"{stage} input"
            )
            if record_batch_t is None:
                return
            write_fn(record_batch_t, node_ids)
    except _PipelineCancelled:
        return
    except BaseException as error:
        _report_failure(failure_queue, cancel_event, stage, None, error)


def _wait_for_pipeline(
    processes: Sequence[Process],
    threads: Sequence[Thread],
    failure_queue: Queue,
    cancel_event: Any,
    timeout: float = PREDICT_QUEUE_TIMEOUT,
) -> None:
    """Wait for normal pipeline completion while monitoring failures."""
    deadline = time.monotonic() + timeout
    while True:
        _check_pipeline_health(processes, failure_queue, cancel_event)

        alive_processes = [p for p in processes if p.is_alive()]
        alive_threads = [t for t in threads if t.is_alive()]
        if not alive_processes and not alive_threads:
            break
        if time.monotonic() >= deadline:
            process_ids = [p.pid for p in alive_processes]
            thread_names = [t.name for t in alive_threads]
            raise TimeoutError(
                "TDM retrieval pipeline stalled during completion; "
                f"processes={process_ids}, threads={thread_names}."
            )
        time.sleep(_PIPELINE_POLL_INTERVAL)

    for process in processes:
        process.join(timeout=0)
    for thread in threads:
        thread.join(timeout=0)
    _raise_background_failure(failure_queue)


def _cleanup_pipeline(
    processes: Sequence[Process],
    threads: Sequence[Thread],
    queues: Sequence[Queue],
    cancel_event: Any,
) -> None:
    """Cancel and reap pipeline components within bounded deadlines."""
    cancel_event.set()
    graceful_deadline = time.monotonic() + _PIPELINE_CLEANUP_TIMEOUT
    while time.monotonic() < graceful_deadline:
        if not any(p.is_alive() for p in processes) and not any(
            t.is_alive() for t in threads
        ):
            break
        time.sleep(_PIPELINE_POLL_INTERVAL)

    surviving_processes = [p for p in processes if p.is_alive()]
    for process in surviving_processes:
        try:
            process.terminate()
        except Exception:
            logger.exception("Failed to terminate retrieval process %s.", process.pid)

    terminate_deadline = time.monotonic() + _PIPELINE_TERMINATE_TIMEOUT
    while time.monotonic() < terminate_deadline and any(
        p.is_alive() for p in surviving_processes
    ):
        time.sleep(_PIPELINE_POLL_INTERVAL)

    surviving_processes = [p for p in surviving_processes if p.is_alive()]
    for process in surviving_processes:
        try:
            process.kill()
        except Exception:
            logger.exception("Failed to kill retrieval process %s.", process.pid)

    kill_deadline = time.monotonic() + _PIPELINE_KILL_TIMEOUT
    while time.monotonic() < kill_deadline and any(
        p.is_alive() for p in surviving_processes
    ):
        time.sleep(_PIPELINE_POLL_INTERVAL)
    for process in surviving_processes:
        if process.is_alive():
            logger.warning("Retrieval process %s survived SIGKILL.", process.pid)

    for process in processes:
        try:
            process.join(timeout=0)
        except Exception:
            logger.exception("Failed to reap retrieval process %s.", process.pid)
    for thread in threads:
        try:
            thread.join(timeout=0)
            if thread.is_alive():
                logger.warning("Retrieval thread %s did not terminate.", thread.name)
        except Exception:
            logger.exception("Failed to join retrieval thread %s.", thread.name)
    for data_queue in queues:
        try:
            cancel_join_thread = getattr(data_queue, "cancel_join_thread", None)
            if cancel_join_thread is not None:
                cancel_join_thread()
            close = getattr(data_queue, "close", None)
            if close is not None:
                close()
        except Exception:
            logger.exception("Failed to close a retrieval pipeline queue.")


def _validate_and_commit_writer(
    writer: BaseWriter,
    pipeline_succeeded: bool,
    expected_batches: int,
    expected_rows: int,
    written_batches: int,
    written_rows: int,
) -> None:
    """Commit output only after non-empty completeness validation."""
    if not pipeline_succeeded:
        raise RuntimeError("TDM retrieval pipeline failed; output was not committed.")
    if expected_rows == 0:
        raise RuntimeError("TDM retrieval input is empty; output was not committed.")
    if expected_batches != written_batches or expected_rows != written_rows:
        raise RuntimeError(
            "TDM retrieval output is incomplete; "
            f"submitted={expected_batches} batches/{expected_rows} rows, "
            f"written={written_batches} batches/{written_rows} rows."
        )
    writer.close()


def tdm_retrieval(
    predict_input_path: str,
    predict_output_path: str,
    scripted_model_path: str,
    recall_num: int,
    n_cluster: int = 2,
    reserved_columns: Optional[str] = None,
    batch_size: Optional[int] = None,
    is_profiling: bool = False,
    debug_level: int = 0,
    dataset_type: Optional[str] = None,
    writer_type: Optional[str] = None,
    num_worker_per_level: int = 1,
) -> None:
    """Evaluate EasyRec TDM model.

    Args:
        predict_input_path (str): inference input data path.
        predict_output_path (str): inference output data path.
        scripted_model_path (str): path to scripted model.
        recall_num (int): recall item num per user.
        n_cluster (int): tree cluster num.
        reserved_columns (str, optional): columns to reserved in output.
        batch_size (int, optional): predict batch_size.
        is_profiling (bool): profiling predict process or not.
        debug_level (int, optional): debug level for debug parsed inputs etc.
        dataset_type (str, optional): dataset type, default use the type in pipeline.
        writer_type (int, optional): data writer type, default will be same as
            dataset_type in data_config.
        num_worker_per_level (int): num data generate worker per tree level.
    """
    reserved_cols: Optional[list[str]] = None
    if reserved_columns is not None:
        reserved_cols = [x.strip() for x in reserved_columns.split(",")]

    pipeline_config = config_util.load_pipeline_config(
        os.path.join(scripted_model_path, "pipeline.config")
    )
    if batch_size:
        pipeline_config.data_config.batch_size = batch_size
    if dataset_type:
        pipeline_config.data_config.dataset_type = getattr(DatasetType, dataset_type)

    device_and_backend = init_process_group()
    device: torch.device = device_and_backend[0]
    sparse_dtype: torch.dtype = torch.int32 if device.type == "cuda" else torch.int64

    is_rank_zero = int(os.environ.get("RANK", 0)) == 0
    is_local_rank_zero = int(os.environ.get("LOCAL_RANK", 0)) == 0

    data_config = pipeline_config.data_config
    data_config.ClearField("label_fields")
    data_config.drop_remainder = False
    # Build feature
    features = _create_features(list(pipeline_config.feature_configs), data_config)

    infer_data_config = copy.copy(data_config)
    infer_data_config.num_workers = 1
    infer_dataloader = create_dataloader(
        infer_data_config,
        features,
        predict_input_path,
        reserved_columns=["ALL_COLUMNS"],
        mode=Mode.PREDICT,
        debug_level=debug_level,
    )
    infer_iterator = infer_dataloader.get_iterator()  # pyre-ignore[16]

    if writer_type is None:
        writer_type = DatasetType.Name(data_config.dataset_type).replace(
            "Dataset", "Writer"
        )
    writer: BaseWriter = create_writer(
        predict_output_path,
        writer_type,
        quota_name=data_config.odps_data_quota_name,
    )

    # disable jit compile， as it compile too slow now.
    if "PYTORCH_TENSOREXPR_FALLBACK" not in os.environ:
        os.environ["PYTORCH_TENSOREXPR_FALLBACK"] = "2"
    model: torch.jit.ScriptModule = torch.jit.load(
        os.path.join(scripted_model_path, "scripted_model.pt"), map_location=device
    )
    model.eval()

    if is_local_rank_zero:
        plogger = ProgressLogger(desc="Predicting", miniters=10)

    if is_profiling:
        if is_rank_zero:
            logger.info(str(model))
        prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=10, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                os.path.join(scripted_model_path, "predict_trace")
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        prof.start()

    parser = DataParser(features)

    sampler_config = pipeline_config.data_config.tdm_sampler
    item_id_field: str = sampler_config.item_id_field
    max_level: int = len(sampler_config.layer_num_sample)
    first_recall_layer = int(math.ceil(math.log(2 * n_cluster * recall_num, n_cluster)))

    dataset = infer_dataloader.dataset
    # pyre-ignore [16]
    fields = dataset.input_fields
    # pyre-ignore [29]
    predict_sampler = TDMPredictSampler(
        sampler_config, fields, batch_size, is_training=False
    )
    predict_sampler.init_cluster(
        num_client_per_rank=(max_level - first_recall_layer) * num_worker_per_level
    )
    predict_sampler.launch_server()

    num_class = pipeline_config.model_config.num_class
    pos_prob_name: str = "probs1" if num_class == 2 else "probs"

    expected_batches = 0
    expected_rows = 0
    written_batches = 0
    total = 0
    recall = 0

    def _forward(
        batch: Batch,
        record_batch_t: RecordBatchTensor,
        node_ids: pa.Array,
        layer_id: int,
    ) -> Tuple[RecordBatchTensor, pa.Array]:
        with torch.no_grad():
            parsed_inputs = batch.to_dict(sparse_dtype=sparse_dtype)
            # when predicting with a model exported using INPUT_TILE,
            #  we set the batch size tensor to 1 to disable tiling.
            parsed_inputs["batch_size"] = torch.tensor(1, dtype=torch.int64)
            predictions = model(parsed_inputs, device)

            gt_node_ids = record_batch_t.get()[item_id_field]
            cur_batch_size = len(gt_node_ids)
            probs = predictions[pos_prob_name].reshape(cur_batch_size, -1)
            if layer_id == max_level - 1:
                k = recall_num
                candidate_ids = node_ids.to_numpy(zero_copy_only=False).reshape(
                    cur_batch_size, -1
                )
                sort_prob_index = torch.argsort(-probs, dim=1).cpu().numpy()
                sort_cand_ids = np.take_along_axis(
                    candidate_ids, sort_prob_index, axis=1
                )
                node_ids = []
                for i in range(cur_batch_size):
                    _, unique_indices = np.unique(sort_cand_ids[i], return_index=True)
                    node_ids.append(
                        np.take(sort_cand_ids[i], np.sort(unique_indices)[:k])
                    )
                node_ids = pa.array(node_ids)
            else:
                k = 2 * recall_num
                _, topk_indices_in_group = torch.topk(probs, k, dim=1)
                topk_indices = topk_indices_in_group + torch.arange(
                    cur_batch_size, device=device
                ).unsqueeze(1) * probs.size(1)
                topk_indices = topk_indices.reshape(-1).cpu().numpy()
                node_ids = node_ids.take(topk_indices)

            return record_batch_t, node_ids

    def _write(record_batch_t: RecordBatchTensor, node_ids: pa.Array) -> None:
        nonlocal written_batches
        nonlocal total
        nonlocal recall
        output_dict = OrderedDict()
        reserve_batch_record = record_batch_t.get()
        if reserve_batch_record is None:
            raise RuntimeError("TDM retrieval output lost its reserved input batch.")
        gt_node_ids = reserve_batch_record[item_id_field]
        cur_batch_size = len(gt_node_ids)
        if reserved_cols is not None:
            for c in reserved_cols:
                output_dict[c] = reserve_batch_record[c]
        output_dict["recall_ids"] = node_ids
        writer.write(output_dict)

        # calculate precision and recall
        retrieval_result = np.any(
            np.equal(
                gt_node_ids.to_numpy(zero_copy_only=False)[:, None],
                np.array(node_ids.to_pylist()),
            ),
            axis=1,
        )
        written_batches += 1
        total += cur_batch_size
        recall += np.sum(retrieval_result)

    in_queues = [Queue(maxsize=2) for _ in range(max_level - first_recall_layer + 1)]
    out_queues = [Queue(maxsize=2) for _ in range(max_level - first_recall_layer)]
    failure_queue = Queue()
    cancel_event = Event()
    all_queues = [*in_queues, *out_queues, failure_queue]

    data_p_list: List[Process] = []
    forward_t_list: List[Thread] = []
    write_t: Optional[Thread] = None
    pipeline_succeeded = False
    i_step = 0
    try:
        for i in range(max_level - first_recall_layer):
            for j in range(num_worker_per_level):
                p = Process(
                    target=_tdm_predict_data_worker,
                    args=(
                        predict_sampler,
                        parser,
                        first_recall_layer,
                        n_cluster,
                        in_queues[i],
                        out_queues[i],
                        i == 0,
                        i * num_worker_per_level + j,
                        cancel_event,
                        failure_queue,
                    ),
                )
                p.start()
                data_p_list.append(p)

        for i in range(max_level - first_recall_layer):
            downstream_consumer_count = (
                num_worker_per_level if i < max_level - first_recall_layer - 1 else 1
            )
            t = Thread(
                target=_forward_loop,
                args=(
                    out_queues[i],
                    in_queues[i + 1],
                    i + first_recall_layer,
                    num_worker_per_level,
                    downstream_consumer_count,
                    _forward,
                    cancel_event,
                    failure_queue,
                ),
                name=f"tdm-forward-{i + first_recall_layer}",
                daemon=True,
            )
            t.start()
            forward_t_list.append(t)
    except BaseException:
        _cleanup_pipeline(
            data_p_list,
            forward_t_list,
            all_queues,
            cancel_event,
        )
        if is_profiling:
            prof.stop()
        raise

    def _check_health() -> None:
        """Check background pipeline health from the main thread."""
        _check_pipeline_health(data_p_list, failure_queue, cancel_event)

    try:
        while True:
            try:
                batch = next(infer_iterator)
                reserve_batch_record = batch.reserves.get()
                if reserve_batch_record is None:
                    raise RuntimeError("TDM retrieval input has no reserved batch.")
                _queue_put(
                    in_queues[0],
                    (batch.reserves, None),
                    cancel_event,
                    "input producer",
                    health_check=_check_health,
                )
                expected_batches += 1
                expected_rows += len(reserve_batch_record)
                if i_step == 0:
                    # Initialize distributed writers synchronously on the first batch.
                    record_batch_t, node_ids = _queue_get(
                        in_queues[-1],
                        cancel_event,
                        "first output",
                        health_check=_check_health,
                    )
                    if record_batch_t is None:
                        raise RuntimeError(
                            "TDM retrieval completed before producing its first output."
                        )
                    _write(record_batch_t, node_ids)
                    write_t = Thread(
                        target=_write_loop,
                        args=(in_queues[-1], _write, cancel_event, failure_queue),
                        name="tdm-writer",
                        daemon=True,
                    )
                    write_t.start()
                if is_local_rank_zero:
                    plogger.log(i_step)
                if is_profiling:
                    prof.step()
                i_step += 1
                _raise_background_failure(failure_queue)
            except StopIteration:
                break

        for _ in range(num_worker_per_level):
            _queue_put(
                in_queues[0],
                (None, None),
                cancel_event,
                "input completion",
                health_check=_check_health,
            )

        if write_t is None:
            record_batch_t, _ = _queue_get(
                in_queues[-1],
                cancel_event,
                "empty input completion",
                health_check=_check_health,
            )
            if record_batch_t is not None:
                raise RuntimeError("Empty TDM retrieval produced unexpected output.")

        pipeline_threads = [*forward_t_list]
        if write_t is not None:
            pipeline_threads.append(write_t)
        _wait_for_pipeline(data_p_list, pipeline_threads, failure_queue, cancel_event)
        pipeline_succeeded = True
    except _PipelineCancelled as error:
        _raise_background_failure(failure_queue, wait=True)
        raise RuntimeError(
            "TDM retrieval pipeline was cancelled without an error."
        ) from error
    finally:
        pipeline_threads = [*forward_t_list]
        if write_t is not None:
            pipeline_threads.append(write_t)
        _cleanup_pipeline(
            data_p_list,
            pipeline_threads,
            all_queues,
            cancel_event,
        )
        if is_profiling:
            prof.stop()

    if not pipeline_succeeded:
        raise RuntimeError("TDM retrieval pipeline did not complete successfully.")

    metric_t = torch.tensor(
        [expected_batches, expected_rows, written_batches, total, recall],
        dtype=torch.int64,
        device=device,
    )
    dist.all_reduce(metric_t, op=ReduceOp.SUM)
    global_expected_batches = int(metric_t[0].cpu().item())
    global_expected_rows = int(metric_t[1].cpu().item())
    global_written_batches = int(metric_t[2].cpu().item())
    global_written_rows = int(metric_t[3].cpu().item())
    global_recall = int(metric_t[4].cpu().item())
    _validate_and_commit_writer(
        writer,
        pipeline_succeeded,
        global_expected_batches,
        global_expected_rows,
        global_written_batches,
        global_written_rows,
    )
    recall_ratio = global_recall / global_written_rows

    if is_rank_zero:
        logger.info(f"Retrieval Finished. Recall:{recall_ratio}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scripted_model_path",
        type=str,
        default=None,
        help="scripted model to be evaled, if not specified, use the checkpoint",
    )
    parser.add_argument(
        "--predict_input_path",
        type=str,
        default=None,
        help="inference data input path",
    )
    parser.add_argument(
        "--predict_output_path",
        type=str,
        default=None,
        help="inference data output path",
    )
    parser.add_argument(
        "--reserved_columns",
        type=str,
        default=None,
        help="column names to reserved in output",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="predict batch size, default will use batch size in config.",
    )
    parser.add_argument(
        "--is_profiling",
        action="store_true",
        default=False,
        help="profiling predict progress.",
    )
    parser.add_argument(
        "--debug_level",
        type=int,
        default=0,
        help="debug level for debug parsed inputs etc.",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default=None,
        help="dataset type, default will use dataset type in config.",
    )
    parser.add_argument(
        "--writer_type",
        type=str,
        default=None,
        help="data writer type, default will be same as dataset_type in data_config.",
    )
    parser.add_argument(
        "--recall_num", type=int, default=200, help="recall item num per user."
    )
    parser.add_argument("--n_cluster", type=int, default=2, help="tree cluster num.")
    parser.add_argument(
        "--num_worker_per_level",
        type=int,
        default=1,
        help="num data generate worker per tree level.",
    )
    args, extra_args = parser.parse_known_args()

    tdm_retrieval(
        predict_input_path=args.predict_input_path,
        predict_output_path=args.predict_output_path,
        scripted_model_path=args.scripted_model_path,
        recall_num=args.recall_num,
        n_cluster=args.n_cluster,
        reserved_columns=args.reserved_columns,
        batch_size=args.batch_size,
        is_profiling=args.is_profiling,
        debug_level=args.debug_level,
        dataset_type=args.dataset_type,
        writer_type=args.writer_type,
        num_worker_per_level=args.num_worker_per_level,
    )
