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
from collections import OrderedDict
from multiprocessing import Event, Process, Queue
from threading import Thread
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import torch
from torch import distributed as dist
from torch.distributed import ReduceOp

from tzrec.constant import Mode
from tzrec.datasets.data_parser import DataParser
from tzrec.datasets.dataset import BaseWriter, create_writer
from tzrec.datasets.sampler import TDMPredictSampler
from tzrec.datasets.utils import Batch, RecordBatchTensor
from tzrec.main import _create_features, create_dataloader
from tzrec.protos.data_pb2 import DatasetType
from tzrec.utils import config_util, predict_util
from tzrec.utils.dist_util import init_process_group
from tzrec.utils.logging_util import ProgressLogger, logger


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
            record_batch_t, node_ids = predict_util.queue_get_interruptibly(
                in_queue, cancel_event, f"{stage}[{worker_id}] input"
            )

            if record_batch_t is None:
                predict_util.queue_put_interruptibly(
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

            predict_util.queue_put_interruptibly(
                out_queue,
                (batch, record_batch_t, updated_inputs[item_id_field]),
                cancel_event,
                f"{stage}[{worker_id}] output",
            )
    except predict_util.PredictPipelineCancelled:
        return
    except BaseException as error:
        predict_util.report_failure(
            failure_queue, cancel_event, stage, worker_id, error
        )
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
            batch, record_batch_t, node_ids = predict_util.queue_get_interruptibly(
                data_queue, cancel_event, f"{stage}[{layer_id}] input"
            )
            if batch is None:
                completed_producers += 1
                continue
            pred = forward_fn(batch, record_batch_t, node_ids, layer_id)
            predict_util.queue_put_interruptibly(
                pred_queue,
                pred,
                cancel_event,
                f"{stage}[{layer_id}] output",
            )
        for _ in range(downstream_consumer_count):
            predict_util.queue_put_interruptibly(
                pred_queue,
                (None, None),
                cancel_event,
                f"{stage}[{layer_id}] completion",
            )
    except predict_util.PredictPipelineCancelled:
        return
    except BaseException as error:
        predict_util.report_failure(failure_queue, cancel_event, stage, layer_id, error)


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
            record_batch_t, node_ids = predict_util.queue_get_interruptibly(
                pred_queue, cancel_event, f"{stage} input"
            )
            if record_batch_t is None:
                return
            write_fn(record_batch_t, node_ids)
    except predict_util.PredictPipelineCancelled:
        return
    except BaseException as error:
        predict_util.report_failure(failure_queue, cancel_event, stage, None, error)


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
        config_util.set_inference_batch_size(pipeline_config.data_config, batch_size)
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

    reserved_input_columns = [
        "ALL_EFFECTIVE_COLUMNS",
        data_config.tdm_sampler.item_id_field,
        *(reserved_cols or []),
    ]
    infer_data_config = copy.copy(data_config)
    infer_data_config.num_workers = 1
    infer_dataloader = create_dataloader(
        infer_data_config,
        features,
        predict_input_path,
        reserved_columns=reserved_input_columns,
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
        predict_util.cleanup_pipeline(
            data_p_list,
            forward_t_list,
            all_queues,
            cancel_event,
        )
        if is_profiling:
            try:
                prof.stop()
            except Exception:
                logger.exception("Failed to stop the retrieval profiler.")
        raise

    def _check_health() -> None:
        """Check background pipeline health from the main thread."""
        predict_util.check_pipeline_health(data_p_list, failure_queue, cancel_event)

    try:
        while True:
            try:
                batch = next(infer_iterator)
                reserve_batch_record = batch.reserves.get()
                if reserve_batch_record is None:
                    raise RuntimeError("TDM retrieval input has no reserved batch.")
                predict_util.queue_put_interruptibly(
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
                    record_batch_t, node_ids = predict_util.queue_get_interruptibly(
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
                predict_util.raise_background_failure(failure_queue)
            except StopIteration:
                break

        for _ in range(num_worker_per_level):
            predict_util.queue_put_interruptibly(
                in_queues[0],
                (None, None),
                cancel_event,
                "input completion",
                health_check=_check_health,
            )

        if write_t is None:
            record_batch_t, _ = predict_util.queue_get_interruptibly(
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
        predict_util.wait_for_pipeline(
            data_p_list, pipeline_threads, failure_queue, cancel_event
        )
        pipeline_succeeded = True
    except predict_util.PredictPipelineCancelled as error:
        predict_util.raise_background_failure(failure_queue, wait=True)
        raise RuntimeError(
            "TDM retrieval pipeline was cancelled without an error."
        ) from error
    finally:
        pipeline_threads = [*forward_t_list]
        if write_t is not None:
            pipeline_threads.append(write_t)
        predict_util.cleanup_pipeline(
            data_p_list,
            pipeline_threads,
            all_queues,
            cancel_event,
        )
        if is_profiling:
            try:
                prof.stop()
            except Exception:
                logger.exception("Failed to stop the retrieval profiler.")

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
    predict_util.validate_and_commit_writer(
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
