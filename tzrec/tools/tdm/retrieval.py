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
import time
from collections import OrderedDict
from multiprocessing import Process, Queue
from threading import Thread
from typing import Dict, Optional, Tuple

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
from tzrec.main import _create_features, _get_dataloader, init_process_group
from tzrec.protos.data_pb2 import DatasetType
from tzrec.utils import config_util
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
) -> None:
    item_id_field = sampler._item_id_field
    sampler.init(worker_id)
    sampler.init_sampler(n_cluster)

    while True:
        record_batch_t, node_ids = in_queue.get()

        if record_batch_t is None:
            out_queue.put((None, None, None))
            time.sleep(10)
            break

        record_batch = record_batch_t.get()
        if is_first_layer:
            sampler.init_sampler(1)

            gt_node_ids = record_batch[item_id_field]
            cur_batch_size = len(gt_node_ids)
            node_ids = sampler.get(pa.array([-1] * cur_batch_size))[item_id_field]

            # skip layers before first_recall_layer
            sampler.init_sampler(n_cluster)
            for _ in range(1, first_recall_layer):
                sampled_result_dict = sampler.get(node_ids)
                node_ids = sampled_result_dict[item_id_field]

        sampled_result_dict = sampler.get(node_ids)
        updated_inputs = update_data(record_batch, sampled_result_dict)
        output_data = data_parser.parse(updated_inputs)
        batch = data_parser.to_batch(output_data, force_no_tile=True)

        out_queue.put((batch, record_batch_t, updated_inputs[item_id_field]))


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
    infer_dataloader = _get_dataloader(
        infer_data_config,
        features,
        predict_input_path,
        reserved_columns=["ALL_COLUMNS"],
        mode=Mode.PREDICT,
        debug_level=debug_level,
    )
    infer_iterator = iter(infer_dataloader)

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
                        np.take(sort_cand_ids[i], np.sort(unique_indices)[:k]).tolist()
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

    def _forward_loop(data_queue: Queue, pred_queue: Queue, layer_id: int) -> None:
        stop_cnt = 0
        while True:
            batch, record_batch_t, node_ids = data_queue.get()
            if batch is None:
                stop_cnt += 1
                if stop_cnt == num_worker_per_level:
                    for _ in range(num_worker_per_level):
                        pred_queue.put((None, None))
                    break
                else:
                    continue
            assert batch is not None
            pred = _forward(batch, record_batch_t, node_ids, layer_id)
            pred_queue.put(pred)

    def _write_loop(pred_queue: Queue, metric_queue: Queue) -> None:
        total = 0
        recall = 0
        while True:
            record_batch_t, node_ids = pred_queue.get()
            if record_batch_t is None:
                break

            output_dict = OrderedDict()
            reserve_batch_record = record_batch_t.get()
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
                    node_ids.to_numpy(),
                ),
                axis=1,
            )
            total += cur_batch_size
            recall += np.sum(retrieval_result)
        metric_queue.put((total, recall))

    in_queues = [Queue(maxsize=2) for _ in range(max_level - first_recall_layer + 1)]
    out_queues = [Queue(maxsize=2) for _ in range(max_level - first_recall_layer)]
    metric_queue = Queue(maxsize=1)

    data_p_list = []
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
                ),
            )
            p.start()
            data_p_list.append(p)

    forward_t_list = []
    for i in range(max_level - first_recall_layer):
        t = Thread(
            target=_forward_loop,
            args=(out_queues[i], in_queues[i + 1], i + first_recall_layer),
        )
        t.start()
        forward_t_list.append(t)

    write_t = Thread(
        target=_write_loop, args=(in_queues[len(in_queues) - 1], metric_queue)
    )
    write_t.start()

    i_step = 0
    while True:
        try:
            batch = next(infer_iterator)
            in_queues[0].put((batch.reserves, None))
            if is_local_rank_zero:
                plogger.log(i_step)
            if is_profiling:
                prof.step()
            i_step += 1
        except StopIteration:
            break

    for _ in range(num_worker_per_level):
        in_queues[0].put((None, None))
    for p in data_p_list:
        p.join()
    for t in forward_t_list:
        t.join()
    write_t.join()
    writer.close()

    total, recall = metric_queue.get()
    total_t = torch.tensor(total, device=device)
    recall_t = torch.tensor(recall, device=device)
    dist.all_reduce(total_t, op=ReduceOp.SUM)
    dist.all_reduce(recall_t, op=ReduceOp.SUM)
    # pyre-ignore [6]
    recall_ratio = recall_t.cpu().item() / total_t.cpu().item()

    if is_profiling:
        prof.stop()
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
        "--recall_num",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--n_cluster",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--num_worker_per_level",
        type=int,
        default=1,
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
        num_worker_per_level=args.num_worker_per_level,
    )
