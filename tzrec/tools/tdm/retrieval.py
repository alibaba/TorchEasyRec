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
import math
import os
from collections import OrderedDict
from typing import Dict, Optional

import numpy as np
import pyarrow as pa
import torch
from torch import distributed as dist
from torch.distributed import ReduceOp

from tzrec.constant import Mode
from tzrec.datasets.data_parser import DataParser
from tzrec.datasets.dataset import create_reader, create_writer
from tzrec.datasets.sampler import TDMPredictSampler
from tzrec.main import _create_features, _get_dataloader, init_process_group
from tzrec.protos.data_pb2 import DatasetType
from tzrec.utils import config_util
from tzrec.utils.logging_util import ProgressLogger, logger


def gen_data_for_tdm_retrieval(
    data_input_path: str,
    data_output_path: str,
    item_id_field: str,
    gt_item_id_field: str,
    root_id: int,
    writer_type: Optional[str] = None,
    save_to_output_path: bool = True,
) -> Dict[str, pa.Array]:
    """Transform eval data for tdm retrieval."""
    reader = create_reader(data_input_path, 256)
    eval_data_dict = {}
    for data_dict in reader.to_batches():
        for k, v in data_dict.items():
            if k not in eval_data_dict:
                eval_data_dict[k] = v
            else:
                eval_data_dict[k] = pa.concat_arrays([eval_data_dict[k], v])

    eval_data_dict[gt_item_id_field] = eval_data_dict[item_id_field]
    eval_data_dict[item_id_field] = pa.array(
        [root_id] * len(eval_data_dict[gt_item_id_field])
    )

    if save_to_output_path:
        writer = create_writer(data_output_path, writer_type)
        writer.write(eval_data_dict)

    return eval_data_dict


def update_data(
    input_data: pa.RecordBatch, sampled_data: Dict[str, pa.Array], num: int
) -> Dict[str, pa.Array]:
    """Update input data based on sampled data.

    Args:
        input_data (pa.RecordBatch): raw input data.
        sampled_data (dict): sampled data.
        num (int): each user's expansion count.

    Returns:
        updated data.
    """
    item_fea_fields = sampled_data.keys()
    all_fea_fields = set(input_data.column_names)
    user_fea_fields = all_fea_fields - item_fea_fields

    updated_data = {}
    for item_fea in item_fea_fields:
        updated_data[item_fea] = sampled_data[item_fea]

    for user_fea in user_fea_fields:
        _user_fea_array = input_data[user_fea]
        index = np.repeat(np.arange(len(_user_fea_array)), num)

        expand_user_fea = _user_fea_array.take(index)
        updated_data[user_fea] = expand_user_fea

    return updated_data


def tdm_retrieval(
    predict_input_path: str,
    predict_output_path: str,
    scripted_model_path: str,
    recall_num: int,
    gt_item_id_field: str,
    n_cluster: int = 2,
    reserved_columns: Optional[str] = None,
    batch_size: Optional[int] = None,
    is_profiling: bool = False,
    debug_level: int = 0,
    dataset_type: Optional[str] = None,
    writer_type: Optional[str] = None,
) -> None:
    """Evaluate EasyRec TDM model.

    Args:
        predict_input_path (str): inference input data path.
        predict_output_path (str): inference output data path.
        scripted_model_path (str): path to scripted model.
        recall_num (int): recall item num per user.
        n_cluster (int): tree cluster num.
        gt_item_id_field (str): ground true item id field.
        reserved_columns (str, optional): columns to reserved in output.
        batch_size (int, optional): predict batch_size.
        is_profiling (bool): profiling predict process or not.
        debug_level (int, optional): debug level for debug parsed inputs etc.
        dataset_type (str, optional): dataset type, default use the type in pipeline.
        writer_type (int, optional): data writer type, default will be same as
            dataset_type in data_config.
    """
    reserved_cols = None
    if reserved_columns is not None:
        reserved_cols = [x.strip() for x in reserved_columns.split(",")]

    pipeline_config = config_util.load_pipeline_config(
        os.path.join(scripted_model_path, "pipeline.config")
    )
    if batch_size:
        pipeline_config.data_config.batch_size = batch_size
    if dataset_type:
        pipeline_config.data_config.dataset_type = getattr(DatasetType, dataset_type)

    device, _ = init_process_group()
    sparse_dtype = torch.int32 if device.type == "cuda" else torch.int64

    is_rank_zero = int(os.environ.get("RANK", 0)) == 0
    is_local_rank_zero = int(os.environ.get("LOCAL_RANK", 0)) == 0

    data_config = pipeline_config.data_config
    data_config.ClearField("label_fields")
    data_config.drop_remainder = False
    # Build feature
    features = _create_features(list(pipeline_config.feature_configs), data_config)

    infer_dataloader = _get_dataloader(
        data_config,
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
    writer = create_writer(
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
    item_id_field = sampler_config.item_id_field
    max_level = len(sampler_config.layer_num_sample)
    first_recall_layer = int(math.log(2 * recall_num, n_cluster)) + 1

    dataset = infer_dataloader.dataset
    # pyre-ignore [16]
    fields = dataset.input_fields
    # pyre-ignore [29]
    pos_sampler = TDMPredictSampler(
        sampler_config, fields, batch_size, is_training=False
    )
    pos_sampler.init_cluster(num_client_per_rank=1)
    pos_sampler.launch_server()
    pos_sampler.init()
    i_step = 0

    num_class = pipeline_config.model_config.num_class
    pos_prob_name = "probs1" if num_class == 2 else "probs"

    recall = 0
    total = 0
    while True:
        try:
            # TBD (hongsheng.jhs): prefetch data on memcpy stream
            batch = next(infer_iterator)
            reserve_batch_record = batch.reserves.get()
            node_ids = reserve_batch_record[item_id_field]
            cur_batch_size = len(node_ids)

            expand_num = n_cluster**first_recall_layer
            pos_sampler.init_sampler(expand_num)

            for layer in range(first_recall_layer, max_level):
                sampled_result_dict = pos_sampler.get(node_ids)
                updated_inputs = update_data(
                    reserve_batch_record, sampled_result_dict, expand_num
                )
                parsed_inputs = parser.parse(updated_inputs)
                for key, value in parsed_inputs.items():
                    if value.dtype in [torch.int32, torch.int64]:
                        parsed_inputs[key] = value.to(sparse_dtype)
                predictions = model(parsed_inputs, device)

                probs = predictions[pos_prob_name].reshape(cur_batch_size, -1)

                if layer == max_level - 1:
                    k = recall_num
                    candidate_ids = (
                        updated_inputs[item_id_field]
                        .to_numpy()
                        .reshape(cur_batch_size, -1)
                    )
                    sort_prob_index = torch.argsort(-probs, dim=1).cpu().numpy()
                    sort_cand_ids = np.take_along_axis(
                        candidate_ids, sort_prob_index, axis=1
                    )
                    node_ids = []
                    for i in range(cur_batch_size):
                        _, unique_indices = np.unique(
                            sort_cand_ids[i], return_index=True
                        )
                        node_ids.append(
                            np.take(
                                sort_cand_ids[i], np.sort(unique_indices)[:k]
                            ).tolist()
                        )
                else:
                    k = 2 * recall_num
                    _, topk_indices_in_group = torch.topk(probs, k, dim=1)
                    topk_indices = (
                        topk_indices_in_group
                        + torch.arange(cur_batch_size)
                        .unsqueeze(1)
                        .to(topk_indices_in_group.device)
                        * expand_num
                    )
                    topk_indices = topk_indices.reshape(-1).cpu().numpy()
                    node_ids = updated_inputs[item_id_field].take(topk_indices)

                if layer == first_recall_layer:
                    pos_sampler.init_sampler(n_cluster)
                    expand_num = n_cluster * k

            output_dict = OrderedDict()
            if reserved_cols is not None:
                for c in reserved_cols:
                    output_dict[c] = reserve_batch_record[c]
            output_dict["recall_ids"] = pa.array(node_ids)
            writer.write(output_dict)

            # calculate precision and recall
            gt_node_ids = reserve_batch_record[gt_item_id_field].to_numpy()
            retrieval_result = np.any(np.equal(gt_node_ids[:, None], node_ids), axis=1)
            total += len(gt_node_ids)
            recall += np.sum(retrieval_result)

            if is_rank_zero:
                plogger.log(i_step)
            if is_profiling:
                prof.step()
            i_step += 1
        except StopIteration:
            break

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
        "--gt_item_id_field",
        type=str,
        default=None,
        help="grond true item id field",
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
    args, extra_args = parser.parse_known_args()

    tdm_retrieval(
        predict_input_path=args.predict_input_path,
        predict_output_path=args.predict_output_path,
        scripted_model_path=args.scripted_model_path,
        recall_num=args.recall_num,
        gt_item_id_field=args.gt_item_id_field,
        n_cluster=args.n_cluster,
        reserved_columns=args.reserved_columns,
        batch_size=args.batch_size,
        is_profiling=args.is_profiling,
        debug_level=args.debug_level,
        dataset_type=args.dataset_type,
    )
