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
import os
from collections import OrderedDict
from typing import Any, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import torch
from numpy.linalg import norm
from torch import distributed as dist

from tzrec.datasets.dataset import create_reader, create_writer
from tzrec.main import init_process_group
from tzrec.utils import faiss_util
from tzrec.utils.logging_util import logger


def batch_hitrate(
    src_ids: List[Any],  # pyre-ignore [2]
    recall_ids: npt.NDArray,
    gt_items: List[List[str]],
    max_num_interests: int,
    num_interests: Optional[List[int]] = None,
) -> Tuple[List[float], List[List[str]], float, float]:
    """Compute hitrate of a batch of src ids.

    Args:
        src_ids (list): trigger id, a list.
        recall_ids (NDArray): recalled ids by src_ids, a numpy array.
        gt_items (list): batch of ground truth item ids list, a list of list.
        max_num_interests (int): max number of interests.
        num_interests (list): some models have different number of interests.

    Returns:
        hitrates (list): hitrate of src_ids, a list.
        hit_ids (list): hit cases, a list of list.
        hits (int): total hit counts of a batch of src ids, a scalar.
        gt_count (int): total ground truth items num of a batch of src ids, a scalar.
    """
    hit_ids = []
    hitrates = []
    hits = 0.0
    gt_count = 0.0
    for idx, src_id in enumerate(src_ids):
        recall_id = recall_ids[idx]
        gt_item = set(gt_items[idx])
        gt_items_size = len(gt_item)
        hit_id_set = set()
        if gt_items_size == 0:  # just skip invalid record.
            logger.warning(
                "Id {:d} has no related items sequence, just skip.".format(src_id)
            )
            continue
        for interest_id in range(max_num_interests):
            if num_interests and interest_id >= num_interests[idx]:
                break
            hit_id_set |= set(recall_id[interest_id]) & gt_item
        hit_count = float(len(hit_id_set))
        hitrates.append(hit_count / gt_items_size)
        hits += hit_count
        gt_count += gt_items_size
        hit_ids.append(list(hit_id_set))
    return hitrates, hit_ids, hits, gt_count


def interest_merge(
    user_emb: npt.NDArray,
    recall_distances: npt.NDArray,
    recall_ids: npt.NDArray,
    top_k: int,
    num_interests: int,
    index_type: str,
) -> Tuple[npt.NDArray, npt.NDArray]:
    """Merge the recall results of different interests.

    Args:
        user_emb (NDArray): user embedding.
        recall_distances (NDArray): recall distances.
        recall_ids (NDArray): recall ids.
        top_k (int): top k candidates.
        num_interests (int): number of interests.
        index_type (str): index type.

    Returns:
        recall_distances (NDArray): merged recall distances.
        recall_ids(NDArray): merged recall ids.
    """
    # In case of all-zero query vector, the corresponding knn results
    # should be removed since faiss returns random target for all-zero query.
    if index_type.endswith("IP"):
        recall_distances = np.minimum(
            recall_distances,
            np.tile(
                (
                    (norm(user_emb, axis=-1, keepdims=True) != 0.0).astype("float") * 2
                    - 1
                )
                * 1e32,
                (1, top_k),
            ),
        )
    else:  # L2 distance
        recall_distances = np.maximum(
            recall_distances,
            np.tile(
                (
                    (norm(user_emb, axis=-1, keepdims=True) == 0.0).astype("float") * 2
                    - 1
                )
                * 1e32,
                (1, top_k),
            ),
        )
    recall_distances_flat = recall_distances.reshape(
        [-1, num_interests * recall_distances.shape[-1]]
    )
    recall_ids_flat = recall_ids.reshape(
        [-1, args.num_interests * recall_ids.shape[-1]]
    )

    sort_idx = np.argsort(recall_distances_flat, axis=-1)
    if index_type.endswith("IP"):  # inner product should be sorted in descending order
        sort_idx = sort_idx[:, ::-1]

    recall_distances_flat_sorted = recall_distances_flat[
        np.arange(recall_distances_flat.shape[0])[:, np.newaxis], sort_idx
    ]
    recall_ids_flat_sorted = recall_ids_flat[
        np.arange(recall_ids_flat.shape[0])[:, np.newaxis], sort_idx
    ]

    # get unique candidates
    recall_distances_flat_sorted_pad = np.concatenate(
        [
            recall_distances_flat_sorted,
            np.zeros((recall_distances_flat_sorted.shape[0], 1)),
        ],
        axis=-1,
    )
    # compute diff value between consecutive distances
    recall_distances_diff = (
        recall_distances_flat_sorted_pad[:, 0:-1]
        - recall_distances_flat_sorted_pad[:, 1:]
    )

    if index_type.endswith("IP"):
        pad_value = -1e32
    else:
        pad_value = 1e32

    # zero diff positions are dulipcated values, so we pad them with a pad value
    recall_distances_unique = np.where(
        recall_distances_diff == 0, pad_value, recall_distances_flat_sorted
    )
    # sort again to get the unique candidates, duplicated values are -1e32(IP)
    # or 1e32(L2), so they are moved to the end
    sort_idx_new = np.argsort(recall_distances_unique, axis=-1)
    if index_type.endswith("IP"):
        sort_idx_new = sort_idx_new[:, ::-1]

    recall_distances = recall_distances_flat_sorted[
        np.arange(recall_distances_flat_sorted.shape[0])[:, np.newaxis],
        sort_idx_new[:, 0:top_k],
    ]
    recall_ids = recall_ids_flat_sorted[
        np.arange(recall_ids_flat_sorted.shape[0])[:, np.newaxis],
        sort_idx_new[:, 0:top_k],
    ]

    recall_distances = recall_distances.reshape([-1, 1, recall_distances.shape[-1]])
    recall_ids = recall_ids.reshape([-1, 1, recall_distances.shape[-1]])

    return recall_distances, recall_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--user_gt_input",
        type=str,
        default=None,
        help="Path to user groudtruth & embedding table with columns [request_id, "
        "gt_items, user_tower_emb]",
    )
    parser.add_argument(
        "--item_embedding_input",
        type=str,
        default=None,
        help="Path to item embedding table with columns [item_id, item_tower_emb]",
    )
    parser.add_argument(
        "--total_hitrate_output",
        type=str,
        default=None,
        help="Path to hitrate table with columns [hitrate]",
    )
    parser.add_argument(
        "--hitrate_details_output",
        type=str,
        default=None,
        help="Path to hitrate detail table with columns [id, topk_ids, "
        "topk_dists, hitrate, hit_ids]",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="batch size.",
    )
    parser.add_argument(
        "--index_type",
        type=str,
        default="IVFFlatIP",
        choices=["IVFFlatIP", "IVFFlatL2"],
        help="index type.",
    )
    parser.add_argument(
        "--top_k", type=int, default=200, help="use top k search result."
    )
    parser.add_argument(
        "--topk_across_interests",
        action="store_true",
        default=False,
        help="select topk candidates across all interests.",
    )
    parser.add_argument(
        "--ivf_nlist", type=int, default=1000, help="nlist of IVFFlat index."
    )
    parser.add_argument(
        "--ivf_nprobe", type=int, default=800, help="nprobe of IVFFlat index."
    )
    parser.add_argument(
        "--item_id_field",
        type=str,
        default="item_id",
        help="item id field name in item embedding table.",
    )
    parser.add_argument(
        "--item_embedding_field",
        type=str,
        default="item_tower_emb",
        help="item embedding field name in item embedding table.",
    )
    parser.add_argument(
        "--request_id_field",
        type=str,
        default="request_id",
        help="request id field name in user gt table.",
    )
    parser.add_argument(
        "--gt_items_field",
        type=str,
        default="gt_items",
        help="gt items field name in user gt table.",
    )
    parser.add_argument(
        "--user_embedding_field",
        type=str,
        default="user_tower_emb",
        help="user embedding field name in user gt table.",
    )
    parser.add_argument(
        "--num_interests",
        type=int,
        default=1,
        help="max user embedding num for each request.",
    )
    parser.add_argument(
        "--num_interests_field",
        type=str,
        default=None,
        help="valid user embedding num for each request in user gt table.",
    )
    parser.add_argument(
        "--reader_type",
        type=str,
        default=None,
        choices=["OdpsReader", "CsvReader", "ParquetReader"],
        help="input path reader type.",
    )
    parser.add_argument(
        "--writer_type",
        type=str,
        default=None,
        choices=["OdpsWriter", "CsvWriter", "ParquetWriter"],
        help="output path writer type.",
    )
    parser.add_argument(
        "--odps_data_quota_name",
        type=str,
        default="pay-as-you-go",
        help="maxcompute storage api/tunnel data quota name.",
    )
    args, extra_args = parser.parse_known_args()

    device, backend = init_process_group()
    worker_id = int(os.environ.get("RANK", 0))
    num_workers = int(os.environ.get("WORLD_SIZE", 1))
    is_rank_zero = int(os.environ.get("RANK", 0)) == 0
    is_local_rank_zero = int(os.environ.get("LOCAL_RANK", 0)) == 0

    selected_cols = [
        args.request_id_field,
        args.gt_items_field,
        args.user_embedding_field,
    ]
    if args.num_interests_field is not None:
        selected_cols.append(args.num_interests_field)
    reader = create_reader(
        input_path=args.user_gt_input,
        batch_size=args.batch_size,
        selected_cols=selected_cols,
        reader_type=args.reader_type,
        quota_name=args.odps_data_quota_name,
    )
    writer_type = args.writer_type
    if not writer_type:
        # pyre-ignore [16]
        writer_type = reader.__class__.__name__.replace("Reader", "Writer")

    index, index_id_map = faiss_util.build_faiss_index(
        args.item_embedding_input,
        id_field=args.item_id_field,
        embedding_field=args.item_embedding_field,
        index_type=args.index_type,
        batch_size=args.batch_size,
        ivf_nlist=args.ivf_nlist,
        reader_type=args.reader_type,
        odps_data_quota_name=args.odps_data_quota_name,
    )
    index.nprobe = args.ivf_nprobe

    details_writer = None
    if args.hitrate_details_output:
        details_writer = create_writer(
            args.hitrate_details_output,
            writer_type,
            quota_name=args.odps_data_quota_name,
        )

    if args.topk_across_interests:
        print("args.topk_across_interests is True")

    # calculate hitrate
    total_count = 0
    total_hits = 0.0
    total_gt_count = 0.0
    for i, data in enumerate(reader.to_batches(worker_id, num_workers)):
        request_id = data[args.request_id_field]
        gt_items = data[args.gt_items_field]
        if not pa.types.is_list(gt_items.type):
            gt_items = gt_items.cast(pa.string())
            gt_items = pa.compute.split_pattern(gt_items, ",")

        user_emb = data[args.user_embedding_field]
        user_emb_type = user_emb.type
        if pa.types.is_list(user_emb_type):
            if pa.types.is_list(user_emb_type.value_type):
                user_emb = user_emb.values
        else:
            user_emb = user_emb.cast(pa.string())
            if args.num_interests > 1:
                user_emb = pa.compute.split_pattern(user_emb, ";").values
            user_emb = pa.compute.split_pattern(user_emb, ",")
        user_emb = user_emb.cast(pa.list_(pa.float32()), safe=False)
        user_emb = np.stack(user_emb.to_numpy(zero_copy_only=False))

        recall_distances, recall_ids = faiss_util.search_faiss_index(
            index, index_id_map, user_emb, args.top_k
        )

        # pick topk candidates across all interests
        if args.topk_across_interests:
            recall_distances, recall_ids = interest_merge(
                user_emb,
                recall_distances,
                recall_ids,
                args.top_k,
                args.num_interests,
                args.index_type,
            )
        else:  # pick topk candidates for each interest
            recall_distances = recall_distances.reshape(
                [-1, args.num_interests, recall_distances.shape[-1]]
            )
            recall_ids = recall_ids.reshape(
                [-1, args.num_interests, recall_distances.shape[-1]]
            )

        num_interests_per_req = None
        if args.num_interests_field:
            num_interests_per_req = data[args.num_interests_field]

        hitrates, hit_ids, hits, gt_count = batch_hitrate(
            request_id.tolist(),
            recall_ids,
            gt_items.tolist(),
            args.num_interests if not args.topk_across_interests else 1,
            num_interests_per_req.tolist() if num_interests_per_req else None,
        )

        total_hits += hits
        total_gt_count += gt_count
        total_count += len(request_id)

        if is_local_rank_zero and i % 10 == 0:
            logger.info(f"Compute {total_count} hitrates...")

        if details_writer:
            details_writer.write(
                OrderedDict(
                    [
                        ("id", request_id),
                        (
                            "topk_ids",
                            pa.array(
                                recall_ids.tolist(),
                                type=pa.list_(pa.list_((pa.string()))),
                            ),
                        ),
                        (
                            "topk_dists",
                            pa.array(
                                recall_distances.tolist(),
                                type=pa.list_(pa.list_(pa.float32())),
                            ),
                        ),
                        ("hitrate", pa.array(hitrates)),
                        ("hit_ids", pa.array(hit_ids, type=pa.list_(pa.string()))),
                    ]
                )
            )

    if details_writer:
        details_writer.close()

    # reduce hitrate
    total_hits_t = torch.tensor(total_hits, device=device)
    total_gt_count_t = torch.tensor(total_gt_count, device=device)
    dist.all_reduce(total_hits_t)
    dist.all_reduce(total_gt_count_t)

    # output hitrate
    total_hitrate = (total_hits_t / total_gt_count_t).cpu().item()
    if is_rank_zero:
        logger.info(f"Total hitrate: {total_hitrate}")
        if args.hitrate_details_output:
            hitrate_writer = create_writer(
                args.total_hitrate_output,
                writer_type,
                quota_name=args.odps_data_quota_name,
            )
            hitrate_writer.write({"hitrate": pa.array([total_hitrate])})
            hitrate_writer.close()
