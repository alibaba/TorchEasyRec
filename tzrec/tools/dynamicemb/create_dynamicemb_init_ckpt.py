# Copyright (c) 2025, Alibaba Group;
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
import json
import multiprocessing as mp
import os
from collections import defaultdict
from dataclasses import dataclass
from queue import Queue
from typing import List, Optional

import numpy as np
import pyfg
from dynamicemb.batched_dynamicemb_tables import (
    encode_checkpoint_file_path,
    encode_meta_json_file_path,
)
from dynamicemb.dynamicemb_config import DynamicEmbScoreStrategy
from dynamicemb.planner import DynamicEmbParameterConstraints

from tzrec.datasets.dataset import BaseReader, create_reader
from tzrec.features.feature import MAX_HASH_BUCKET_SIZE
from tzrec.main import _create_features, _create_model
from tzrec.models.model import TrainWrapper
from tzrec.utils import config_util
from tzrec.utils.logging_util import logger


@dataclass
class DynamicEmbTableInitInfo:
    """Dynamic Embedding Info need Init by Table."""

    # init table path
    table_path: str
    # embedding save dir
    save_paths: List[str]
    # need dump embedding score or not.
    need_dump_scores: List[bool]
    # embedding dimension
    embedding_dim: int
    # embedding field separator
    separator: str


def _read_loop(
    reader: BaseReader,
    embedding_dim: int,
    separator: str,
    world_size: int,
    output_queues: List[mp.Queue],
    worker_id: int,
    worker_num: int,
) -> None:
    id_field, embedding_field = reader.schema[0].name, reader.schema[1].name
    fg_json = [
        {
            "feature_name": id_field,
            "feature_type": "id_feature",
            "expression": f"item:{id_field}",
            "value_type": "string",
            "value_dim": 1,
            "need_prefix": False,
            # do hash bucket for consistency with training
            "hash_bucket_size": MAX_HASH_BUCKET_SIZE,
        },
        {
            "feature_name": embedding_field,
            "feature_type": "raw_feature",
            "expression": f"item:{embedding_field}",
            "value_type": "float",
            "value_dim": embedding_dim,
            "separator": separator,
        },
    ]
    # pyre-ignore [16]
    fg_handler = pyfg.FgArrowHandler({"features": fg_json}, 2)

    read_rows = 0
    for i_batch, data in enumerate(reader.to_batches(worker_id, worker_num)):
        fg_output, status = fg_handler.process_arrow(data)
        assert status.ok(), status.message()
        keys = fg_output[id_field].np_values
        embs = fg_output[embedding_field].dense_values

        keys_div_ws = keys % world_size
        for i in range(world_size):
            mask = keys_div_ws == i
            if np.any(mask):
                mask_keys = keys[mask]
                mask_embs = embs[mask]
                output_queues[i].put((mask_keys, mask_embs))

        read_rows += len(keys)
        if i_batch % 1000 == 0 and i_batch > 0:
            logger.info(f"reader worker [{worker_id}] finish {read_rows} rows.")

    for i in range(world_size):
        output_queues[i].put((None, None))


def _write_loop(
    input_queue: mp.Queue,
    emb_name: str,
    rank: int,
    world_size: int,
    save_paths: List[str],
    need_dump_scores: List[bool],
    reader_worker_num: int,
) -> None:
    assert len(save_paths) == len(need_dump_scores)
    fkeys = []
    fvalues = []
    fscores = []
    for save_dir, need_dump_score in zip(save_paths, need_dump_scores):
        fkeys.append(
            open(
                encode_checkpoint_file_path(
                    save_dir, emb_name, rank, world_size, "keys"
                ),
                "wb",
            )
        )
        fvalues.append(
            open(
                encode_checkpoint_file_path(
                    save_dir, emb_name, rank, world_size, "values"
                ),
                "wb",
            )
        )
        if need_dump_score:
            fscores.append(
                open(
                    encode_checkpoint_file_path(
                        save_dir, emb_name, rank, world_size, "scores"
                    ),
                    "wb",
                )
            )
        else:
            fscores.append(None)

    exit_cnt = 0
    prev_logger_rows = 0
    write_rows = 0
    while True:
        keys, embs = input_queue.get()
        if keys is None or embs is None:
            exit_cnt += 1
            if exit_cnt == reader_worker_num:
                break
            continue
        for fkey in fkeys:
            fkey.write(keys.astype(np.int64).tobytes())
        for fvalue in fvalues:
            fvalue.write(embs.astype(np.float32).tobytes())
        for fscore in fscores:
            if fscore is not None:
                fscore.write(np.zeros((len(keys),), dtype=np.uint64).tobytes())
        write_rows += len(keys)
        if write_rows - prev_logger_rows >= 1000000:
            prev_logger_rows = write_rows
            logger.info(f"writer worker [{rank}] finish {write_rows} rows.")

    for fkey in fkeys:
        fkey.close()
    for fvalue in fvalues:
        fvalue.close()
    for fscore in fscores:
        if fscore is not None:
            fscore.close()


def _init_one_emb(
    emb_name: str,
    init_info: DynamicEmbTableInitInfo,
    world_size: int,
    reader_worker_num: Optional[int],
    reader_type: str,
    odps_data_quota_name: str,
) -> None:
    q_list = [mp.Queue(maxsize=3) for _ in range(args.world_size)]

    read_p_list = []
    reader = create_reader(
        init_info.table_path,
        batch_size=20000,
        reader_type=reader_type,
        quota_name=odps_data_quota_name,
        compression="ZSTD",
        rebalance=False,  # we do not rebalance parquet file for each worker
    )
    if reader_worker_num is None:
        reader_worker_num = world_size
        cpu_count = os.cpu_count()
        if cpu_count is not None:
            reader_worker_num = max(cpu_count - world_size, reader_worker_num)
    num_files = reader.num_files()
    if num_files is not None:
        reader_worker_num = min(reader_worker_num, num_files)

    for i in range(reader_worker_num):
        p = mp.Process(
            target=_read_loop,
            args=(
                reader,
                init_info.embedding_dim,
                init_info.separator,
                world_size,
                q_list,
                i,
                reader_worker_num,
            ),
        )
        p.start()
        read_p_list.append(p)

    write_p_list = []
    for i in range(args.world_size):
        write_p = mp.Process(
            target=_write_loop,
            args=(
                q_list[i],
                emb_name,
                i,
                world_size,
                init_info.save_paths,
                init_info.need_dump_scores,
                reader_worker_num,
            ),
        )
        write_p.start()
        write_p_list.append(write_p)

    for i, p in enumerate(read_p_list):
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"reader worker[{i}] for embedding [{emb_name}] failed.")
    for i, p in enumerate(write_p_list):
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"writer worker[{i}] for embedding [{emb_name}] failed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create init dynamic embedding ckpt from embedding init table."
    )
    parser.add_argument(
        "--pipeline_config_path",
        type=str,
        default=None,
        help="Path to pipeline config file.",
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=1,
        help="training world size.",
    )
    parser.add_argument(
        "--reader_worker_num",
        type=int,
        default=None,
        help="reader worker number, default is cpu count.",
    )
    parser.add_argument(
        "--reader_type",
        type=str,
        default=None,
        choices=["OdpsReader", "CsvReader", "ParquetReader"],
        help="input path reader type.",
    )
    parser.add_argument(
        "--save_dir", type=str, default="./", help="init ckpt save directory."
    )
    parser.add_argument(
        "--separator", type=str, default=",", help="embedding separator."
    )
    parser.add_argument(
        "--odps_data_quota_name",
        type=str,
        default=None,
        help="maxcompute storage api/tunnel data quota name.",
    )
    args, extra_args = parser.parse_known_args()

    ckpt_dir = os.path.join(args.save_dir, "model.ckpt-0")
    pipeline_config = config_util.load_pipeline_config(args.pipeline_config_path)

    data_config = pipeline_config.data_config
    # Build feature
    features = _create_features(list(pipeline_config.feature_configs), data_config)

    # Build model
    model = _create_model(
        pipeline_config.model_config,
        features,
        list(data_config.label_fields),
        sample_weights=list(data_config.sample_weight_fields),
    )
    model = TrainWrapper(model)

    # Extract module with dynamic embedding
    dyemb_name_to_mod_options = defaultdict(list)
    q = Queue()
    q.put(("", model))
    while not q.empty():
        path, m = q.get()
        if hasattr(m, "parameter_constraints"):
            for fqn, const in m.parameter_constraints(path).items():
                if isinstance(const, DynamicEmbParameterConstraints):
                    mod_path, emb_name = fqn.rsplit(".", 1)
                    dyemb_name_to_mod_options[emb_name].append(
                        ("model." + mod_path, const.dynamicemb_options)
                    )

        else:
            for name, child in m.named_children():
                q.put((f"{path}{name}.", child))

    # Extract dynamic embedding to init_table
    dyemb_name_to_init_info = {}
    dynamicemb_load_table_names = defaultdict(list)
    for feature in features:
        # not support WIDE embedding now.
        if feature.config.HasField("dynamicemb"):
            if feature.config.dynamicemb.HasField("init_table"):
                emb_config = feature.emb_config
                assert (
                    emb_config is not None
                    and emb_config.name in dyemb_name_to_mod_options
                )
                save_paths = []
                need_dump_scores = []
                for mod_path, options in dyemb_name_to_mod_options[emb_config.name]:
                    dynamicemb_load_table_names[mod_path].append(emb_config.name)
                    save_path = os.path.join(ckpt_dir, "dynamicemb", mod_path)
                    os.makedirs(save_path, exist_ok=True)
                    with open(
                        encode_meta_json_file_path(save_path, emb_config.name), "w"
                    ) as f:
                        f.write(json.dumps({}))
                    save_paths.append(save_path)
                    need_dump_scores.append(
                        options.score_strategy != DynamicEmbScoreStrategy.TIMESTAMP
                    )

                dyemb_name_to_init_info[emb_config.name] = DynamicEmbTableInitInfo(
                    table_path=feature.config.dynamicemb.init_table,
                    save_paths=save_paths,
                    need_dump_scores=need_dump_scores,
                    embedding_dim=emb_config.embedding_dim,
                    separator=args.separator,
                )

    for emb_name, init_info in dyemb_name_to_init_info.items():
        logger.info(f"Start init embedding [{emb_name}].")
        _init_one_emb(
            emb_name,
            init_info,
            args.world_size,
            args.reader_worker_num,
            args.reader_type,
            args.odps_data_quota_name or data_config.odps_data_quota_name,
        )
        logger.info(f"Finish init embedding [{emb_name}].")

    with open(os.path.join(ckpt_dir, "meta"), "w") as f:
        json.dump(
            {
                "load_model": False,
                "load_optim": False,
                "dynamicemb_load_table_names": dynamicemb_load_table_names,
                "dynamicemb_load_optim": False,
            },
            f,
        )
