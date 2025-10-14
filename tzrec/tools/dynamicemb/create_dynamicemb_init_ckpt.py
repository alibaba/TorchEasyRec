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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from queue import Queue
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pyfg
from dynamicemb import dump_load
from dynamicemb.dynamicemb_config import DynamicEmbScoreStrategy
from dynamicemb.planner import DynamicEmbParameterConstraints

from tzrec.datasets.dataset import create_reader
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


_DYN_EMB_QUEUE_TYPE = Queue[Tuple[Optional[npt.NDArray], Optional[npt.NDArray]]]


def _read_loop(
    table_path: str,
    embedding_dim: int,
    separator: str,
    world_size: int,
    output_queues: List[_DYN_EMB_QUEUE_TYPE],
    reader_type: Optional[str],
    odps_data_quota_name: Optional[str],
) -> None:
    reader = create_reader(
        table_path,
        batch_size=20000,
        reader_type=reader_type,
        quota_name=odps_data_quota_name,
    )
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
    for data in reader.to_batches():
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

    for i in range(world_size):
        output_queues[i].put((None, None))


def _write_loop(
    input_queue: _DYN_EMB_QUEUE_TYPE,
    emb_name: str,
    rank: int,
    world_size: int,
    save_paths: List[str],
    need_dump_scores: List[bool],
) -> None:
    assert len(save_paths) == len(need_dump_scores)
    fkeys = []
    fvalues = []
    fscores = []
    for save_dir, need_dump_score in zip(save_paths, need_dump_scores):
        fkeys.append(
            open(
                dump_load.encode_key_file_path(save_dir, emb_name, rank, world_size),
                "wb",
            )
        )
        fvalues.append(
            open(
                dump_load.encode_value_file_path(save_dir, emb_name, rank, world_size),
                "wb",
            )
        )
        if need_dump_score:
            fscores.append(
                open(
                    dump_load.encode_score_file_path(
                        save_dir, emb_name, rank, world_size
                    ),
                    "wb",
                )
            )
        else:
            fscores.append(None)

    while True:
        keys, embs = input_queue.get()
        if keys is None or embs is None:
            break
        for fkey in fkeys:
            fkey.write(keys.astype(np.int64).tobytes())
        for fvalue in fvalues:
            fvalue.write(embs.astype(np.float32).tobytes())
        for fscore in fscores:
            if fscore is not None:
                fscore.write(np.zeros((len(keys),), dtype=np.uint64).tobytes())

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
    reader_type: str,
    odps_data_quota_name: str,
) -> None:
    q_list = [Queue(maxsize=3) for _ in range(args.world_size)]

    future_list = []
    with ThreadPoolExecutor(max_workers=1 + world_size) as executor:
        future_list.append(
            executor.submit(
                _read_loop,
                table_path=init_info.table_path,
                embedding_dim=init_info.embedding_dim,
                separator=init_info.separator,
                world_size=world_size,
                output_queues=q_list,
                reader_type=reader_type,
                odps_data_quota_name=odps_data_quota_name,
            )
        )
        for i in range(args.world_size):
            future_list.append(
                executor.submit(
                    _write_loop,
                    input_queue=q_list[i],
                    emb_name=emb_name,
                    rank=i,
                    world_size=world_size,
                    save_paths=init_info.save_paths,
                    need_dump_scores=init_info.need_dump_scores,
                )
            )
        for future in future_list:
            future.result()


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
        default="pay-as-you-go",
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
        emb_config = feature.emb_config
        # not support WIDE embedding now.
        if emb_config is not None and emb_config.name in dyemb_name_to_mod_options:
            if feature.config.dynamicemb.HasField("init_table"):
                save_paths = []
                need_dump_scores = []
                for mod_path, options in dyemb_name_to_mod_options[emb_config.name]:
                    # TODO: use mod_path as dynamicemb_load_table_names key,
                    # when dynamicemb support it.
                    dynamicemb_load_table_names[mod_path.rsplit(".", 1)[1]].append(
                        emb_config.name
                    )
                    save_path = os.path.join(ckpt_dir, "dynamicemb", mod_path)
                    os.makedirs(save_path, exist_ok=True)
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

    init_p_dict = {}
    for emb_name, init_info in dyemb_name_to_init_info.items():
        p = mp.Process(
            target=_init_one_emb,
            args=(
                emb_name,
                init_info,
                args.world_size,
                args.reader_type,
                args.odps_data_quota_name,
            ),
        )
        p.start()
        logger.info(f"Start init embedding [{emb_name}].")
        init_p_dict[emb_name] = p

    for emb_name, p in init_p_dict.items():
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"init worker for embedding [{emb_name}] failed.")
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
