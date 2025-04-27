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

import os
from typing import Optional, Tuple

import faiss
import numpy as np
import numpy.typing as npt
import pyarrow as pa

from tzrec.datasets.dataset import create_reader
from tzrec.utils.logging_util import logger


def build_faiss_index(
    embedding_input_path: str,
    id_field: str,
    embedding_field: str,
    index_type: str,
    batch_size: int = 1024,
    ivf_nlist: int = 1000,
    hnsw_M: int = 32,
    hnsw_efConstruction: int = 200,
    reader_type: Optional[str] = None,
    odps_data_quota_name: str = "pay-as-you-go",
) -> Tuple[faiss.Index, npt.NDArray]:
    """Build faiss index.

    Args:
        embedding_input_path (str): path to embedding table.
        id_field (str): id field name in table.
        embedding_field (str): embedding field name in table.
        index_type (str): index type, available is ["IVFFlatIP", "HNSWFlatIP",
            "IVFFlatL2", "HNSWFlatL2"].
        batch_size (int): table read batch_size.
        ivf_nlist (int): nlist of IVFFlat index.
        hnsw_M (int): M of HNSWFlat index.
        hnsw_efConstruction (int): efConstruction of HNSWFlat index.
        reader_type (str, optional): specify the input path reader type,
            if we cannot infer from input_path.
        odps_data_quota_name (str): maxcompute storage api/tunnel data quota name.

    Returns:
        index (faiss.Index): faiss index.
        index_id_map (NDArray): a list of embedding ids for mapping
            continuous ids to origin id.
    """
    is_local_rank_zero = int(os.environ.get("LOCAL_RANK", 0)) == 0

    reader = create_reader(
        input_path=embedding_input_path,
        batch_size=batch_size,
        selected_cols=[id_field, embedding_field],
        reader_type=reader_type,
        quota_name=odps_data_quota_name,
    )

    index_id_map = []
    embeddings = []
    embedding_dim = None
    for i, data in enumerate(reader.to_batches()):
        eid_data = data[id_field]
        emb_data = data[embedding_field]

        index_id_map.extend(eid_data.tolist())
        if not pa.types.is_list(emb_data.type):
            emb_data = emb_data.cast(pa.string())
            emb_data = pa.compute.split_pattern(emb_data, ",")
        emb_data = emb_data.cast(pa.list_(pa.float32()), safe=False)
        embeddings.append(np.stack(emb_data.to_numpy(zero_copy_only=False)))
        if embedding_dim is None:
            embedding_dim = len(emb_data[0])

        if is_local_rank_zero and i % 100 == 0:
            logger.info(f"Reading {len(index_id_map)} embeddings...")

    if is_local_rank_zero:
        logger.info("Building faiss index...")
    if index_type.endswith("IP"):
        # pyre-ignore [16]
        quantizer = faiss.IndexFlatIP(embedding_dim)
        # pyre-ignore [16]
        metric_type = faiss.METRIC_INNER_PRODUCT
    elif index_type.endswith("L2"):
        # pyre-ignore [16]
        quantizer = faiss.IndexFlatL2(embedding_dim)
        # pyre-ignore [16]
        metric_type = faiss.METRIC_L2
    else:
        raise ValueError(f"Unknown metric_type in index {index_type}.")

    if index_type.startswith("IVFFlat"):
        # pyre-ignore [16]
        index = faiss.IndexIVFFlat(quantizer, embedding_dim, ivf_nlist, metric_type)
    elif index_type.startswith("HNSWFlat"):
        # pyre-ignore [16]
        index = faiss.IndexHNSWFlat(embedding_dim, hnsw_M, metric_type)
        index.hnsw.efConstruction = hnsw_efConstruction
    else:
        raise ValueError(f"Unknown index_type: {index_type}")

    # pyre-ignore [16]
    if faiss.get_num_gpus() > 0:
        # pyre-ignore [16]
        res = faiss.StandardGpuResources()
        # pyre-ignore [16]
        index = faiss.index_cpu_to_gpu(res, int(os.environ.get("LOCAL_RANK", 0)), index)

    embeddings = np.concatenate(embeddings)
    if index_type.startswith("IVFFlat"):
        index.train(embeddings)

    index.add(embeddings)
    if is_local_rank_zero:
        logger.info("Build embeddings finished.")

    return index, np.array(index_id_map, dtype=str)


def write_faiss_index(
    index: faiss.Index, index_id_map: npt.NDArray, output_dir: str
) -> None:
    """Write faiss index.

    Args:
        index (faiss.Index): faiss index.
        index_id_map (NDArray): a list of embedding ids
            for mapping continuous ids to origin id.
        output_dir (str): index output dir.
    """
    # pyre-ignore [16]
    faiss.write_index(index, os.path.join(output_dir, "faiss_index"))
    with open(os.path.join(output_dir, "id_mapping"), "w") as f:
        for eid in index_id_map:
            f.write(f"{eid}\n")


def search_faiss_index(
    index: faiss.Index, index_id_map: npt.NDArray, query: npt.NDArray, k: int
) -> Tuple[npt.NDArray, npt.NDArray]:
    """Search faiss index.

    Args:
        index (faiss.Index): faiss index.
        index_id_map (NDArray): a list of embedding ids for mapping
            continuous ids to origin id.
        query (NDArray): search query.
        k (int): top k.

    Returns:
        distances (NDArray): a array of distances.
        ids (NDArray): a array of ids.
    """
    distances, faiss_ids = index.search(query, k)
    ids = index_id_map[faiss_ids]
    return distances, ids
