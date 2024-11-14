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

import collections
import multiprocessing as mp
import os
import time
from multiprocessing.connection import Connection
from typing import Any, List, Optional

import numpy as np
import numpy.typing as npt
import pyarrow as pa
from sklearn.cluster import KMeans

from tzrec.datasets.dataset import create_reader
from tzrec.tools.tdm.gen_tree import tree_builder
from tzrec.tools.tdm.gen_tree.tree_builder import TDMTreeNode
from tzrec.utils.logging_util import logger


class TreeCluster:
    """Cluster based on emb vec.

    Args:
        item_input_path (str): The file path where the item information is stored.
        item_id_field (str): The column name representing item_id in the file.
        attr_fields (List[str]): The column names representing the features in the file.
        output_dir (str): The output file.
        parallel (int): The number of CPU cores for parallel processing.
        n_cluster (int): The branching factor of the nodes in the tree.
    """

    def __init__(
        self,
        item_input_path: str,
        item_id_field: str,
        attr_fields: Optional[str] = None,
        raw_attr_fields: Optional[str] = None,
        output_dir: Optional[str] = None,
        embedding_field: str = "item_emb",
        parallel: int = 16,
        n_cluster: int = 2,
        **kwargs: Any,
    ) -> None:
        self.item_input_path = item_input_path
        self.mini_batch = 1024

        self.data = None
        self.leaf_nodes = None

        self.parallel = parallel
        self.queue = None
        self.timeout = 5
        self.codes = None
        self.output_dir = output_dir
        self.n_clusters = n_cluster

        self.item_id_field = item_id_field
        self.attr_fields = []
        self.raw_attr_fields = []
        if attr_fields:
            self.attr_fields = [x.strip() for x in attr_fields.split(",")]
        if raw_attr_fields:
            self.raw_attr_fields = [x.strip() for x in raw_attr_fields.split(",")]

        self.embedding_field = embedding_field

        self.dataset_kwargs = {}
        if "odps_data_quota_name" in kwargs:
            self.dataset_kwargs["quota_name"] = kwargs["odps_data_quota_name"]

    def _read(self) -> None:
        t1 = time.time()
        data = list()

        self.leaf_nodes = []

        selected_cols = (
            {self.item_id_field, self.embedding_field}
            | set(self.attr_fields)
            | set(self.raw_attr_fields)
        )
        reader = create_reader(
            self.item_input_path,
            4096,
            selected_cols=list(selected_cols),
            **self.dataset_kwargs,
        )

        for data_dict in reader.to_batches():
            ids = data_dict[self.item_id_field].cast(pa.int64()).to_pylist()
            data += data_dict[self.embedding_field].to_pylist()

            batch_tree_nodes = []
            for one_id in ids:
                batch_tree_nodes.append(TDMTreeNode(item_id=one_id))

            for attr in self.attr_fields:
                attr_data = data_dict[attr]
                for i in range(len(batch_tree_nodes)):
                    batch_tree_nodes[i].attrs.append(attr_data[i])

            for attr in self.raw_attr_fields:
                attr_data = data_dict[attr]
                for i in range(len(batch_tree_nodes)):
                    batch_tree_nodes[i].raw_attrs.append(attr_data[i])

            self.leaf_nodes.extend(batch_tree_nodes)

        if isinstance(data[0], str):
            data = [eval(i) for i in data]
        self.data = np.array(data)
        t2 = time.time()

        logger.info(
            "Read data done, {} records read, elapsed: {}".format(
                len(self.leaf_nodes), t2 - t1
            )
        )

    def train(self, save_tree: bool = False) -> TDMTreeNode:
        """Cluster data."""
        self._read()
        # The (code, index) stored in the queue represent the node number
        # in the current class’s tree and the index of the item belonging
        # to this class, respectively.
        queue = mp.Queue()
        queue.put((0, np.arange(len(self.leaf_nodes))))
        processes = []
        pipes = []
        for _ in range(self.parallel):
            parent_conn, child_conn = mp.Pipe()
            p = mp.Process(target=self._train, args=(child_conn, queue))
            processes.append(p)
            pipes.append(parent_conn)
            p.start()

        self.codes = np.zeros((len(self.leaf_nodes),), dtype=np.int64)
        for pipe in pipes:
            codes = pipe.recv()
            for i in range(len(codes)):
                if codes[i] > 0:
                    self.leaf_nodes[i].tree_code = codes[i]

        for p in processes:
            p.join()

        assert queue.empty()
        builder = tree_builder.TreeBuilder(self.output_dir, self.n_clusters)
        root = builder.build(self.leaf_nodes, save_tree)
        return root

    def _train(self, pipe: Connection, queue: mp.Queue) -> None:
        last_size = -1
        catch_time = 0
        processed = False
        code = np.zeros((len(self.leaf_nodes),), dtype=np.int64)
        parent_code = None
        index = None
        while True:
            for _ in range(3):
                try:
                    parent_code, index = queue.get(timeout=self.timeout)
                except Exception as _:
                    index = None
                if index is not None:
                    break

            if index is None:
                if processed and (last_size <= self.mini_batch or catch_time >= 3):
                    logger.info("Process {} exits".format(os.getpid()))
                    break
                else:
                    logger.info(
                        "Got empty job, pid: {}, time: {}".format(
                            os.getpid(), catch_time
                        )
                    )
                    catch_time += 1
                    continue

            processed = True
            catch_time = 0
            last_size = len(index)
            if last_size <= self.mini_batch:
                self._mini_batch(parent_code, index, code)
            else:
                start = time.time()
                sub_index = self._cluster(index)
                logger.info(
                    "Train iteration done, parent_code:{}, "
                    "data size: {}, elapsed time: {}".format(
                        parent_code, len(index), time.time() - start
                    )
                )
                self.timeout = int(0.4 * self.timeout + 0.6 * (time.time() - start))
                if self.timeout < 5:
                    self.timeout = 5

                for i in range(self.n_clusters):
                    if len(sub_index[i]) > 1:
                        queue.put((self.n_clusters * parent_code + i + 1, sub_index[i]))

        process_count = 0
        for c in code:
            if c > 0:
                process_count += 1
        logger.info("Process {} process {} items".format(os.getpid(), process_count))
        pipe.send(code)

    def _mini_batch(
        self, parent_code: int, index: npt.NDArray, code: npt.NDArray
    ) -> None:
        dq = collections.deque()
        dq.append((parent_code, index))
        batch_size = len(index)
        tstart = time.time()
        while dq:
            parent_code, index = dq.popleft()

            if len(index) <= self.n_clusters:
                for i in range(len(index)):
                    code[index[i]] = self.n_clusters * parent_code + i + 1
                continue

            sub_index = self._cluster(index)
            for i in range(self.n_clusters):
                if len(sub_index[i]) > 1:
                    dq.append((self.n_clusters * parent_code + i + 1, sub_index[i]))
                elif len(sub_index[i]) > 0:
                    for j in range(len(sub_index[i])):
                        code[sub_index[i][j]] = (
                            self.n_clusters * parent_code + i + j + 1
                        )
        logger.info(
            "Minibatch, batch size: {}, elapsed: {}".format(
                batch_size, time.time() - tstart
            )
        )

    def _cluster(self, index: npt.NDArray) -> List[npt.NDArray]:
        data = self.data[index]
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=0).fit(data)
        labels = kmeans.labels_
        sub_indices = []
        remain_index = []
        ave_num = int(len(index) / self.n_clusters)

        for i in range(self.n_clusters):
            sub_i = np.where(labels == i)[0]
            sub_index = index[sub_i]
            if len(sub_index) <= ave_num:
                sub_indices.append(sub_index)
            else:
                distances = kmeans.transform(data[sub_i])[:, i]
                sorted_index = sub_index[np.argsort(distances)]
                sub_indices.append(sorted_index[:ave_num])
                remain_index.extend(list(sorted_index[ave_num:]))
        idx = 0
        remain_index = np.array(remain_index)

        # reblance index
        while idx < self.n_clusters and len(remain_index) > 0:
            if len(sub_indices[idx]) >= ave_num:
                idx += 1
            else:
                diff = min(len(remain_index), ave_num - len(sub_indices[idx]))
                remain_data = self.data[remain_index]
                distances = kmeans.transform(remain_data)[:, idx]
                sorted_index = remain_index[np.argsort(distances)]

                # Supplement the data by sorting the distances
                # to the current cluster centers in ascending order.
                sub_indices[idx] = np.append(
                    sub_indices[idx], np.array(sorted_index[0:diff])
                )
                remain_index = sorted_index[diff:]
                idx += 1
        if len(remain_index) > 0:
            sub_indices[0] = np.append(sub_indices[0], remain_index)

        return sub_indices
