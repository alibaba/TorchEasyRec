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

from typing import Any, List, Optional

import pyarrow as pa

from tzrec.datasets.dataset import create_reader
from tzrec.tools.tdm.gen_tree.tree_builder import TDMTreeNode, TreeBuilder


class TreeGenerator:
    """Generate tree and train file.

    Args:
        item_input_path(str): The file path where the item information is stored.
        item_id_field(str): The column name representing item_id in the file.
        cate_id_field(str): The column name representing the category in the file.
        attr_fields(List[str]): The column names representing the features in the file.
        tree_output_file(str): The tree output file.
        n_cluster(int): The branching factor of the nodes in the tree.
    """

    def __init__(
        self,
        item_input_path: str,
        item_id_field: str,
        cate_id_field: str,
        attr_fields: Optional[str] = None,
        raw_attr_fields: Optional[str] = None,
        tree_output_dir: Optional[str] = None,
        n_cluster: int = 2,
        **kwargs: Any,
    ) -> None:
        self.item_input_path = item_input_path
        self.item_id_field = item_id_field
        self.cate_id_field = cate_id_field
        self.attr_fields: Optional[List[str]] = None
        self.raw_attr_fields: Optional[List[str]] = None
        if attr_fields:
            self.attr_fields = [x.strip() for x in attr_fields.split(",")]
        if raw_attr_fields:
            self.raw_attr_fields = [x.strip() for x in raw_attr_fields.split(",")]
        self.tree_output_dir = tree_output_dir
        self.n_cluster = n_cluster

        self.dataset_kwargs = {}
        if "odps_data_quota_name" in kwargs:
            self.dataset_kwargs["quota_name"] = kwargs["odps_data_quota_name"]

    def generate(self, save_tree: bool = False) -> TDMTreeNode:
        """Generate tree."""
        item_fea = self._read()
        root = self._init_tree(item_fea, save_tree)
        return root

    def _read(self) -> List[TDMTreeNode]:
        leaf_nodes = []

        selected_cols = (
            {self.item_id_field, self.cate_id_field}
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
            ids = data_dict[self.item_id_field].cast(pa.int64())
            cates = data_dict[self.cate_id_field].cast(pa.string()).fill_null("")

            batch_tree_nodes = []
            for one_id, one_cate in zip(ids, cates):
                batch_tree_nodes.append(TDMTreeNode(item_id=one_id, cate=one_cate))

            if self.attr_fields is not None:
                for attr in self.attr_fields:
                    attr_data = data_dict[attr]
                    for i in range(len(batch_tree_nodes)):
                        batch_tree_nodes[i].attrs.append(attr_data[i])

            if self.raw_attr_fields is not None:
                for attr in self.raw_attr_fields:
                    attr_data = data_dict[attr]
                    for i in range(len(batch_tree_nodes)):
                        batch_tree_nodes[i].raw_attrs.append(attr_data[i])

            leaf_nodes.extend(batch_tree_nodes)

        return leaf_nodes

    def _init_tree(self, leaf_nodes: List[TDMTreeNode], save_tree: bool) -> TDMTreeNode:
        leaf_nodes.sort(key=lambda x: (x.cate, x.item_id))

        def gen_code(
            start: int, end: int, code: int, leaf_nodes: List[TDMTreeNode]
        ) -> None:
            if end <= start:
                return
            if end == start + 1:
                leaf_nodes[start].tree_code = code
                return
            for i in range(self.n_cluster):
                left = int(start + i * (end - start) / self.n_cluster)
                right = int(start + (i + 1) * (end - start) / self.n_cluster)
                gen_code(
                    left, right, self.n_cluster * code + self.n_cluster - i, leaf_nodes
                )

        gen_code(0, len(leaf_nodes), 0, leaf_nodes)

        builder = TreeBuilder(self.tree_output_dir, self.n_cluster)
        root = builder.build(leaf_nodes, save_tree)
        return root
