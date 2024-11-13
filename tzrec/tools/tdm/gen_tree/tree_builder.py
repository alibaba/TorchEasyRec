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

import math
import os
import pickle
from collections import Counter
from typing import List, Optional

import pyarrow as pa
import pyarrow.compute as pc
from anytree import NodeMixin
from anytree.exporter.dictexporter import DictExporter

from tzrec.utils.logging_util import logger


class BaseClass(object):
    """Tree base class."""

    pass


class TDMTreeNode(BaseClass, NodeMixin):
    """TDM tree node."""

    def __init__(
        self,
        tree_code: int = -1,
        item_id: Optional[int] = None,
        cate: Optional[str] = None,
        attrs: Optional[List[pa.Scalar]] = None,
        raw_attrs: Optional[List[pa.Scalar]] = None,
        parent: Optional["TDMTreeNode"] = None,
        children: Optional[List["TDMTreeNode"]] = None,
    ) -> None:
        super(TDMTreeNode, self).__init__()
        self.tree_code = tree_code
        self.item_id = item_id
        self.cate = cate
        self.attrs = attrs or []
        self.raw_attrs = raw_attrs or []

        self.attrs_list = []
        self.raw_attrs_list = []

        self.parent = parent
        if children:
            self.children = children

    def set_parent(self, parent: "TDMTreeNode") -> None:
        """Set parent."""
        self.parent = parent

    def set_children(self, children: List["TDMTreeNode"]) -> None:
        """Set children."""
        self.children = children


class TreeBuilder:
    """Build tree base codes.

    Args:
        output_dir(str): tree output file.
        n_cluster(int): The branching factor of the nodes in the tree.
    """

    def __init__(self, output_dir: Optional[str] = ".", n_cluster: int = 2) -> None:
        self.output_dir = output_dir
        self.n_cluster = n_cluster

    def build(
        self,
        leaf_nodes: List[TDMTreeNode],
        save_tree: bool = False,
    ) -> TDMTreeNode:
        """Build tree."""
        # pull all leaf nodes to the last level
        min_code = (
            self.n_cluster ** math.ceil(math.log(len(leaf_nodes), self.n_cluster)) - 1
        )
        max_code = 0
        max_item_id = 0
        for i in range(len(leaf_nodes)):
            while leaf_nodes[i].tree_code < min_code:
                leaf_nodes[i].tree_code = leaf_nodes[i].tree_code * self.n_cluster + 1
            max_code = max(leaf_nodes[i].tree_code, max_code)
            leaf_item_id = leaf_nodes[i].item_id
            assert leaf_item_id is not None
            max_item_id = max(leaf_item_id, max_item_id)

        tree_nodes: List[Optional[TDMTreeNode]] = [None for _ in range(max_code + 1)]
        logger.info("start gen code_list")

        for leaf_node in leaf_nodes:
            tree_nodes[leaf_node.tree_code] = leaf_node
            ancestors = self._ancestors(leaf_node.tree_code)
            for ancestor in ancestors:
                if tree_nodes[ancestor] is None:
                    tree_nodes[ancestor] = TDMTreeNode(tree_code=ancestor)
                ancestor_node = tree_nodes[ancestor]
                assert ancestor_node is not None
                ancestor_node.attrs_list.append(leaf_node.attrs)
                ancestor_node.raw_attrs_list.append(leaf_node.raw_attrs)

        for code in range(max_code + 1):
            node = tree_nodes[code]
            if node is None:
                continue
            assert node is not None
            if node.item_id is None:
                node.attrs = self._column_modes(node.attrs_list)
                node.raw_attrs = self._column_means(node.raw_attrs_list)
                node.item_id = max_item_id + code + 1

            if code > 0:
                ancestor = int((code - 1) / self.n_cluster)
                ancestor_node = tree_nodes[ancestor]
                assert ancestor_node is not None
                node.set_parent(ancestor_node)

            node.attrs_list = []
            node.raw_attrs_list = []

        root_node = tree_nodes[0]
        assert root_node is not None
        if save_tree:
            self.save_tree(root_node)
        return root_node

    def _column_modes(self, matrix: List[List[pa.Scalar]]) -> List[pa.Scalar]:
        transposed_matrix = list(zip(*matrix))
        modes = []
        for column in transposed_matrix:
            if pa.types.is_string(column[0].type):
                filtered_column = [x for x in column if x]
                if filtered_column:
                    most_common = Counter(filtered_column).most_common(1)[0][0]
                    modes.append(most_common)
                else:
                    modes.append(pa.scalar(""))
            else:
                mode = pc.mode(list(column))
                if len(mode) > 0:
                    modes.append(mode[0])
                else:
                    # null value with column dtype
                    modes.append(column[0])
        return modes

    def _column_means(self, matrix: List[List[pa.Scalar]]) -> List[pa.Scalar]:
        transposed_matrix = list(zip(*matrix))
        means = []
        for column in transposed_matrix:
            means.append(pc.mean(list(column)).cast(column[0].type))
        return means

    def save_tree(self, root: TDMTreeNode) -> None:
        """Save tree."""
        assert self.output_dir is not None, "if save tree, must set output_dir."
        path = os.path.join(self.output_dir, "tree.pkl")
        logger.info(f"save tree to {path}")
        exporter = DictExporter()
        data = exporter.export(root)
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def _ancestors(self, code: int) -> List[int]:
        ancs = []
        while code > 0:
            code = int((code - 1) / self.n_cluster)
            ancs.append(code)
        return ancs
