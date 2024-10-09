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
import pickle
from collections import OrderedDict
from typing import Any, Callable, Iterator, List, Optional, Tuple

import pyarrow as pa
from anytree.importer.dictimporter import DictImporter
from anytree.iterators.abstractiter import AbstractIter
from anytree.walker import Walker

from tzrec.datasets.dataset import create_writer
from tzrec.tools.tdm.gen_tree.tree_builder import TDMTreeClass
from tzrec.utils.logging_util import logger


class LevelOrderIter(AbstractIter):
    """Level-order traversal tree."""

    @staticmethod
    def _iter(
        children: List[TDMTreeClass],
        filter_: Callable[[TDMTreeClass], bool],
        stop: Callable[[TDMTreeClass], bool],
        maxlevel: int,
    ) -> Iterator[Tuple[TDMTreeClass, int]]:
        level = 1
        while children:
            next_children = []
            for child in children:
                if filter_(child):
                    yield child, level
                next_children += AbstractIter._get_children(child.children, stop)
            children = next_children
            level += 1
            if AbstractIter._abort_at_level(level, maxlevel):
                break


class TreeSearch(object):
    """Convert anytree to nodes and edges.

    Args:
        tree_path(str): tree file path.
        output_file(str): nodes and edges output file.
        chile_num(int): The branching factor of the nodes in the tree.
    """

    def __init__(
        self,
        output_file: str,
        tree_path: Optional[str] = None,
        root: Optional[TDMTreeClass] = None,
        child_num: int = 2,
        **kwargs: Any,
    ) -> None:
        self.child_num = child_num
        if root is not None:
            self.root = root
        elif tree_path is not None:
            self._load(tree_path)
        else:
            raise ValueError("Either root or tree_path must be provided.")

        assert self.root is not None, "Either root or tree_path must be provided."

        self.travel_list = []
        self.nodes = []

        self.level_code = [[]]
        self.max_level = 0

        self.output_file = output_file

        self.dataset_kwargs = {}
        if "odps_data_quota_name" in kwargs:
            self.dataset_kwargs["quota_name"] = kwargs["odps_data_quota_name"]

        self._get_nodes()

    def _load(self, path: str) -> None:
        """Load tree."""
        logger.info("Begin load tree.")
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.root = DictImporter().import_(data)

    def _get_nodes(self) -> None:
        """Get node info."""
        logger.info("Begin iter tree.")

        for node, level in LevelOrderIter(self.root):
            if level - 1 > self.max_level:
                self.max_level = level - 1
                self.level_code.append([])
            self.level_code[self.max_level].append(node)

        tree_walker = Walker()
        logger.info("Begin Travel Tree.")
        for leaf_node in self.level_code[-1]:
            paths_0, paths_1, _ = tree_walker.walk(leaf_node, self.root)
            paths = list(paths_0) + [paths_1]
            travel = [i.item_id for i in paths]
            self.travel_list.append(travel)

    def save(self) -> None:
        """Save tree info."""
        if self.output_file.startswith("odps://"):
            node_writer = create_writer(
                self.output_file + "node_table", **self.dataset_kwargs
            )
            ids = []
            weight = []
            features = []
            for level, nodes in enumerate(self.level_code):
                for node in nodes:
                    ids.append(node.item_id)
                    weight.append(1.0)
                    fea = [level, node.item_id]
                    if node.attrs:
                        fea.append(node.attrs)
                    if node.raw_attrs:
                        fea.append(node.raw_attrs)
                    features.append(",".join(map(str, fea)))
            node_table_dict = OrderedDict()
            node_table_dict["id"] = pa.array(ids)
            node_table_dict["weight"] = pa.array(weight)
            node_table_dict["features"] = pa.array(features)
            node_writer.write(node_table_dict)

            edge_writer = create_writer(
                self.output_file + "edge_table", **self.dataset_kwargs
            )
            src_ids = []
            dst_ids = []
            weight = []
            for travel in self.travel_list:
                # do not include edge from leaf to root
                for i in range(self.max_level - 1):
                    src_ids.append(travel[0])
                    dst_ids.append(travel[i + 1])
                    weight.append(1.0)
            edge_table_dict = OrderedDict()
            edge_table_dict["src_id"] = pa.array(src_ids)
            edge_table_dict["dst_id"] = pa.array(dst_ids)
            edge_table_dict["weight"] = pa.array(weight)
            edge_writer.write(edge_table_dict)

        else:
            if not os.path.exists(self.output_file):
                os.makedirs(self.output_file)
            with open(os.path.join(self.output_file, "node_table.txt"), "w") as f:
                f.write("id:int64\tweight:float\tfeature:string\n")
                for level, nodes in enumerate(self.level_code):
                    for node in nodes:
                        f.write(f"{node.item_id}\t{1.0}\t")
                        fea = [level, node.item_id]
                        if node.attrs:
                            fea.append(node.attrs)
                        if node.raw_attrs:
                            fea.append(node.raw_attrs)
                        f.write(",".join(map(str, fea)) + "\n")

            with open(os.path.join(self.output_file, "edge_table.txt"), "w") as f:
                f.write("src_id:int64\tdst_id:int64\tweight:float\n")
                for travel in self.travel_list:
                    # do not include edge from leaf to root
                    for i in range(self.max_level - 1):
                        f.write(f"{travel[0]}\t{travel[i+1]}\t{1.0}\n")

    def save_predict_edge(self) -> None:
        """Save edge info for prediction."""
        if self.output_file.startswith("odps://"):
            writer = create_writer(
                self.output_file + "predict_edge_table", **self.dataset_kwargs
            )
            src_ids = []
            dst_ids = []
            weight = []
            for i in range(self.max_level):
                for node in self.level_code[i]:
                    for child in node.children:
                        src_ids.append(node.item_id)
                        dst_ids.append(child.item_id)
                        weight.append(1.0)
            edge_table_dict = OrderedDict()
            edge_table_dict["src_id"] = pa.array(src_ids)
            edge_table_dict["dst_id"] = pa.array(dst_ids)
            edge_table_dict["weight"] = pa.array(weight)
            writer.write(edge_table_dict)
        else:
            with open(
                os.path.join(self.output_file, "predict_edge_table.txt"), "w"
            ) as f:
                f.write("src_id:int64\tdst_id:int64\tweight:float\n")
                for i in range(self.max_level):
                    for node in self.level_code[i]:
                        for child in node.children:
                            f.write(f"{node.item_id}\t{child.item_id}\t{1.0}\n")

    def save_serving_tree(self, tree_output_dir: str) -> None:
        """Save tree info for serving."""
        if not os.path.exists(tree_output_dir):
            os.makedirs(tree_output_dir)
        with open(os.path.join(tree_output_dir, "serving_tree"), "w") as f:
            f.write(f"{self.max_level + 1} {self.child_num}\n")
            for _, nodes in enumerate(self.level_code):
                for node in nodes:
                    f.write(f"{node.tree_code} {node.item_id}\n")
