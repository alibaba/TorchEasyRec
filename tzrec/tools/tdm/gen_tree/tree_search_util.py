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
from tzrec.tools.tdm.gen_tree.tree_builder import TDMTreeNode
from tzrec.utils.env_util import use_hash_node_id
from tzrec.utils.logging_util import logger


class LevelOrderIter(AbstractIter):
    """Level-order traversal tree."""

    @staticmethod
    def _iter(
        children: List[TDMTreeNode],
        filter_: Callable[[TDMTreeNode], bool],
        stop: Callable[[TDMTreeNode], bool],
        maxlevel: int,
    ) -> Iterator[Tuple[TDMTreeNode, int]]:
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


def _add_suffix_to_odps_table(table_path: str, suffix: str) -> str:
    str_list = table_path.split("/")
    str_list[4] = str_list[4] + suffix
    return "/".join(str_list)


class TreeSearch(object):
    """Convert anytree to nodes and edges.

    Args:
        output_file (str): nodes and edges output file.
        tree_path (str): tree file path.
        root (TDMTreeNode): root node of tree.
        chile_num (int): The branching factor of the nodes in the tree.
    """

    def __init__(
        self,
        output_file: str,
        tree_path: Optional[str] = None,
        root: Optional[TDMTreeNode] = None,
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
        logger.info(
            f"Tree Level: {self.max_level + 1}, Tree Cluster: {self.child_num}."
        )

        tree_walker = Walker()
        logger.info("Begin Travel Tree.")
        for leaf_node in self.level_code[-1]:
            paths_0, paths_1, _ = tree_walker.walk(leaf_node, self.root)
            paths = list(paths_0) + [paths_1]
            travel = [i.item_id for i in paths]
            self.travel_list.append(travel)

    def save(self, attr_delimiter: str = ",") -> None:
        """Save tree info."""
        if self.output_file.startswith("odps://"):
            output_path = _add_suffix_to_odps_table(self.output_file, "_node_table")
            node_writer = create_writer(output_path, **self.dataset_kwargs)
            ids = []
            weight = []
            features = []
            first_node = True
            for level, nodes in enumerate(self.level_code):
                for node in nodes:
                    fea = [level, node.item_id]
                    if node.attrs:
                        fea.append(
                            attr_delimiter.join(
                                map(lambda x: str(x) if x.is_valid else "", node.attrs)
                            )
                        )
                    if node.raw_attrs:
                        fea.append(
                            attr_delimiter.join(
                                map(
                                    lambda x: str(x) if x.is_valid else 0,
                                    node.raw_attrs,
                                )
                            )
                        )

                    # add a node with id -1 for graph-learn to get root node
                    if first_node:
                        ids.append("-1" if use_hash_node_id() else -1)
                        weight.append(1.0)
                        features.append(
                            attr_delimiter.join(["-1"] + list(map(str, fea[1:])))
                        )
                        first_node = False

                    ids.append(node.item_id)
                    weight.append(1.0)
                    features.append(attr_delimiter.join(map(str, fea)))

            node_table_dict = OrderedDict()
            node_table_dict["id"] = pa.array(ids)
            node_table_dict["weight"] = pa.array(weight)
            node_table_dict["features"] = pa.array(features)
            node_writer.write(node_table_dict)
            node_writer.close()

            output_path = _add_suffix_to_odps_table(self.output_file, "_edge_table")
            edge_writer = create_writer(output_path, **self.dataset_kwargs)
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
            edge_writer.close()

        else:
            if not os.path.exists(self.output_file):
                os.makedirs(self.output_file)
            with open(os.path.join(self.output_file, "node_table.txt"), "w") as f:
                id_type = "string" if use_hash_node_id() else "int64"
                f.write(f"id:{id_type}\tweight:float\tfeature:string\n")
                first_node = True
                for level, nodes in enumerate(self.level_code):
                    for node in nodes:
                        fea = [level, node.item_id]
                        if node.attrs:
                            fea.append(
                                attr_delimiter.join(
                                    map(
                                        lambda x: str(x) if x.is_valid else "",
                                        node.attrs,
                                    )
                                )
                            )
                        if node.raw_attrs:
                            fea.append(
                                attr_delimiter.join(
                                    map(
                                        lambda x: str(x) if x.is_valid else 0,
                                        node.raw_attrs,
                                    )
                                )
                            )
                        # add a node with id -1 for graph-learn to get root node
                        if first_node:
                            f.write(
                                f"-1\t1.0\t-1{attr_delimiter}{attr_delimiter.join(map(str, fea[1:]))}\n"  # NOQA
                            )
                            first_node = False
                        f.write(
                            f"{node.item_id}\t1.0\t{attr_delimiter.join(map(str, fea))}\n"  # NOQA
                        )

            with open(os.path.join(self.output_file, "edge_table.txt"), "w") as f:
                id_type = "string" if use_hash_node_id() else "int64"
                f.write(f"src_id:{id_type}\tdst_id:{id_type}\tweight:float\n")
                for travel in self.travel_list:
                    # do not include edge from leaf to root
                    for i in range(self.max_level - 1):
                        f.write(f"{travel[0]}\t{travel[i + 1]}\t{1.0}\n")

    def save_predict_edge(self) -> None:
        """Save edge info for prediction."""
        if self.output_file.startswith("odps://"):
            output_path = _add_suffix_to_odps_table(
                self.output_file, "_predict_edge_table"
            )
            writer = create_writer(output_path, **self.dataset_kwargs)
            # add a edge from -1 to root for graph-learn to get root node
            src_ids = ["-1" if use_hash_node_id() else -1]
            dst_ids = [self.root.item_id]
            weight = [1.0]
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
            writer.close()
        else:
            with open(
                os.path.join(self.output_file, "predict_edge_table.txt"), "w"
            ) as f:
                id_type = "string" if use_hash_node_id() else "int64"
                f.write(f"src_id:{id_type}\tdst_id:{id_type}\tweight:float\n")
                # add a edge from  with id -1 to root for graph-learn to get root node
                f.write(f"-1\t{self.root.item_id}\t1.0\n")
                for i in range(self.max_level):
                    for node in self.level_code[i]:
                        for child in node.children:
                            f.write(f"{node.item_id}\t{child.item_id}\t{1.0}\n")

    def save_node_feature(
        self,
        item_id_field: str,
        attr_fields: Optional[str] = None,
        raw_attr_fields: Optional[str] = None,
    ) -> None:
        """Save feature of tree node for serving."""
        if self.output_file.startswith("odps://"):
            output_path = _add_suffix_to_odps_table(self.output_file, "_node_feature")
            writer_type = "OdpsWriter"
        else:
            output_path = os.path.join(self.output_file, "node_feature")
            writer_type = "ParquetWriter"
        writer = create_writer(
            output_path, writer_type=writer_type, **self.dataset_kwargs
        )

        attr_field_names = (
            [x.strip() for x in attr_fields.split(",")] if attr_fields else []
        )
        raw_attr_field_names = (
            [x.strip() for x in raw_attr_fields.split(",")] if raw_attr_fields else []
        )
        attr_names = [item_id_field] + attr_field_names + raw_attr_field_names
        attr_values = [[] for _ in range(len(attr_names))]
        for _, nodes in enumerate(self.level_code):
            for node in nodes:
                for i, attr_value in enumerate(
                    [pa.scalar(node.item_id)] + node.attrs + node.raw_attrs
                ):
                    attr_values[i].append(attr_value)
        attr_dict = OrderedDict(
            zip(attr_names, [pa.array(attr_arr) for attr_arr in attr_values])
        )
        writer.write(attr_dict)
        writer.close()

    def save_serving_tree(self, tree_output_dir: str) -> None:
        """Save tree info for serving."""
        if not os.path.exists(tree_output_dir):
            os.makedirs(tree_output_dir)
        with open(os.path.join(tree_output_dir, "serving_tree"), "w") as f:
            f.write(f"{self.max_level + 1} {self.child_num}\n")
            for _, nodes in enumerate(self.level_code):
                for node in nodes:
                    f.write(f"{node.tree_code} {node.item_id}\n")
