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
from collections import Counter
from typing import List, Optional

import numpy as np
import numpy.typing as npt
from anytree import NodeMixin
from anytree.exporter.dictexporter import DictExporter

from tzrec.utils.logging_util import logger


class BaseClass(object):
    """Tree base class."""

    pass


class TDMTreeClass(BaseClass, NodeMixin):
    """TDM tree."""

    def __init__(
        self,
        tree_code: int,
        emb_vec: List[float],
        item_id: Optional[int] = None,
        attrs: Optional[str] = None,
        raw_attrs: Optional[str] = None,
        parent: Optional["TDMTreeClass"] = None,
        children: Optional[List["TDMTreeClass"]] = None,
    ) -> None:
        super(TDMTreeClass, self).__init__()
        self.tree_code = tree_code
        self.emb_vec = emb_vec
        self.item_id = item_id
        self.attrs = attrs
        self.raw_attrs = raw_attrs
        self.parent = parent
        if children:
            self.children = children

    def set_parent(self, parent: "TDMTreeClass") -> None:
        """Set parent."""
        self.parent = parent

    def set_children(self, children: "TDMTreeClass") -> None:
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
        ids: npt.NDArray,
        codes: npt.NDArray,
        attrs: npt.NDArray,
        raw_attrs: npt.NDArray,
        data: npt.NDArray,
        save_tree: bool = False,
    ) -> TDMTreeClass:
        """Build tree."""
        max_item_id = max(ids) + 1
        # sort by code
        argindex = np.argsort(codes)
        codes = codes[argindex]
        ids = ids[argindex]
        attrs = attrs[argindex]
        raw_attrs = raw_attrs[argindex]
        data = data[argindex]

        # pull all leaf nodes to the last level
        min_code = 0
        max_code = codes[-1]
        while max_code > 0:
            min_code = min_code * self.n_cluster + 1
            max_code = int((max_code - 1) / self.n_cluster)

        max_code = 0
        for i in range(len(codes)):
            while codes[i] < min_code:
                codes[i] = codes[i] * self.n_cluster + 1
            max_code = max(codes[i], max_code)

        class Item:
            def __init__(self, item_id=None, attrs=None, raw_attrs=None, emb=None):
                self.item_id = item_id
                self.attrs = attrs
                self.raw_attrs = raw_attrs
                self.emb = emb

        code_list: List[Optional[Item]] = [None for _ in range(max_code + 1)]
        node_dict = {}
        logger.info("start gen code_list")

        for _id, code, attr, raw_attr, datum in zip(ids, codes, attrs, raw_attrs, data):
            code_list[code] = Item(_id, attr, raw_attr, datum)
            ancestors = self._ancestors(code)
            for ancestor in ancestors:
                code_list[ancestor] = Item(attrs=[], raw_attrs=[], emb=[])

        # If there are embedding vectors, the embedding vector of the parent node
        # is the average of the embedding vectors of all its child nodes,
        # and the ID feature is determined by the mode.
        for code in range(max_code, -1, -1):
            code_item = code_list[code]
            if code_item is None:
                continue
            assert code_item is not None
            if not isinstance(code_item.item_id, np.integer):
                if data[0].size != 0:
                    code_item.emb = np.mean(code_item.emb, axis=0)
                mode_attr = self._column_modes(code_item.attrs)
                code_item.attrs = ",".join(mode_attr)

                if len(code_item.raw_attrs[0]) > 0:
                    mean_raw_attr = np.nanmean(code_item.raw_attrs, axis=0)
                    mean_raw_attr = mean_raw_attr.astype(str)
                    mean_raw_attr[mean_raw_attr == "nan"] = ""
                    code_item.raw_attrs = ",".join(mean_raw_attr)
                else:
                    code_item.raw_attrs = ""
            if code > 0:
                ancestors = self._ancestors(code)
                for ancestor in ancestors:
                    ancestor_code_item = code_list[ancestor]
                    assert ancestor_code_item is not None
                    if data[0].size != 0:
                        ancestor_code_item.emb.append(code_item.emb)
                    ancestor_code_item.attrs.append(code_item.attrs.split(","))
                    if code_item.raw_attrs:
                        raw_fea = np.array(code_item.raw_attrs.split(","))
                        raw_fea[raw_fea == ""] = np.nan
                        ancestor_code_item.raw_attrs.append(raw_fea.astype(float))
                    else:
                        ancestor_code_item.raw_attrs.append("")

        logger.info("start gen node_dict")
        for code in range(0, max_code + 1):
            code_item = code_list[code]
            if code_item is None:
                continue
            assert code_item is not None
            if isinstance(code_item.item_id, np.integer):
                node_dict[code] = TDMTreeClass(
                    code,
                    emb_vec=[],
                    item_id=int(code_item.item_id),
                    attrs=code_item.attrs,
                    raw_attrs=code_item.raw_attrs,
                )
            else:
                node_dict[code] = TDMTreeClass(
                    code,
                    emb_vec=[],
                    item_id=code + max_item_id,
                    attrs=code_item.attrs,
                    raw_attrs=code_item.raw_attrs,
                )
            if code > 0:
                ancestor = int((code - 1) / self.n_cluster)
                node_dict[code].set_parent(node_dict[ancestor])

        if save_tree:
            self.save_tree(node_dict[0])
        return node_dict[0]

    def _column_modes(self, matrix: List[List[str]]) -> List[str]:
        transposed_matrix = list(zip(*matrix))
        modes = []

        for column in transposed_matrix:
            filtered_column = [x for x in column if x]
            if filtered_column:
                most_common = Counter(filtered_column).most_common(1)[0][0]
                modes.append(most_common)
            else:
                modes.append("")

        return modes

    def save_tree(self, root: TDMTreeClass) -> None:
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
