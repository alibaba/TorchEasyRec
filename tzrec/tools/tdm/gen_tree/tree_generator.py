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

from typing import Dict, List, Optional, Union

import numpy as np
import pyarrow as pa

from tzrec.datasets.dataset import create_reader
from tzrec.tools.tdm.gen_tree.tree_builder import TDMTreeClass, TreeBuilder


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
        tree_output_file: Optional[str] = None,
        n_cluster: int = 2,
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
        self.tree_output_file = tree_output_file
        self.n_cluster = n_cluster

    def generate(self, save_tree: bool = False) -> TDMTreeClass:
        """Generate tree."""
        item_fea = self._read()
        root = self._init_tree(item_fea, save_tree)
        return root

    def _read(self) -> Dict[str, List[Union[int, float, str]]]:
        item_fea = {"ids": [], "cates": [], "attrs": [], "raw_attrs": []}
        reader = create_reader(self.item_input_path, 4096)
        for data_dict in reader.to_batches():
            item_fea["ids"] += data_dict[self.item_id_field].to_pylist()
            item_fea["cates"] += (
                data_dict[self.cate_id_field]
                .cast(pa.string())
                .fill_null("")
                .to_pylist()
            )
            tmp_attr = []
            if self.attr_fields is not None:
                # pyre-ignore [16]
                for attr in self.attr_fields:
                    tmp_attr.append(
                        data_dict[attr].cast(pa.string()).fill_null("").to_pylist()
                    )
                item_fea["attrs"] += [",".join(map(str, i)) for i in zip(*tmp_attr)]
            else:
                item_fea["attrs"] += [""] * len(
                    data_dict[self.item_id_field].to_pylist()
                )

            tmp_raw_attr = []
            if self.raw_attr_fields is not None:
                for attr in self.raw_attr_fields:
                    tmp_raw_attr.append(
                        data_dict[attr].cast(pa.string()).fill_null("").to_pylist()
                    )
                item_fea["raw_attrs"] += [
                    ",".join(map(str, i)) for i in zip(*tmp_raw_attr)
                ]
            else:
                item_fea["raw_attrs"] += [""] * len(
                    data_dict[self.item_id_field].to_pylist()
                )

        return item_fea

    def _init_tree(
        self, item_fea: Dict[str, List[Union[int, float, str]]], save_tree: bool
    ) -> TDMTreeClass:
        class Item:
            def __init__(self, item_id, cat_id, attrs=None, raw_attrs=None):
                self.item_id = item_id
                self.cat_id = cat_id
                self.attrs = attrs
                self.raw_attrs = raw_attrs
                self.code = 0

            def __lt__(self, other):
                return self.cat_id < other.cat_id or (
                    self.cat_id == other.cat_id and self.item_id < other.item_id
                )

        items = []
        for item_id, cat_id, attrs, raw_attrs in zip(
            item_fea["ids"], item_fea["cates"], item_fea["attrs"], item_fea["raw_attrs"]
        ):
            items.append(Item(item_id, cat_id, attrs, raw_attrs))
        items.sort()

        def gen_code(start: int, end: int, code: int, items: List[Item]) -> None:
            if end <= start:
                return
            if end == start + 1:
                items[start].code = code
                return
            for i in range(self.n_cluster):
                left = int(start + i * (end - start) / self.n_cluster)
                right = int(start + (i + 1) * (end - start) / self.n_cluster)
                gen_code(left, right, self.n_cluster * code + self.n_cluster - i, items)

        gen_code(0, len(items), 0, items)
        ids = np.array([item.item_id for item in items])
        codes = np.array([item.code for item in items])
        attrs = np.array([item.attrs if item.attrs else "" for item in items])
        raw_attrs = np.array(
            [item.raw_attrs if item.raw_attrs else "" for item in items]
        )
        data = np.array([[] for i in range(len(ids))])

        builder = TreeBuilder(self.tree_output_file, self.n_cluster)
        root = builder.build(ids, codes, attrs, raw_attrs, data, save_tree)
        return root
