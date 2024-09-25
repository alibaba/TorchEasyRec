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

import unittest
from typing import List, Tuple

import numpy as np
import numpy.typing as npt

from tzrec.tools.tdm.gen_tree.tree_builder import TreeBuilder
from tzrec.tools.tdm.gen_tree.tree_search_util import LevelOrderIter


class TreeBuilderTest(unittest.TestCase):
    def gen_ids_codes(
        self,
    ) -> Tuple[
        npt.NDArray,
        npt.NDArray,
        npt.NDArray,
        npt.NDArray,
        npt.NDArray,
    ]:
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
        for i in range(10):
            items.append(Item(item_id=i, cat_id=i, attrs=f"{i}", raw_attrs=f"{0.1*i}"))
        items.sort()

        def gen_code(start: int, end: int, code: int, items: List[Item]) -> None:
            if end <= start:
                return
            if end == start + 1:
                items[start].code = code
                return
            for i in range(2):
                left = int(start + i * (end - start) / 2)
                right = int(start + (i + 1) * (end - start) / 2)
                gen_code(left, right, 2 * code + 2 - i, items)

        gen_code(0, len(items), 0, items)
        ids = np.array([item.item_id for item in items])
        codes = np.array([item.code for item in items])
        attrs = np.array([item.attrs if item.attrs else "" for item in items])
        raw_attrs = np.array(
            [item.raw_attrs if item.raw_attrs else "" for item in items]
        )
        data = np.array([[] for i in range(len(ids))])

        return ids, codes, attrs, raw_attrs, data

    def test_treebuilder(self) -> None:
        ids, codes, attrs, raw_attrs, data = self.gen_ids_codes()
        builder = TreeBuilder()
        root = builder.build(ids, codes, attrs, raw_attrs, data, False)
        true_ids = list(range(10, 25)) + list(range(9, -1, -1))
        true_levels = [1] + [2] * 2 + [3] * 4 + [4] * 8 + [5] * 10
        true_leaf_feas = [f"{i},{0.1*i}" for i in true_ids[-10:]]
        ids = []
        levels = []
        leaf_feas = []
        for node, level in LevelOrderIter(root):
            levels.append(level)
            ids.append(node.item_id)
            if level == 5:
                leaf_feas.append(node.attrs + "," + node.raw_attrs)
        self.assertEqual(true_ids, ids)
        self.assertEqual(true_levels, levels)
        self.assertEqual(true_leaf_feas, leaf_feas)


if __name__ == "__main__":
    unittest.main()
