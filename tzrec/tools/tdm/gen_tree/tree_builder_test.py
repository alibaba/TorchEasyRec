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
from typing import List

import pyarrow as pa

from tzrec.tools.tdm.gen_tree.tree_builder import TDMTreeNode, TreeBuilder
from tzrec.tools.tdm.gen_tree.tree_search_util import LevelOrderIter


class TreeBuilderTest(unittest.TestCase):
    def gen_ids_codes(self) -> List[TDMTreeNode]:
        leaf_nodes = []
        for i in range(10):
            leaf_nodes.append(
                TDMTreeNode(
                    item_id=i,
                    cate=i,
                    attrs=[pa.scalar(i)],
                    raw_attrs=[pa.scalar(0.1 * i)],
                )
            )
        leaf_nodes.sort(key=lambda x: (x.cate, x.item_id))

        def gen_code(
            start: int, end: int, code: int, leaf_nodes: List[TDMTreeNode]
        ) -> None:
            if end <= start:
                return
            if end == start + 1:
                leaf_nodes[start].tree_code = code
                return
            for i in range(2):
                left = int(start + i * (end - start) / 2)
                right = int(start + (i + 1) * (end - start) / 2)
                gen_code(left, right, 2 * code + 2 - i, leaf_nodes)

        gen_code(0, len(leaf_nodes), 0, leaf_nodes)

        return leaf_nodes

    def test_treebuilder(self) -> None:
        leaf_nodes = self.gen_ids_codes()
        builder = TreeBuilder()
        root = builder.build(leaf_nodes, False)
        true_ids = list(range(10, 25)) + list(range(9, -1, -1))
        true_levels = [1] + [2] * 2 + [3] * 4 + [4] * 8 + [5] * 10
        true_leaf_feas = [[pa.scalar(i), pa.scalar(0.1 * i)] for i in true_ids[-10:]]
        ids = []
        levels = []
        leaf_feas = []
        for node, level in LevelOrderIter(root):
            levels.append(level)
            ids.append(node.item_id)
            if level == 5:
                leaf_feas.append(node.attrs + node.raw_attrs)
        self.assertEqual(true_ids, ids)
        self.assertEqual(true_levels, levels)
        self.assertEqual(true_leaf_feas, leaf_feas)


if __name__ == "__main__":
    unittest.main()
