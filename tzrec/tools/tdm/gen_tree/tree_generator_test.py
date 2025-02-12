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

import csv
import os
import tempfile
import unittest

import pyarrow as pa
from parameterized import parameterized

from tzrec.tools.tdm.gen_tree.tree_generator import TreeGenerator
from tzrec.tools.tdm.gen_tree.tree_search_util import LevelOrderIter


class TreeGeneratortest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_file = tempfile.NamedTemporaryFile(
            mode="w", delete=False, newline="", suffix=".csv"
        )
        writer = csv.writer(self.tmp_file)
        writer.writerow(["item_id", "cate_id", "raw_a", "str_a"])
        for i in range(10):
            writer.writerow([i, i, 0.1 * i, f"我们{i}"])
        self.tmp_file.close()

    def tearDown(self) -> None:
        os.remove(self.tmp_file.name)
        os.environ.pop("USE_HASH_NODE_ID", None)

    @parameterized.expand([[False], [True]])
    def test_tree_generator(self, use_hash_id: bool) -> None:
        if use_hash_id:
            os.environ["USE_HASH_NODE_ID"] = "1"
        generator = TreeGenerator(
            item_input_path=self.tmp_file.name,
            item_id_field="item_id",
            cate_id_field="cate_id",
            attr_fields="cate_id,str_a",
            raw_attr_fields="raw_a",
        )
        root = generator.generate(save_tree=False)
        true_ids = list(range(10, 25)) + list(range(9, -1, -1))
        true_levels = [1] + [2] * 2 + [3] * 4 + [4] * 8 + [5] * 10
        true_leaf_feas = [
            [pa.scalar(i), pa.scalar(f"我们{i}"), pa.scalar(0.1 * i)]
            for i in true_ids[-10:]
        ]
        ids = []
        levels = []
        leaf_feas = []
        for node, level in LevelOrderIter(root):
            levels.append(level)
            ids.append(node.item_id)
            if level == 5:
                leaf_feas.append(node.attrs + node.raw_attrs)
        if use_hash_id:
            self.assertEqual(
                [str(x) if x < 10 else f"nonleaf#{x - 10}" for x in true_ids], ids
            )
        else:
            self.assertEqual(true_ids, ids)
        self.assertEqual(true_levels, levels)
        self.assertEqual(true_leaf_feas, leaf_feas)


if __name__ == "__main__":
    unittest.main()
