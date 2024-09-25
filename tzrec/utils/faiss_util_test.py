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
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds

from tzrec.utils import faiss_util


class FaissUtilTest(unittest.TestCase):
    def test_faiss_util(self):
        t = pa.Table.from_arrays(
            [
                pa.array(np.random.randint(0, 100000, (10000,))),
                pa.array(np.random.random((10000, 16)).tolist()),
            ],
            names=["item_id", "item_tower_emb"],
        )
        with tempfile.TemporaryDirectory(prefix="tzrec_") as test_dir:
            ds.write_dataset(
                t,
                test_dir,
                format="parquet",
                max_rows_per_file=5000,
                max_rows_per_group=5000,
            )
            index, index_id_map = faiss_util.build_faiss_index(
                os.path.join(test_dir, "*.parquet"),
                id_field="item_id",
                embedding_field="item_tower_emb",
                index_type="IVFFlatIP",
            )
            faiss_util.write_faiss_index(index, index_id_map, test_dir)
            distances, ids = faiss_util.search_faiss_index(
                index, index_id_map, np.random.random((50, 16)), k=200
            )
            self.assertTrue(os.path.exists(os.path.join(test_dir, "faiss_index")))
            self.assertTrue(os.path.exists(os.path.join(test_dir, "id_mapping")))
            self.assertEqual(distances.shape, (50, 200))
            self.assertEqual(ids.shape, (50, 200))


if __name__ == "__main__":
    unittest.main()
