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

import multiprocessing as mp
import os
import shutil
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
from parameterized import parameterized
from torch.utils.data import DataLoader

from tzrec.datasets.parquet_dataset import ParquetDataset, ParquetWriter
from tzrec.features.feature import create_features
from tzrec.protos import data_pb2, feature_pb2


class ParquetDatasetTest(unittest.TestCase):
    @parameterized.expand([[5000], [10000]])
    def test_parquet_dataset(self, num_rows):
        feature_cfgs = [
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(feature_name="id_a")
            ),
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(feature_name="tag_b")
            ),
            feature_pb2.FeatureConfig(
                raw_feature=feature_pb2.RawFeature(feature_name="raw_c")
            ),
            feature_pb2.FeatureConfig(
                raw_feature=feature_pb2.RawFeature(feature_name="raw_d", value_dim=2)
            ),
        ]
        features = create_features(feature_cfgs)

        t = pa.Table.from_arrays(
            [
                pa.array(["unused"] * num_rows),
                pa.array(["1"] * num_rows),
                pa.array(["2\x033"] * num_rows),
                pa.array([4] * num_rows),
                pa.array([[5.0, 6.0]] * num_rows, type=pa.list_(pa.float32())),
                pa.array([0] * num_rows),
            ],
            names=["unused", "id_a", "tag_b", "raw_c", "raw_d", "label"],
        )
        with tempfile.TemporaryDirectory(prefix="tzrec_") as test_dir:
            ds.write_dataset(
                t,
                test_dir,
                format="parquet",
                max_rows_per_file=5000,
                max_rows_per_group=5000,
            )

            dataset = ParquetDataset(
                data_config=data_pb2.DataConfig(
                    batch_size=4,
                    dataset_type=data_pb2.DatasetType.ParquetDataset,
                    fg_mode=data_pb2.FgMode.FG_NONE,
                    label_fields=["label"],
                ),
                features=features,
                input_path=f"{test_dir}/*.parquet",
            )
            self.assertEqual(len(dataset.input_fields), 5)
            dataloader = DataLoader(
                dataset=dataset,
                batch_size=None,
                num_workers=2,
                pin_memory=True,
                collate_fn=lambda x: x,
            )
            iterator = iter(dataloader)
            for _ in range(10):
                data = next(iterator)
            self.assertEqual(
                sorted(data.to_dict().keys()),
                [
                    "id_a.lengths",
                    "id_a.values",
                    "label",
                    "raw_c.values",
                    "raw_d.values",
                    "tag_b.lengths",
                    "tag_b.values",
                ],
            )


class ParquetWriterTest(unittest.TestCase):
    def setUp(self):
        if not os.path.exists("./tmp"):
            os.makedirs("./tmp")
        self.test_dir = tempfile.mkdtemp(prefix="tzrec_", dir="./tmp")

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_parquet_writer(self):
        def _writer_worker(rank):
            os.environ["RANK"] = str(rank)
            writer = ParquetWriter(self.test_dir)
            for _ in range(5):
                writer.write(
                    {
                        "int_a": pa.array(np.random.randint(0, 100, 128)),
                        "float_b": pa.array(np.random.random(128)),
                    }
                )
            writer.close()

        world_size = 2
        procs = []
        for rank in range(world_size):
            p = mp.Process(
                target=_writer_worker,
                args=(rank,),
            )
            p.start()
            procs.append(p)
        for i, p in enumerate(procs):
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"writer worker-{i} failed.")
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "part-0.parquet")))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "part-1.parquet")))


if __name__ == "__main__":
    unittest.main()
