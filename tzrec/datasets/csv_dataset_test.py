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
from parameterized import parameterized
from pyarrow import csv
from torch.utils.data import DataLoader

from tzrec.datasets.csv_dataset import CsvDataset, CsvWriter
from tzrec.features.feature import FgMode, create_features
from tzrec.protos import data_pb2, feature_pb2


class CsvDatasetTest(unittest.TestCase):
    @parameterized.expand([[True, 10000], [False, 10000], [False, 5000]])
    def test_csv_dataset(self, with_header, num_rows):
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
                raw_feature=feature_pb2.RawFeature(feature_name="raw_d")
            ),
        ]
        features = create_features(feature_cfgs)

        t = pa.Table.from_arrays(
            [
                pa.array(["unused"] * num_rows),
                pa.array(["1"] * num_rows),
                pa.array(["2\x033"] * num_rows),
                pa.array([4] * num_rows),
                pa.array([5.0] * num_rows),
                pa.array([0] * num_rows),
            ],
            names=["unused", "id_a", "tag_b", "raw_c", "raw_d", "label"],
        )
        with tempfile.TemporaryDirectory(prefix="tzrec_") as test_dir:
            for i in range(2):
                csv.write_csv(
                    t,
                    os.path.join(test_dir, f"part-{i}.csv"),
                    csv.WriteOptions(include_header=with_header),
                )
            data_config = data_pb2.DataConfig(
                batch_size=4,
                dataset_type=data_pb2.DatasetType.CsvDataset,
                fg_mode=data_pb2.FgMode.FG_NONE,
                label_fields=["label"],
                with_header=with_header,
            )
            if not with_header:
                data_config.input_fields.extend(
                    [
                        data_pb2.Field(input_name="unused"),
                        data_pb2.Field(input_name="id_a"),
                        data_pb2.Field(input_name="tag_b"),
                        data_pb2.Field(input_name="raw_c"),
                        data_pb2.Field(input_name="raw_d"),
                        data_pb2.Field(input_name="label"),
                    ]
                )
            dataset = CsvDataset(
                data_config,
                features=features,
                input_path=f"{test_dir}/*.csv",
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

    def test_csv_dataset_with_all_nulls(self):
        feature_cfgs = [
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="id_a", expression="user:id_a", hash_bucket_size=1000
                )
            ),
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="tag_b", expression="user:tag_b", hash_bucket_size=100
                )
            ),
            feature_pb2.FeatureConfig(
                raw_feature=feature_pb2.RawFeature(
                    feature_name="raw_c", expression="user:raw_c"
                )
            ),
            feature_pb2.FeatureConfig(
                raw_feature=feature_pb2.RawFeature(
                    feature_name="raw_d", expression="user:raw_d"
                )
            ),
        ]
        features = create_features(feature_cfgs, fg_mode=FgMode.FG_DAG)

        t = pa.Table.from_arrays(
            [
                pa.array(["unused"] * 10000),
                pa.array(["1"] * 10000),
                pa.array([None] * 10000),
                pa.array([4] * 10000),
                pa.array([None] * 10000),
                pa.array([0] * 10000),
            ],
            names=["unused", "id_a", "tag_b", "raw_c", "raw_d", "label"],
        )
        with tempfile.TemporaryDirectory(prefix="tzrec_") as test_dir:
            for i in range(2):
                csv.write_csv(
                    t,
                    os.path.join(test_dir, f"part-{i}.csv"),
                    csv.WriteOptions(include_header=False),
                )
            data_config = data_pb2.DataConfig(
                batch_size=4,
                dataset_type=data_pb2.DatasetType.CsvDataset,
                fg_mode=data_pb2.FgMode.FG_DAG,
                label_fields=["label"],
            )
            data_config.input_fields.extend(
                [
                    data_pb2.Field(input_name="unused"),
                    data_pb2.Field(input_name="id_a"),
                    data_pb2.Field(
                        input_name="tag_b", input_type=data_pb2.FieldType.STRING
                    ),
                    data_pb2.Field(input_name="raw_c"),
                    data_pb2.Field(
                        input_name="raw_d", input_type=data_pb2.FieldType.DOUBLE
                    ),
                    data_pb2.Field(input_name="label"),
                ]
            )
            dataset = CsvDataset(
                data_config,
                features=features,
                input_path=f"{test_dir}/*.csv",
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


class CsvWriterTest(unittest.TestCase):
    def setUp(self):
        if not os.path.exists("./tmp"):
            os.makedirs("./tmp")
        self.test_dir = tempfile.mkdtemp(prefix="tzrec_", dir="./tmp")

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_csv_writer(self):
        def _writer_worker(rank):
            os.environ["RANK"] = str(rank)
            writer = CsvWriter(self.test_dir)
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
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "part-0.csv")))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "part-1.csv")))


if __name__ == "__main__":
    unittest.main()
