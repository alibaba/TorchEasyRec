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

from tzrec.constant import Mode
from tzrec.datasets.csv_dataset import CsvDataset, CsvWriter
from tzrec.datasets.dataset import create_dataloader
from tzrec.features.feature import FgMode, create_features
from tzrec.protos import data_pb2, feature_pb2
from tzrec.tests.utils import (
    drain_sample,
    raw_c_features,
    sample_data_config,
    stop_sample,
)
from tzrec.utils.test_util import make_test_dir, parameterized_name_func


class CsvDatasetTest(unittest.TestCase):
    @parameterized.expand(
        [
            [True, 10000, False],  # with_header, num_rows, use_input_fields_str
            [False, 10000, False],
            [False, 5000, False],
            [False, 10000, True],  # test with input_fields_str
            [False, 5000, True],  # test with input_fields_str
        ]
    )
    def test_csv_dataset(self, with_header, num_rows, use_input_fields_str):
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
                if use_input_fields_str:
                    data_config.input_fields_str = (
                        "unused:STRING;id_a:STRING;tag_b:STRING;"
                        "raw_c:INT64;raw_d:DOUBLE;label:INT64"
                    )
                else:
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
        self.test_dir = make_test_dir()

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

    def test_csv_writer_chunked_array(self):
        os.environ["RANK"] = "0"
        writer = CsvWriter(self.test_dir)
        writer.write(
            {
                "int_a": pa.chunked_array([pa.array([1, 2, 3]), pa.array([4, 5])]),
                "float_b": pa.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            }
        )
        writer.close()
        table = csv.read_csv(os.path.join(self.test_dir, "part-0.csv"))
        self.assertEqual(table.num_rows, 5)
        self.assertEqual(table.column("int_a").to_pylist(), [1, 2, 3, 4, 5])


class CsvSampleShuffleTest(unittest.TestCase):
    """Sample-level shuffle on the single-rank CSV path."""

    def _write_csv(self, test_dir, num_rows, num_files=2):
        rows_per_file = num_rows // num_files
        for f in range(num_files):
            path = os.path.join(test_dir, f"part-{f}.csv")
            with open(path, "w") as fh:
                fh.write("raw_c,label\n")
                for i in range(f * rows_per_file, (f + 1) * rows_per_file):
                    fh.write(f"{float(i)},{i % 2}\n")

    @parameterized.expand(
        [
            ("rolling_workers", data_pb2.SAMPLE_ROLLING, 2),
            ("window_workers", data_pb2.SAMPLE_WINDOW, 2),
            ("rolling_no_workers", data_pb2.SAMPLE_ROLLING, 0),
        ],
        name_func=parameterized_name_func,
    )
    def test_sample_shuffle_epochs(self, _name, mode, num_workers):
        with tempfile.TemporaryDirectory(prefix="tzrec_") as test_dir:
            self._write_csv(test_dir, 100)
            dataloader = create_dataloader(
                sample_data_config(
                    data_pb2.DatasetType.CsvDataset, mode, num_workers, with_header=True
                ),
                raw_c_features(),
                f"{test_dir}/*.csv",
                mode=Mode.TRAIN,
            )
            iterator = None
            try:
                for _ in range(2):
                    iterator = dataloader.get_iterator()
                    sizes, values, _ = drain_sample(iterator)
                    self.assertEqual(sum(sizes), 100)
                    self.assertEqual(sorted(sizes), [4] + [8] * 12)
                    self.assertEqual(sorted(values), list(range(100)))
            finally:
                stop_sample(dataloader, iterator)

    def test_sample_shuffle_uses_actual_worker_count(self):
        with tempfile.TemporaryDirectory(prefix="tzrec_") as test_dir:
            self._write_csv(test_dir, 100, num_files=1)
            # 1 file caps the dataloader at one worker; the plan follows it
            dataloader = create_dataloader(
                sample_data_config(
                    data_pb2.DatasetType.CsvDataset,
                    data_pb2.SAMPLE_ROLLING,
                    4,
                    with_header=True,
                ),
                raw_c_features(),
                f"{test_dir}/*.csv",
                mode=Mode.TRAIN,
            )
            self.assertEqual(dataloader.num_workers, 1)
            self.assertEqual(dataloader.dataset._sample_shuffle._num_consumers, 1)
            iterator = None
            try:
                iterator = dataloader.get_iterator()
                sizes, values, _ = drain_sample(iterator)
                self.assertEqual(sum(sizes), 100)
                self.assertEqual(sorted(values), list(range(100)))
            finally:
                stop_sample(dataloader, iterator)

    def test_sample_shuffle_rejects_multi_rank(self):
        with tempfile.TemporaryDirectory(prefix="tzrec_") as test_dir:
            self._write_csv(test_dir, 20)
            dataset = CsvDataset(
                sample_data_config(
                    data_pb2.DatasetType.CsvDataset,
                    data_pb2.SAMPLE_ROLLING,
                    1,
                    with_header=True,
                ),
                raw_c_features(),
                f"{test_dir}/*.csv",
                mode=Mode.TRAIN,
            )
            with self.assertRaisesRegex(ValueError, "single rank"):
                dataset._reader.create_read_plan(0, 2, 1)

    def test_sample_shuffle_rejects_batch_cost_size(self):
        with tempfile.TemporaryDirectory(prefix="tzrec_") as test_dir:
            self._write_csv(test_dir, 20)
            with self.assertRaisesRegex(ValueError, "batch_cost_size"):
                create_dataloader(
                    sample_data_config(
                        data_pb2.DatasetType.CsvDataset,
                        data_pb2.SAMPLE_ROLLING,
                        1,
                        with_header=True,
                        sample_cost_field="raw_c",
                        batch_cost_size=1024,
                    ),
                    raw_c_features(),
                    f"{test_dir}/*.csv",
                    mode=Mode.TRAIN,
                )


if __name__ == "__main__":
    unittest.main()
