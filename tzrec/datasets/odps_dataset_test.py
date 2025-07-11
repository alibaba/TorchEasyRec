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
import time
import unittest

import numpy as np
import pyarrow as pa
from odps import ODPS
from parameterized import parameterized
from torch import distributed as dist
from torch.utils.data import DataLoader

from tzrec.datasets.odps_dataset import OdpsDataset, OdpsWriter, _create_odps_account
from tzrec.features.feature import FgMode, create_features
from tzrec.protos import data_pb2, feature_pb2, sampler_pb2
from tzrec.utils import test_util
from tzrec.utils.misc_util import get_free_port, random_name


class OdpsDatasetTest(unittest.TestCase):
    def setUp(self):
        self.o = None
        self.test_project = os.environ.get("CI_ODPS_PROJECT_NAME", None)
        if "ODPS_CONFIG_FILE_PATH" in os.environ:
            with open(os.environ["ODPS_CONFIG_FILE_PATH"], "r") as f:
                for line in f.readlines():
                    values = line.split("=", 1)
                    if len(values) == 2 and values[0] == "project_name":
                        self.test_project = values[1].strip()
        self.test_suffix = random_name()

    def tearDown(self):
        if self.o is not None:
            self.o.delete_table(f"test_odps_dataset_{self.test_suffix}", if_exists=True)
            self.o.delete_table(f"test_odps_sampler_{self.test_suffix}", if_exists=True)
        self.o = None
        os.environ.pop("USE_HASH_NODE_ID", None)

    def _create_test_table_and_feature_cfgs(self, has_lookup=True):
        self.o.delete_table(f"test_odps_dataset_{self.test_suffix}", if_exists=True)
        t = self.o.create_table(
            f"test_odps_dataset_{self.test_suffix}",
            (
                "unused string, id_a string, tag_b string, raw_c bigint, raw_d double, "
                "raw_e int, raw_f float, raw_g array<float>, map_h map<string,bigint>, "
                "label bigint",
                "dt string",
            ),
            if_not_exists=True,
            hints={"odps.sql.type.system.odps2": "true"},
        )
        for dt in ["20240319", "20240320"]:
            with t.open_writer(partition=f"dt={dt}", create_partition=True) as writer:
                writer.write(
                    [
                        [
                            "unused",
                            "1",
                            "\x1d".join(["2\x1d3"] * 10000),
                            4,
                            5.0,
                            3,
                            4.0,
                            [1.0, 2.0, 3.0],
                            {"1": 1, "2": 2},
                            0,
                        ]
                    ]
                    * 10000
                )
        feature_cfgs = [
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="id_a", expression="item:id_a", num_buckets=100000
                )
            ),
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="tag_b", expression="user:tag_b", num_buckets=10
                )
            ),
            feature_pb2.FeatureConfig(
                raw_feature=feature_pb2.RawFeature(
                    feature_name="raw_c", expression="item:raw_c"
                )
            ),
            feature_pb2.FeatureConfig(
                raw_feature=feature_pb2.RawFeature(
                    feature_name="raw_d", expression="item:raw_d"
                )
            ),
            feature_pb2.FeatureConfig(
                raw_feature=feature_pb2.RawFeature(
                    feature_name="raw_e", expression="user:raw_e"
                )
            ),
            feature_pb2.FeatureConfig(
                raw_feature=feature_pb2.RawFeature(
                    feature_name="raw_f", expression="user:raw_f"
                )
            ),
            feature_pb2.FeatureConfig(
                raw_feature=feature_pb2.RawFeature(
                    feature_name="raw_g", expression="user:raw_g", value_dim=3
                ),
            ),
        ]
        if has_lookup:
            feature_cfgs.append(
                feature_pb2.FeatureConfig(
                    lookup_feature=feature_pb2.LookupFeature(
                        feature_name="lookup_h", map="user:map_h", key="item:id_a"
                    ),
                )
            )
        return feature_cfgs

    @parameterized.expand([[False], [True]])
    @unittest.skipIf(
        "ODPS_CONFIG_FILE_PATH" not in os.environ
        and "ALIBABA_CLOUD_ECS_METADATA" not in os.environ,
        "odps config not found",
    )
    def test_odps_dataset(self, is_orderby_partition):
        account, odps_endpoint = _create_odps_account()
        self.o = ODPS(
            account=account,
            project=self.test_project,
            endpoint=odps_endpoint,
        )
        feature_cfgs = self._create_test_table_and_feature_cfgs()
        features = create_features(feature_cfgs, fg_mode=FgMode.FG_DAG)

        dataset = OdpsDataset(
            data_config=data_pb2.DataConfig(
                batch_size=8196,
                dataset_type=data_pb2.DatasetType.OdpsDataset,
                fg_mode=FgMode.FG_DAG,
                label_fields=["label"],
                is_orderby_partition=is_orderby_partition,
                odps_data_quota_name="",
            ),
            features=features,
            input_path=f"odps://{self.test_project}/tables/test_odps_dataset_{self.test_suffix}/dt=20240319&dt=20240320",
        )
        self.assertEqual(len(dataset.input_fields), 9)
        self.assertEqual(
            len(list(dataset._reader._input_to_sess.values())[0]),
            2 if is_orderby_partition else 1,
        )
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=None,
            num_workers=2,
            pin_memory=True,
            collate_fn=lambda x: x,
        )
        iterator = iter(dataloader)
        for _ in range(2):
            data = next(iterator)
            data_dict = data.to_dict()
            self.assertEqual(
                sorted(data_dict.keys()),
                [
                    "id_a.lengths",
                    "id_a.values",
                    "label",
                    "lookup_h.values",
                    "raw_c.values",
                    "raw_d.values",
                    "raw_e.values",
                    "raw_f.values",
                    "raw_g.values",
                    "tag_b.lengths",
                    "tag_b.values",
                ],
            )
            self.assertEqual(len(data_dict["id_a.lengths"]), 8196)

    @parameterized.expand([["bigint"], ["string"], ["int"]])
    @unittest.skipIf(
        "ODPS_CONFIG_FILE_PATH" not in os.environ
        and "ALIBABA_CLOUD_ECS_METADATA" not in os.environ,
        "odps config not found",
    )
    def test_odps_dataset_with_sampler(self, id_type):
        account, odps_endpoint = _create_odps_account()
        self.o = ODPS(
            account=account,
            project=self.test_project,
            endpoint=odps_endpoint,
        )
        if id_type == "string":
            os.environ["USE_HASH_NODE_ID"] = "1"
        feature_cfgs = self._create_test_table_and_feature_cfgs(has_lookup=False)

        self.o.delete_table(f"test_odps_sampler_{self.test_suffix}", if_exists=True)
        t = self.o.create_table(
            f"test_odps_sampler_{self.test_suffix}",
            (
                f"id {id_type}, weight double, features string",
                "dt string, alpha string",
            ),
            if_not_exists=True,
            hints={"odps.sql.type.system.odps2": "true"},
        )
        with t.open_writer(
            partition="dt=20240319,alpha=1", create_partition=True
        ) as writer:
            writer.write([[i, 1.0, f"{i}:4:5.0"] for i in range(10000)])

        features = create_features(
            feature_cfgs, fg_mode=FgMode.FG_DAG, neg_fields=["id_a", "raw_c", "raw_d"]
        )
        dataset = OdpsDataset(
            data_config=data_pb2.DataConfig(
                batch_size=8196,
                dataset_type=data_pb2.DatasetType.OdpsDataset,
                fg_mode=FgMode.FG_DAG,
                label_fields=["label"],
                odps_data_quota_name="",
                negative_sampler=sampler_pb2.NegativeSampler(
                    input_path=f"odps://{self.test_project}/tables/test_odps_sampler_{self.test_suffix}/dt=20240319/alpha=1",
                    num_sample=100,
                    attr_fields=["id_a", "raw_c", "raw_d"],
                    item_id_field="id_a",
                ),
            ),
            features=features,
            input_path=f"odps://{self.test_project}/tables/test_odps_dataset_{self.test_suffix}/dt=20240319&dt=20240320",
        )
        dataset.launch_sampler_cluster(2)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=None,
            num_workers=2,
            pin_memory=True,
            collate_fn=lambda x: x,
        )
        iterator = iter(dataloader)
        for _ in range(2):
            data = next(iterator)
            data_dict = data.to_dict()
            self.assertEqual(
                sorted(data_dict.keys()),
                [
                    "id_a.lengths",
                    "id_a.values",
                    "label",
                    "raw_c.values",
                    "raw_d.values",
                    "raw_e.values",
                    "raw_f.values",
                    "raw_g.values",
                    "tag_b.lengths",
                    "tag_b.values",
                ],
            )
            self.assertEqual(len(data_dict["id_a.lengths"]), 8296)
            self.assertEqual(len(data_dict["raw_c.values"]), 8296)
            self.assertEqual(len(data_dict["raw_e.values"]), 8196)


class OdpsWriterTest(unittest.TestCase):
    def setUp(self):
        self.o = None
        self.test_project = os.environ.get("CI_ODPS_PROJECT_NAME", None)
        if "ODPS_CONFIG_FILE_PATH" in os.environ:
            with open(os.environ["ODPS_CONFIG_FILE_PATH"], "r") as f:
                for line in f.readlines():
                    values = line.split("=", 1)
                    if len(values) == 2 and values[0] == "project_name":
                        self.test_project = values[1].strip()
        self.test_suffix = random_name()

    def tearDown(self):
        if self.o is not None:
            self.o.delete_table(f"test_odps_dataset_{self.test_suffix}", if_exists=True)
        self.o = None

    @parameterized.expand(
        [["", 2, 2], ["/dt=20240401", 2, 2], ["/dt=20240401", 2, 1]],
        name_func=test_util.parameterized_name_func,
    )
    @unittest.skipIf(
        "ODPS_CONFIG_FILE_PATH" not in os.environ
        and "ALIBABA_CLOUD_ECS_METADATA" not in os.environ,
        "odps config not found",
    )
    def test_odps_writer(self, partition_spec, default_world_size, writer_world_size):
        account, odps_endpoint = _create_odps_account()
        self.o = ODPS(
            account=account,
            project=self.test_project,
            endpoint=odps_endpoint,
        )

        def _writer_worker(rank, port):
            os.environ["RANK"] = str(rank)
            os.environ["WORLD_SIZE"] = str(default_world_size)
            os.environ["MASTER_ADDR"] = "127.0.0.1"
            os.environ["MASTER_PORT"] = str(port)
            dist.init_process_group(backend="gloo")
            time.sleep(rank)  # prevent get credential failed
            writer = OdpsWriter(
                f"odps://{self.test_project}/tables/test_odps_dataset_{self.test_suffix}{partition_spec}",
                quota_name="",
                world_size=writer_world_size,
            )
            for _ in range(5):
                writer.write(
                    {
                        "int_a": pa.array(np.random.randint(0, 100, 128)),
                        "float_b": pa.array(np.random.random(128)),
                    }
                )
            writer.close()

        writer_world_size = 2
        port = get_free_port()
        procs = []
        for rank in range(writer_world_size):
            p = mp.Process(
                target=_writer_worker,
                args=(rank, port),
            )
            p.start()
            procs.append(p)
        for i, p in enumerate(procs):
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"writer worker-{i} failed.")

        t = self.o.get_table(f"test_odps_dataset_{self.test_suffix}")
        if partition_spec:
            partition = ",".join(partition_spec.strip("/").split("/"))
        else:
            partition = None
        with t.open_reader(partition=partition) as reader:
            self.assertEqual(reader.count, 1280)


if __name__ == "__main__":
    unittest.main()
