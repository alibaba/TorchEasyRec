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
import unittest

from odps import ODPS
from torch.utils.data import DataLoader

from tzrec.datasets.odps_dataset_v1 import OdpsDatasetV1
from tzrec.features.feature import BaseFeature
from tzrec.protos import data_pb2, feature_pb2
from tzrec.utils import config_util
from tzrec.utils.misc_util import random_name


class OdpsDatasetV1Test(unittest.TestCase):
    def setUp(self):
        self.odps_config = {}
        self.o = None
        if "ODPS_CONFIG_FILE_PATH" in os.environ:
            with open(os.environ["ODPS_CONFIG_FILE_PATH"], "r") as f:
                for line in f.readlines():
                    values = line.split("=", 1)
                    if len(values) == 2:
                        self.odps_config[values[0]] = values[1].strip()
            self.o = ODPS(
                access_id=self.odps_config["access_id"],
                secret_access_key=self.odps_config["access_key"],
                project=self.odps_config["project_name"],
                endpoint=self.odps_config["end_point"],
            )
        self.test_suffix = random_name()

    def tearDown(self):
        if self.o is not None:
            self.o.delete_table(f"test_odps_dataset_{self.test_suffix}", if_exists=True)

    @unittest.skipIf("ODPS_CONFIG_FILE_PATH" not in os.environ, "odps config not found")
    def test_odps_dataset(self):
        self.o.delete_table(f"test_odps_dataset_{self.test_suffix}", if_exists=True)
        t = self.o.create_table(
            f"test_odps_dataset_{self.test_suffix}",
            "unused int, id_a string, tag_b string, raw_c bigint, "
            "raw_d double, label bigint",
            if_not_exists=True,
        )
        with t.open_writer() as writer:
            writer.write([[0, "1", "2\x033", 4, 5.0, 0]] * 10000)

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
        features = []
        for cfg in feature_cfgs:
            feature_cls_name = config_util.which_msg(cfg, "feature")
            features.append(BaseFeature.create_class(feature_cls_name)(cfg))

        dataset = OdpsDatasetV1(
            data_config=data_pb2.DataConfig(
                batch_size=4,
                dataset_type=data_pb2.DatasetType.OdpsDataset,
                fg_mode=data_pb2.FgMode.FG_NONE,
                label_fields=["label"],
            ),
            features=features,
            input_path=f'odps://{self.odps_config["project_name"]}/tables/test_odps_dataset_{self.test_suffix}',
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


if __name__ == "__main__":
    unittest.main()
