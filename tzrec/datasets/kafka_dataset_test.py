# Copyright (c) 2026, Alibaba Group;
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
import time
import unittest

import pyarrow as pa
from alibabacloud_alikafka20190916 import models as alikafka_models
from alibabacloud_alikafka20190916.client import Client as AliKafkaClient
from alibabacloud_credentials.client import Client as CredClient
from alibabacloud_tea_openapi import models as openapi_models
from alibabacloud_tea_util import models as util_models
from confluent_kafka import Producer
from torch.utils.data import DataLoader

from tzrec.datasets.kafka_dataset import KafkaDataset
from tzrec.features.feature import FgMode, create_features
from tzrec.protos import data_pb2, feature_pb2
from tzrec.utils.checkpoint_util import update_dataloder_state
from tzrec.utils.logging_util import logger
from tzrec.utils.misc_util import random_name


class KafkaDatasetTest(unittest.TestCase):
    def setUp(self):
        credential = CredClient()
        config = openapi_models.Config(credential=credential)
        self.region = os.environ.get("CI_REGION")
        self.instance_id = os.environ.get("CI_ALIKAFKA_INSTANCE_ID")
        self.brokers = os.environ.get("CI_ALIKAFKA_BROKERS")

        config.endpoint = f"alikafka.{self.region}.aliyuncs.com"
        self.client = AliKafkaClient(config)
        self.test_topic = f"tzrec_kafka_test_v{random_name()}"

    def tearDown(self):
        try:
            req = alikafka_models.DeleteTopicRequest(
                instance_id=self.instance_id,
                topic=self.test_topic,
                region_id=self.region,
            )
            runtime = util_models.RuntimeOptions()
            self.client.delete_topic_with_options(req, runtime)
        except Exception as e:
            logger.error(e)

    def _create_test_table_and_feature_cfgs(self, has_lookup=True):
        create_topic_req = alikafka_models.CreateTopicRequest(
            instance_id=self.instance_id,
            topic=self.test_topic,
            remark=self.test_topic,
            region_id=self.region,
            partition_num=4,
        )
        runtime = util_models.RuntimeOptions()
        self.client.create_topic_with_options(create_topic_req, runtime)

        while True:
            runtime = util_models.RuntimeOptions()
            get_topic_request = alikafka_models.GetTopicListRequest(
                instance_id=self.instance_id,
                region_id=self.region,
                topic=self.test_topic,
            )
            resp = self.client.get_topic_list(get_topic_request)
            if resp.body.topic_list.topic_vo[0].status == 0:
                break
            time.sleep(10)

        input_fields = [
            data_pb2.Field(input_name="unused", input_type=data_pb2.STRING),
            data_pb2.Field(input_name="id_a", input_type=data_pb2.STRING),
            data_pb2.Field(input_name="tag_b", input_type=data_pb2.STRING),
            data_pb2.Field(input_name="raw_c", input_type=data_pb2.INT64),
            data_pb2.Field(input_name="raw_d", input_type=data_pb2.DOUBLE),
            data_pb2.Field(input_name="raw_e", input_type=data_pb2.INT32),
            data_pb2.Field(input_name="raw_f", input_type=data_pb2.FLOAT),
            data_pb2.Field(input_name="raw_g", input_type=data_pb2.ARRAY_FLOAT),
            data_pb2.Field(input_name="map_h", input_type=data_pb2.MAP_STRING_INT64),
            data_pb2.Field(input_name="label", input_type=data_pb2.INT64),
        ]
        config = {"bootstrap.servers": self.brokers}
        producer = Producer(config)

        for _ in range(10000):
            record_batch = pa.record_batch(
                {
                    "unused": pa.array(["unused"] * 128, type=pa.string()),
                    "id_a": pa.array(["1"] * 128, type=pa.string()),
                    "tag_b": pa.array(["2\x1d3"] * 128, type=pa.string()),
                    "raw_c": pa.array([4] * 128, type=pa.int64()),
                    "raw_d": pa.array([5.0] * 128, type=pa.float64()),
                    "raw_e": pa.array([3] * 128, type=pa.int32()),
                    "raw_f": pa.array([4.0] * 128, type=pa.float32()),
                    "raw_g": pa.array(
                        [[1.0, 2.0, 3.0]] * 128, type=pa.list_(pa.float32())
                    ),
                    "map_h": pa.array(
                        [{"1": 1, "2": 2}] * 128, type=pa.map_(pa.string(), pa.int64())
                    ),
                    "label": pa.array([0] * 128, type=pa.int64()),
                }
            )
            producer.produce(topic=self.test_topic, value=record_batch.serialize())
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
        return feature_cfgs, input_fields

    @unittest.skipIf(
        "CI_ALIKAFKA_INSTANCE_ID" not in os.environ, "ci kafka is not exists."
    )
    def test_kafka_dataset(self):
        feature_cfgs, input_fields = self._create_test_table_and_feature_cfgs()
        features = create_features(feature_cfgs, fg_mode=FgMode.FG_DAG)

        dataset = KafkaDataset(
            data_config=data_pb2.DataConfig(
                batch_size=8196,
                dataset_type=data_pb2.DatasetType.KafkaDataset,
                input_fields=input_fields,
                fg_mode=FgMode.FG_DAG,
                label_fields=["label"],
            ),
            features=features,
            input_path=f"kafka://{self.brokers}/{self.test_topic}?group.id=tzrec_test_group&auto.offset.reset=earliest",
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

    @unittest.skipIf(
        "CI_ALIKAFKA_INSTANCE_ID" not in os.environ, "ci kafka is not exists."
    )
    def test_kafka_dataset_checkpoint_metadata(self):
        feature_cfgs, input_fields = self._create_test_table_and_feature_cfgs(
            has_lookup=False
        )
        features = create_features(feature_cfgs, fg_mode=FgMode.FG_DAG)

        dataset = KafkaDataset(
            data_config=data_pb2.DataConfig(
                batch_size=8196,
                dataset_type=data_pb2.DatasetType.KafkaDataset,
                input_fields=input_fields,
                fg_mode=FgMode.FG_DAG,
                label_fields=["label"],
            ),
            features=features,
            input_path=f"kafka://{self.brokers}/{self.test_topic}?group.id=tzrec_test_group&auto.offset.reset=earliest",
        )
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=None,
            num_workers=2,
            pin_memory=True,
            collate_fn=lambda x: x,
        )
        iterator = iter(dataloader)
        batch = next(iterator)

        # Verify checkpoint_info is present and has correct format
        self.assertIsNotNone(batch.checkpoint_info)
        self.assertIsInstance(batch.checkpoint_info, dict)

        # Checkpoint keys should be in format "{topic}:{partition}"
        for key, value in batch.checkpoint_info.items():
            self.assertIn(":", key)
            topic_part, partition_str = key.rsplit(":", 1)
            self.assertEqual(topic_part, self.test_topic)
            self.assertTrue(partition_str.isdigit())
            # Value should be a non-negative offset
            self.assertIsInstance(value, int)
            self.assertGreaterEqual(value, 0)

    @unittest.skipIf(
        "CI_ALIKAFKA_INSTANCE_ID" not in os.environ, "ci kafka is not exists."
    )
    def test_kafka_dataset_checkpoint_resume(self):
        feature_cfgs, input_fields = self._create_test_table_and_feature_cfgs(
            has_lookup=False
        )
        features = create_features(feature_cfgs, fg_mode=FgMode.FG_DAG)

        # First, read some batches and capture checkpoint info
        dataset1 = KafkaDataset(
            data_config=data_pb2.DataConfig(
                batch_size=8196,
                dataset_type=data_pb2.DatasetType.KafkaDataset,
                input_fields=input_fields,
                fg_mode=FgMode.FG_DAG,
                label_fields=["label"],
            ),
            features=features,
            input_path=f"kafka://{self.brokers}/{self.test_topic}?group.id=tzrec_test_group&auto.offset.reset=earliest",
        )
        dataloader1 = DataLoader(
            dataset=dataset1,
            batch_size=None,
            num_workers=2,
            pin_memory=True,
            collate_fn=lambda x: x,
        )
        iterator1 = iter(dataloader1)

        # Read first batch and capture checkpoint
        checkpoint_state_acc = {}
        batch1 = next(iterator1)
        first_checkpoint_state = batch1.checkpoint_info.copy()
        update_dataloder_state(checkpoint_state_acc, first_checkpoint_state)
        self.assertIsNotNone(first_checkpoint_state)
        self.assertGreater(len(first_checkpoint_state), 0)
        # Read more batches
        for _ in range(8):
            batch1 = next(iterator1)
            update_dataloder_state(checkpoint_state_acc, batch1.checkpoint_info.copy())
        del iterator1
        del dataloader1

        # Now create a new dataset with checkpoint state set
        dataset2 = KafkaDataset(
            data_config=data_pb2.DataConfig(
                batch_size=8196,
                dataset_type=data_pb2.DatasetType.KafkaDataset,
                input_fields=input_fields,
                fg_mode=FgMode.FG_DAG,
                label_fields=["label"],
            ),
            features=features,
            input_path=f"kafka://{self.brokers}/{self.test_topic}?group.id=tzrec_test_group&auto.offset.reset=earliest",
        )
        dataloader2 = DataLoader(
            dataset=dataset2,
            batch_size=None,
            num_workers=2,
            pin_memory=True,
            collate_fn=lambda x: x,
        )
        # Set first batch checkpoint state to resume from saved offsets
        dataloader2.dataset.load_state_dict(first_checkpoint_state)
        iterator2 = iter(dataloader2)

        # Read batch from resumed dataset
        batch2 = next(iterator2)

        # The resumed batch should have offsets greater than the checkpoint
        # (since we resume from checkpoint_offset + 1)
        for key, new_offset in batch2.checkpoint_info.items():
            if key in first_checkpoint_state:
                # New offset should be greater than or equal to first checkpoint offset
                # (equal if we consumed the next message after checkpoint)
                self.assertGreaterEqual(new_offset, first_checkpoint_state[key])
            if key in checkpoint_state_acc:
                # New offset should be less than or equal to acc checkpoint offset
                # (equal if we consumed the next message after checkpoint)
                self.assertLessEqual(new_offset, checkpoint_state_acc[key])


if __name__ == "__main__":
    unittest.main()
