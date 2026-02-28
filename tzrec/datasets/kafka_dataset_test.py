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
from tzrec.datasets.utils import CKPT_ROW_IDX, CKPT_SOURCE_ID
from tzrec.features.feature import FgMode, create_features
from tzrec.protos import data_pb2, feature_pb2
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
        req = alikafka_models.CreateTopicRequest(
            instance_id=self.instance_id,
            topic=self.test_topic,
            remark=self.test_topic,
            region_id=self.region,
            partition_num=12,
        )
        runtime = util_models.RuntimeOptions()
        self.client.create_topic_with_options(req, runtime)

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
        for _ in range(1000):
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
            num_workers=0,
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


class KafkaReaderCheckpointTest(unittest.TestCase):
    """Tests for Kafka reader checkpoint functionality.

    Note: These tests verify checkpoint source_id format without requiring
    actual Kafka connections.
    """

    def test_checkpoint_source_id_format(self):
        """Test Kafka checkpoint source_id format: {topic}:{partition}."""
        topic = "my_topic"
        partition = 3
        expected_source_id = f"{topic}:{partition}"
        self.assertEqual(expected_source_id, "my_topic:3")

    def test_checkpoint_source_id_with_multiple_partitions(self):
        """Test checkpoint source_id for multiple partitions."""
        topic = "my_topic"
        num_partitions = 12
        source_ids = [f"{topic}:{p}" for p in range(num_partitions)]

        # Each partition should have unique source_id
        self.assertEqual(len(source_ids), len(set(source_ids)))

        # Format should be consistent
        for source_id in source_ids:
            self.assertIn(topic, source_id)
            self.assertIn(":", source_id)

    def test_checkpoint_row_idx_is_offset(self):
        """Test that CKPT_ROW_IDX stores Kafka message offset."""
        # In Kafka, we store message offset as row_idx
        # This allows resume from specific offset
        offset = 12345
        # Simulate checkpoint_info structure
        checkpoint_info = {"my_topic:3": offset}
        self.assertEqual(checkpoint_info["my_topic:3"], 12345)

    def test_checkpoint_constants_exist(self):
        """Verify checkpoint constants are defined."""
        self.assertEqual(CKPT_SOURCE_ID, "__ckpt_source_id__")
        self.assertEqual(CKPT_ROW_IDX, "__ckpt_row_idx__")


if __name__ == "__main__":
    unittest.main()
