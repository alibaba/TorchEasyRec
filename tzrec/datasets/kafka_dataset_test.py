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


import io
import multiprocessing as mp
import os
import shutil
import signal
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pyarrow as pa
from alibabacloud_alikafka20190916 import models as alikafka_models
from alibabacloud_alikafka20190916.client import Client as AliKafkaClient
from alibabacloud_credentials.client import Client as CredClient
from alibabacloud_tea_openapi import models as openapi_models
from alibabacloud_tea_util import models as util_models
from confluent_kafka import KafkaException, Producer, TopicPartition
from parameterized import param, parameterized
from torch.utils.data import DataLoader

from tzrec.constant import Mode
from tzrec.datasets import kafka_dataset
from tzrec.datasets.kafka_dataset import (
    KafkaDataset,
    _build_projection,
    _offsets_for_times_batched,
    _parse_kafka_uri,
    resolve_start_offsets,
    validate_sample_shuffle_config,
)
from tzrec.datasets.utils import CKPT_ROW_IDX, CKPT_SOURCE_ID, DATA_TIMESTAMP
from tzrec.features.feature import FgMode, create_features
from tzrec.protos import data_pb2, feature_pb2
from tzrec.protos.eval_pb2 import EvalConfig
from tzrec.protos.optimizer_pb2 import DenseOptimizer
from tzrec.protos.train_pb2 import TrainConfig
from tzrec.utils.checkpoint_util import update_dataloder_state
from tzrec.utils.logging_util import logger
from tzrec.utils.misc_util import random_name
from tzrec.utils.test_util import make_test_dir, parameterized_name_func


class ParseKafkaUriTest(unittest.TestCase):
    """Unit tests for _parse_kafka_uri."""

    def test_parse_without_start_timestamp(self):
        uri = "kafka://broker:9092/topic?group.id=g1&auto.offset.reset=earliest"
        topic, params, start_ts = _parse_kafka_uri(uri)
        self.assertEqual(topic, "topic")
        self.assertEqual(params["bootstrap.servers"], "broker:9092")
        self.assertEqual(params["group.id"], "g1")
        self.assertIsNone(start_ts)
        self.assertNotIn("start.timestamp.ms", params)

    def test_parse_with_start_timestamp(self):
        uri = "kafka://broker:9092/topic?group.id=g1&start.timestamp.ms=1711929600000"
        topic, params, start_ts = _parse_kafka_uri(uri)
        self.assertEqual(topic, "topic")
        self.assertEqual(start_ts, 1711929600000)
        # start.timestamp.ms should be removed from consumer params
        self.assertNotIn("start.timestamp.ms", params)

    def test_parse_with_zero_timestamp(self):
        uri = "kafka://broker:9092/topic?group.id=g1&start.timestamp.ms=0"
        _, _, start_ts = _parse_kafka_uri(uri)
        self.assertEqual(start_ts, 0)

    def test_parse_with_negative_timestamp_raises(self):
        uri = "kafka://broker:9092/topic?group.id=g1&start.timestamp.ms=-1"
        with self.assertRaises(ValueError):
            _parse_kafka_uri(uri)

    def test_parse_with_non_integer_timestamp_raises(self):
        uri = "kafka://broker:9092/topic?group.id=g1&start.timestamp.ms=abc"
        with self.assertRaises(ValueError):
            _parse_kafka_uri(uri)


def _resolved(partition, offset, topic="t"):
    """Build a TopicPartition as returned by offsets_for_times."""
    tp = TopicPartition(topic, partition)
    tp.offset = offset
    return tp


class OffsetsForTimesBatchedTest(unittest.TestCase):
    """Unit tests for _offsets_for_times_batched (broker-free, mocked)."""

    START_TS = 1711929600000

    def test_empty_partitions_returns_empty(self):
        consumer = mock.MagicMock()
        result = _offsets_for_times_batched(
            consumer, [], self.START_TS, 30, 30.0, 3, sleep_fn=mock.MagicMock()
        )
        self.assertEqual(result, {})
        consumer.offsets_for_times.assert_not_called()

    def test_single_chunk_all_resolved(self):
        seen_offsets = []

        def _fake(chunk, timeout):
            # offsets_for_times receives the target timestamp in tp.offset.
            seen_offsets.extend(tp.offset for tp in chunk)
            return [_resolved(tp.partition, tp.partition + 100) for tp in chunk]

        consumer = mock.MagicMock()
        consumer.offsets_for_times.side_effect = _fake
        parts = [TopicPartition("t", p) for p in range(5)]
        result = _offsets_for_times_batched(
            consumer, parts, self.START_TS, 30, 30.0, 3, sleep_fn=mock.MagicMock()
        )
        self.assertEqual(consumer.offsets_for_times.call_count, 1)
        self.assertEqual(result, {("t", p): p + 100 for p in range(5)})
        # target timestamp was set on every partition before the call.
        self.assertEqual(seen_offsets, [self.START_TS] * 5)

    def test_chunking_boundary(self):
        consumer = mock.MagicMock()
        consumer.offsets_for_times.side_effect = lambda chunk, timeout: [
            _resolved(tp.partition, tp.partition + 100) for tp in chunk
        ]
        parts = [TopicPartition("t", p) for p in range(65)]
        result = _offsets_for_times_batched(
            consumer, parts, self.START_TS, 30, 30.0, 3, sleep_fn=mock.MagicMock()
        )
        chunk_sizes = [
            len(call.args[0]) for call in consumer.offsets_for_times.call_args_list
        ]
        self.assertEqual(chunk_sizes, [30, 30, 5])
        self.assertEqual(len(result), 65)

    def test_partial_resolution_filters_negative(self):
        # broker returns -1 when the timestamp is past the last message.
        consumer = mock.MagicMock()
        consumer.offsets_for_times.return_value = [
            _resolved(0, 100),
            _resolved(1, -1),
            _resolved(2, 102),
        ]
        parts = [TopicPartition("t", p) for p in range(3)]
        result = _offsets_for_times_batched(
            consumer, parts, self.START_TS, 30, 30.0, 3, sleep_fn=mock.MagicMock()
        )
        self.assertEqual(result, {("t", 0): 100, ("t", 2): 102})
        self.assertNotIn(("t", 1), result)

    def test_retry_then_success(self):
        sleep_fn = mock.MagicMock()
        consumer = mock.MagicMock()
        consumer.offsets_for_times.side_effect = [
            KafkaException("transient timeout"),
            [_resolved(0, 100)],
        ]
        parts = [TopicPartition("t", 0)]
        result = _offsets_for_times_batched(
            consumer, parts, self.START_TS, 30, 30.0, 3, sleep_fn=sleep_fn
        )
        self.assertEqual(result, {("t", 0): 100})
        self.assertEqual(consumer.offsets_for_times.call_count, 2)
        self.assertEqual(sleep_fn.call_count, 1)

    def test_retry_exhausted_raises(self):
        sleep_fn = mock.MagicMock()
        consumer = mock.MagicMock()
        consumer.offsets_for_times.side_effect = KafkaException("persistent timeout")
        parts = [TopicPartition("t", 0)]
        with self.assertRaises(KafkaException):
            _offsets_for_times_batched(
                consumer, parts, self.START_TS, 30, 30.0, 3, sleep_fn=sleep_fn
            )
        # 1 initial attempt + 3 retries = 4 calls; sleep before each retry = 3.
        self.assertEqual(consumer.offsets_for_times.call_count, 4)
        self.assertEqual(sleep_fn.call_count, 3)

    def test_retry_isolated_per_chunk(self):
        sleep_fn = mock.MagicMock()
        # chunk 0 fails once then succeeds; chunk 1 succeeds first try.
        outcomes = [
            KafkaException("blip"),
            [_resolved(0, 100)],
            [_resolved(1, 101)],
        ]
        consumer = mock.MagicMock()
        consumer.offsets_for_times.side_effect = outcomes
        parts = [TopicPartition("t", 0), TopicPartition("t", 1)]
        result = _offsets_for_times_batched(
            consumer, parts, self.START_TS, 1, 30.0, 3, sleep_fn=sleep_fn
        )
        self.assertEqual(result, {("t", 0): 100, ("t", 1): 101})
        self.assertEqual(consumer.offsets_for_times.call_count, 3)
        self.assertEqual(sleep_fn.call_count, 1)

    def test_env_defaults_resolved_at_call_time(self):
        seen = []

        def _fake(chunk, timeout):
            seen.append((len(chunk), timeout))
            return [_resolved(tp.partition, tp.partition) for tp in chunk]

        consumer = mock.MagicMock()
        consumer.offsets_for_times.side_effect = _fake
        parts = [TopicPartition("t", p) for p in range(5)]
        env = {
            "TZREC_KAFKA_OFFSETS_FOR_TIMES_BATCH_SIZE": "2",
            "TZREC_KAFKA_OFFSETS_FOR_TIMES_TIMEOUT": "0.5",
        }
        with mock.patch.dict(os.environ, env):
            result = _offsets_for_times_batched(
                consumer, parts, self.START_TS, sleep_fn=mock.MagicMock()
            )
        self.assertEqual(seen, [(2, 0.5), (2, 0.5), (1, 0.5)])
        self.assertEqual(len(result), 5)

    def test_env_retries_resolved_at_call_time(self):
        consumer = mock.MagicMock()
        consumer.offsets_for_times.side_effect = KafkaException("timeout")
        parts = [TopicPartition("t", 0)]
        with mock.patch.dict(
            os.environ, {"TZREC_KAFKA_OFFSETS_FOR_TIMES_RETRIES": "0"}
        ):
            with self.assertRaises(KafkaException):
                _offsets_for_times_batched(
                    consumer, parts, self.START_TS, sleep_fn=mock.MagicMock()
                )
        self.assertEqual(consumer.offsets_for_times.call_count, 1)


class BuildProjectionTest(unittest.TestCase):
    """Unit tests for _build_projection."""

    DST = pa.schema(
        [
            pa.field("id_a", pa.string()),
            pa.field("raw_c", pa.int64()),
            pa.field("label", pa.int64()),
        ]
    )

    def test_identical_schema_needs_no_projection(self):
        self.assertIsNone(_build_projection(self.DST, self.DST))

    def test_upstream_added_column(self):
        src = pa.schema(
            [
                pa.field("id_a", pa.string()),
                pa.field("raw_c", pa.int64()),
                pa.field("label", pa.int64()),
                pa.field("new_col", pa.float32()),
            ]
        )
        self.assertEqual(_build_projection(src, self.DST), [0, 1, 2])

    def test_upstream_reordered_column(self):
        src = pa.schema(
            [
                pa.field("label", pa.int64()),
                pa.field("new_col", pa.float32()),
                pa.field("id_a", pa.string()),
                pa.field("raw_c", pa.int64()),
            ]
        )
        projection = _build_projection(src, self.DST)
        assert projection is not None
        self.assertEqual(projection, [2, 3, 0])
        batch = pa.record_batch(
            [
                pa.array([7], type=pa.int64()),
                pa.array([1.5], type=pa.float32()),
                pa.array(["a"], type=pa.string()),
                pa.array([3], type=pa.int64()),
            ],
            schema=src,
        )
        projected = pa.RecordBatch.from_arrays(
            [batch.column(i) for i in projection], names=self.DST.names
        )
        self.assertEqual(
            projected.to_pydict(), {"id_a": ["a"], "raw_c": [3], "label": [7]}
        )

    def test_missing_column_raises(self):
        src = pa.schema([pa.field("id_a", pa.string()), pa.field("label", pa.int64())])
        with self.assertRaisesRegex(ValueError, r"missing column\(s\) \['raw_c'\]"):
            _build_projection(src, self.DST)

    def test_changed_column_type_raises(self):
        src = pa.schema(
            [
                pa.field("id_a", pa.string()),
                pa.field("raw_c", pa.int32()),
                pa.field("label", pa.int64()),
            ]
        )
        with self.assertRaisesRegex(ValueError, r"changed column type\(s\).*raw_c"):
            _build_projection(src, self.DST)

    @parameterized.expand(
        [
            param(
                "list",
                dst_type=pa.list_(pa.float32()),
                src_type=pa.list_(pa.field("element", pa.float32(), False)),
            ),
            param(
                "map",
                dst_type=pa.map_(pa.string(), pa.int64()),
                src_type=pa.map_(
                    pa.field("key", pa.string(), False),
                    pa.field("value", pa.int64(), False),
                ),
            ),
        ]
    )
    def test_nested_nullability_is_normalized(self, _name, dst_type, src_type):
        """A nullability-only delta is accepted, and casting makes batches combine."""
        dst = pa.schema([pa.field("raw_g", dst_type)])
        src = pa.schema([pa.field("raw_g", src_type)])
        projection = _build_projection(src, dst)
        assert projection is not None
        self.assertEqual(projection, [0])

        value = [1.0, 2.0] if pa.types.is_list(dst_type) else [("k", 1)]
        src_batch = pa.record_batch([pa.array([value], type=src_type)], schema=src)
        dst_batch = pa.record_batch([pa.array([value], type=dst_type)], schema=dst)
        projected = pa.RecordBatch.from_arrays(
            [src_batch.column(i) for i in projection], names=dst.names
        ).cast(dst)
        self.assertEqual(len(pa.Table.from_batches([projected, dst_batch])), 2)


class KafkaDatasetConfigTest(unittest.TestCase):
    """data_config fields reach the reader, broker-free."""

    @parameterized.expand(
        [
            param("train", mode=Mode.TRAIN, expected_shuffle=True),
            param("eval", mode=Mode.EVAL, expected_shuffle=False),
        ]
    )
    def test_shuffle_config_reaches_reader(self, _name, mode, expected_shuffle):
        data_config = data_pb2.DataConfig(
            batch_size=4,
            dataset_type=data_pb2.DatasetType.KafkaDataset,
            fg_mode=data_pb2.FgMode.FG_NONE,
            label_fields=["label"],
            shuffle=True,
            shuffle_buffer_size=64,
            input_fields=[
                data_pb2.Field(input_name="id_a", input_type=data_pb2.STRING),
                data_pb2.Field(input_name="label", input_type=data_pb2.INT64),
            ],
        )
        features = create_features(
            [
                feature_pb2.FeatureConfig(
                    id_feature=feature_pb2.IdFeature(feature_name="id_a")
                )
            ]
        )
        dataset = KafkaDataset(
            data_config=data_config,
            features=features,
            input_path="kafka://broker:9092/topic?group.id=tzrec_test_group",
            mode=mode,
        )
        self.assertEqual(dataset._reader._shuffle, expected_shuffle)
        self.assertEqual(dataset._reader._shuffle_buffer_size, 64)


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

    def _create_test_table_and_feature_cfgs(self, embedded_schema=False):
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

        input_fields_str = (
            "unused:STRING;id_a:STRING;tag_b:STRING;raw_c:INT64;"
            "raw_d:DOUBLE;raw_e:INT32;raw_f:FLOAT;raw_g:ARRAY<FLOAT>;"
            "map_h:MAP<STRING,INT64>;label:INT64"
        )
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

        for i in range(10000):
            columns = {
                "unused": pa.array(["unused"] * 128, type=pa.string()),
                "id_a": pa.array(["1"] * 128, type=pa.string()),
                "tag_b": pa.array(["2\x1d3"] * 128, type=pa.string()),
                "raw_c": pa.array([4] * 128, type=pa.int64()),
                "raw_d": pa.array([5.0] * 128, type=pa.float64()),
                "raw_e": pa.array([3] * 128, type=pa.int32()),
                "raw_f": pa.array([4.0] * 128, type=pa.float32()),
                "raw_g": pa.array([[1.0, 2.0, 3.0]] * 128, type=pa.list_(pa.float32())),
                "map_h": pa.array(
                    [{"1": 1, "2": 2}] * 128, type=pa.map_(pa.string(), pa.int64())
                ),
                "label": pa.array([0] * 128, type=pa.int64()),
            }
            # emulate an upstream schema change: a new column appears and the
            # existing ones are reordered. Alternating per message so that a
            # single consume() call mixes both schemas.
            if embedded_schema and i % 2 == 1:
                columns["added_i"] = pa.array([6] * 128, type=pa.int64())
                columns = dict(reversed(list(columns.items())))
            record_batch = pa.record_batch(columns)
            if embedded_schema:
                # real producers stamp key-value metadata on the embedded schema
                record_batch = record_batch.replace_schema_metadata({b"pandas": b"{}"})
                # Serialize with embedded schema using IPC stream format
                sink = io.BytesIO()
                with pa.ipc.new_stream(sink, record_batch.schema) as writer:
                    writer.write_batch(record_batch)
                value = sink.getvalue()
            else:
                value = record_batch.serialize()
            producer.produce(topic=self.test_topic, value=value)
        producer.flush()
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
            feature_pb2.FeatureConfig(
                lookup_feature=feature_pb2.LookupFeature(
                    feature_name="lookup_h", map="user:map_h", key="item:id_a"
                ),
            ),
        ]
        return feature_cfgs, input_fields, input_fields_str

    @parameterized.expand([[False, False], [True, False], [True, True]])
    @unittest.skipIf(
        os.environ.get("CI_ALIKAFKA_INSTANCE_ID", "") == "", "ci kafka is not exists."
    )
    def test_kafka_dataset(self, use_input_fields_str, embedded_schema):
        feature_cfgs, input_fields, input_fields_str = (
            self._create_test_table_and_feature_cfgs(embedded_schema=embedded_schema)
        )
        features = create_features(feature_cfgs, fg_mode=FgMode.FG_DAG)

        data_config = data_pb2.DataConfig(
            batch_size=8196,
            dataset_type=data_pb2.DatasetType.KafkaDataset,
            fg_mode=FgMode.FG_DAG,
            label_fields=["label"],
        )
        if not embedded_schema:
            if use_input_fields_str:
                data_config.input_fields_str = input_fields_str
            else:
                data_config.input_fields.extend(input_fields)

        dataset = KafkaDataset(
            data_config=data_config,
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
        os.environ.get("CI_ALIKAFKA_INSTANCE_ID", "") == "", "ci kafka is not exists."
    )
    def test_kafka_dataset_checkpoint_metadata(self):
        feature_cfgs, input_fields, _ = self._create_test_table_and_feature_cfgs()
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

        # event-time in positive Unix-epoch seconds (-1.0 only without timestamps)
        self.assertIsInstance(batch.data_timestamp, float)
        self.assertGreater(batch.data_timestamp, 0)

    @unittest.skipIf(
        os.environ.get("CI_ALIKAFKA_INSTANCE_ID", "") == "", "ci kafka is not exists."
    )
    def test_kafka_dataset_checkpoint_resume(self):
        feature_cfgs, input_fields, _ = self._create_test_table_and_feature_cfgs()
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

    @unittest.skipIf(
        os.environ.get("CI_ALIKAFKA_INSTANCE_ID", "") == "", "ci kafka is not exists."
    )
    def test_kafka_dataset_start_timestamp(self):
        """Test that start.timestamp.ms skips messages produced before the timestamp."""
        feature_cfgs, input_fields, _ = self._create_test_table_and_feature_cfgs()
        features = create_features(feature_cfgs, fg_mode=FgMode.FG_DAG)

        # Record the timestamp (ms) after the first batch of messages is produced
        # by _create_test_table_and_feature_cfgs, then produce a second batch.
        # The start timestamp should cause the consumer to skip the first batch.
        start_ts_ms = int(time.time() * 1000)
        time.sleep(2)

        # Produce a second batch of messages after the timestamp boundary
        config = {"bootstrap.servers": self.brokers}
        producer = Producer(config)
        second_batch_count = 100
        for _ in range(second_batch_count):
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
                        [{"1": 1, "2": 2}] * 128,
                        type=pa.map_(pa.string(), pa.int64()),
                    ),
                    "label": pa.array([1] * 128, type=pa.int64()),
                }
            )
            producer.produce(topic=self.test_topic, value=record_batch.serialize())
        producer.flush()

        # Use start.timestamp.ms to skip the first batch of 10000 messages
        dataset = KafkaDataset(
            data_config=data_pb2.DataConfig(
                batch_size=8196,
                dataset_type=data_pb2.DatasetType.KafkaDataset,
                input_fields=input_fields,
                fg_mode=FgMode.FG_DAG,
                label_fields=["label"],
            ),
            features=features,
            input_path=(
                f"kafka://{self.brokers}/{self.test_topic}"
                f"?group.id=tzrec_test_group"
                f"&auto.offset.reset=earliest"
                f"&start.timestamp.ms={start_ts_ms}"
            ),
        )
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=None,
            num_workers=0,
            pin_memory=True,
            collate_fn=lambda x: x,
        )
        iterator = iter(dataloader)
        batch = next(iterator)

        # Verify labels are from the second batch (label=1, not label=0).
        # The first batch produced by _create_test_table_and_feature_cfgs uses
        # label=0, and the second batch produced above uses label=1.
        data_dict = batch.to_dict()
        labels = data_dict["label"]
        for label in labels:
            self.assertEqual(label, 1)

    @unittest.skipIf(
        os.environ.get("CI_ALIKAFKA_INSTANCE_ID", "") == "", "ci kafka is not exists."
    )
    def test_kafka_dataset_heartbeat_no_max_poll_exceeded(self):
        """Test heartbeat prevents MAX_POLL_EXCEEDED during long idle."""
        feature_cfgs, input_fields, _ = self._create_test_table_and_feature_cfgs()
        features = create_features(feature_cfgs, fg_mode=FgMode.FG_DAG)

        # Use a low max.poll.interval.ms (10s) so idle_threshold = 8s.
        # Sleep 12s between reads to exceed max.poll.interval.ms.
        # Without the heartbeat, this would trigger MAX_POLL_EXCEEDED.
        dataset = KafkaDataset(
            data_config=data_pb2.DataConfig(
                batch_size=8196,
                dataset_type=data_pb2.DatasetType.KafkaDataset,
                input_fields=input_fields,
                fg_mode=FgMode.FG_DAG,
                label_fields=["label"],
            ),
            features=features,
            input_path=(
                f"kafka://{self.brokers}/{self.test_topic}"
                f"?group.id=tzrec_test_group"
                f"&auto.offset.reset=earliest"
                f"&session.timeout.ms=6000"
                f"&max.poll.interval.ms=10000"
            ),
        )
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=None,
            num_workers=0,
            pin_memory=True,
            collate_fn=lambda x: x,
        )
        iterator = iter(dataloader)

        # Read first batch
        batch1 = next(iterator)
        data_dict1 = batch1.to_dict()
        self.assertEqual(len(data_dict1["id_a.lengths"]), 8196)

        # Sleep longer than max.poll.interval.ms (10s).
        # Heartbeat should activate at 8s (80% of 10s) and keep alive.
        time.sleep(12)

        # Read second batch — should succeed without MAX_POLL_EXCEEDED
        batch2 = next(iterator)
        data_dict2 = batch2.to_dict()
        self.assertEqual(len(data_dict2["id_a.lengths"]), 8196)
        self.assertEqual(
            sorted(data_dict2.keys()),
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


def make_batch(n, start_id=0, partition=0, rows_per_msg=16, offset_start=0):
    """Build a record batch whose every column derives from a unique row id."""
    ids = np.arange(start_id, start_id + n, dtype=np.int64)
    offsets = offset_start + ids // rows_per_msg
    return pa.RecordBatch.from_arrays(
        [
            pa.array(ids),
            pa.array(ids.astype(np.float64) * 1.5),
            pa.array([f"str_{i}" for i in ids]),
            pa.array([[int(i), int(i + 1)] for i in ids]),
            pa.array(ids % 2),
            pa.array(ids.astype(np.float32) * 0.25),
            pa.array([f"topic:{partition}"] * n),
            pa.array(offsets),
            pa.array(ids.astype(np.float64) + 0.5),
        ],
        names=[
            "id",
            "feat",
            "str_f",
            "list_f",
            "label",
            "weight",
            CKPT_SOURCE_ID,
            CKPT_ROW_IDX,
            DATA_TIMESTAMP,
        ],
    )


def check_row_correspondence(tc, batch, partition=0, offset_start=0, rows_per_msg=16):
    """All columns of an emitted batch must follow the row id permutation."""
    ids = batch.column("id").to_pylist()
    feats = batch.column("feat").to_pylist()
    strs = batch.column("str_f").to_pylist()
    lists = batch.column("list_f").to_pylist()
    labels = batch.column("label").to_pylist()
    weights = batch.column("weight").to_pylist()
    sources = batch.column(CKPT_SOURCE_ID).to_pylist()
    offsets = batch.column(CKPT_ROW_IDX).to_pylist()
    tss = batch.column(DATA_TIMESTAMP).to_pylist()
    for i, f, s, li, la, w, src, off, ts in zip(
        ids, feats, strs, lists, labels, weights, sources, offsets, tss
    ):
        tc.assertAlmostEqual(f, i * 1.5)
        tc.assertEqual(s, f"str_{i}")
        tc.assertEqual(li, [i, i + 1])
        tc.assertEqual(la, i % 2)
        tc.assertAlmostEqual(w, i * 0.25, places=5)
        tc.assertEqual(src, f"topic:{partition}")
        tc.assertEqual(off, offset_start + i // rows_per_msg)
        tc.assertAlmostEqual(ts, i + 0.5)


class SamplePoolTest(unittest.TestCase):
    def _pool(self, capacity, batch_size, mode, seed=0):
        rng = np.random.default_rng(seed)
        return kafka_dataset.SamplePool(capacity, batch_size, mode, rng=rng)

    @parameterized.expand(
        [
            (data_pb2.SAMPLE_WINDOW,),
            (data_pb2.SAMPLE_ROLLING,),
        ],
        name_func=parameterized_name_func,
    )
    def test_invalid_config(self, mode):
        with self.assertRaises(ValueError):
            kafka_dataset.SamplePool(0, 16, mode)
        with self.assertRaises(ValueError):
            kafka_dataset.SamplePool(16, 0, mode)
        with self.assertRaises(ValueError):
            kafka_dataset.SamplePool(16, 16, 999)

    @parameterized.expand(
        [
            (data_pb2.SAMPLE_WINDOW,),
            (data_pb2.SAMPLE_ROLLING,),
        ],
        name_func=parameterized_name_func,
    )
    def test_no_emit_until_full(self, mode):
        pool = self._pool(128, 16, mode)
        # empty pool
        self.assertIsNone(pool.emit())
        # fewer rows than one batch: retained, nothing emitted
        pool.add(make_batch(8))
        self.assertIsNone(pool.emit())
        self.assertEqual(pool.total_rows, 8)
        # still short of capacity
        pool.add(make_batch(64, start_id=8))
        self.assertIsNone(pool.emit())
        # fill past capacity: emit one full batch
        pool.add(make_batch(64, start_id=72))
        out = pool.emit()
        self.assertIsNotNone(out)
        self.assertEqual(out.num_rows, 16)
        if mode == data_pb2.SAMPLE_WINDOW:
            # a frozen window keeps emitting its remaining full batches
            self.assertIsNotNone(pool.emit())
        else:
            # rolling waits for a refill to capacity
            self.assertIsNone(pool.emit())

    @parameterized.expand(
        [(1,), (16,)],
        name_func=parameterized_name_func,
    )
    def test_batch_size_one_and_capacity_one_batch(self, capacity_batches):
        batch_size = 8
        pool = self._pool(
            batch_size * capacity_batches, batch_size, data_pb2.SAMPLE_ROLLING
        )
        pool.add(make_batch(batch_size * capacity_batches))
        out = pool.emit()
        self.assertIsNotNone(out)
        self.assertEqual(out.num_rows, batch_size)

    def test_window_emits_all_full_batches_without_new_data(self):
        pool = self._pool(96, 16, data_pb2.SAMPLE_WINDOW)
        pool.add(make_batch(100))
        seen = []
        while True:
            out = pool.emit()
            if out is None:
                break
            seen.append(out)
        self.assertEqual(sum(b.num_rows for b in seen), 96)  # 6 full batches
        self.assertEqual(pool.total_rows, 4)  # leftover retained

    def test_window_arrivals_not_mixed_into_frozen_window(self):
        pool = self._pool(64, 16, data_pb2.SAMPLE_WINDOW)
        pool.add(make_batch(64, start_id=0))
        first = pool.emit()
        self.assertIsNotNone(first)
        first_ids = set(first.column("id").to_pylist())
        # arrivals during the frozen phase
        pool.add(make_batch(32, start_id=1000))
        rest = []
        while True:
            out = pool.emit()
            if out is None:
                break
            rest.append(out)
        frozen_ids = set(first_ids)
        for b in rest:
            frozen_ids.update(b.column("id").to_pylist())
        self.assertTrue(all(i < 64 for i in frozen_ids))
        # the 32 new rows wait for the next full window
        self.assertEqual(pool.total_rows, 32)
        self.assertIsNone(pool.emit())
        pool.add(make_batch(32, start_id=2000))
        out = pool.emit()
        self.assertIsNotNone(out)
        self.assertTrue(all(i >= 1000 for i in out.column("id").to_pylist()))

    def test_rolling_extract_then_refill(self):
        # deterministic rng: the first batch is a fixed subset
        pool = self._pool(32, 16, data_pb2.SAMPLE_ROLLING, seed=7)
        pool.add(make_batch(32, start_id=0))
        first = pool.emit()
        self.assertIsNotNone(first)
        first_ids = set(first.column("id").to_pylist())
        self.assertEqual(len(first_ids), 16)
        self.assertEqual(pool.total_rows, 16)
        # no emit before the refill reaches capacity
        self.assertIsNone(pool.emit())
        pool.add(make_batch(16, start_id=32))
        second = pool.emit()
        self.assertIsNotNone(second)
        second_ids = set(second.column("id").to_pylist())
        # the second batch mixes old and new samples and repeats nothing
        self.assertEqual(len(second_ids & first_ids), 0)
        self.assertTrue(any(i < 32 for i in second_ids))
        self.assertTrue(any(i >= 32 for i in second_ids))
        self.assertEqual(pool.total_rows, 16)

    def test_row_correspondence_and_conservation(self):
        for mode in (data_pb2.SAMPLE_WINDOW, data_pb2.SAMPLE_ROLLING):
            pool = self._pool(96, 16, mode)
            total_in = 0
            for i in range(6):
                pool.add(
                    make_batch(
                        32, start_id=32 * i, partition=i % 3, offset_start=100 * (i % 3)
                    )
                )
                total_in += 32
            emitted = []
            while True:
                out = pool.emit()
                if out is None:
                    if mode == data_pb2.SAMPLE_WINDOW:
                        break
                    # rolling: refill past capacity and continue
                    if pool.total_rows < 96:
                        break
                if out is None:
                    break
                emitted.append(out)
            for batch in emitted:
                # per-row correspondence is per source partition
                for part in range(3):
                    cols = batch.filter(
                        pa.compute.equal(
                            batch.column(CKPT_SOURCE_ID), pa.scalar(f"topic:{part}")
                        )
                    )
                    check_row_correspondence(
                        self, cols, partition=part, offset_start=100 * part
                    )
            rows_out = sum(b.num_rows for b in emitted)
            self.assertEqual(rows_out + pool.total_rows, total_in)
            self.assertEqual(pool.rows_in, total_in)
            self.assertEqual(pool.rows_out, rows_out)

    def test_msg_split_and_batch_spanning_offsets(self):
        pool = self._pool(40, 16, data_pb2.SAMPLE_WINDOW)
        # one message of 40 rows (all at offset 7) spans multiple batches
        pool.add(make_batch(40, rows_per_msg=40, offset_start=7))
        offsets_seen = set()
        while True:
            out = pool.emit()
            if out is None:
                break
            offsets_seen.update(out.column(CKPT_ROW_IDX).to_pylist())
        # 40 rows -> 2 full batches, 8 retained; all carry offset 7
        self.assertEqual(offsets_seen, {7})
        self.assertEqual(pool.total_rows, 8)

    def test_window_boundary_compaction_bounds_memory(self):
        # reviewer scenario: capacity 32, batch 8, arrivals of 16 rows while
        # a window is frozen; the next window is full right when the previous
        # one ends, so add()-time compaction never fires
        pool = self._pool(32, 8, data_pb2.SAMPLE_WINDOW)
        next_id = 0
        total_out = 0
        for _ in range(25):
            pool.add(make_batch(16, start_id=next_id))
            next_id += 16
            pool.add(make_batch(16, start_id=next_id))
            next_id += 16
            # freeze + drain the window, with arrivals mid-drain
            for i in range(10):
                out = pool.emit()
                if out is None:
                    break
                total_out += out.num_rows
                if i == 1:
                    pool.add(make_batch(16, start_id=next_id))
                    next_id += 16
        self.assertGreater(pool.num_compactions, 0)
        self.assertLessEqual(len(pool._chunks), 10)
        self.assertEqual(pool.rows_in, next_id)
        self.assertEqual(pool.rows_in - pool.rows_out, pool.total_rows)

    def test_output_batches_survive_compaction(self):
        pool = self._pool(32, 8, data_pb2.SAMPLE_ROLLING)
        next_id = 0
        produced = []
        for _ in range(40):
            pool.add(make_batch(16, start_id=next_id))
            next_id += 16
            while True:
                out = pool.emit()
                if out is None:
                    break
                produced.append(out)
        self.assertGreater(pool.num_compactions, 0)
        # batches emitted before a compaction stay valid and row-correct
        seen = set()
        for b in produced:
            check_row_correspondence(self, b)
            seen.update(b.column("id").to_pylist())
        self.assertEqual(len(seen), len(produced) * 8)

    def test_null_and_nested_columns(self):
        pool = self._pool(32, 16, data_pb2.SAMPLE_WINDOW)
        ids = np.arange(32, dtype=np.int64)
        rb = pa.RecordBatch.from_arrays(
            [
                pa.array(ids),
                pa.array([None if i % 3 == 0 else f"v{i}" for i in ids]),
                pa.array(
                    [None if i % 4 == 0 else [int(i)] * (i % 3) for i in ids],
                    type=pa.list_(pa.int64()),
                ),
                pa.array(["topic:0"] * 32),
                pa.array(ids // 16),
                pa.array(ids.astype(np.float64)),
            ],
            names=[
                "id",
                "str_f",
                "list_f",
                CKPT_SOURCE_ID,
                CKPT_ROW_IDX,
                DATA_TIMESTAMP,
            ],
        )
        pool.add(rb)
        out = pool.emit()
        self.assertIsNotNone(out)
        for i, s, li, off in zip(
            out.column("id").to_pylist(),
            out.column("str_f").to_pylist(),
            out.column("list_f").to_pylist(),
            out.column(CKPT_ROW_IDX).to_pylist(),
        ):
            self.assertEqual(s, None if i % 3 == 0 else f"v{i}")
            self.assertEqual(li, None if i % 4 == 0 else [i] * (i % 3))
            self.assertEqual(off, i // 16)


class CheckpointInfoTest(unittest.TestCase):
    """checkpoint_info of a sample-shuffled batch reflects its actual content."""

    def _dataset(self):
        cfg = _data_config()
        return KafkaDataset(
            data_config=cfg,
            features=_features(),
            input_path=_fake_uri(),
            mode=Mode.TRAIN,
        )

    def _build(self, *batches):
        t = pa.Table.from_batches(batches).combine_chunks()
        rb = t.to_batches(max_chunksize=t.num_rows)[0]
        return self._dataset()._build_batch(
            {name: rb.column(i) for i, name in enumerate(rb.schema.names)}
        )

    def test_cross_partition_batch_reports_max_per_partition(self):
        # partition 0 contributes rows of offsets 3..5, partition 1 of 7..8
        b0 = make_batch(6, start_id=0, partition=0, rows_per_msg=2, offset_start=3)
        b1 = make_batch(6, start_id=0, partition=1, rows_per_msg=3, offset_start=7)
        batch = self._build(b0, b1)
        self.assertEqual(batch.checkpoint_info, {"topic:0": 5, "topic:1": 8})

    def test_message_split_across_batches_keeps_its_offset(self):
        # one message (offset 7) split across two output batches
        b = make_batch(40, rows_per_msg=40, offset_start=7)
        t = pa.Table.from_batches([b]).combine_chunks()
        rbs = t.to_batches(max_chunksize=16)
        for rb in rbs[:2]:
            batch = self._dataset()._build_batch(
                {name: rb.column(i) for i, name in enumerate(rb.schema.names)}
            )
            self.assertEqual(batch.checkpoint_info, {"topic:0": 7})

    def test_event_time_is_max_per_batch(self):
        b = make_batch(8, start_id=0)
        batch = self._build(b)
        self.assertAlmostEqual(batch.data_timestamp, 7.5)


_FAKE_ID_BASE = 10**9


class _FakeMessage:
    def __init__(self, payload, partition, offset, ts_ms):
        self._payload = payload
        self._partition = partition
        self._offset = offset
        self._ts_ms = ts_ms

    def error(self):
        return None

    def value(self):
        return self._payload

    def partition(self):
        return self._partition

    def offset(self):
        return self._offset

    def timestamp(self):
        return (1, self._ts_ms)


class _FakeTopicMetadata:
    def __init__(self, num_partitions):
        self.partitions = {p: None for p in range(num_partitions)}
        self.error = None


class _FakeClusterMetadata:
    def __init__(self, topic, num_partitions):
        self.topics = {topic: _FakeTopicMetadata(num_partitions)}


class FakeKafkaConsumer:
    """Deterministic in-process fake of confluent_kafka.Consumer.

    Injected as the explicit ``consumer_factory`` of the code under test;
    the reader context (reader_id, num_readers) arrives as constructor
    arguments. Extra URI query params flow into this config:
      fake.num_partitions (default 2), fake.rows_per_msg (default 3),
      fake.total_msgs (per partition, default 1000000),
      fake.msg_interval_secs (default 0, i.e. saturated),
      fake.ts_offset (offset returned by offsets_for_times, default 7),
      fake.low_watermark (default 0, raise to simulate retention loss),
      fake.rebalance_after_msgs (re-assign this reader's partitions after
      serving this many messages, replaying from the start offsets),
      fake.grow_to_partitions (on the first rebalance, also assign a
      partition created after startup, which must fail loudly).

    Partition p is served to reader r iff p % num_readers == r. The message
    at (p, o) carries rows_per_msg rows with globally unique ids
    p * 10**9 + o * rows_per_msg + i and label = id % 2.
    """

    def __init__(self, config, reader_id=0, num_readers=1):
        self._num_partitions = int(config.get("fake.num_partitions", "2"))
        self._rows_per_msg = int(config.get("fake.rows_per_msg", "3"))
        self._total_msgs = int(config.get("fake.total_msgs", "1000000"))
        self._interval = float(config.get("fake.msg_interval_secs", "0"))
        self._ts_offset = int(config.get("fake.ts_offset", "7"))
        self._low_watermark = int(config.get("fake.low_watermark", "0"))
        self._reader_id = reader_id
        self._num_readers = num_readers
        self._rebalance_after = int(config.get("fake.rebalance_after_msgs", "0"))
        self._grow_to = int(config.get("fake.grow_to_partitions", "0"))
        self._rebalances = 0
        self._served = 0
        self._positions = {}
        self._assignment = []
        self._on_assign = None
        self._closed = False

    def subscribe(self, topics, on_assign=None, on_revoke=None):
        self._topic = topics[0]
        self._on_assign = on_assign
        self._on_revoke = on_revoke
        parts = self._my_partitions()
        if on_assign is not None and parts:
            from confluent_kafka import TopicPartition

            on_assign(self, [TopicPartition(self._topic, p) for p in parts])

    def _my_partitions(self):
        return [
            p
            for p in range(self._num_partitions)
            if p % self._num_readers == self._reader_id
        ]

    def _maybe_rebalance(self):
        if (
            self._rebalance_after > 0
            and self._served >= self._rebalance_after
            and self._rebalances == 0
            and self._on_assign is not None
        ):
            from confluent_kafka import TopicPartition

            self._rebalances += 1
            num_parts = self._num_partitions
            if self._grow_to > num_parts:
                num_parts = self._grow_to
            parts = [
                p for p in range(num_parts) if p % self._num_readers == self._reader_id
            ]
            # revoke + re-assign: re-seek to this run's start offsets and
            # replay everything read so far
            if self._on_revoke is not None:
                self._on_revoke(
                    self,
                    [TopicPartition(self._topic, p) for p in self._my_partitions()],
                )
            self._on_assign(self, [TopicPartition(self._topic, p) for p in parts])

    def assign(self, partitions):
        for tp in partitions:
            self._positions[tp.partition] = tp.offset
        self._assignment = [(tp.topic, tp.partition) for tp in partitions]

    def assignment(self):
        from confluent_kafka import TopicPartition

        return [TopicPartition(t, p) for t, p in self._assignment]

    def pause(self, partitions):
        pass

    def resume(self, partitions):
        pass

    def poll(self, timeout=0):
        return None

    def consume(self, num_messages, timeout=1.0):
        self._maybe_rebalance()
        if self._interval > 0:
            time.sleep(self._interval)
        out = []
        parts = sorted(self._positions.keys())
        if not parts:
            time.sleep(timeout)
            return out
        ts_ms = int(time.time() * 1000)
        idx = 0
        while len(out) < num_messages:
            p = parts[idx % len(parts)]
            idx += 1
            o = self._positions[p]
            if o >= self._total_msgs:
                if all(self._positions[x] >= self._total_msgs for x in parts):
                    break
                continue
            rpm = self._rows_per_msg
            ids = np.arange(rpm, dtype=np.int64) + p * _FAKE_ID_BASE + o * rpm
            rb = pa.RecordBatch.from_arrays(
                [pa.array(ids), pa.array(ids % 2)], names=["id", "label"]
            )
            out.append(_FakeMessage(rb.serialize().to_pybytes(), p, o, ts_ms))
            self._positions[p] = o + 1
            self._served += 1
        return out

    def get_watermark_offsets(self, tp, timeout=60.0):
        return (self._low_watermark, self._total_msgs)

    def offsets_for_times(self, tps, timeout=30.0):
        from confluent_kafka import TopicPartition

        return [TopicPartition(tp.topic, tp.partition, self._ts_offset) for tp in tps]

    def list_topics(self, topic=None, timeout=60.0):
        return _FakeClusterMetadata(topic, self._num_partitions)

    def close(self):
        self._closed = True


def _fake_uri(**params):
    qs = "&".join(f"fake.{k}={v}" for k, v in params.items())
    return (
        f"kafka://fake:9092/topic?group.id=tzrec_test&auto.offset.reset=earliest&{qs}"
    )


def _data_config(**overrides):
    cfg = data_pb2.DataConfig(
        batch_size=8,
        dataset_type=data_pb2.DatasetType.KafkaDataset,
        fg_mode=data_pb2.FgMode.FG_NONE,
        label_fields=["label"],
        shuffle=True,
        shuffle_mode=data_pb2.ShuffleMode.SAMPLE_ROLLING,
        shuffle_buffer_size=2,
        num_workers=2,
        input_fields=[
            data_pb2.Field(input_name="id", input_type=data_pb2.INT64),
            data_pb2.Field(input_name="label", input_type=data_pb2.INT64),
        ],
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _features():
    return create_features(
        [feature_pb2.FeatureConfig(id_feature=feature_pb2.IdFeature(feature_name="id"))]
    )


def _ids_of(batch):
    d = batch.to_dict()
    return sorted(int(v) for v in d["id.values"].tolist())


class ValidateSampleShuffleConfigTest(unittest.TestCase):
    @parameterized.expand(
        [
            param("zero_batch", batch_size=0),
            param("zero_buffer", shuffle_buffer_size=0),
            param("batch_cost", batch_cost_size=1024),
        ],
        name_func=parameterized_name_func,
    )
    def test_invalid_configs(self, _name, **overrides):
        cfg = _data_config(**overrides)
        with self.assertRaises(ValueError):
            validate_sample_shuffle_config(cfg)

    def test_auto_commit_passthrough(self):
        # same as BATCH: the default is injected, explicit values pass
        # through to librdkafka untouched, and sample mode does not reject them
        _, params, _ = _parse_kafka_uri(_fake_uri())
        self.assertEqual(params["enable.auto.commit"], "false")
        _, params, _ = _parse_kafka_uri(
            "kafka://fake:9092/topic?group.id=g&enable.auto.commit=true"
        )
        self.assertEqual(params["enable.auto.commit"], "true")
        validate_sample_shuffle_config(_data_config())

    def test_enabled_only_for_train_shuffle_sample_mode(self):
        from tzrec.datasets.dataset import sample_shuffle_enabled

        self.assertTrue(sample_shuffle_enabled(_data_config(), Mode.TRAIN))
        self.assertFalse(sample_shuffle_enabled(_data_config(), Mode.EVAL))
        self.assertFalse(sample_shuffle_enabled(_data_config(), Mode.PREDICT))
        self.assertFalse(
            sample_shuffle_enabled(_data_config(shuffle=False), Mode.TRAIN)
        )
        self.assertFalse(
            sample_shuffle_enabled(
                _data_config(shuffle_mode=data_pb2.ShuffleMode.BATCH), Mode.TRAIN
            )
        )
        # a legacy config that never mentions the new field stays BATCH
        legacy = data_pb2.DataConfig()
        self.assertEqual(legacy.shuffle_mode, data_pb2.ShuffleMode.BATCH)
        self.assertFalse(sample_shuffle_enabled(legacy, Mode.TRAIN))


class ResolveStartOffsetsTest(unittest.TestCase):
    def _resolve(self, input_path, checkpoint_state=None):
        return resolve_start_offsets(
            input_path, checkpoint_state, consumer_factory=FakeKafkaConsumer
        )

    def test_from_checkpoint(self):
        starts = self._resolve(_fake_uri(), {"topic:0": 41, "topic:1": 42})
        self.assertEqual(starts, {0: 42, 1: 43})

    def test_from_timestamp(self):
        starts = self._resolve(_fake_uri() + "&start.timestamp.ms=123456", None)
        self.assertEqual(starts, {0: 7, 1: 7})

    def test_offset_reset_latest_and_earliest(self):
        # no auto.offset.reset: librdkafka default is latest
        base = "kafka://fake:9092/topic?group.id=tzrec_test"
        starts = self._resolve(base + "&fake.total_msgs=100", None)
        self.assertEqual(starts, {0: 100, 1: 100})
        starts = self._resolve(
            base + "&fake.total_msgs=100&auto.offset.reset=earliest", None
        )
        self.assertEqual(starts, {0: 0, 1: 0})

    @parameterized.expand(
        [
            param("earliest", reset="earliest", expected=0),
            param("smallest", reset="smallest", expected=0),
            param("beginning", reset="beginning", expected=0),
            param("latest", reset="latest", expected=100),
            param("largest", reset="largest", expected=100),
            param("end", reset="end", expected=100),
        ],
        name_func=parameterized_name_func,
    )
    def test_offset_reset_aliases(self, _name, reset, expected):
        base = "kafka://fake:9092/topic?group.id=tzrec_test"
        starts = self._resolve(
            base + f"&fake.total_msgs=100&auto.offset.reset={reset}", None
        )
        self.assertEqual(starts, {0: expected, 1: expected})

    @parameterized.expand(
        [param("error", reset="error"), param("unknown", reset="bogus")],
        name_func=parameterized_name_func,
    )
    def test_offset_reset_error_values_raise(self, _name, reset):
        base = "kafka://fake:9092/topic?group.id=tzrec_test"
        with self.assertRaisesRegex(ValueError, "auto.offset.reset"):
            self._resolve(
                base + f"&fake.total_msgs=100&auto.offset.reset={reset}", None
            )

    def test_timestamp_fallback_honors_reset_alias(self):
        # timestamp resolution finds nothing -> fall back to auto.offset.reset
        base = "kafka://fake:9092/topic?group.id=tzrec_test"
        starts = self._resolve(
            base + "&fake.total_msgs=100&fake.ts_offset=-1"
            "&start.timestamp.ms=123456&auto.offset.reset=smallest",
            None,
        )
        self.assertEqual(starts, {0: 0, 1: 0})

    def test_checkpoint_out_of_range_returned_as_is(self):
        # matches the BATCH path: checkpoint + 1 without watermark checks;
        # librdkafka applies auto.offset.reset for out-of-range offsets
        starts = self._resolve(
            _fake_uri(low_watermark=100, total_msgs=1000),
            {"topic:0": 41, "topic:1": 5000},
        )
        self.assertEqual(starts, {0: 42, 1: 5001})

    def test_timestamp_retention_error(self):
        # the timestamp branch keeps its own retention check
        with self.assertRaises(ValueError):
            self._resolve(
                _fake_uri(low_watermark=100) + "&start.timestamp.ms=123456", None
            )


class ReaderDispatchTest(unittest.TestCase):
    """KafkaReader.to_batches selects the BATCH or SAMPLE path."""

    def _reader(self, table_schema=None):
        return kafka_dataset.KafkaReader(
            _fake_uri(),
            batch_size=8,
            shuffle=True,
            shuffle_buffer_size=2,
            input_fields=[
                pa.field("id", pa.int64()),
                pa.field("label", pa.int64()),
            ],
        )

    def _configure(self, reader, shared, stop, fatal):
        reader._configure_sample_shuffle(shared, stop, fatal)

    def _payload(self, reader, n=2):
        # payloads must match the dispatch schema exactly; a mismatch reads
        # garbage columns instead of raising
        ids = np.arange(n, dtype=np.int64)
        rb = pa.RecordBatch.from_arrays(
            [
                pa.array(ids),
                pa.array(ids % 2),
                pa.array(["topic:0"] * n),
                pa.array(ids // 16),
                pa.array(ids.astype(np.float64)),
            ],
            schema=reader._sample_table_schema,
        )
        return rb.serialize().to_pybytes()

    def test_sample_pickup_and_metadata_columns(self):
        ctx = mp.get_context("spawn")
        shared = ctx.Queue()
        reader = self._reader()
        self._configure(reader, shared, ctx.Event(), ctx.Event())
        shared.put(self._payload(reader))
        input_data = next(reader.to_batches(0, 2, 1))
        self.assertEqual(input_data["id"].to_pylist(), [0, 1])
        self.assertEqual(input_data["label"].to_pylist(), [0, 1])
        self.assertEqual(input_data[CKPT_SOURCE_ID].to_pylist(), ["topic:0"] * 2)
        self.assertIn(DATA_TIMESTAMP, input_data)

    def test_sample_path_does_not_read_kafka_or_shuffle_batches(self):
        ctx = mp.get_context("spawn")
        shared = ctx.Queue()
        reader = self._reader()
        self._configure(reader, shared, ctx.Event(), ctx.Event())

        def _must_not_read(*_args, **_kwargs):
            raise AssertionError("sample path must not read Kafka in the worker")

        reader._reader = _must_not_read
        reader._arrow_reader_iter = _must_not_read
        shared.put(self._payload(reader))
        input_data = next(reader.to_batches(0, 1, 1))
        self.assertEqual(input_data["id"].to_pylist(), [0, 1])

    def test_batch_path_still_uses_worker_reader_and_batch_shuffle(self):
        reader = self._reader()
        calls = []

        def fake_reader(worker_id, num_workers):
            calls.append((worker_id, num_workers))
            return iter([make_batch(8)])

        reader._reader = fake_reader
        batches = list(reader.to_batches(1, 2, 1))
        self.assertEqual(calls, [(1, 2)])
        # a single full batch flows through the batch-shuffle buffer
        self.assertEqual(sum(len(b["id"]) for b in batches), 8)

    def test_fatal_event_raises(self):
        ctx = mp.get_context("spawn")
        fatal = ctx.Event()
        reader = self._reader()
        self._configure(reader, ctx.Queue(), ctx.Event(), fatal)
        fatal.set()
        it = reader.to_batches(0, 1, 1)
        with self.assertRaisesRegex(RuntimeError, "coordinator exited unexpectedly"):
            next(it)

    def test_stop_event_ends_the_pass(self):
        ctx = mp.get_context("spawn")
        shared = ctx.Queue()
        stop = ctx.Event()
        reader = self._reader()
        self._configure(reader, shared, stop, ctx.Event())
        it = reader.to_batches(0, 1, 1)
        stop.set()
        with self.assertRaises(StopIteration):
            next(it)


class CoordinatorTest(unittest.TestCase):
    """In-process coordinator with a stubbed reader set."""

    def _coordinator_kwargs(self, ctx, mode, dispatch_maxsize, num_readers=0):
        schema = pa.schema(
            [
                pa.field("id", pa.int64()),
                pa.field(CKPT_SOURCE_ID, pa.string()),
                pa.field(CKPT_ROW_IDX, pa.int64()),
                pa.field(DATA_TIMESTAMP, pa.float64()),
            ]
        )
        return dict(
            input_path=_fake_uri(),
            num_readers=num_readers,
            start_offsets={0: 0, 1: 0},
            input_q=ctx.Queue(),
            stop_event=ctx.Event(),
            shared_dispatch_queue=ctx.Queue(maxsize=dispatch_maxsize),
            mode=mode,
            capacity_rows=16,
            batch_size=8,
            input_table_schema=schema,
            reader_schema=pa.schema([pa.field("id", pa.int64())]),
            # FakeConsumer payloads carry [id, label]; schema-less decode must
            # match exactly or pyarrow misreads the buffer
            full_schema=pa.schema(
                [pa.field("id", pa.int64()), pa.field("label", pa.int64())]
            ),
            has_embedded_schema=False,
            trainer_pid=os.getpid(),
        )

    def _coordinator(
        self,
        mode=data_pb2.ShuffleMode.SAMPLE_ROLLING,
        dispatch_maxsize=0,
        num_readers=0,
        input_maxsize=0,
        consumer_factory=None,
    ):
        ctx = mp.get_context("spawn")
        kwargs = self._coordinator_kwargs(ctx, mode, dispatch_maxsize, num_readers)
        if input_maxsize:
            kwargs["input_q"] = ctx.Queue(maxsize=input_maxsize)
        if consumer_factory is not None:
            kwargs["consumer_factory"] = consumer_factory
        coord = kafka_dataset._Coordinator(**kwargs)
        if num_readers == 0:
            coord._stop_event = ctx.Event()
        return coord

    def _feed(self, coord, n_rows, partition=0, offset_start=0):
        # the payload must match the coordinator's input table schema exactly
        ids = np.arange(n_rows, dtype=np.int64)
        rb = pa.RecordBatch.from_arrays(
            [
                pa.array(ids),
                pa.array([f"topic:{partition}"] * n_rows),
                pa.array(offset_start + ids // 16),
                pa.array(ids.astype(np.float64)),
            ],
            names=["id", CKPT_SOURCE_ID, CKPT_ROW_IDX, DATA_TIMESTAMP],
        )
        coord._input_q.put(rb.serialize().to_pybytes())

    def test_dispatch(self):
        coord = self._coordinator()
        self._feed(coord, 64)
        coord._service_input(1.0)
        emitted = coord._emit_and_dispatch()
        payload = coord._shared_dispatch_queue.get(timeout=1.0)
        rb = pa.ipc.read_record_batch(payload, coord._input_table_schema)
        self.assertEqual(rb.num_rows, 8)
        for _ in range(emitted - 1):
            coord._shared_dispatch_queue.get(timeout=1.0)
        coord._shared_dispatch_queue.close()
        coord._shared_dispatch_queue.join_thread()

    def test_pending_sent_once_when_capacity_frees(self):
        coord = self._coordinator(dispatch_maxsize=1)
        self._feed(coord, 64)
        coord._service_input(1.0)
        # one slot: the first batch fills it, the second is stashed as pending
        self.assertEqual(coord._emit_and_dispatch(), 1)
        first = coord._shared_dispatch_queue.get(timeout=1.0)
        pending = coord._pending_dispatch
        self.assertIsNotNone(pending)
        self.assertNotEqual(first, pending)
        # freeing the slot sends the stashed payload exactly once; the next
        # batch becomes the new pending one
        self.assertEqual(coord._emit_and_dispatch(), 1)
        self.assertEqual(coord._shared_dispatch_queue.get(timeout=1.0), pending)
        self.assertNotEqual(coord._pending_dispatch, pending)

    def test_pool_full_gates_input(self):
        coord = self._coordinator(dispatch_maxsize=1)
        self.addCleanup(coord._input_q.close)
        self.addCleanup(coord._input_q.cancel_join_thread)
        for _ in range(5):
            self._feed(coord, 32)
        t = threading.Thread(target=coord.run, daemon=True)
        t.start()
        try:
            time.sleep(1.0)
            # pool never exceeds capacity + one in-flight table; the
            # undigested input stays on the bounded queue (backpressure)
            self.assertLessEqual(coord._pool.total_rows, 16 + 32)
            self.assertFalse(coord._input_q.empty())
        finally:
            coord._stop_event.set()
            t.join(timeout=10.0)
        self.assertFalse(t.is_alive())

    def test_shutdown_reaps_all_readers_within_budget(self):
        coord = self._coordinator(
            num_readers=6, input_maxsize=1, consumer_factory=FakeKafkaConsumer
        )
        coord._start_readers()
        self.assertTrue(all(p.is_alive() for p in coord._readers))
        # let the readers fill the bounded input queue and block on it
        time.sleep(2.0)
        t0 = time.monotonic()
        coord._shutdown_readers()
        elapsed = time.monotonic() - t0
        self.assertTrue(all(not p.is_alive() for p in coord._readers))
        self.assertLess(elapsed, 12.0)

    def test_stop_event_exits_with_blocked_output(self):
        coord = self._coordinator(dispatch_maxsize=1)
        self._feed(coord, 64)
        coord._service_input(1.0)
        coord._emit_and_dispatch()
        # dispatch queue is full and a payload is stashed as pending
        self.assertIsNotNone(coord._pending_dispatch)
        t = threading.Thread(target=coord.run, daemon=True)
        t.start()
        time.sleep(0.3)
        assert t.is_alive()
        coord._stop_event.set()
        t.join(timeout=10.0)
        self.assertFalse(t.is_alive())

    def test_shutdown_budget_does_not_scale_with_reader_count(self):
        clock = [0.0]

        class _SlowReader:
            def __init__(self):
                self.terminated = False
                self.alive = True

            def join(self, timeout=None):
                # never exits by itself; consume whatever budget it is given
                clock[0] += timeout or 0.0

            def is_alive(self):
                return self.alive

            def terminate(self):
                self.terminated = True

        for n in (1, 6):
            clock[0] = 0.0
            coord = self._coordinator()
            coord._readers = [_SlowReader() for _ in range(n)]
            with mock.patch.object(kafka_dataset.time, "monotonic", lambda: clock[0]):
                coord._shutdown_readers()
            # one shared 6s graceful + 2s forced budget, regardless of n
            self.assertAlmostEqual(clock[0], 8.0, places=6)
            self.assertTrue(all(r.terminated for r in coord._readers))

    def test_idle_does_not_spin(self):
        coord = self._coordinator()
        calls = []
        orig = coord._service_input

        def spy(timeout):
            calls.append(timeout)
            return orig(timeout)

        coord._service_input = spy
        t = threading.Thread(target=coord.run, daemon=True)
        t.start()
        time.sleep(1.0)
        coord._stop_event.set()
        t.join(timeout=10.0)
        self.assertFalse(t.is_alive())
        # ~10 wakeups/s tick; a zero-timeout spin would produce thousands
        self.assertLess(len(calls), 50)
        self.assertTrue(all(0 < x <= 0.11 for x in calls))


class SampleShuffleEndToEndTest(unittest.TestCase):
    """Full stack: create_dataloader + spawned coordinator + fake consumers."""

    def setUp(self):
        self._client = None
        # the reader captures the module default at construction and passes
        # it explicitly to the spawned coordinator
        patcher = mock.patch.object(
            kafka_dataset, "_default_consumer_factory", FakeKafkaConsumer
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        if self._client is not None:
            self._client.stop()

    def _run_round(self, cfg_kwargs, uri_kwargs, n_batches, timeout=60.0):
        from tzrec.datasets.dataset import create_dataloader

        def _alarm(signum, frame):
            raise TimeoutError("timed out waiting for batches")

        cfg = _data_config(**cfg_kwargs)
        dataloader = create_dataloader(
            cfg,
            _features(),
            _fake_uri(**uri_kwargs),
            mode=Mode.TRAIN,
        )
        self._client = dataloader.dataset._sample_shuffle
        it = dataloader.get_iterator()
        batches = []
        old_handler = signal.signal(signal.SIGALRM, _alarm)
        signal.setitimer(signal.ITIMER_REAL, timeout)
        try:
            while len(batches) < n_batches:
                batches.append(next(it))
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)
        return batches

    @parameterized.expand(
        [
            param("rolling", mode=data_pb2.ShuffleMode.SAMPLE_ROLLING),
            param("window", mode=data_pb2.ShuffleMode.SAMPLE_WINDOW),
        ],
        name_func=parameterized_name_func,
    )
    def test_batches_flow(self, _name, mode):
        batches = self._run_round(
            dict(shuffle_mode=mode, num_workers=2),
            dict(num_partitions=2, rows_per_msg=3),
            n_batches=4,
        )
        seen = set()
        for b in batches:
            ids = _ids_of(b)
            self.assertEqual(len(ids), 8)
            self.assertEqual(len(set(ids)), 8)
            seen.update(ids)
            self.assertGreaterEqual(b.data_timestamp, 0)
            # checkpoint_info must equal each partition's max offset actually
            # present in this batch
            expected = {}
            for i in ids:
                part = i // _FAKE_ID_BASE
                off = (i % _FAKE_ID_BASE) // 3
                expected[part] = max(expected.get(part, -1), off)
            self.assertEqual(
                b.checkpoint_info, {f"topic:{p}": v for p, v in expected.items()}
            )
        # no duplicates across batches without any rebalance
        self.assertEqual(len(seen), 8 * len(batches))
        for i in seen:
            self.assertLess(i, 2 * _FAKE_ID_BASE + 10**6)

    def test_num_workers_zero(self):
        batches = self._run_round(
            dict(num_workers=0),
            dict(num_partitions=1, rows_per_msg=4),
            n_batches=2,
        )
        for b in batches:
            self.assertEqual(len(_ids_of(b)), 8)

    def test_checkpoint_state_accumulates_max_offsets(self):
        batches = self._run_round(
            dict(num_workers=2), dict(num_partitions=2, rows_per_msg=4), n_batches=4
        )
        state = {}
        for b in batches:
            update_dataloder_state(state, b.checkpoint_info)
        # the accumulated state equals the per-partition max over trained
        # batches, and a stale smaller offset never moves it back
        expected = {}
        for b in batches:
            for i in _ids_of(b):
                part = i // _FAKE_ID_BASE
                off = (i % _FAKE_ID_BASE) // 4
                expected[part] = max(expected.get(part, -1), off)
        self.assertEqual(state, {f"topic:{p}": v for p, v in expected.items()})
        if 0 in expected:
            update_dataloder_state(state, {"topic:0": 0})
            self.assertEqual(state["topic:0"], expected[0])

    def test_rebalance_replays_and_recovers(self):
        batches = self._run_round(
            dict(num_workers=1),
            dict(num_partitions=1, rows_per_msg=2, rebalance_after_msgs=20),
            n_batches=6,
        )
        seen = [i for b in batches for i in _ids_of(b)]
        # the replay after rebalance makes ids repeat across batches
        self.assertLess(len(set(seen)), len(seen))
        state = {}
        for b in batches:
            update_dataloder_state(state, b.checkpoint_info)
        self.assertEqual(set(state.keys()), {"topic:0"})
        self.assertGreaterEqual(state["topic:0"], 0)

    def test_new_partition_after_startup_fails(self):
        from tzrec.datasets.dataset import create_dataloader

        def _alarm(signum, frame):
            raise TimeoutError("partition expansion did not fail the pipeline")

        dataloader = create_dataloader(
            _data_config(num_workers=1),
            _features(),
            _fake_uri(num_partitions=1, rebalance_after_msgs=200, grow_to_partitions=2),
            mode=Mode.TRAIN,
        )
        self._client = dataloader.dataset._sample_shuffle
        it = dataloader.get_iterator()
        next(it)  # flowing before the rebalance
        old_handler = signal.signal(signal.SIGALRM, _alarm)
        signal.setitimer(signal.ITIMER_REAL, 30.0)
        try:
            with self.assertRaisesRegex(
                RuntimeError, "coordinator exited unexpectedly"
            ):
                while True:
                    next(it)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)

    def test_coordinator_death_propagates(self):
        from tzrec.datasets.dataset import create_dataloader

        def _alarm(signum, frame):
            raise TimeoutError("worker did not observe coordinator death")

        dataloader = create_dataloader(
            _data_config(num_workers=1),
            _features(),
            _fake_uri(num_partitions=1),
            mode=Mode.TRAIN,
        )
        self._client = dataloader.dataset._sample_shuffle
        it = dataloader.get_iterator()
        next(it)  # pipeline is flowing
        self._client._proc.terminate()
        old_handler = signal.signal(signal.SIGALRM, _alarm)
        signal.setitimer(signal.ITIMER_REAL, 30.0)
        try:
            with self.assertRaisesRegex(
                RuntimeError, "coordinator exited unexpectedly"
            ):
                while True:
                    next(it)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)

    def test_repeated_create_and_close(self):
        for _ in range(2):
            batches = self._run_round(
                dict(num_workers=1), dict(num_partitions=1), n_batches=1
            )
            self.assertEqual(len(_ids_of(batches[0])), 8)
            self._client.stop()
            self._client = None

    def test_batch_mode_creates_no_background_resources(self):
        from tzrec.datasets.dataset import create_dataloader

        # the fake URI carries fake.* keys that a real Consumer rejects, so
        # the batch path must run against the injected fake too; otherwise
        # its worker dies at startup and the handle assertion below would
        # still pass
        dataloader = create_dataloader(
            _data_config(shuffle_mode=data_pb2.ShuffleMode.BATCH),
            _features(),
            _fake_uri(num_partitions=1),
            mode=Mode.TRAIN,
            checkpoint_state={"topic:0": 0},
        )
        self.assertIsNone(dataloader.dataset._sample_shuffle)
        it = dataloader.get_iterator()
        try:
            data = next(it)
            self.assertEqual(len(_ids_of(data)), 8)
        finally:
            # persistent workers outlive the iterator unless shut down
            it._shutdown_workers()


class HandleLifecycleTest(unittest.TestCase):
    """SampleShuffleHandle lifecycle without real processes."""

    def _handle(self):
        ctx = mp.get_context("spawn")
        h = kafka_dataset.SampleShuffleHandle.__new__(kafka_dataset.SampleShuffleHandle)
        h.stop_event = ctx.Event()
        h.fatal_event = ctx.Event()
        h._proc = None
        h._atexit_cleanup = None
        return h

    def test_watchdog_sets_fatal_event_on_unexpected_exit(self):
        h = self._handle()

        class _DeadProc:
            exitcode = 1

            def is_alive(self):
                return False

            def join(self, timeout=None):
                return None

        h._proc = _DeadProc()
        h._start_watchdog()
        deadline = time.monotonic() + 5.0
        while not h.fatal_event.is_set():
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.02)
        h._proc = None

    def test_stop_is_clean(self):
        h = self._handle()

        class _Proc:
            exitcode = 0

            def __init__(self, stop_event):
                self._stop = stop_event
                self.terminated = False
                # simulate the coordinator reacting to the shared stop event
                threading.Thread(
                    target=lambda: (self._stop.wait(), setattr(self, "_alive", False)),
                    daemon=True,
                ).start()
                self._alive = True

            def is_alive(self):
                return self._alive

            def join(self, timeout=None):
                for _ in range(int((timeout or 0) * 100) + 1):
                    if not self._alive:
                        return
                    time.sleep(0.01)

            def terminate(self):
                self.terminated = True
                self._alive = False

        proc = _Proc(h.stop_event)
        h._proc = proc
        t0 = time.monotonic()
        h.stop()
        # a clean stop returns promptly without terminating the process
        self.assertLess(time.monotonic() - t0, 10.0)
        self.assertFalse(proc.terminated)
        self.assertFalse(h.fatal_event.is_set())

    def _prepared_dataset(self):
        # the reader captures the module default at construction and hands it
        # explicitly to the spawned coordinator
        patcher = mock.patch.object(
            kafka_dataset, "_default_consumer_factory", FakeKafkaConsumer
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        cfg = _data_config(num_workers=1)
        ds = KafkaDataset(
            data_config=cfg,
            features=_features(),
            input_path=_fake_uri(num_partitions=1),
            mode=Mode.TRAIN,
        )
        self.addCleanup(ds.stop_loading)
        return ds

    def test_prepare_keeps_reader_and_delays_input_q(self):
        ds = self._prepared_dataset()
        reader_before = ds._reader
        ds.prepare_for_loading(None)
        # the reader object is kept and configured in place
        self.assertIs(ds._reader, reader_before)
        self.assertIsNotNone(reader_before._sample_queue)
        handle = ds._sample_shuffle
        # the reader -> coordinator queue is not created during prepare
        self.assertIsNone(handle.input_q)
        ds.start_loading()
        input_q = handle.input_q
        self.assertIsNotNone(input_q)
        # repeated start keeps the same queue and process
        ds.start_loading()
        self.assertIs(handle.input_q, input_q)

    def test_start_does_not_re_resolve_offsets(self):
        ds = self._prepared_dataset()
        ds.prepare_for_loading(None)
        number = ds._reader  # no-op; keep the reader referenced
        assert number is not None
        with mock.patch.object(
            kafka_dataset, "resolve_start_offsets", side_effect=AssertionError
        ):
            # start only spawns; latest is never re-interpreted here
            ds.start_loading()

    def test_start_failure_cleans_input_q(self):
        ds = self._prepared_dataset()
        ds.prepare_for_loading(None)
        handle = ds._sample_shuffle
        with mock.patch.object(
            handle._ctx, "Process", side_effect=RuntimeError("boom")
        ):
            with self.assertRaises(RuntimeError):
                ds.start_loading()
        self.assertIsNone(handle.input_q)
        # stopping a never-started handle is a no-op
        ds.stop_loading()

    def test_stop_without_start_is_noop(self):
        self._handle().stop()


class TrainLoopSampleShuffleTest(unittest.TestCase):
    """Trainer-side integration with a mocked model and pipeline."""

    def setUp(self):
        self.test_dir = make_test_dir()
        self.addCleanup(shutil.rmtree, self.test_dir)

    def _run_loop(self, n_steps, ts_per_step=None):
        """Run _train_and_evaluate with mocks; return recorded maybe_save calls."""
        from tzrec.main import _train_and_evaluate

        model = mock.Mock()
        model.module.model.compute_train_metric.return_value = {}
        optimizer = mock.Mock()
        optimizer.params = {}
        optimizer.param_groups = []
        train_dataloader = mock.Mock()
        train_dataloader.get_iterator.return_value = iter([object()])
        train_dataloader.dataset._data_config = _data_config()
        train_dataloader.dataset.is_sample_shuffled.return_value = True
        ts_per_step = ts_per_step or [-1.0] * (n_steps + 1)
        batches = [
            SimpleNamespace(
                checkpoint_info={"topic:0": i, "topic:1": 100 - i},
                data_timestamp=ts,
            )
            for i, ts in enumerate(ts_per_step)
        ]
        pipeline = mock.Mock()
        pipeline.progress.side_effect = [
            ({"loss": __import__("torch").tensor(1.0)}, {}, b) for b in batches
        ]
        calls = []

        def maybe_save_impl(
            step, model_, optimizer_, dataloader_state, dense_ema, **kw
        ):
            calls.append(
                dict(
                    step=step,
                    kw=kw,
                    state=dict(dataloader_state) if dataloader_state else None,
                )
            )
            return False

        ckpt_manager = mock.Mock()
        ckpt_manager.maybe_save.side_effect = maybe_save_impl
        exporter = mock.Mock()
        exporter.enabled = False
        train_config = TrainConfig(
            num_steps=n_steps,
            save_checkpoints_steps=1,
            use_tensorboard=False,
            dense_optimizer=DenseOptimizer(),
        )
        with (
            mock.patch.dict(os.environ, {"RANK": "1", "LOCAL_RANK": "1"}),
            mock.patch("tzrec.main.create_train_pipeline", return_value=pipeline),
            mock.patch("tzrec.main.OnlineDenseExportManager", return_value=exporter),
        ):
            _train_and_evaluate(
                model=model,
                optimizer=optimizer,
                train_dataloader=train_dataloader,
                eval_dataloader=None,
                lr_scheduler=[],
                model_dir=self.test_dir,
                train_config=train_config,
                eval_config=EvalConfig(),
                ckpt_manager=ckpt_manager,
            )
        return calls, train_dataloader

    def test_checkpoint_info_max_merges_stepwise(self):
        calls, _ = self._run_loop(4)
        # per-step accumulation keeps the per-partition max so far
        states = [c["state"] for c in calls if c["state"] is not None]
        self.assertEqual(states[-1]["topic:0"], 3)
        self.assertEqual(states[-1]["topic:1"], 100)
        for prev, cur in zip(states, states[1:]):
            self.assertGreaterEqual(cur["topic:0"], prev["topic:0"])

    def test_event_time_max_so_far(self):
        calls, _ = self._run_loop(4, ts_per_step=[100.0, 90.0, 120.0, 110.0, 110.0])
        # shuffled batches report max-so-far event time, never regressing
        seen = [c["kw"]["data_timestamp"] for c in calls]
        self.assertEqual(seen[:4], [100.0, 100.0, 120.0, 120.0])

    def test_loading_stopped_at_exit(self):
        _, train_dataloader = self._run_loop(2)
        train_dataloader.dataset.stop_loading.assert_called_once()


class ReaderConfigTest(unittest.TestCase):
    """The reader entry's default factory builds a real consumer as-is."""

    def test_real_consumer_gets_uri_config_verbatim(self):
        ctx = mp.get_context("spawn")
        stop = ctx.Event()
        stop.set()  # never enter the consume loop
        # confluent_kafka.Consumer raises KafkaException on unknown config
        # keys at construction; nothing may be added beyond the URI config
        kafka_dataset._reader_process_main(
            input_path="kafka://localhost:9092/topic?group.id=g",
            reader_id=0,
            num_readers=1,
            coordinator_pid=os.getpid(),
            start_offsets={0: 0},
            input_q=ctx.Queue(),
            stop_event=stop,
            reader_schema=pa.schema([pa.field("id", pa.int64())]),
            full_schema=pa.schema([pa.field("id", pa.int64())]),
            has_embedded_schema=False,
            batch_size=8,
        )


if __name__ == "__main__":
    unittest.main()
