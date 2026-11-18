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
import os
import pickle
import signal
import time
import unittest
from functools import partial
from types import SimpleNamespace
from unittest import mock

import pyarrow as pa
from alibabacloud_alikafka20190916 import models as alikafka_models
from alibabacloud_alikafka20190916.client import Client as AliKafkaClient
from alibabacloud_credentials.client import Client as CredClient
from alibabacloud_tea_openapi import models as openapi_models
from alibabacloud_tea_util import models as util_models
from confluent_kafka import KafkaError, KafkaException, Producer, TopicPartition
from parameterized import param, parameterized
from torch.utils.data import DataLoader

from tzrec.constant import Mode
from tzrec.datasets import kafka_dataset
from tzrec.datasets.dataset import create_dataloader
from tzrec.datasets.kafka_dataset import (
    KafkaDataset,
    KafkaReader,
    _build_projection,
    _offsets_for_times_batched,
    _parse_kafka_uri,
)
from tzrec.datasets.utils import DATA_TIMESTAMP
from tzrec.features.feature import FgMode, create_features
from tzrec.protos import data_pb2, feature_pb2
from tzrec.tests import utils
from tzrec.utils.checkpoint_util import update_dataloder_state
from tzrec.utils.logging_util import logger
from tzrec.utils.misc_util import random_name
from tzrec.utils.test_util import parameterized_name_func


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


def _msg(partition, offset, value, rows=1, error=None):
    batch = pa.record_batch({"x": pa.array([value] * rows, type=pa.int64())})
    return mock.Mock(
        **{
            "error.return_value": error,
            "value.return_value": batch.serialize(),
            "partition.return_value": partition,
            "offset.return_value": offset,
            "timestamp.return_value": (0, 1000),
        }
    )


def _eof(partition, offset):
    return mock.Mock(
        **{
            "error.return_value": KafkaError(KafkaError._PARTITION_EOF),
            "partition.return_value": partition,
            "offset.return_value": offset,
        }
    )


def _decoded_logs(mock_logger):
    return [
        call.args[0]
        for call in mock_logger.info.call_args_list
        if "decoded_messages=" in call.args[0]
    ]


class KafkaReaderRoundRobinTest(unittest.TestCase):
    """Round-robin reading with mocked consumers."""

    def _make_reader(
        self,
        consumers,
        partitions=(0, 1),
        leaders=None,
        round_robin=True,
        batch_size=1,
        **kwargs,
    ):
        leaders = leaders or {partition: 0 for partition in partitions}
        query_consumer = mock.Mock()
        query_consumer.list_topics.return_value = SimpleNamespace(
            topics={
                "topic": SimpleNamespace(
                    partitions={
                        partition: SimpleNamespace(leader=leaders[partition])
                        for partition in partitions
                    },
                    error=None,
                )
            }
        )
        self.enterContext(
            mock.patch(
                "tzrec.datasets.kafka_dataset.Consumer",
                side_effect=([query_consumer] if round_robin else []) + consumers,
            )
        )
        self.enterContext(
            mock.patch(
                "tzrec.datasets.kafka_dataset.dist.is_initialized",
                return_value=False,
            )
        )
        return KafkaReader(
            "kafka://broker:9092/topic?group.id=g1",
            batch_size,
            input_fields=[pa.field("x", pa.int64())],
            kafka_partition_round_robin=round_robin,
            **kwargs,
        )

    @parameterized.expand(
        [
            param("messages", 1, [1, 3, 2, 4]),
            param("zero_rows", 0, [3, 2, 4]),
        ],
        name_func=parameterized_name_func,
    )
    def test_round_robin_waits_for_valid_message(self, _name, rows, expected):
        consumers = [mock.Mock(), mock.Mock()]
        consumers[0].consume.side_effect = [
            [_msg(0, 0, 1, rows=rows)],
            [_msg(0, 1, 2)],
        ]
        consumers[1].consume.side_effect = [
            [],
            [_msg(1, 0, 0, error=KafkaError(KafkaError._TRANSPORT))],
            [_msg(1, 0, 3)],
            [_msg(1, 1, 4)],
        ]
        for consumer in consumers:
            consumer.get_watermark_offsets.return_value = (0, 2)
        gen = self._make_reader(consumers)._reader(0, 1)
        self.addCleanup(gen.close)
        observed = [next(gen).column("x")[0].as_py() for _ in expected]
        self.assertEqual(observed, expected)
        self.assertEqual(
            [call.kwargs["timeout"] for call in consumers[0].consume.call_args_list],
            [0, 0],
        )
        self.assertEqual(
            [call.kwargs["timeout"] for call in consumers[1].consume.call_args_list],
            [0, 1.0, 0, 0],
        )

    @parameterized.expand(
        [
            param("eof_at_high_watermark", 5),
            param("eof_below_high_watermark", 10),
        ],
        name_func=parameterized_name_func,
    )
    def test_round_robin_skips_caught_up_partition(self, _name, high):
        consumers = [mock.Mock(), mock.Mock()]
        consumers[0].consume.side_effect = [
            [_msg(0, 0, 1)],
            [_msg(0, 1, 2)],
            [_msg(0, 2, 3)],
            [_msg(0, 3, 4)],
        ]
        consumers[1].consume.side_effect = [
            [],
            [_eof(1, 5)],
            [],
            [_msg(1, 5, 9)],
            [],
            [_msg(1, 6, 10)],
        ]
        consumers[1].get_watermark_offsets.return_value = (0, high)
        with mock.patch("tzrec.datasets.kafka_dataset.logger") as mock_logger:
            gen = self._make_reader(consumers)._reader(0, 1)
            self.addCleanup(gen.close)
            observed = [next(gen).column("x")[0].as_py() for _ in range(4)]
            self.assertEqual(observed, [1, 2, 3, 9])
            consumers[1].get_watermark_offsets.return_value = (0, 10)
            observed = [next(gen).column("x")[0].as_py() for _ in range(2)]
        self.assertEqual(observed, [4, 10])
        self.assertEqual(
            [call.kwargs["timeout"] for call in consumers[1].consume.call_args_list],
            [0, 1.0, 0, 0, 0, 1.0],
        )
        mock_logger.error.assert_not_called()

    @parameterized.expand(
        [
            param(
                "caught_up_takes_partial",
                (0, 2),
                [2, 4, 2],
                [mock.call(1, timeout=0), mock.call(2, timeout=0)],
            ),
            param(
                "backlogged_fills_turn",
                (0, 10),
                [2, 4, 4],
                [
                    mock.call(1, timeout=0),
                    mock.call(2, timeout=0),
                    mock.call(1, timeout=1.0),
                ],
            ),
        ],
        name_func=parameterized_name_func,
    )
    def test_round_robin_msg_per_turn_follows_rows_estimate(
        self, _name, watermark, expected_rows, expected_calls
    ):
        consumers = [mock.Mock(), mock.Mock()]
        consumers[0].consume.side_effect = [
            [_msg(0, 0, 1, rows=2)],
            [_msg(0, 1, 2, rows=2)],
            [_msg(0, 2, 3, rows=2)],
        ]
        consumers[0].get_watermark_offsets.return_value = watermark
        consumers[1].consume.return_value = [
            _msg(1, 0, 4, rows=2),
            _msg(1, 1, 5, rows=2),
        ]
        gen = self._make_reader(consumers, batch_size=8)._reader(0, 1)
        self.addCleanup(gen.close)
        observed = [next(gen).num_rows for _ in expected_rows]
        self.assertEqual(observed, expected_rows)
        self.assertEqual(consumers[0].consume.call_args_list, expected_calls)

    def test_round_robin_sleeps_when_all_caught_up(self):
        consumers = [mock.Mock(), mock.Mock()]
        consumers[0].consume.side_effect = [[_eof(0, 3)]] + [[]] * 8
        consumers[1].consume.side_effect = [[_eof(1, 7)]] + [[]] * 8
        consumers[0].get_watermark_offsets.return_value = (0, 3)
        consumers[1].get_watermark_offsets.return_value = (0, 7)

        class _StopLoop(Exception):
            pass

        sleep = mock.Mock(side_effect=_StopLoop)
        with mock.patch.object(kafka_dataset.time, "sleep", sleep):
            gen = self._make_reader(consumers)._reader(0, 1)
            self.addCleanup(gen.close)
            with self.assertRaises(_StopLoop):
                next(gen)
        sleep.assert_called_once_with(0.01)
        for consumer in consumers:
            for call in consumer.consume.call_args_list:
                self.assertEqual(call.kwargs["timeout"], 0)

    @parameterized.expand(
        [
            param(
                "multiple_brokers_non_contiguous_uneven",
                leaders={7: 0, 1: 1, 4: 0, 3: 2, 9: 1, 6: 2, 12: 0, 10: 2},
                expected=[[1, 4, 6], [7, 9, 10], [3, 12]],
            ),
            param(
                "leader_unavailable",
                leaders={0: -1, 1: 0, 2: 0},
                expected=[[0, 2], [1]],
            ),
        ],
        name_func=parameterized_name_func,
    )
    def test_partition_split(self, _name, leaders, expected):
        for worker_id, owned in enumerate(expected):
            consumers = [mock.Mock() for _ in owned]
            consumers[0].consume.return_value = [_msg(owned[0], 0, 1)]
            reader = self._make_reader(
                consumers, partitions=list(leaders), leaders=leaders
            )
            gen = reader._reader(worker_id, len(expected))
            self.addCleanup(gen.close)
            next(gen)
            self.assertEqual(
                [
                    [tp.partition for tp in consumer.assign.call_args.args[0]]
                    for consumer in consumers
                ],
                [[partition] for partition in owned],
            )
            gen.close()

    def test_insufficient_partitions_raises(self):
        reader = self._make_reader([], partitions=(0, 1))
        gen = reader._reader(0, 3)
        self.addCleanup(gen.close)
        with self.assertRaisesRegex(ValueError, "fewer than the 3"):
            next(gen)

    def test_initialization_failure_closes_consumers(self):
        consumers = [mock.Mock(), mock.Mock()]
        consumers[1].assign.side_effect = RuntimeError("assign failed")
        gen = self._make_reader(consumers)._reader(0, 1)
        self.addCleanup(gen.close)
        with self.assertRaisesRegex(RuntimeError, "assign failed"):
            next(gen)
        for consumer in consumers:
            consumer.close.assert_called_once()

    def test_logs_decoded_messages(self):
        consumers = [mock.Mock(), mock.Mock()]
        consumers[0].consume.return_value = [_msg(0, 0, 1)]
        consumers[1].consume.return_value = [_msg(1, 0, 2)]
        reader = self._make_reader(consumers, kafka_log_every_n_messages=1)
        with (
            mock.patch.dict(os.environ, {"RANK": "0"}),
            mock.patch("tzrec.datasets.kafka_dataset.logger") as mock_logger,
        ):
            gen = reader._reader(0, 1)
            self.addCleanup(gen.close)
            next(gen)
            next(gen)
            logs = _decoded_logs(mock_logger)
        self.assertEqual(
            logs,
            [
                "KafkaReader[rank-0|worker-0] decoded_messages=1 "
                "partitions={'topic:0': 1}",
                "KafkaReader[rank-0|worker-0] decoded_messages=2 "
                "partitions={'topic:0': 1, 'topic:1': 1}",
            ],
        )

    def test_decode_failure_closes_consumers(self):
        consumers = [mock.Mock(), mock.Mock()]
        message = _msg(0, 0, 1)
        message.value.return_value = b"not-arrow"
        consumers[0].consume.return_value = [message]
        gen = self._make_reader(consumers)._reader(0, 1)
        self.addCleanup(gen.close)
        with self.assertRaises(pa.ArrowInvalid):
            next(gen)
        for consumer in consumers:
            consumer.close.assert_called_once()

    def test_to_batches_close_failure_still_closes_others(self):
        consumers = [mock.Mock(), mock.Mock()]
        consumers[0].consume.return_value = [_msg(0, 0, 5)]
        consumers[1].close.side_effect = RuntimeError("close failed")
        iterator = self._make_reader(consumers).to_batches(0, 1)
        self.addCleanup(iterator.close)
        self.assertEqual(next(iterator)["x"].to_pylist(), [5])
        iterator.close()
        for consumer in consumers:
            consumer.close.assert_called_once()

    def _construct_with_dist(self, rank, consumer_cls, broadcast):
        with (
            mock.patch(
                "tzrec.datasets.kafka_dataset.dist_util.get_dataset_process_group",
                return_value=mock.sentinel.dataset_pg,
            ),
            mock.patch.multiple(
                "tzrec.datasets.kafka_dataset.dist",
                is_initialized=mock.Mock(return_value=True),
                get_world_size=mock.Mock(return_value=2),
                get_rank=mock.Mock(return_value=rank),
                broadcast_object_list=broadcast,
            ),
            mock.patch("tzrec.datasets.kafka_dataset.Consumer", new=consumer_cls),
        ):
            return KafkaReader(
                "kafka://broker:9092/topic?group.id=g1",
                1,
                input_fields=[pa.field("x", pa.int64())],
                kafka_partition_round_robin=True,
            )

    def test_partition_order_broadcasts_between_ranks(self):
        query_consumer = mock.Mock()
        query_consumer.list_topics.return_value = SimpleNamespace(
            topics={
                "topic": SimpleNamespace(
                    partitions={
                        0: SimpleNamespace(leader=1),
                        1: SimpleNamespace(leader=0),
                        2: SimpleNamespace(leader=1),
                    },
                    error=None,
                )
            }
        )
        consumer_cls = mock.Mock(return_value=query_consumer)
        broadcast = mock.Mock()
        rank0_reader = self._construct_with_dist(0, consumer_cls, broadcast)
        broadcast.assert_called_once_with(
            [None, [1, 0, 2]], group=mock.sentinel.dataset_pg
        )
        payload = broadcast.call_args.args[0]
        self.assertEqual(rank0_reader._round_robin_partition_order, [1, 0, 2])
        query_consumer.close.assert_called_once()

        def _broadcast(object_list, group):
            self.assertIs(group, mock.sentinel.dataset_pg)
            object_list[:] = payload

        rank1_consumer_cls = mock.Mock()
        rank1_reader = self._construct_with_dist(1, rank1_consumer_cls, _broadcast)
        rank1_consumer_cls.assert_not_called()
        self.assertEqual(
            rank1_reader._round_robin_partition_order,
            rank0_reader._round_robin_partition_order,
        )

    def test_partition_order_query_failure_propagates(self):
        query_consumer = mock.Mock()
        query_consumer.list_topics.return_value = SimpleNamespace(
            topics={"topic": SimpleNamespace(partitions={}, error="unknown topic")}
        )
        consumer_cls = mock.Mock(return_value=query_consumer)
        broadcast = mock.Mock()
        with self.assertRaisesRegex(ValueError, "unknown topic"):
            self._construct_with_dist(0, consumer_cls, broadcast)
        payload = broadcast.call_args.args[0]
        self.assertIn("unknown topic", payload[0])
        query_consumer.close.assert_called_once()

        def _broadcast(object_list, group):
            self.assertIs(group, mock.sentinel.dataset_pg)
            object_list[:] = payload

        consumer_cls.reset_mock()
        with self.assertRaisesRegex(ValueError, "unknown topic"):
            self._construct_with_dist(1, consumer_cls, _broadcast)
        consumer_cls.assert_not_called()

    @parameterized.expand(
        [
            ("subscribe_parent_exit", False, True),
            ("round_robin_parent_exit", True, True),
            ("subscribe_main_process", False, False),
            ("round_robin_main_process", True, False),
        ],
        name_func=parameterized_name_func,
    )
    def test_idle_poll_parent_lifecycle(self, _name, round_robin, has_parent):
        consumer = mock.Mock()
        idle_polls = 2 if round_robin else 1
        consumer.consume.side_effect = [[]] * idle_polls + [[_msg(0, 0, 7)]]
        consumer.get_watermark_offsets.return_value = (0, 1)
        reader = self._make_reader([consumer], partitions=(0,), round_robin=round_robin)
        with (
            mock.patch.object(
                kafka_dataset.mp,
                "parent_process",
                return_value=SimpleNamespace(pid=123) if has_parent else None,
            ),
            mock.patch.object(kafka_dataset.os, "getppid", side_effect=[123, 456]),
        ):
            gen = reader._reader()
            try:
                if has_parent:
                    with self.assertRaises(StopIteration):
                        next(gen)
                else:
                    self.assertEqual(next(gen).column("x").to_pylist(), [7])
            finally:
                gen.close()
        self.assertEqual(
            consumer.consume.call_count, idle_polls if has_parent else idle_polls + 1
        )
        consumer.close.assert_called_once()

    def test_read_plan_uses_global_reader_ids(self):
        reader = self._make_reader([], partitions=(0, 1, 2, 3))
        reader.load_state_dict({"topic:0": 12})
        plan = pickle.loads(pickle.dumps(reader.create_read_plan(1, 2, 2)))
        self.assertEqual(plan.num_readers, 2)
        self.assertEqual([task.reader_id for task in plan.tasks], [0, 1])
        self.assertEqual([task.kwargs["worker_id"] for task in plan.tasks], [2, 3])
        self.assertEqual([task.kwargs["num_workers"] for task in plan.tasks], [4, 4])
        for task in plan.tasks:
            task_reader = task.kwargs["reader"]
            self.assertEqual(task_reader._checkpoint_state, {"topic:0": 12})
            self.assertEqual(task_reader._round_robin_partition_order, [0, 1, 2, 3])
        self.assertFalse(plan.is_finite)
        self.assertIn(DATA_TIMESTAMP, plan.schema.names)


class KafkaReaderDecodedMessageLogTest(unittest.TestCase):
    """Decoded-message counting and periodic logging in normal consumption."""

    def _make_reader(self, consumer, **kwargs):
        self.enterContext(
            mock.patch("tzrec.datasets.kafka_dataset.Consumer", return_value=consumer)
        )
        return KafkaReader(
            "kafka://broker:9092/topic?group.id=g1",
            1,
            input_fields=[pa.field("x", pa.int64())],
            **kwargs,
        )

    def test_counts_decoded_messages_per_partition(self):
        consumer = mock.Mock()
        consumer.consume.side_effect = [
            [_msg(0, 0, 1, rows=2)],
            [_msg(1, 0, 2, error="boom"), _msg(0, 1, 3, rows=0)],
            [_msg(1, 1, 4), _msg(1, 2, 5)],
        ]
        with (
            mock.patch.dict(os.environ, {"RANK": "0"}),
            mock.patch("tzrec.datasets.kafka_dataset.logger") as mock_logger,
        ):
            gen = self._make_reader(consumer, kafka_log_every_n_messages=2)._reader(
                1, 2
            )
            self.addCleanup(gen.close)
            # The zero-row message yields no batch, so the second next()
            # runs through the next consume() call as well.
            for _ in range(2):
                next(gen)
            logs = _decoded_logs(mock_logger)
        self.assertEqual(
            logs,
            [
                "KafkaReader[rank-0|worker-1] decoded_messages=2 "
                "partitions={'topic:0': 2}",
                "KafkaReader[rank-0|worker-1] decoded_messages=4 "
                "partitions={'topic:0': 2, 'topic:1': 2}",
            ],
        )

    @parameterized.expand(
        [
            param("unset", {}),
            param("zero", {"kafka_log_every_n_messages": 0}),
        ],
        name_func=parameterized_name_func,
    )
    def test_logging_disabled(self, _name, reader_kwargs):
        consumer = mock.Mock()
        consumer.consume.return_value = [_msg(0, 0, 1), _msg(1, 0, 2)]
        reader = self._make_reader(consumer, **reader_kwargs)
        with mock.patch("tzrec.datasets.kafka_dataset.logger") as mock_logger:
            gen = reader._reader(0, 1)
            self.addCleanup(gen.close)
            next(gen)
            self.assertEqual(_decoded_logs(mock_logger), [])

    def test_counter_resets_per_reader_iteration(self):
        consumer = mock.Mock()
        consumer.consume.return_value = [_msg(0, 0, 1)]
        reader = self._make_reader(consumer, kafka_log_every_n_messages=100)
        for _ in range(2):
            gen = reader._reader(0, 1)
            self.addCleanup(gen.close)
            next(gen)
            self.assertEqual(reader._decoded_message_count, 1)
            self.assertEqual(reader._decoded_partition_counts, {"topic:0": 1})
            gen.close()

    def test_rebalance_preserves_counters(self):
        consumer = mock.Mock()
        consumer.consume.return_value = [_msg(0, 0, 1)]
        reader = self._make_reader(consumer, kafka_log_every_n_messages=100)
        reader.load_state_dict({"topic:0": 9})
        gen = reader._reader(0, 1)
        self.addCleanup(gen.close)
        next(gen)
        on_assign = consumer.subscribe.call_args.kwargs["on_assign"]
        on_assign(consumer, [TopicPartition("topic", 0)])
        # the callback re-seeks to the offset after the checkpointed one
        consumer.assign.assert_called_once()
        (tp,) = consumer.assign.call_args.args[0]
        self.assertEqual((tp.topic, tp.partition, tp.offset), ("topic", 0, 10))
        self.assertEqual(reader._decoded_message_count, 1)
        self.assertEqual(reader._decoded_partition_counts, {"topic:0": 1})
        next(gen)
        self.assertEqual(reader._decoded_message_count, 2)
        self.assertEqual(reader._decoded_partition_counts, {"topic:0": 2})


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
            param(
                "round_robin_train",
                mode=Mode.TRAIN,
                expected_shuffle=True,
                round_robin=True,
            ),
            param(
                "round_robin_eval",
                mode=Mode.EVAL,
                expected_shuffle=False,
                round_robin=True,
                log_every_n_messages=1000,
            ),
        ],
        name_func=parameterized_name_func,
    )
    def test_reader_config_reaches_reader(
        self,
        _name,
        mode,
        expected_shuffle,
        round_robin=None,
        log_every_n_messages=None,
    ):
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
        if round_robin is not None:
            data_config.kafka_partition_round_robin = round_robin
        if log_every_n_messages is not None:
            data_config.kafka_log_every_n_messages = log_every_n_messages
        features = create_features(
            [
                feature_pb2.FeatureConfig(
                    id_feature=feature_pb2.IdFeature(feature_name="id_a")
                )
            ]
        )
        with mock.patch.object(
            KafkaReader, "_get_round_robin_partition_order", return_value=[0, 1]
        ):
            dataset = KafkaDataset(
                data_config=data_config,
                features=features,
                input_path="kafka://broker:9092/topic?group.id=tzrec_test_group",
                mode=mode,
            )
        self.assertEqual(dataset._reader._shuffle, expected_shuffle)
        self.assertEqual(dataset._reader._shuffle_buffer_size, 64)
        self.assertEqual(
            dataset._reader._kafka_partition_round_robin, bool(round_robin)
        )
        self.assertEqual(
            dataset._reader._log_every_n_messages, log_every_n_messages or 0
        )


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

    def _create_test_topic(self):
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

    def _create_test_table_and_feature_cfgs(self, embedded_schema=False):
        self._create_test_topic()
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

    @parameterized.expand(
        [
            ("rolling", "SAMPLE_ROLLING", 2, False),
            ("window", "SAMPLE_WINDOW", 2, False),
            ("round_robin_no_workers", "SAMPLE_ROLLING", 0, True),
        ],
        name_func=parameterized_name_func,
    )
    @unittest.skipIf(
        os.environ.get("CI_ALIKAFKA_INSTANCE_ID", "") == "", "ci kafka is not exists."
    )
    def test_sample_shuffle_checkpoint_resume(
        self, _name, mode, num_workers, round_robin
    ):
        self._create_test_topic()
        producer = Producer({"bootstrap.servers": self.brokers})
        rows_per_message = 24
        messages_per_partition = 16
        metadata = {}
        timestamp_ms = int(time.time() * 1000)

        def on_delivery(row_start, event_time, error, message):
            self.assertIsNone(error)
            for row in range(row_start, row_start + rows_per_message):
                metadata[row] = (
                    f"{self.test_topic}:{message.partition()}",
                    message.offset(),
                    event_time / 1000.0,
                )

        for partition in range(4):
            for offset in range(messages_per_partition):
                row_start = (
                    partition * messages_per_partition + offset
                ) * rows_per_message
                record_batch = pa.record_batch(
                    {
                        "raw_c": list(range(row_start, row_start + rows_per_message)),
                        "label": [0] * rows_per_message,
                    }
                )
                producer.produce(
                    topic=self.test_topic,
                    partition=partition,
                    value=record_batch.serialize(),
                    timestamp=timestamp_ms + offset,
                    on_delivery=partial(on_delivery, row_start, timestamp_ms + offset),
                )
        self.assertEqual(producer.flush(30), 0)
        self.assertEqual(len(metadata), 4 * messages_per_partition * rows_per_message)
        cfg = utils.sample_data_config(
            data_pb2.DatasetType.KafkaDataset,
            mode,
            num_workers,
            kafka_partition_round_robin=round_robin,
        )
        cfg.input_fields.extend(
            [
                data_pb2.Field(input_name="raw_c", input_type=data_pb2.INT64),
                data_pb2.Field(input_name="label", input_type=data_pb2.INT64),
            ]
        )
        path = (
            f"kafka://{self.brokers}/{self.test_topic}"
            "?group.id=tzrec_sample_shuffle&auto.offset.reset=earliest"
        )

        def alarm(signum, frame):
            raise TimeoutError("timed out waiting for sample-shuffle batches")

        old_handler = signal.signal(signal.SIGALRM, alarm)
        self.addCleanup(signal.signal, signal.SIGALRM, old_handler)
        self.addCleanup(signal.setitimer, signal.ITIMER_REAL, 0)
        signal.setitimer(signal.ITIMER_REAL, 120)
        state = {}
        seen = set()
        for resume in (False, True):
            checkpoint = state.copy() if resume else {}
            dataloader = create_dataloader(
                cfg,
                utils.raw_c_features(),
                path,
                mode=Mode.TRAIN,
                checkpoint_state=checkpoint,
            )
            iterator = None
            message_batches = {}
            try:
                iterator = dataloader.get_iterator()
                for batch_index, batch in enumerate(iterator):
                    values = [int(v) for v in batch.to_dict()["raw_c.values"]]
                    self.assertEqual(len(values), 8)
                    self.assertEqual(len(set(values)), 8)
                    self.assertTrue(seen.isdisjoint(values))
                    seen.update(values)
                    expected = {}
                    timestamps = []
                    messages = set()
                    for value in values:
                        source, offset, event_time = metadata[value]
                        if resume:
                            self.assertGreater(offset, checkpoint[source])
                        expected[source] = max(expected.get(source, -1), offset)
                        timestamps.append(event_time)
                        messages.add((source, offset))
                    self.assertEqual(batch.checkpoint_info, expected)
                    self.assertEqual(batch.data_timestamp, max(timestamps))
                    for message in messages:
                        message_batches[message] = message_batches.get(message, 0) + 1
                    update_dataloder_state(state, batch.checkpoint_info)
                    if (resume and batch_index >= 3) or (
                        not resume and len(state) == 4
                    ):
                        break
                self.assertTrue(any(count > 1 for count in message_batches.values()))
            finally:
                utils.stop_sample(dataloader, iterator)
        self.assertEqual(len(state), 4)

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


if __name__ == "__main__":
    unittest.main()
