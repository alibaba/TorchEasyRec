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

import contextlib
import io
import os
import shutil
import unittest
from unittest import mock

import pyarrow as pa
from parameterized import param, parameterized

from tzrec.tools.inspect_kafka import _decode_message, _fetch_messages, inspect_kafka
from tzrec.utils.test_util import make_test_dir, parameterized_name_func

_URI = "kafka://broker:9092/topic?group.id=g1"


def _batch(num_rows=3):
    return pa.record_batch(
        {
            "user_id": pa.array(list(range(num_rows)), type=pa.int64()),
            "tags": pa.array([["a", "b"]] * num_rows, type=pa.list_(pa.string())),
        }
    )


def _ipc_stream(batches, compression=None):
    sink = io.BytesIO()
    options = pa.ipc.IpcWriteOptions(compression=compression)
    with pa.ipc.new_stream(sink, batches[0].schema, options=options) as writer:
        for b in batches:
            writer.write_batch(b)
    return sink.getvalue()


def _message(value, offset, partition=0, ts_ms=1711929600000):
    msg = mock.MagicMock()
    msg.error.return_value = None
    msg.value.return_value = value
    msg.offset.return_value = offset
    msg.partition.return_value = partition
    msg.timestamp.return_value = (1, ts_ms)
    return msg


def _mock_consumer(values, low=0, ts_ms=1711929600000):
    consumer = mock.MagicMock()
    consumer.get_watermark_offsets.return_value = (low, low + len(values))
    state = {}

    def _assign(tps):
        state["pos"] = tps[0].offset

    def _consume(num, timeout):
        start = state["pos"]
        end = min(start + num, low + len(values))
        state["pos"] = end
        return [_message(values[o - low], o, ts_ms=ts_ms) for o in range(start, end)]

    consumer.assign.side_effect = _assign
    consumer.consume.side_effect = _consume
    return consumer


class DecodeMessageTest(unittest.TestCase):
    @parameterized.expand(
        [
            param("single", num_batches=1, compression=None),
            param("multi", num_batches=2, compression=None),
            param("zstd", num_batches=1, compression="zstd"),
        ],
        name_func=parameterized_name_func,
    )
    def test_ipc_stream(self, _name, num_batches, compression):
        batch = _batch().replace_schema_metadata({b"pandas": b"{}"})
        fmt, batches = _decode_message(
            _ipc_stream([batch] * num_batches, compression), None
        )
        self.assertEqual(fmt, "ipc_stream")
        self.assertEqual(len(batches), num_batches)
        self.assertTrue(batches[0].equals(batch))

    def test_schema_less(self):
        batch = _batch()
        fmt, batches = _decode_message(batch.serialize().to_pybytes(), batch.schema)
        self.assertEqual(fmt, "schema_less")
        self.assertTrue(batches[0].equals(batch))

    def test_schema_less_without_schema_raises(self):
        with self.assertRaisesRegex(ValueError, "input_fields_str"):
            _decode_message(_batch().serialize().to_pybytes(), None)

    def test_garbage_with_schema_raises(self):
        with self.assertRaises(pa.ArrowInvalid):
            _decode_message(b"not an arrow message", _batch().schema)


class FetchMessagesTest(unittest.TestCase):
    @parameterized.expand(
        [
            param("last", offset=-1, num_messages=1, expected=[14]),
            param("last_three", offset=-3, num_messages=3, expected=[12, 13, 14]),
            param("absolute", offset=11, num_messages=2, expected=[11, 12]),
            param("clamp_low", offset=0, num_messages=1, expected=[10]),
            param("clamp_high", offset=100, num_messages=3, expected=[14]),
            param("clamp_count", offset=-2, num_messages=5, expected=[13, 14]),
        ],
        name_func=parameterized_name_func,
    )
    def test_offset_resolution(self, _name, offset, num_messages, expected):
        consumer = _mock_consumer([b""] * 5, low=10)
        with mock.patch("tzrec.tools.inspect_kafka.Consumer", return_value=consumer):
            messages = _fetch_messages(_URI, 0, offset, num_messages)
        self.assertEqual([m.offset() for m in messages], expected)
        consumer.subscribe.assert_not_called()
        consumer.close.assert_called_once()

    def test_empty_partition(self):
        consumer = _mock_consumer([], low=10)
        with mock.patch("tzrec.tools.inspect_kafka.Consumer", return_value=consumer):
            self.assertEqual(_fetch_messages(_URI, 0, -1, 1), [])
        consumer.assign.assert_not_called()

    def test_never_commits(self):
        consumer = _mock_consumer([b""])
        with mock.patch(
            "tzrec.tools.inspect_kafka.Consumer", return_value=consumer
        ) as consumer_cls:
            _fetch_messages(_URI + "&enable.auto.commit=true", 0, -1, 1)
        self.assertEqual(consumer_cls.call_args[0][0]["enable.auto.commit"], "false")

    def test_message_error_raises(self):
        consumer = _mock_consumer([b""])
        bad = _message(b"", 0)
        bad.error.return_value = "broker down"
        consumer.consume.side_effect = None
        consumer.consume.return_value = [bad]
        with mock.patch("tzrec.tools.inspect_kafka.Consumer", return_value=consumer):
            with self.assertRaisesRegex(RuntimeError, "broker down"):
                _fetch_messages(_URI, 0, -1, 1)
        consumer.close.assert_called_once()


class InspectKafkaTest(unittest.TestCase):
    def setUp(self):
        self.test_dir = make_test_dir()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _run(self, values, ts_ms=1711929600000, **kwargs):
        consumer = _mock_consumer(values, ts_ms=ts_ms)
        out = io.StringIO()
        with (
            mock.patch(
                "tzrec.tools.inspect_kafka.Consumer", return_value=consumer
            ) as consumer_cls,
            contextlib.redirect_stdout(out),
        ):
            inspect_kafka(_URI, num_messages=len(values), **kwargs)
        self.consumer_cls = consumer_cls
        return out.getvalue()

    def test_ipc_stream(self):
        output = self._run(
            [_ipc_stream([_batch(), _batch()])], num_rows=2, columns=["tags", "x"]
        )
        self.assertIn("format=ipc_stream batches=2 rows=6", output)
        self.assertIn("only reads the first record batch", output)
        self.assertIn("user_id: int64", output)
        self.assertIn("requested columns not in message: ['x']", output)
        self.assertIn('rows (first 2 of 3):\n{"tags": ["a", "b"]}\n', output)

    def test_schema_less(self):
        value = _batch().serialize().to_pybytes()
        output = self._run([value], input_fields_str="user_id:INT64;tags:ARRAY<STRING>")
        self.assertIn("format=schema_less batches=1 rows=3", output)
        self.assertIn('{"user_id": 0, "tags": ["a", "b"]}', output)
        self.assertIn("failed to decode message", self._run([value]))

    def test_ipc_stream_with_input_fields_warns(self):
        output = self._run([_ipc_stream([_batch()])], input_fields_str="user_id:INT64")
        self.assertIn("embedded schema but input_fields is set", output)

    def test_bad_messages_do_not_stop(self):
        output = self._run(
            [None, b"garbage", _ipc_stream([_batch()])], ts_ms=-1, offset=0
        )
        self.assertIn("offset=0 timestamp=-1 (N/A) size=0B", output)
        self.assertEqual(output.count("failed to decode message"), 2)
        self.assertIn("offset=2 timestamp=-1 (N/A) size=", output)
        self.assertIn("format=ipc_stream batches=1 rows=3", output)

    def test_missing_columns(self):
        config_path = os.path.join(self.test_dir, "pipeline.config")
        with open(config_path, "w") as f:
            f.write(
                """
data_config {
    batch_size: 32
    dataset_type: KafkaDataset
    fg_mode: FG_DAG
    label_fields: "label"
}
feature_configs {
    id_feature {
        feature_name: "user_id"
        expression: "user:user_id"
        num_buckets: 100
    }
}
feature_configs {
    id_feature {
        feature_name: "item_id"
        expression: "item:item_id"
        num_buckets: 100
    }
}
"""
            )
        output = self._run([_ipc_stream([_batch()])], pipeline_config_path=config_path)
        self.assertIn(
            "columns required by the model but missing: ['item_id', 'label']", output
        )
        # only the fetch consumer, no schema peek from building the dataset
        self.consumer_cls.assert_called_once()


if __name__ == "__main__":
    unittest.main()
