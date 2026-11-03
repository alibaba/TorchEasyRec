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

import argparse
import json
from datetime import datetime, timezone
from typing import Any, List, Optional, Set, Tuple

import pyarrow as pa
from confluent_kafka import Consumer, TopicPartition

from tzrec.datasets.dataset import BaseDataset
from tzrec.datasets.kafka_dataset import _parse_kafka_uri
from tzrec.datasets.utils import FIELD_TYPE_TO_PA, get_input_fields_proto
from tzrec.main import _create_features
from tzrec.protos import data_pb2
from tzrec.utils import config_util


def _decode_message(
    value: bytes, schema: Optional[pa.Schema]
) -> Tuple[str, List[pa.RecordBatch]]:
    """Decode a kafka message value into arrow record batches.

    Args:
        value (bytes): kafka message value.
        schema (pa.Schema, optional): schema for schema-less messages.

    Returns:
        fmt (str): ``ipc_stream`` for a message with an embedded schema, or
            ``schema_less`` for a ``record_batch.serialize()`` payload.
        batches (list): all record batches in the message.

    Raises:
        ValueError: if the message has no embedded schema and no schema is given.
    """
    try:
        reader = pa.ipc.open_stream(value)
    except (pa.ArrowInvalid, OSError) as e:
        if schema is None:
            raise ValueError(
                "message is not an Arrow IPC stream with an embedded schema "
                f"({e}); pass --input_fields_str or --pipeline_config_path to "
                "decode it as a schema-less record batch."
            ) from e
        return "schema_less", [pa.ipc.read_record_batch(value, schema)]
    try:
        batches = list(reader)
    finally:
        reader.close()
    return "ipc_stream", batches


def _fetch_messages(
    input_path: str, partition: int, offset: int, num_messages: int
) -> List[Any]:
    """Fetch messages from one partition without joining the consumer group.

    Args:
        input_path (str): kafka URI, e.g. kafka://broker:9092/topic?group.id=xxx
        partition (int): partition to read.
        offset (int): absolute start offset if >= 0, otherwise counted back from
            the high watermark (-1 is the last message).
        num_messages (int): max number of messages to fetch.

    Returns:
        list: fetched confluent_kafka messages.
    """
    topic, config, _ = _parse_kafka_uri(input_path)
    # never commit, the group.id may be shared with a training job
    config["enable.auto.commit"] = "false"
    consumer = Consumer(config)
    try:
        low, high = consumer.get_watermark_offsets(
            TopicPartition(topic, partition), timeout=60.0
        )
        if high <= low:
            return []
        start = offset if offset >= 0 else high + offset
        start = min(max(start, low), high - 1)
        num_messages = min(num_messages, high - start)
        consumer.assign([TopicPartition(topic, partition, start)])
        messages = []
        while len(messages) < num_messages:
            batch = consumer.consume(num_messages - len(messages), timeout=10.0)
            if not batch:
                break
            for msg in batch:
                if msg.error():
                    raise RuntimeError(f"error reading kafka message: {msg.error()}")
                messages.append(msg)
        return messages
    finally:
        consumer.close()


def _parse_schema(data_config: data_pb2.DataConfig) -> Optional[pa.Schema]:
    """Build the schema-less message schema from data_config input fields."""
    fields = get_input_fields_proto(data_config)
    if not fields:
        return None
    return pa.schema(
        [pa.field(f.input_name, FIELD_TYPE_TO_PA[f.input_type]) for f in fields]
    )


def inspect_kafka(
    input_path: str,
    partition: int = 0,
    offset: int = -1,
    num_messages: int = 1,
    num_rows: int = 5,
    columns: Optional[List[str]] = None,
    input_fields_str: Optional[str] = None,
    pipeline_config_path: Optional[str] = None,
) -> None:
    """Print the decoded content of kafka messages.

    Args:
        input_path (str): kafka URI, e.g. kafka://broker:9092/topic?group.id=xxx
        partition (int): partition to read.
        offset (int): absolute start offset if >= 0, otherwise counted back from
            the high watermark (-1 is the last message).
        num_messages (int): number of messages to print.
        num_rows (int): number of rows to print per message.
        columns (list, optional): columns to print, default all.
        input_fields_str (str, optional): schema for schema-less messages, in the
            data_config.input_fields_str format.
        pipeline_config_path (str, optional): pipeline config, used as the schema
            fallback and to report the columns the model needs but a message lacks.
    """
    schema = None
    selected_cols: Optional[Set[str]] = None
    if pipeline_config_path:
        pipeline_config = config_util.load_pipeline_config(pipeline_config_path)
        data_config = pipeline_config.data_config
        schema = _parse_schema(data_config)
        features = _create_features(list(pipeline_config.feature_configs), data_config)
        # BaseDataset selects the same columns as KafkaDataset without the
        # KafkaReader peeking partition 0 for a schema.
        dataset = BaseDataset(data_config, features, input_path)
        selected_cols = dataset._selected_input_names
    if input_fields_str:
        schema = _parse_schema(data_pb2.DataConfig(input_fields_str=input_fields_str))

    messages = _fetch_messages(input_path, partition, offset, num_messages)
    if not messages:
        print(f"no messages found in partition {partition}.")
        return

    for msg in messages:
        ts_ms = msg.timestamp()[1]
        ts_str = (
            datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).isoformat()
            if ts_ms >= 0
            else "N/A"
        )
        value = msg.value() or b""
        header = (
            f"=== partition={msg.partition()} offset={msg.offset()} "
            f"timestamp={ts_ms} ({ts_str}) size={len(value)}B"
        )
        try:
            fmt, batches = _decode_message(value, schema)
        except Exception as e:
            print(f"{header}\nfailed to decode message: {e}\n")
            continue
        num_total_rows = sum(len(b) for b in batches)
        print(f"{header} format={fmt} batches={len(batches)} rows={num_total_rows}")
        if len(batches) > 1:
            print(
                "WARNING: KafkaDataset only reads the first record batch of a "
                "message, the other batches are dropped."
            )
        if fmt == "ipc_stream" and schema is not None:
            print(
                "WARNING: message has an embedded schema but input_fields is set, "
                "KafkaDataset will decode it as schema-less."
            )
        if not batches:
            print()
            continue

        batch = batches[0]
        print(f"schema:\n{batch.schema.to_string()}")
        if selected_cols is not None:
            missing = sorted(selected_cols - set(batch.schema.names))
            print(f"columns required by the model but missing: {missing}")

        if columns:
            not_found = [c for c in columns if c not in batch.schema.names]
            if not_found:
                print(f"requested columns not in message: {not_found}")
            batch = batch.select([c for c in columns if c in batch.schema.names])
        print(f"rows (first {min(num_rows, len(batch))} of {len(batch)}):")
        for row in batch.slice(0, num_rows).to_pylist():
            print(json.dumps(row, default=str, ensure_ascii=False))
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="kafka uri, e.g. kafka://broker:9092/topic?group.id=xxx",
    )
    parser.add_argument(
        "--partition", type=int, default=0, help="kafka partition to read."
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=-1,
        help="start offset; negative counts back from the latest, -1 is the last "
        "message.",
    )
    parser.add_argument(
        "--num_messages", type=int, default=1, help="number of messages to print."
    )
    parser.add_argument(
        "--num_rows", type=int, default=5, help="number of rows to print per message."
    )
    parser.add_argument(
        "--columns",
        type=str,
        default=None,
        help="comma separated columns to print, default all.",
    )
    parser.add_argument(
        "--input_fields_str",
        type=str,
        default=None,
        help="schema for schema-less messages, e.g. user_id:INT64;tags:ARRAY<STRING>",
    )
    parser.add_argument(
        "--pipeline_config_path",
        type=str,
        default=None,
        help="pipeline config, used as schema fallback and to check the columns "
        "required by the model.",
    )
    args, extra_args = parser.parse_known_args()
    inspect_kafka(
        args.input_path,
        partition=args.partition,
        offset=args.offset,
        num_messages=args.num_messages,
        num_rows=args.num_rows,
        columns=[c.strip() for c in args.columns.split(",")] if args.columns else None,
        input_fields_str=args.input_fields_str,
        pipeline_config_path=args.pipeline_config_path,
    )
