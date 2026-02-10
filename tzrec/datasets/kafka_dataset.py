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


import math
from typing import Any, Dict, Iterator, List, Optional, Tuple
from urllib.parse import parse_qsl, urlparse

import pyarrow as pa
from confluent_kafka import Consumer

from tzrec.datasets.dataset import BaseDataset, BaseReader
from tzrec.datasets.utils import FIELD_TYPE_TO_PA
from tzrec.features.feature import BaseFeature
from tzrec.protos import data_pb2


def _parse_kafka_uri(uri: str) -> Tuple[str, Dict[str, Any]]:
    """Parse kafka URI into configuration dict.

    Args:
        uri: kafka://broker:9092/topic?group.id=xxx&auto.offset.reset=earliest

    Returns:
        Dict containing broker, topic, and consumer config
    """
    parsed = urlparse(uri)
    if parsed.scheme != "kafka":
        raise ValueError(f"Invalid kafka URI scheme: {parsed.scheme}")

    broker = parsed.netloc
    if not broker:
        raise ValueError("Kafka broker not specified in URI")

    topic = parsed.path.lstrip("/")
    if not topic:
        raise ValueError("Kafka topic not specified in URI")

    params = dict(parse_qsl(parsed.query))
    if "group.id" not in params:
        raise ValueError("Consumer group not specified in URI (use ?group.id=xxx)")

    if "debug" not in params:
        params["debug"] = "assignor,conf"
    params["bootstrap.servers"] = broker

    return topic, params


class KafkaDataset(BaseDataset):
    """Dataset for reading data from Kafka (each message is a serialized Arrow batch).

    Args:
        data_config (DataConfig): an instance of DataConfig.
        features (list): list of features.
        input_path (str): kafka URI, e.g. kafka://broker:9092/topic?group=xxx
    """

    def __init__(
        self,
        data_config: data_pb2.DataConfig,
        features: List[BaseFeature],
        input_path: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(data_config, features, input_path, **kwargs)

        input_fields = []
        for f in self._data_config.input_fields:
            pa_type = FIELD_TYPE_TO_PA.get(f.input_type)
            input_fields.append(pa.field(f.input_name, pa_type))

        # pyre-ignore [29]
        self._reader = KafkaReader(
            input_path,
            self._batch_size,
            list(self._selected_input_names) if self._selected_input_names else None,
            self._data_config.drop_remainder,
            input_fields=input_fields,
        )
        self._init_input_fields()


class KafkaReader(BaseReader):
    """Kafka reader class.

    Args:
        input_path (str): kafka URI, e.g. kafka://broker:9092/topic?group=xxx
        batch_size (int): batch size.
        selected_cols (list): selection column names.
        drop_remainder (bool): drop last batch.
        input_fields (list): list of pa.Field for schema definition.
    """

    def __init__(
        self,
        input_path: str,
        batch_size: int,
        selected_cols: Optional[List[str]] = None,
        drop_remainder: bool = False,
        input_fields: Optional[List[pa.Field]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            input_path,
            batch_size,
            selected_cols,
            drop_remainder,
        )

        self._drop_columns = []
        if input_fields:
            self._full_schema = pa.schema(input_fields)
            if self._selected_cols:
                fields = []
                self._ordered_cols = []
                for field in input_fields:
                    if field.name in self._selected_cols:
                        fields.append(field)
                        self._ordered_cols.append(field.name)
                    else:
                        self._drop_columns.append(field.name)
                self._schema = pa.schema(fields)
            else:
                self._schema = pa.schema(input_fields)
                self._ordered_cols = [f.name for f in input_fields]
        else:
            raise ValueError("input_fields must be specified for KafkaReader. ")

    @property
    def schema(self) -> pa.Schema:
        """Table schema."""
        return self._schema

    def _reader(
        self, worker_id: int = 0, num_workers: int = 1
    ) -> Iterator[pa.RecordBatch]:
        """Read Arrow batches from Kafka.

        Args:
            worker_id: Worker ID
            num_workers: Total number of workers

        Yields:
            PyArrow RecordBatch
        """
        topic, config = _parse_kafka_uri(self._input_path)
        consumer = Consumer(config)
        consumer.subscribe([topic])

        batch_size_per_msg = None
        try:
            while True:
                num_messages = (
                    int(math.ceil(self._batch_size / batch_size_per_msg))
                    if batch_size_per_msg
                    else 2
                )
                messages = consumer.consume(num_messages)

                current_batch_size = 0
                record_batchs = []
                for msg in messages:
                    msg_data = msg.value()
                    record_batch = pa.ipc.read_record_batch(msg_data, self._full_schema)
                    current_batch_size += len(record_batch)
                    record_batchs.append(record_batch)

                # estimate batch_size per message
                if batch_size_per_msg is None:
                    batch_size_per_msg = current_batch_size / len(record_batch)
                else:
                    batch_size_per_msg = (
                        0.9 * batch_size_per_msg
                        + 0.1 * current_batch_size / len(record_batch)
                    )

                # combine into one record batch
                t = (
                    pa.Table.from_batches(record_batchs)
                    .drop_columns(self._drop_columns)
                    .combine_chunks()
                )
                for batch in t.to_batches():
                    yield batch

        finally:
            consumer.close()

    def to_batches(
        self, worker_id: int = 0, num_workers: int = 1
    ) -> Iterator[Dict[str, pa.Array]]:
        """Get batch iterator.

        Args:
            worker_id: Worker ID
            num_workers: Total number of workers

        Yields:
            Dict of column name to PyArrow Array
        """
        reader = self._reader(worker_id, num_workers)
        yield from self._arrow_reader_iter(reader)
