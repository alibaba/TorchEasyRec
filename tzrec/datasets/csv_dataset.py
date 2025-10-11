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


import glob
import os
import random
import time
from collections import OrderedDict
from typing import Any, Dict, Iterator, List, Optional

import pyarrow as pa
import pyarrow.dataset as ds
from pyarrow import csv

from tzrec.constant import Mode
from tzrec.datasets.dataset import BaseDataset, BaseReader, BaseWriter
from tzrec.datasets.utils import FIELD_TYPE_TO_PA
from tzrec.features.feature import BaseFeature
from tzrec.protos import data_pb2


class CsvDataset(BaseDataset):
    """Dataset for reading data with csv format.

    Args:
        data_config (DataConfig): an instance of DataConfig.
        features (list): list of features.
        input_path (str): data input path.
    """

    def __init__(
        self,
        data_config: data_pb2.DataConfig,
        features: List[BaseFeature],
        input_path: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(data_config, features, input_path, **kwargs)
        column_names = None
        column_types = {}
        if not self._data_config.with_header:
            column_names = [f.input_name for f in self._data_config.input_fields]
        for f in self._data_config.input_fields:
            if f.HasField("input_type"):
                if f.input_type in FIELD_TYPE_TO_PA:
                    column_types[f.input_name] = FIELD_TYPE_TO_PA[f.input_type]
                else:
                    raise ValueError(
                        f"{f.input_type} of column [{f.input_name}] "
                        "is not supported by CsvDataset."
                    )
        # pyre-ignore [29]
        self._reader = CsvReader(
            input_path,
            self._batch_size,
            list(self._selected_input_names) if self._selected_input_names else None,
            self._data_config.drop_remainder,
            shuffle=self._data_config.shuffle and self._mode == Mode.TRAIN,
            shuffle_buffer_size=self._data_config.shuffle_buffer_size,
            column_names=column_names,
            delimiter=self._data_config.delimiter,
            column_types=column_types,
        )
        self._init_input_fields()


class CsvReader(BaseReader):
    """Csv reader class.

    Args:
        input_path (str): data input path.
        batch_size (int): batch size.
        selected_cols (list): selection column names.
        drop_remainder (bool): drop last batch.
        shuffle (bool): shuffle data or not.
        shuffle_buffer_size (int): buffer size for shuffle.
        column_names (list): set column name if csv without header.
        delimiter (str): csv delimiter.
    """

    def __init__(
        self,
        input_path: str,
        batch_size: int,
        selected_cols: Optional[List[str]] = None,
        drop_remainder: bool = False,
        shuffle: bool = False,
        shuffle_buffer_size: int = 32,
        column_names: Optional[List[str]] = None,
        delimiter: str = ",",
        column_types: Optional[Dict[str, pa.DataType]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            input_path,
            batch_size,
            selected_cols,
            drop_remainder,
            shuffle,
            shuffle_buffer_size,
        )
        self._csv_fmt = ds.CsvFileFormat(
            parse_options=pa.csv.ParseOptions(delimiter=delimiter),
            convert_options=pa.csv.ConvertOptions(column_types=column_types),
            read_options=csv.ReadOptions(
                column_names=column_names, block_size=64 * 1024 * 1024
            ),
        )
        self._input_files = []
        for input_path in self._input_path.split(","):
            self._input_files.extend(glob.glob(input_path))
        if len(self._input_files) == 0:
            raise RuntimeError(f"No csv files exist in {self._input_path}.")
        dataset = ds.dataset(self._input_files[0], format=self._csv_fmt)
        self._ordered_cols = None
        if self._selected_cols is not None:
            fields = []
            self._ordered_cols = []
            for field in dataset.schema:
                # pyre-ignore [58]
                if field.name in self._selected_cols:
                    fields.append(field)
                    self._ordered_cols.append(field.name)
            self._schema = pa.schema(fields)
        else:
            self._schema = dataset.schema

    @property
    def schema(self) -> pa.Schema:
        """Table schema."""
        return self._schema

    def to_batches(
        self, worker_id: int = 0, num_workers: int = 1
    ) -> Iterator[Dict[str, pa.Array]]:
        """Get batch iterator."""
        input_files = self._input_files[worker_id::num_workers]
        if self._shuffle:
            random.shuffle(input_files)
        if len(input_files) > 0:
            dataset = ds.dataset(input_files, format=self._csv_fmt)
            reader = dataset.to_batches(
                batch_size=self._batch_size, columns=self._ordered_cols
            )
            yield from self._arrow_reader_iter(reader)

    def num_files(self) -> int:
        """Get number of files in the dataset."""
        return len(self._input_files)


class CsvWriter(BaseWriter):
    """Csv writer class.

    Args:
        output_path (str): data output path.
    """

    def __init__(self, output_path: str, **kwargs: Any) -> None:
        rank = int(os.environ.get("RANK", 0))
        if rank == 0:
            if not os.path.exists(output_path):
                os.makedirs(output_path)
        else:
            while not os.path.exists(output_path):
                time.sleep(1)
        output_path = os.path.join(output_path, f"part-{rank}.csv")
        super().__init__(output_path)
        self._writer = None

    def write(self, output_dict: OrderedDict[str, pa.Array]) -> None:
        """Write a batch of data."""
        if not self._lazy_inited:
            schema = []
            for k, v in output_dict.items():
                schema.append((k, v.type))
            self._writer = csv.CSVWriter(self._output_path, schema=pa.schema(schema))
            self._lazy_inited = True
        record_batch = pa.RecordBatch.from_arrays(
            list(output_dict.values()),
            list(output_dict.keys()),
        )
        self._writer.write(record_batch)

    def close(self) -> None:
        """Close and commit data."""
        if self._writer is not None:
            self._writer.close()
        super().close()
