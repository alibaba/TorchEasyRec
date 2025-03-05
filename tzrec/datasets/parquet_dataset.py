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
from pyarrow import parquet

from tzrec.constant import Mode
from tzrec.datasets.dataset import BaseDataset, BaseReader, BaseWriter
from tzrec.features.feature import BaseFeature
from tzrec.protos import data_pb2


def _reader_iter(
    input_files: List[str],
    batch_size: int,
) -> Iterator[pa.RecordBatch]:
    for input_file in input_files:
        parquet_file = parquet.ParquetFile(input_file)
        for batch in parquet_file.iter_batches(batch_size):
            yield batch


class ParquetDataset(BaseDataset):
    """Dataset for reading data with parquet format.

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
        # pyre-ignore [29]
        self._reader = ParquetReader(
            input_path,
            self._batch_size,
            list(self._selected_input_names) if self._selected_input_names else None,
            self._data_config.drop_remainder,
            shuffle=self._data_config.shuffle and self._mode == Mode.TRAIN,
            shuffle_buffer_size=self._data_config.shuffle_buffer_size,
        )
        self._init_input_fields()


class ParquetReader(BaseReader):
    """Parquet reader class.

    Args:
        input_path (str): data input path.
        batch_size (int): batch size.
        selected_cols (list): selection column names.
        drop_remainder (bool): drop last batch.
        shuffle (bool): shuffle data or not.
        shuffle_buffer_size (int): buffer size for shuffle.
    """

    def __init__(
        self,
        input_path: str,
        batch_size: int,
        selected_cols: Optional[List[str]] = None,
        drop_remainder: bool = False,
        shuffle: bool = False,
        shuffle_buffer_size: int = 32,
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
        self._ordered_cols = None
        self.schema = []
        self._input_files = []
        for input_path in self._input_path.split(","):
            self._input_files.extend(glob.glob(input_path))
        if len(self._input_files) == 0:
            raise RuntimeError(f"No parquet files exist in {self._input_path}.")
        parquet_file = parquet.ParquetFile(self._input_files[0])
        if self._selected_cols:
            self._ordered_cols = []
            for field in parquet_file.schema_arrow:
                # pyre-ignore [58]
                if field.name in selected_cols:
                    self.schema.append(field)
                    self._ordered_cols.append(field.name)
        else:
            self.schema = parquet_file.schema_arrow

    def to_batches(
        self, worker_id: int = 0, num_workers: int = 1
    ) -> Iterator[Dict[str, pa.Array]]:
        """Get batch iterator."""
        input_files = self._input_files[worker_id::num_workers]
        if self._shuffle:
            random.shuffle(input_files)
        if len(input_files) > 0:
            reader = _reader_iter(input_files, self._batch_size)
            yield from self._arrow_reader_iter(reader)

    def num_files(self) -> int:
        """Get number of files in the dataset."""
        return len(self._input_files)


class ParquetWriter(BaseWriter):
    """Parquet writer class.

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
        output_path = os.path.join(output_path, f"part-{rank}.parquet")
        super().__init__(output_path)
        self._writer = None

    def write(self, output_dict: OrderedDict[str, pa.Array]) -> None:
        """Write a batch of data."""
        if not self._lazy_inited:
            schema = []
            for k, v in output_dict.items():
                schema.append((k, v.type))
            self._writer = parquet.ParquetWriter(
                self._output_path, schema=pa.schema(schema)
            )
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
