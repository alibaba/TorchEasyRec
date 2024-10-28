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
from typing import Any, Dict, Iterator, List, Optional

import common_io
import pyarrow as pa
from odps import options

from tzrec.datasets.dataset import BaseDataset, BaseReader
from tzrec.features.feature import BaseFeature
from tzrec.protos import data_pb2

# pyre-ignore [16]
options.read_timeout = int(os.getenv("TUNNEL_READ_TIMEOUT", "120"))

TYPE2PA = {
    "bigint": pa.int64(),
    "double": pa.float32(),
    "boolean": pa.int64(),
    "string": pa.string(),
    "datetime": pa.int64(),
}


def _pa_read(
    reader: common_io.table.TableReader,
    num_records: int = 1,
    allow_smaller_final_batch: bool = False,
) -> Dict[str, pa.Array]:
    """Read the table and return the rows as a pa.array."""
    reader._check_status()
    left_count = reader._end_pos - reader._read_pos
    if left_count <= 0 or (not allow_smaller_final_batch and left_count < num_records):
        raise common_io.exception.OutOfRangeException("No more data to read.")

    num_records = min(num_records, left_count)
    schema = reader.get_schema()
    result = [[] for _ in schema]
    for _ in range(num_records):
        record = reader._do_read_with_retry()
        reader._read_pos += 1
        for i, v in enumerate(record.values):
            result[i].append(v)

    result_dict = {}
    for i, s in enumerate(schema):
        typestr = s["typestr"]
        result_dict[s["colname"]] = pa.array(result[i], type=TYPE2PA[typestr])

    return result_dict


class OdpsDatasetV1(BaseDataset):
    """Dataset for reading data in Odps(Maxcompute).

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
        self._reader = OdpsReaderV1(
            input_path,
            self._batch_size,
            list(self._selected_input_names),
            self._data_config.drop_remainder,
        )
        self._init_input_fields()

    def _init_input_fields(self) -> None:
        """Init input fields info."""
        self._input_fields = []
        typedict = {
            "bigint": pa.int64(),
            "double": pa.float64(),
            "boolean": pa.int64(),
            "string": pa.string(),
            "datetime": pa.int64(),
        }
        for s in self._reader.schema:
            self._input_fields.append(
                pa.field(name=s["colname"], type=typedict[s["typestr"]])
            )
        # prevent graph-learn parse odps config hang
        os.environ["END_POINT"] = os.environ["ODPS_ENDPOINT"]


class OdpsReaderV1(BaseReader):
    """Odps(Maxcompute) reader class.

    Args:
        input_path (str): data input path.
        batch_size (int): batch size.
        selected_cols (list): selection column names.
        drop_remainder (bool): drop last batch.
    """

    def __init__(
        self,
        input_path: str,
        batch_size: int,
        selected_cols: Optional[List[str]] = None,
        drop_remainder: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(input_path, batch_size, selected_cols, drop_remainder)
        self.schema = []
        reader = common_io.table.TableReader(
            self._input_path.split(",")[0],
        )
        self._ordered_cols = []
        for field in reader.get_schema():
            if not selected_cols or field["colname"] in selected_cols:
                self.schema.append(field)
                self._ordered_cols.append(field["colname"])
        reader.close()

    def _iter_one_table(
        self, input_path: str, worker_id: int = 0, num_workers: int = 1
    ) -> Iterator[Dict[str, pa.Array]]:
        reader = common_io.table.TableReader(
            input_path,
            slice_id=worker_id,
            slice_count=num_workers,
            selected_cols=",".join(self._ordered_cols or []),
        )
        while True:
            try:
                data = _pa_read(
                    reader,
                    num_records=self._batch_size,
                    allow_smaller_final_batch=self._drop_remainder,
                )
            except common_io.exception.OutOfRangeException:
                reader.close()
                break
            yield data

    def to_batches(
        self, worker_id: int = 0, num_workers: int = 1
    ) -> Iterator[Dict[str, pa.Array]]:
        """Get batch iterator."""
        for input_path in self._input_path.split(","):
            yield from self._iter_one_table(input_path, worker_id, num_workers)
