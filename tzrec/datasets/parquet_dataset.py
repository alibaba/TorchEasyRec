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
import sys
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pyarrow as pa
from pyarrow import parquet
from torch import distributed as dist

from tzrec.constant import Mode
from tzrec.datasets.dataset import BaseDataset, BaseReader, BaseWriter
from tzrec.datasets.utils import (
    CKPT_ROW_IDX,
    CKPT_SOURCE_ID,
    calc_remaining_intervals,
    calc_slice_position,
    redistribute_intervals,
)
from tzrec.features.feature import BaseFeature
from tzrec.protos import data_pb2
from tzrec.utils import dist_util
from tzrec.utils.logging_util import logger


def _reader_iter(
    input_files: List[str],
    batch_size: int,
    parquet_metas: List[parquet.FileMetaData],
    columns: Optional[List[str]],
    start: int,
    end: int,
    worker_id: int,
    source_id: Optional[str] = None,
) -> Iterator[pa.RecordBatch]:
    cnt = 0
    global_row_idx = start  # Track absolute global row index
    for input_file in input_files:
        if cnt >= end:
            break

        metadata = parquet_metas.get(input_file, None)
        if metadata is not None and cnt + metadata.num_rows <= start:
            cnt += metadata.num_rows
            continue
        else:
            row_groups = None
            if metadata is not None:
                i = 0
                for i in range(metadata.num_row_groups):
                    row_group_rows = metadata.row_group(i).num_rows
                    if cnt + row_group_rows <= start:
                        cnt += row_group_rows
                        continue
                    else:
                        break
                row_groups = list(range(i, metadata.num_row_groups))

            parquet_file = parquet.ParquetFile(input_file)
            for batch in parquet_file.iter_batches(
                batch_size, row_groups=row_groups, columns=columns, use_threads=False
            ):
                original_batch_len = len(batch)  # Store before any modification
                if cnt + original_batch_len <= start:
                    logger.debug(
                        f"worker {worker_id} skip batch. "
                        f"start: {start}, end: {end}, cnt: {cnt}, len: {original_batch_len}."
                    )
                elif cnt <= start:
                    logger.debug(
                        f"worker {worker_id} yield start batch. "
                        f"start: {start}, end: {end}, cnt: {cnt}, len: {original_batch_len}."
                    )
                    sliced_batch = batch[start - cnt : end - cnt]
                    # Inject checkpoint metadata if source_id is provided
                    if source_id is not None:
                        batch_len = len(sliced_batch)
                        row_indices = list(
                            range(global_row_idx, global_row_idx + batch_len)
                        )
                        sliced_batch = pa.RecordBatch.from_arrays(
                            list(sliced_batch.columns)
                            + [
                                pa.array([source_id] * batch_len, type=pa.string()),
                                pa.array(row_indices, type=pa.int64()),
                            ],
                            names=list(sliced_batch.schema.names)
                            + [CKPT_SOURCE_ID, CKPT_ROW_IDX],
                        )
                        global_row_idx += batch_len
                    yield sliced_batch
                elif cnt + original_batch_len > end:
                    sliced_batch = batch[: end - cnt]
                    # Inject checkpoint metadata if source_id is provided
                    if source_id is not None:
                        batch_len = len(sliced_batch)
                        row_indices = list(
                            range(global_row_idx, global_row_idx + batch_len)
                        )
                        sliced_batch = pa.RecordBatch.from_arrays(
                            list(sliced_batch.columns)
                            + [
                                pa.array([source_id] * batch_len, type=pa.string()),
                                pa.array(row_indices, type=pa.int64()),
                            ],
                            names=list(sliced_batch.schema.names)
                            + [CKPT_SOURCE_ID, CKPT_ROW_IDX],
                        )
                        global_row_idx += batch_len
                    yield sliced_batch
                else:
                    # Inject checkpoint metadata if source_id is provided
                    if source_id is not None:
                        batch_len = len(batch)
                        row_indices = list(
                            range(global_row_idx, global_row_idx + batch_len)
                        )
                        batch = pa.RecordBatch.from_arrays(
                            list(batch.columns)
                            + [
                                pa.array([source_id] * batch_len, type=pa.string()),
                                pa.array(row_indices, type=pa.int64()),
                            ],
                            names=list(batch.schema.names)
                            + [CKPT_SOURCE_ID, CKPT_ROW_IDX],
                        )
                        global_row_idx += batch_len
                    yield batch

                cnt += original_batch_len
                if cnt >= end:
                    break
            parquet_file.close()


def _get_metadata(input_file: str) -> Tuple[str, parquet.FileMetaData]:
    parquet_file = parquet.ParquetFile(input_file)
    metadata = parquet_file.metadata
    parquet_file.close()
    return input_file, metadata


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
            drop_redundant_bs_eq_one=self._mode != Mode.PREDICT,
            sample_cost_field=self._data_config.sample_cost_field,
            batch_cost_size=self._data_config.batch_cost_size,
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
        drop_redundant_bs_eq_one (bool): drop last redundant batch with batch_size
            equal one to prevent train_eval hung.
        rebalance (bool): rebalance parquet rows to equal number for each worker.
        sample_cost_field (str): sample cost field name.
        batch_cost_size (int): batch cost limit size.
    """

    def __init__(
        self,
        input_path: str,
        batch_size: int,
        selected_cols: Optional[List[str]] = None,
        drop_remainder: bool = False,
        shuffle: bool = False,
        shuffle_buffer_size: int = 32,
        drop_redundant_bs_eq_one: bool = False,
        rebalance: bool = True,
        sample_cost_field: Optional[str] = None,
        batch_cost_size: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            input_path,
            batch_size,
            selected_cols,
            drop_remainder,
            shuffle,
            shuffle_buffer_size,
            sample_cost_field=sample_cost_field,
            batch_cost_size=batch_cost_size,
        )
        self._pg = dist_util.get_dist_object_pg()
        self._drop_redundant_bs_eq_one = drop_redundant_bs_eq_one
        self._rebalance = rebalance

        self._ordered_cols = None
        self._input_files = []
        for input_path in self._input_path.split(","):
            self._input_files.extend(glob.glob(input_path))
        if len(self._input_files) == 0:
            raise RuntimeError(f"No parquet files exist in {self._input_path}.")

        parquet_file = parquet.ParquetFile(self._input_files[0])
        if self._selected_cols:
            fields = []
            self._ordered_cols = []
            for field in parquet_file.schema_arrow:
                # pyre-ignore [58]
                if field.name in selected_cols:
                    fields.append(field)
                    self._ordered_cols.append(field.name)
            self._schema = pa.schema(fields)
        else:
            self._schema = parquet_file.schema_arrow
        parquet_file.close()

        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        # get parquet metadata
        self._parquet_metas = {}
        parquet_metas_per_rank = {}
        self._num_rows = []
        if self._rebalance:
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                for k, v in executor.map(
                    _get_metadata, self._input_files[rank::world_size]
                ):
                    parquet_metas_per_rank[k] = v
            if self._pg is not None:
                parquet_metas_list = [None] * world_size
                dist.all_gather_object(parquet_metas_list, parquet_metas_per_rank)
                for v in parquet_metas_list:
                    self._parquet_metas.update(v)
            else:
                self._parquet_metas = parquet_metas_per_rank

            for input_file in self._input_files:
                self._num_rows.append(self._parquet_metas[input_file].num_rows)

    @property
    def schema(self) -> pa.Schema:
        """Table schema."""
        return self._schema

    def to_batches(
        self, worker_id: int = 0, num_workers: int = 1
    ) -> Iterator[Dict[str, pa.Array]]:
        """Get batch iterator."""
        if len(self._input_files) == 0:
            return

        total_rows = sum(self._num_rows) if self._rebalance else 0

        # Check if we're resuming from checkpoint
        if self._checkpoint_state and self._rebalance:
            # Calculate remaining intervals from checkpoint
            remaining = calc_remaining_intervals(
                self._checkpoint_state,
                self._input_path,
                total_rows,
            )
            if not remaining:
                return  # All data already consumed

            # Redistribute remaining intervals among current workers
            worker_intervals = redistribute_intervals(
                remaining, worker_id, num_workers
            )
            if not worker_intervals:
                return  # This worker has no data to process

            # Generate source_id for this worker's intervals
            for start, end in worker_intervals:
                source_id = f"{self._input_path}:{start}"
                reader = _reader_iter(
                    self._input_files,
                    self._batch_size,
                    self._parquet_metas,
                    self._ordered_cols,
                    start,
                    end,
                    worker_id,
                    source_id,
                )
                yield from self._arrow_reader_iter(reader)
        else:
            # Normal case: no checkpoint or not rebalancing
            start, end = 0, sys.maxsize
            source_id = None
            if self._rebalance:
                start, end, _ = calc_slice_position(
                    total_rows,
                    worker_id,
                    num_workers,
                    self._batch_size,
                    self._drop_redundant_bs_eq_one,
                )
                source_id = f"{self._input_path}:{start}"

            reader = _reader_iter(
                self._input_files
                if self._rebalance
                else self._input_files[worker_id::num_workers],
                self._batch_size,
                self._parquet_metas,
                self._ordered_cols,
                start,
                end,
                worker_id,
                source_id,
            )
            yield from self._arrow_reader_iter(reader)

    def num_files(self) -> Optional[int]:
        """Get number of files in the dataset."""
        if self._rebalance:
            # We will rebalance the Parquet file into num_worker parts,
            # eliminating the need for num_files to determine the number
            # of workers to use.
            return None
        else:
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
