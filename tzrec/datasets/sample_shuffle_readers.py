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

"""Module-level read tasks for the sample-level shuffle read plan.

Every function is a generator called by a reader process as
``fn(stop_event=..., **kwargs)``. The parameters must be picklable because
the plan travels to the spawned coordinator; resources that are not picklable
are created inside the reader process instead of being handed over.
"""

from typing import Any, Dict, Iterator, List, Optional, Tuple

import pyarrow as pa


def read_parquet(
    stop_event: Any,
    input_files: List[str],
    batch_size: int,
    parquet_metas: Dict[str, Any],
    columns: Optional[List[str]],
    intervals: List[Tuple[int, int]],
    source_prefix: str,
    reader_id: int,
) -> Iterator[pa.RecordBatch]:
    """Read assigned parquet row intervals from the shared file list.

    Args:
        stop_event: unused; the reader loop checks it between batches.
        input_files: all parquet files of the dataset, in a stable order.
        batch_size: rows per scanner batch.
        parquet_metas: per-file metadata used to skip whole row groups.
        columns: selected columns, or None for all.
        intervals: absolute (start, end) row intervals of this task.
        source_prefix: checkpoint source id prefix.
        reader_id: reader slot, used in debug logs.

    Yields:
        parquet record batches with checkpoint metadata.
    """
    from tzrec.datasets.parquet_dataset import _reader_iter

    for start, end in intervals:
        yield from _reader_iter(
            input_files,
            batch_size,
            parquet_metas,
            columns,
            start,
            end,
            reader_id,
            source_id=f"{source_prefix}:{start}",
        )
