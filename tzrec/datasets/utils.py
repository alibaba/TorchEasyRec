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

from dataclasses import dataclass, field
from typing import Dict, Optional, List

import numpy.typing as npt
import pyarrow as pa
import torch
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor, KeyedTensor
from torchrec.streamable import Pipelineable

from tzrec.protos.data_pb2 import FieldType

BASE_DATA_GROUP = "__BASE__"
NEG_DATA_GROUP = "__NEG__"
CROSS_NEG_DATA_GROUP = "__CNEG__"

C_SAMPLE_MASK = "__SAMPLE_MASK__"
C_NEG_SAMPLE_MASK = "__NEG_SAMPLE_MASK__"

FIELD_TYPE_TO_PA = {
    FieldType.INT32: pa.int32(),
    FieldType.INT64: pa.int64(),
    FieldType.FLOAT: pa.float32(),
    FieldType.DOUBLE: pa.float64(),
    FieldType.STRING: pa.string(),
}


@dataclass
class ParsedData:
    """Internal parsed data structure."""

    name: str


@dataclass
class SparseData(ParsedData):
    """Internal data structure for sparse feature."""

    values: npt.NDArray
    lengths: npt.NDArray
    weights: Optional[npt.NDArray] = None


@dataclass
class DenseData(ParsedData):
    """Internal data structure for dense feature."""

    values: npt.NDArray


@dataclass
class SequenceSparseData(ParsedData):
    """Internal data structure for sequence sparse feature."""

    values: npt.NDArray
    lengths: npt.NDArray
    seq_lengths: npt.NDArray


@dataclass
class SequenceDenseData(ParsedData):
    """Internal data structure for sequence dense feature."""

    values: npt.NDArray
    seq_lengths: npt.NDArray


class RecordBatchTensor:
    """PyArrow RecordBatch use Tensor as buffer.

    For efficient transfer data between processes, e.g., mp.Queue.
    """

    def __init__(self, record_batch: Optional[pa.RecordBatch] = None) -> None:
        self._schema = None
        self._buff = None
        if record_batch:
            self._schema = record_batch.schema
            self._buff = torch.UntypedStorage.from_buffer(
                record_batch.serialize(), dtype=torch.uint8
            )

    def get(self) -> Optional[pa.RecordBatch]:
        """Get RecordBatch."""
        if self._buff is not None:
            # pyre-ignore[16]
            return pa.ipc.read_record_batch(
                pa.foreign_buffer(self._buff.data_ptr(), self._buff.size()),
                self._schema,
            )
        else:
            return None


@dataclass
class Batch(Pipelineable):
    """Input Batch."""

    # key of dense_features is group name
    dense_features: Dict[str, KeyedTensor] = field(default_factory=dict)
    # key of sparse_features is group name
    sparse_features: Dict[str, KeyedJaggedTensor] = field(default_factory=dict)
    # key of sequence_dense_features is feature name
    sequence_dense_features: Dict[str, JaggedTensor] = field(default_factory=dict)
    # key of labels is label name
    labels: Dict[str, torch.Tensor] = field(default_factory=dict)
    # reserved inputs [for predict]
    reserves: RecordBatchTensor = field(default_factory=RecordBatchTensor)
    # size for user side input tile when do inference and INPUT_TILE=2 or 3
    tile_size: int = field(default=-1)
    # sample_weight
    sample_weights: Dict[str, torch.Tensor] = field(default_factory=dict)

    def to(self, device: torch.device, non_blocking: bool = False) -> "Batch":
        """Copy to specified device."""
        return Batch(
            dense_features={
                k: v.to(device=device, non_blocking=non_blocking)
                for k, v in self.dense_features.items()
            },
            sparse_features={
                k: v.to(device=device, non_blocking=non_blocking)
                for k, v in self.sparse_features.items()
            },
            sequence_dense_features={
                k: v.to(device=device, non_blocking=non_blocking)
                for k, v in self.sequence_dense_features.items()
            },
            labels={
                k: v.to(device=device, non_blocking=non_blocking)
                for k, v in self.labels.items()
            },
            reserves=self.reserves,
            tile_size=self.tile_size,
            sample_weights={
                k: v.to(device=device, non_blocking=non_blocking)
                for k, v in self.sample_weights.items()
            },
        )

    def record_stream(self, stream: torch.Stream) -> None:
        """Record which streams have used the tensor."""
        for v in self.dense_features.values():
            v.record_stream(stream)
        for v in self.sparse_features.values():
            v.record_stream(stream)
        for v in self.sequence_dense_features.values():
            v.record_stream(stream)
        for v in self.labels.values():
            v.record_stream(stream)
        for v in self.sample_weights.values():
            v.record_stream(stream)

    def pin_memory(self) -> "Batch":
        """Copy to pinned memory."""
        # TODO(hongsheng.jhs): KeyedTensor do not have pin_memory()
        dense_features = {}
        for k, v in self.dense_features.items():
            dense_features[k] = KeyedTensor(
                keys=v.keys(),
                length_per_key=v.length_per_key(),
                values=v.values().pin_memory(),
                key_dim=v.key_dim(),
            )
        sequence_dense_features = {}
        for k, v in self.sequence_dense_features.items():
            weights = v._weights
            lengths = v._lengths
            offsets = v._offsets
            sequence_dense_features[k] = JaggedTensor(
                values=v.values().pin_memory(),
                weights=weights.pin_memory() if weights is not None else None,
                lengths=lengths.pin_memory() if lengths is not None else None,
                offsets=offsets.pin_memory() if offsets is not None else None,
            )
        return Batch(
            dense_features=dense_features,
            sparse_features={
                k: v.pin_memory() for k, v in self.sparse_features.items()
            },
            sequence_dense_features=sequence_dense_features,
            labels={k: v.pin_memory() for k, v in self.labels.items()},
            reserves=self.reserves,
            tile_size=self.tile_size,
            sample_weights={k: v.pin_memory() for k, v in self.sample_weights.items()},
        )

    def to_dict(
        self, sparse_dtype: Optional[torch.dtype] = None
    ) -> Dict[str, torch.Tensor]:
        """Convert to feature tensor dict."""
        tensor_dict = {}
        for x in self.dense_features.values():
            for k, v in x.to_dict().items():
                tensor_dict[f"{k}.values"] = v
        for x in self.sparse_features.values():
            if sparse_dtype:
                x = KeyedJaggedTensor(
                    keys=x.keys(),
                    values=x.values().to(sparse_dtype),
                    lengths=x.lengths().to(sparse_dtype),
                    weights=x.weights_or_none(),
                )
            for k, v in x.to_dict().items():
                tensor_dict[f"{k}.values"] = v.values()
                tensor_dict[f"{k}.lengths"] = v.lengths()
                if v.weights_or_none() is not None:
                    tensor_dict[f"{k}.weights"] = v.weights()
        for k, v in self.sequence_dense_features.items():
            tensor_dict[f"{k}.values"] = v.values()
            tensor_dict[f"{k}.lengths"] = v.lengths()
        for k, v in self.labels.items():
            tensor_dict[f"{k}"] = v
        for k, v in self.sample_weights.items():
            tensor_dict[f"{k}"] = v
        if self.tile_size > 0:
            tensor_dict["batch_size"] = torch.tensor(self.tile_size, dtype=torch.int64)
        return tensor_dict

    def to_list(
        self,
        sparse_dtype: Optional[torch.dtype] = None
    ) -> Dict[str, torch.Tensor]:
        """Convert to feature tensor list.
        used in export,we will skip the labels.
        """
        tensor_dict = {}
        for x in self.dense_features.values():
            for k, v in x.to_dict().items():
                tensor_dict[f"{k}.values"] = v
        for x in self.sparse_features.values():
            if sparse_dtype:
                x = KeyedJaggedTensor(
                    keys=x.keys(),
                    values=x.values().to(sparse_dtype),
                    lengths=x.lengths().to(sparse_dtype),
                    weights=x.weights_or_none(),
                )
            for k, v in x.to_dict().items():
                tensor_dict[f"{k}.values"] = v.values()
                tensor_dict[f"{k}.lengths"] = v.lengths()
                if v.weights_or_none() is not None:
                    tensor_dict[f"{k}.weights"] = v.weights()
        for k, v in self.sequence_dense_features.items():
            tensor_dict[f"{k}.values"] = v.values()
            tensor_dict[f"{k}.lengths"] = v.lengths()
        if self.tile_size > 0:
            tensor_dict["batch_size"] = torch.tensor(self.tile_size, dtype=torch.int64)
        sorted_dict = {k: tensor_dict[k] for k in sorted(tensor_dict)}
        values_list = list(sorted_dict.values())
        return values_list