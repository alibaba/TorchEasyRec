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

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.compute as pc
import torch
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor, KeyedTensor
from torchrec.streamable import Pipelineable

from tzrec.protos import data_pb2
from tzrec.protos.data_pb2 import FieldType
from tzrec.utils.logging_util import logger

BASE_DATA_GROUP = "__BASE__"
NEG_DATA_GROUP = "__NEG__"
CROSS_NEG_DATA_GROUP = "__CNEG__"

C_SAMPLE_MASK = "__SAMPLE_MASK__"
C_NEG_SAMPLE_MASK = "__NEG_SAMPLE_MASK__"

HARD_NEG_INDICES = "hard_neg_indices"
CAND_POS_LENGTHS = "cand_pos_lengths"

# Checkpoint metadata column names injected into RecordBatch
CKPT_SOURCE_ID = "__ckpt_source_id__"  # string column for checkpoint source identifier
CKPT_ROW_IDX = "__ckpt_row_idx__"  # int64 column for absolute row index
# transient event-time column (Unix-epoch seconds, -1 when unavailable); its
# per-batch max is surfaced on Batch.data_timestamp.
DATA_TIMESTAMP = "__data_timestamp__"  # float64 column, event-time (seconds)


def inject_checkpoint_metadata(
    batch: pa.RecordBatch,
    source_id: str,
    global_row_idx: int,
) -> Tuple[pa.RecordBatch, int]:
    """Inject checkpoint metadata (source_id and row_idx) into a batch.

    Args:
        batch: The input record batch.
        source_id: The source identifier for checkpointing.
        global_row_idx: The current global row index.

    Returns:
        A tuple of (new_batch_with_metadata, updated_global_row_idx).
    """
    batch_len = len(batch)
    row_indices = list(range(global_row_idx, global_row_idx + batch_len))
    new_batch = pa.RecordBatch.from_arrays(
        list(batch.columns)
        + [
            pa.array([source_id] * batch_len, type=pa.string()),
            pa.array(row_indices, type=pa.int64()),
        ],
        names=list(batch.schema.names) + [CKPT_SOURCE_ID, CKPT_ROW_IDX],
    )
    return new_batch, global_row_idx + batch_len


FIELD_TYPE_TO_PA = {
    FieldType.INT32: pa.int32(),
    FieldType.INT64: pa.int64(),
    FieldType.FLOAT: pa.float32(),
    FieldType.DOUBLE: pa.float64(),
    FieldType.STRING: pa.string(),
    FieldType.ARRAY_INT32: pa.list_(pa.int32()),
    FieldType.ARRAY_INT64: pa.list_(pa.int64()),
    FieldType.ARRAY_FLOAT: pa.list_(pa.float32()),
    FieldType.ARRAY_DOUBLE: pa.list_(pa.float64()),
    FieldType.ARRAY_STRING: pa.list_(pa.string()),
    FieldType.ARRAY_ARRAY_INT32: pa.list_(pa.list_(pa.int32())),
    FieldType.ARRAY_ARRAY_INT64: pa.list_(pa.list_(pa.int64())),
    FieldType.ARRAY_ARRAY_FLOAT: pa.list_(pa.list_(pa.float32())),
    FieldType.ARRAY_ARRAY_DOUBLE: pa.list_(pa.list_(pa.float64())),
    FieldType.ARRAY_ARRAY_STRING: pa.list_(pa.list_(pa.string())),
    FieldType.MAP_STRING_INT32: pa.map_(pa.string(), pa.int32()),
    FieldType.MAP_STRING_INT64: pa.map_(pa.string(), pa.int64()),
    FieldType.MAP_STRING_FLOAT: pa.map_(pa.string(), pa.float32()),
    FieldType.MAP_STRING_DOUBLE: pa.map_(pa.string(), pa.float64()),
    FieldType.MAP_STRING_STRING: pa.map_(pa.string(), pa.string()),
    FieldType.MAP_INT64_INT32: pa.map_(pa.int64(), pa.int32()),
    FieldType.MAP_INT64_INT64: pa.map_(pa.int64(), pa.int64()),
    FieldType.MAP_INT64_FLOAT: pa.map_(pa.int64(), pa.float32()),
    FieldType.MAP_INT64_DOUBLE: pa.map_(pa.int64(), pa.float64()),
    FieldType.MAP_INT64_STRING: pa.map_(pa.int64(), pa.string()),
    FieldType.MAP_INT32_INT32: pa.map_(pa.int32(), pa.int32()),
    FieldType.MAP_INT32_INT64: pa.map_(pa.int32(), pa.int64()),
    FieldType.MAP_INT32_FLOAT: pa.map_(pa.int32(), pa.float32()),
    FieldType.MAP_INT32_DOUBLE: pa.map_(pa.int32(), pa.float64()),
    FieldType.MAP_INT32_STRING: pa.map_(pa.int32(), pa.string()),
}

# Type name mapping from ODPS-style type str to FieldType enum
# Note: Aliases INT/INT32 and BIGINT/INT64 are handled by normalizing the type string
TYPE_STR_TO_FIELD_TYPE = {
    # Basic types (use canonical names INT32/INT64)
    "INT32": FieldType.INT32,
    "INT64": FieldType.INT64,
    "STRING": FieldType.STRING,
    "FLOAT": FieldType.FLOAT,
    "DOUBLE": FieldType.DOUBLE,
    # Array types (use canonical INT32/INT64 inside)
    "ARRAY<INT32>": FieldType.ARRAY_INT32,
    "ARRAY<INT64>": FieldType.ARRAY_INT64,
    "ARRAY<STRING>": FieldType.ARRAY_STRING,
    "ARRAY<FLOAT>": FieldType.ARRAY_FLOAT,
    "ARRAY<DOUBLE>": FieldType.ARRAY_DOUBLE,
    # Nested array types
    "ARRAY<ARRAY<INT32>>": FieldType.ARRAY_ARRAY_INT32,
    "ARRAY<ARRAY<INT64>>": FieldType.ARRAY_ARRAY_INT64,
    "ARRAY<ARRAY<STRING>>": FieldType.ARRAY_ARRAY_STRING,
    "ARRAY<ARRAY<FLOAT>>": FieldType.ARRAY_ARRAY_FLOAT,
    "ARRAY<ARRAY<DOUBLE>>": FieldType.ARRAY_ARRAY_DOUBLE,
    # Map types (use canonical INT32/INT64 inside)
    "MAP<STRING,INT32>": FieldType.MAP_STRING_INT32,
    "MAP<STRING,INT64>": FieldType.MAP_STRING_INT64,
    "MAP<STRING,STRING>": FieldType.MAP_STRING_STRING,
    "MAP<STRING,FLOAT>": FieldType.MAP_STRING_FLOAT,
    "MAP<STRING,DOUBLE>": FieldType.MAP_STRING_DOUBLE,
    "MAP<INT64,INT32>": FieldType.MAP_INT64_INT32,
    "MAP<INT64,INT64>": FieldType.MAP_INT64_INT64,
    "MAP<INT64,STRING>": FieldType.MAP_INT64_STRING,
    "MAP<INT64,FLOAT>": FieldType.MAP_INT64_FLOAT,
    "MAP<INT64,DOUBLE>": FieldType.MAP_INT64_DOUBLE,
    "MAP<INT32,INT32>": FieldType.MAP_INT32_INT32,
    "MAP<INT32,INT64>": FieldType.MAP_INT32_INT64,
    "MAP<INT32,STRING>": FieldType.MAP_INT32_STRING,
    "MAP<INT32,FLOAT>": FieldType.MAP_INT32_FLOAT,
    "MAP<INT32,DOUBLE>": FieldType.MAP_INT32_DOUBLE,
}


def _normalize_type_str(type_str: str) -> str:
    """Normalize type string.

    1. Converting to uppercase
    2. Removing spaces
    3. Replacing ODPS aliases: BIGINT->INT64, INT->INT32
       (handles both BIGINT/INT64 and INT/INT32 as valid inputs)

    Args:
        type_str: type string to normalize

    Returns:
        normalized type string
    """
    normalized = type_str.upper().strip()
    normalized = re.sub(r"\s+", "", normalized)
    # Use word boundaries to match whole words only
    normalized = re.sub(r"\bBIGINT\b", "INT64", normalized)
    normalized = re.sub(r"\bINT\b", "INT32", normalized)
    return normalized


def get_input_fields_proto(
    data_config: data_pb2.DataConfig,
) -> List[data_pb2.Field]:
    """Get input fields from data_config.input_fields_str or data_config.input_fields.

    If input_fields_str is specified, parse it and return the fields.
    Otherwise, return data_config.input_fields directly.

    Args:
        data_config: DataConfig proto message

    Returns:
        List of Field proto messages
    """
    if data_config.HasField("input_fields_str") and data_config.input_fields_str:
        input_fields_str = data_config.input_fields_str.strip()
        if not input_fields_str:
            return []

        fields = []
        # Split by semicolon, filter out empty parts
        field_parts = [p.strip() for p in input_fields_str.split(";") if p.strip()]
        for field_part in field_parts:
            if ":" not in field_part:
                raise ValueError(
                    f"Invalid input_fields_str format: '{field_part}'. "
                    "Expected format: 'field_name:field_type'"
                )
            name, type_str = field_part.split(":", 1)
            name = name.strip()
            type_str = type_str.strip()

            if not name:
                raise ValueError(
                    f"Empty field name in input_fields_str: '{field_part}'"
                )
            if not type_str:
                raise ValueError(
                    f"Empty field type in input_fields_str: '{field_part}'"
                )

            # Normalize the type string
            normalized_type = _normalize_type_str(type_str)

            if normalized_type not in TYPE_STR_TO_FIELD_TYPE:
                raise ValueError(
                    f"Unknown field type '{type_str}' "
                    f"(normalized: '{normalized_type}') for field '{name}'. "
                    f"Supported types: {list(TYPE_STR_TO_FIELD_TYPE.keys())}"
                )

            field_proto = data_pb2.Field()
            field_proto.input_name = name
            field_proto.input_type = TYPE_STR_TO_FIELD_TYPE[normalized_type]
            fields.append(field_proto)

        return fields
    else:
        # Return the existing input_fields
        return list(data_config.input_fields)


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
    key_lengths: npt.NDArray
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

    # key of dense_features is data group name
    dense_features: Dict[str, KeyedTensor] = field(default_factory=dict)
    # key of sparse_features is data group name
    sparse_features: Dict[str, KeyedJaggedTensor] = field(default_factory=dict)
    # key of sequence_mulval_lengths is data group name
    #
    # for multi-value sequence, we flatten it, then store values & accumate lengths
    # into sparse_features, store key_lengths & seq_lengths into sequence_mulval_lengths
    #
    # e.g.
    # for the sequence `click_seq`: [[[3, 4], [5]], [6, [7, 8]]]
    # we can denote it in jagged formular with:
    #   values: [3, 4, 5, 6, 7, 8]
    #   key_lengths: [2, 1, 1, 2]
    #   seq_lengths: [2, 2]
    # then:
    #   sparse_features[dg]['click_seq'].values() = [3, 4, 5, 6, 7, 8]  # values
    #   sparse_features[dg]['click_seq'].lengths() = [3, 3]  # accumate lengths
    #   sequence_mulval_lengths[dg]['click_seq'].values() = [2, 1, 1, 2]  # key_lengths
    #   sequence_mulval_lengths[dg]['click_seq'].lengths() = [2, 2]  # seq_lengths
    sequence_mulval_lengths: Dict[str, KeyedJaggedTensor] = field(default_factory=dict)
    # key of sequence_dense_features is feature name
    sequence_dense_features: Dict[str, JaggedTensor] = field(default_factory=dict)
    # key of labels is label name
    labels: Dict[str, torch.Tensor] = field(default_factory=dict)
    # key of jagged_labels is label name
    jagged_labels: Dict[str, JaggedTensor] = field(default_factory=dict)
    # reserved inputs [for predict]
    reserves: RecordBatchTensor = field(default_factory=RecordBatchTensor)
    # size for user side input tile when do inference and INPUT_TILE=2 or 3
    tile_size: int = field(default=-1)
    # sample_weight
    sample_weights: Dict[str, torch.Tensor] = field(default_factory=dict)

    additional_infos: Dict[str, torch.Tensor] = field(default_factory=dict)
    # dummy batch or not
    dummy: bool = field(default=False)
    # checkpoint info: {source_key: max_abs_row}
    checkpoint_info: Optional[Dict[str, int]] = field(default=None)
    # max event-time (Unix-epoch seconds) in this batch, -1.0 when unavailable
    data_timestamp: float = field(default=-1.0)

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
            sequence_mulval_lengths={
                k: v.to(device=device, non_blocking=non_blocking)
                for k, v in self.sequence_mulval_lengths.items()
            },
            sequence_dense_features={
                k: v.to(device=device, non_blocking=non_blocking)
                for k, v in self.sequence_dense_features.items()
            },
            labels={
                k: v.to(device=device, non_blocking=non_blocking)
                for k, v in self.labels.items()
            },
            jagged_labels={
                k: v.to(device=device, non_blocking=non_blocking)
                for k, v in self.jagged_labels.items()
            },
            reserves=self.reserves,
            tile_size=self.tile_size,
            sample_weights={
                k: v.to(device=device, non_blocking=non_blocking)
                for k, v in self.sample_weights.items()
            },
            additional_infos={
                k: v.to(device=device, non_blocking=non_blocking)
                for k, v in self.additional_infos.items()
            },
            dummy=self.dummy,
            checkpoint_info=self.checkpoint_info,
            data_timestamp=self.data_timestamp,
        )

    def record_stream(self, stream: torch.Stream) -> None:
        """Record which streams have used the tensor."""
        for v in self.dense_features.values():
            # pyre-ignore [6]
            v.record_stream(stream)
        for v in self.sparse_features.values():
            # pyre-ignore [6]
            v.record_stream(stream)
        for v in self.sequence_mulval_lengths.values():
            # pyre-ignore [6]
            v.record_stream(stream)
        for v in self.sequence_dense_features.values():
            # pyre-ignore [6]
            v.record_stream(stream)
        for v in self.labels.values():
            v.record_stream(stream)
        for v in self.jagged_labels.values():
            # pyre-ignore [6]
            v.record_stream(stream)
        for v in self.sample_weights.values():
            v.record_stream(stream)
        for v in self.additional_infos.values():
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
        jagged_labels = {}
        for k, v in self.jagged_labels.items():
            weights = v._weights
            lengths = v._lengths
            offsets = v._offsets
            jagged_labels[k] = JaggedTensor(
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
            sequence_mulval_lengths={
                k: v.pin_memory() for k, v in self.sequence_mulval_lengths.items()
            },
            sequence_dense_features=sequence_dense_features,
            labels={k: v.pin_memory() for k, v in self.labels.items()},
            jagged_labels=jagged_labels,
            reserves=self.reserves,
            tile_size=self.tile_size,
            sample_weights={k: v.pin_memory() for k, v in self.sample_weights.items()},
            additional_infos={
                k: v.pin_memory() for k, v in self.additional_infos.items()
            },
            dummy=self.dummy,
            checkpoint_info=self.checkpoint_info,
            data_timestamp=self.data_timestamp,
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
        for x in self.sequence_mulval_lengths.values():
            if sparse_dtype:
                x = KeyedJaggedTensor(
                    keys=x.keys(),
                    values=x.values().to(sparse_dtype),
                    lengths=x.lengths().to(sparse_dtype),
                )
            for k, v in x.to_dict().items():
                tensor_dict[f"{k}.key_lengths"] = v.values()
                tensor_dict[f"{k}.lengths"] = v.lengths()
        for k, v in self.sequence_dense_features.items():
            tensor_dict[f"{k}.values"] = v.values()
            tensor_dict[f"{k}.lengths"] = v.lengths()
        for k, v in self.labels.items():
            tensor_dict[f"{k}"] = v
        for k, v in self.jagged_labels.items():
            tensor_dict[f"{k}.values"] = v.values()
            tensor_dict[f"{k}.lengths"] = v.lengths()
        for k, v in self.sample_weights.items():
            tensor_dict[f"{k}"] = v
        if self.tile_size > 0:
            tensor_dict["batch_size"] = torch.tensor(self.tile_size, dtype=torch.int64)

        for k, v in self.additional_infos.items():
            tensor_dict[f"{k}"] = v

        return tensor_dict


def expand_tdm_sample(
    input_data: Dict[str, pa.Array],
    pos_sampled: Dict[str, pa.Array],
    neg_sampled: Dict[str, pa.Array],
    data_config: data_pb2.DataConfig,
) -> Dict[str, pa.Array]:
    """Expand input data with sampled data for TDM.

    Combine the sampled positive and negative samples with the item
    features, then expand the user features based on the original
    user-item relationships, and supplement the corresponding labels
    according to the positive and negative samples. The sampled
    outcomes for each item are contiguous in the sampler output.

    Example::

        user_fea: [1, 2], item_fea: [0.1, 0.2], labels: [1, 1],
        pos_sample: [0.11, 0.12, 0.21, 0.22],
        neg_sample: [-0.11, -0.12, -0.21, -0.22]

        concat item_fea:
            [0.1, 0.2, 0.11, 0.12, 0.21, 0.22, -0.11, -0.12, -0.21, -0.22]
        duplicate user_fea preserving user-item relationship:
            [1, 2, 1, 1, 2, 2, 1, 1, 2, 2]
        expand label:
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]

    Mutates ``input_data`` in place and returns it.
    """
    item_fea_names = pos_sampled.keys()
    all_fea_names = input_data.keys()
    label_fields = set(data_config.label_fields)
    user_fea_names = all_fea_names - item_fea_names - label_fields

    for item_fea_name in item_fea_names:
        input_data[item_fea_name] = pa.concat_arrays(
            [
                input_data[item_fea_name],
                pos_sampled[item_fea_name],
                neg_sampled[item_fea_name],
            ]
        )

    # In the sampling results, the sampled outcomes for each item are contiguous.
    batch_size = len(input_data[list(label_fields)[0]])
    num_pos_sampled = len(pos_sampled[list(item_fea_names)[0]])
    num_neg_sampled = len(neg_sampled[list(item_fea_names)[0]])
    user_pos_index = np.repeat(np.arange(batch_size), num_pos_sampled // batch_size)
    user_neg_index = np.repeat(np.arange(batch_size), num_neg_sampled // batch_size)
    for user_fea_name in user_fea_names:
        user_fea = input_data[user_fea_name]
        pos_expand_user_fea = user_fea.take(user_pos_index)
        neg_expand_user_fea = user_fea.take(user_neg_index)
        input_data[user_fea_name] = pa.concat_arrays(
            [
                input_data[user_fea_name],
                pos_expand_user_fea,
                neg_expand_user_fea,
            ]
        )

    for label_field in label_fields:
        input_data[label_field] = pa.concat_arrays(
            [
                input_data[label_field].cast(pa.int64()),
                pa.array([1] * num_pos_sampled, type=pa.int64()),
                pa.array([0] * num_neg_sampled, type=pa.int64()),
            ]
        )

    return input_data


def build_sampler_input(
    input_data: Dict[str, pa.Array],
    item_id_field: Optional[str],
    user_id_field: Optional[str],
    seq_delim: str,
) -> Dict[str, pa.Array]:
    """Shallow-copy input_data with item_id (and user_id) flattened for the sampler.

    When `item_id_field` is a grouped sequence sub-feature, per-row
    positives (delimited string or list array) are flattened to 1D and
    `user_id_field` (if any) is expanded by per-row positive count.
    Scalar item_id (`seq_delim=""`) falls through unchanged. The
    caller's `input_data` is not mutated.

    Args:
        input_data: per-row input column dict.
        item_id_field: sampler config's `item_id_field`, or None.
        user_id_field: sampler config's `user_id_field`, or None.
        seq_delim: candidate sequence's `sequence_delim`, or "" when
            `item_id_field` is a top-level scalar feature.

    Returns:
        A new shallow-copy dict with item_id flattened and user_id
        expanded when both apply.
    """
    sampler_input = dict(input_data)
    if item_id_field is None or not seq_delim:
        return sampler_input

    raw = input_data[item_id_field]
    if pa.types.is_string(raw.type) or pa.types.is_large_string(raw.type):
        pos_lists = pc.split_pattern(raw, seq_delim)
    elif pa.types.is_list(raw.type) or pa.types.is_large_list(raw.type):
        pos_lists = raw
    else:
        # Scalar (e.g. int64 single-positive for DSSM): pass through.
        return sampler_input

    sampler_input[item_id_field] = pc.list_flatten(pos_lists)

    if user_id_field is not None and user_id_field in input_data:
        counts = pc.list_value_length(pos_lists).to_numpy()
        row_indices = pa.array(np.repeat(np.arange(len(counts)), counts))
        sampler_input[user_id_field] = pc.take(input_data[user_id_field], row_indices)
    return sampler_input


def combine_negs_to_candidate_sequence(
    pos_data: pa.Array,
    negs: pa.Array,
    seq_delim: str,
) -> Tuple[pa.Array, np.ndarray]:
    """Append `negs` to row B-1 of `pos_data`; output type matches input type.

    Per-row layout (block-(B-1)-suffix):
        row i < B-1:  pos_i values only
        row B-1:      pos_{B-1} values + appended negs

    Output type matches input type: `list<T>` / `large_list<T>` pos
    produces the same list type out (Arrow-native via
    `ListArray.from_arrays`); delimited string pos produces string out.

    Simple-vs-hard neg distinction is invisible in the output layout;
    callers use `HARD_NEG_INDICES` from the sampler to attribute hard
    negs to queries downstream.

    String-path input convention: canonical delimiter-separated rows
    (non-null, non-empty). `count_substring + 1` is structurally
    identical to `len(split_pattern)` for those inputs.

    Args:
        pos_data: per-row positives, either delimited string or list array.
        negs: sampled negatives; flat array or `list<T>` of 1-element
            lists (flattened internally before appending).
        seq_delim: delimiter for the string path; ignored for the list
            path.

    Returns:
        (combined: pa.Array (same type as pos_data, length B),
         pos_lengths: int32 np.ndarray (length B), per-row positive
         count only -- not including the appended negs.)
    """
    # Sampler wraps each scalar in a 1-element list for list-typed attrs.
    if pa.types.is_list(negs.type) or pa.types.is_large_list(negs.type):
        negs = pc.list_flatten(negs)
    n_negs = len(negs)

    if pa.types.is_list(pos_data.type) or pa.types.is_large_list(pos_data.type):
        # list path -- Arrow-native.
        pos_lengths = pc.list_value_length(pos_data).to_numpy().astype(np.int32)
        if len(pos_data) == 0:
            if n_negs > 0:
                logger.warning(
                    "combine_negs_to_candidate_sequence: empty pos_data with "
                    "%d negs; negs dropped.",
                    n_negs,
                )
            return pos_data, pos_lengths
        if n_negs == 0:
            return pos_data, pos_lengths
        inner_type = pos_data.type.value_type
        if negs.type != inner_type:
            negs = negs.cast(inner_type, safe=False)
        new_values = pa.concat_arrays([pc.list_flatten(pos_data), negs])
        new_lengths = pos_lengths.copy()
        new_lengths[-1] += n_negs
        is_large = pa.types.is_large_list(pos_data.type)
        offset_dtype = np.int64 if is_large else np.int32
        new_offsets = pa.array(
            np.concatenate([[0], new_lengths.cumsum()]).astype(offset_dtype),
            type=pa.int64() if is_large else pa.int32(),
        )
        list_cls = pa.LargeListArray if is_large else pa.ListArray
        return list_cls.from_arrays(new_offsets, new_values), pos_lengths
    else:
        # string path -- count_substring for pos_lengths, modify last row only.
        pos_str = pos_data.cast(pa.string())
        pos_lengths = (
            pc.add(pc.count_substring(pos_str, pattern=seq_delim), 1)
            .to_numpy()
            .astype(np.int32)
        )
        rows = pos_str.to_pylist()
        if len(rows) == 0:
            if n_negs > 0:
                logger.warning(
                    "combine_negs_to_candidate_sequence: empty pos_data with "
                    "%d negs; negs dropped.",
                    n_negs,
                )
            return pa.array(rows, type=pa.string()), pos_lengths
        if n_negs == 0:
            return pa.array(rows, type=pa.string()), pos_lengths
        negs_str = seq_delim.join(negs.cast(pa.string()).to_pylist())
        rows[-1] = rows[-1] + seq_delim + negs_str
        return pa.array(rows, type=pa.string()), pos_lengths


def calc_slice_position(
    row_count: int,
    slice_id: int,
    slice_count: int,
    batch_size: int,
    drop_redundant_bs_eq_one: bool,
    pre_total_remain: int = 0,
) -> Tuple[int, int, int]:
    """Calc table read position according to the slice information.

    Args:
        row_count (int): table total row count.
        slice_id (int): worker id.
        slice_count (int): total worker number.
        batch_size (int): batch_size.
        drop_redundant_bs_eq_one (bool): drop last redundant batch with batch_size
            equal one to prevent train_eval hung.
        pre_total_remain (int): remaining total count in pre-table is
            insufficient to meet the batch_size requirement for each worker.

    Return:
        start (int): start row position in table.
        end (int): start row position in table.
        total_remain (int): remaining total count in curr-table is
            insufficient to meet the batch_size requirement for each worker.
    """
    pre_remain_size = int(pre_total_remain / slice_count)
    pre_remain_split_point = pre_total_remain % slice_count

    size = int((row_count + pre_total_remain) / slice_count)
    split_point = (row_count + pre_total_remain) % slice_count
    if slice_id < split_point:
        start = slice_id * (size + 1)
        end = start + (size + 1)
    else:
        start = split_point * (size + 1) + (slice_id - split_point) * size
        end = start + size

    real_start = (
        start - pre_remain_size * slice_id - min(pre_remain_split_point, slice_id)
    )
    real_end = (
        end
        - pre_remain_size * (slice_id + 1)
        - min(pre_remain_split_point, slice_id + 1)
    )
    # when (end - start) % bz = 1 on some workers and
    # (end - start) % bz = 0 on other workers, train_eval will hang
    if (
        drop_redundant_bs_eq_one
        and split_point != 0
        and (end - start) % batch_size == 1
        and size % batch_size == 0
    ):
        real_end = real_end - 1
        split_point = 0
    return real_start, real_end, (size % batch_size) * slice_count + split_point


def calc_remaining_intervals(
    checkpoint_state: Optional[Dict[str, int]],
    input_path: str,
    total_rows: int,
) -> List[Tuple[int, int]]:
    """Calculate remaining intervals from checkpoint state.

    The checkpoint key format is `{input_path}:{start}` where `start` is the
    beginning of the worker's range. From sorted starts + total_rows, we can
    infer the original ranges and calculate remaining intervals.

    Args:
        checkpoint_state (dict): dict mapping source_id to max consumed row index.
        input_path (str): the input path to filter checkpoint entries.
        total_rows (int): total number of rows in the dataset.

    Returns:
        List of (start, end) tuples representing remaining intervals.
    """
    if not checkpoint_state:
        return [(0, total_rows)]  # No checkpoint, all data remaining

    # Parse checkpoint keys: "{input_path}:{start}" -> (start, consumed)
    # Filter by input_path matching
    entries = []  # [(start, consumed), ...]
    for key, consumed in checkpoint_state.items():
        last_colon = key.rfind(":")
        if last_colon == -1:
            continue
        key_input_path = key[:last_colon]
        # Match input_path (exact match)
        if key_input_path == input_path:
            start = int(key[last_colon + 1 :])
            entries.append((start, consumed))

    if not entries:
        return [(0, total_rows)]  # No matching checkpoint

    # Sort by start to infer original ranges
    entries.sort(key=lambda x: x[0])

    # Calculate remaining intervals
    remaining = []
    num_entries = len(entries)
    for i, (_, consumed) in enumerate(entries):
        # Infer the end of this worker's range
        if i + 1 < num_entries:
            range_end = entries[i + 1][0]  # Next worker's start
        else:
            range_end = total_rows  # Last worker goes to end

        # Remaining interval is [consumed+1, range_end)
        if consumed + 1 < range_end:
            remaining.append((consumed + 1, range_end))

    return remaining if remaining else []


def calc_slice_intervals(
    total_rows: int,
    worker_id: int,
    num_workers: int,
    batch_size: int = 1,
    drop_redundant_bs_eq_one: bool = False,
    pre_total_remain: int = 0,
    checkpoint_state: Optional[Dict[str, int]] = None,
    input_path: Optional[str] = None,
) -> List[Tuple[int, int]]:
    """Redistribute remaining intervals among workers.

    Flattens all intervals into a total row count, then assigns a portion
    to each worker based on worker_id and num_workers.

    Args:
        total_rows (int): total number of rows in the dataset.
        worker_id: Current worker's ID (0-indexed).
        num_workers: Total number of workers.
        batch_size: batch_size.
        drop_redundant_bs_eq_one: drop last redundant batch with batch_size
            equal one to prevent train_eval hung.
        pre_total_remain (int): remaining total count in pre-table is
            insufficient to meet the batch_size requirement for each worker.
        checkpoint_state (dict): dict mapping source_id to max consumed row index.
        input_path (str): the input path to filter checkpoint entries.

    Returns:
        worker_intervals (list): List of (start, end) tuples assigned to this worker.
        total_remain (int): remaining total count in curr-table is
            insufficient to meet the batch_size requirement for each worker.
    """
    if checkpoint_state:
        intervals = calc_remaining_intervals(checkpoint_state, input_path, total_rows)
        total_rows = sum(end - start for start, end in intervals)

    # Reuse calc_slice_position for worker start/end calculation
    worker_start, worker_end, total_remain = calc_slice_position(
        row_count=total_rows,
        slice_id=worker_id,
        slice_count=num_workers,
        batch_size=batch_size,
        drop_redundant_bs_eq_one=drop_redundant_bs_eq_one,
        pre_total_remain=pre_total_remain,
    )

    if checkpoint_state:
        # Map worker's logical range [worker_start, worker_end) to actual intervals
        result = []
        current_pos = 0
        for interval_start, interval_end in intervals:
            interval_len = interval_end - interval_start
            interval_logical_start = current_pos
            interval_logical_end = current_pos + interval_len

            # Check if this interval overlaps with worker's range
            overlap_start = max(worker_start, interval_logical_start)
            overlap_end = min(worker_end, interval_logical_end)

            if overlap_start < overlap_end:
                # Map back to actual row indices
                actual_start = interval_start + (overlap_start - interval_logical_start)
                actual_end = interval_start + (overlap_end - interval_logical_start)
                result.append((actual_start, actual_end))

            current_pos = interval_logical_end
            if current_pos >= worker_end:
                break
    else:
        result = [(worker_start, worker_end)]

    return result, total_remain


def remove_nullable(field_type: pa.DataType) -> pa.DataType:
    """Recursive removal of the null=False property from lists and nested lists."""
    if pa.types.is_list(field_type):
        # Get element fields
        value_field = field_type.value_field
        # Change the nullable to True
        normalized_value_field = value_field.with_nullable(True)
        # Recursive processing of element types
        normalized_value_type = remove_nullable(normalized_value_field.type)
        # Construct a new list type
        return pa.list_(normalized_value_type)

    else:
        return field_type
