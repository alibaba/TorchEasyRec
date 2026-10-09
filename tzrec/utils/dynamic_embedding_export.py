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

"""Stream dynamic embedding checkpoint arrays without restoring the tables."""

import glob
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import (
    Any,
    BinaryIO,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
)

import numpy as np

from tzrec.utils import checkpoint_util, quant_util
from tzrec.utils.npz_util import StreamingArray


@dataclass(frozen=True)
class _CheckpointShard:
    """Paths and row count of a validated checkpoint shard."""

    rank: int
    world_size: int
    key_file: str
    value_file: str
    score_file: str
    rows: int


def _read_array(
    stream: BinaryIO, dtype: np.dtype, shape: Tuple[int, ...]
) -> np.ndarray:
    """Read one bounded array, including from streams that return short reads."""
    array = np.empty(shape, dtype=dtype)
    buffer = memoryview(array).cast("B")
    offset = 0
    while offset < len(buffer):
        size = stream.readinto(buffer[offset:])
        if not size:
            raise ValueError(
                f"Unexpected EOF in dynamic embedding file {stream.name}: "
                f"expected {len(buffer)} bytes, read {offset}"
            )
        offset += size
    return array


def _iter_column(
    shards: List[_CheckpointShard], field: str, chunk_rows: int
) -> Iterator[np.ndarray]:
    """Read keys or scores in checkpoint rank order."""
    for shard in shards:
        path = getattr(shard, field)
        with open(path, "rb") as stream:
            for start in range(0, shard.rows, chunk_rows):
                yield _read_array(
                    stream, np.dtype(np.int64), (min(chunk_rows, shard.rows - start),)
                )
            if stream.read(1):
                raise ValueError(f"Dynamic embedding file grew during export: {path}")


def _iter_values(
    shards: List[_CheckpointShard],
    embedding_name: str,
    embedding_dim: int,
    chunk_rows: int,
    quant_format: str,
    on_values: Optional[Callable[[str, np.ndarray, np.ndarray], None]],
) -> Iterator[np.ndarray]:
    """Read aligned key/value blocks and pass exported values to both sinks."""
    for shard in shards:
        with (
            open(shard.key_file, "rb") as key_stream,
            open(shard.value_file, "rb") as value_stream,
        ):
            for start in range(0, shard.rows, chunk_rows):
                rows = min(chunk_rows, shard.rows - start)
                keys = _read_array(key_stream, np.dtype(np.int64), (rows,))
                values = _read_array(
                    value_stream, np.dtype(np.float32), (rows, embedding_dim)
                )
                if quant_format:
                    values = quant_util.distributed_quantize_embeddings(
                        values, embedding_dim, embedding_name, quant_format
                    )
                if on_values is not None:
                    on_values(embedding_name, keys, values)
                yield values
                del keys, values
            for stream in (key_stream, value_stream):
                if stream.read(1):
                    raise ValueError(
                        f"Dynamic embedding file grew during export: {stream.name}"
                    )


def _find_shards(
    checkpoint_path: str, embedding_dimensions: Mapping[str, int]
) -> Dict[str, List[_CheckpointShard]]:
    """Resolve table names and validate checkpoint sizes before reading data."""
    dynamicemb_path = os.path.join(checkpoint_path, "dynamicemb")
    key_pattern = re.compile(
        r"^(?P<name>.+)_emb_keys\.rank_(?P<rank>\d+)\.world_size_(?P<world>\d+)$"
    )
    seen_paths = set()
    shards_by_table = defaultdict(list)
    for key_file in sorted(
        glob.glob(os.path.join(dynamicemb_path, "*/*_emb_keys.rank_*.world_size_*"))
    ):
        real_path = os.path.realpath(key_file)
        if real_path in seen_paths:
            continue
        seen_paths.add(real_path)
        match = key_pattern.match(os.path.basename(key_file))
        if match is None:
            continue
        module_fqn = os.path.relpath(
            os.path.dirname(key_file), dynamicemb_path
        ).replace(os.path.sep, ".")
        if module_fqn.startswith("model.model."):
            module_fqn = module_fqn[len("model.") :]
        table_name = match.group("name")
        candidates = [
            checkpoint_util.remap_input_tile_user_key(
                f"{module_fqn}.{collection}.{table_name}"
            )
            for collection in ("embedding_bags", "embeddings")
        ]
        embedding_name = next(
            (name for name in candidates if name in embedding_dimensions), None
        )
        if embedding_name is None:
            continue
        embedding_dim = embedding_dimensions[embedding_name]
        if embedding_dim <= 0:
            raise ValueError(f"Invalid embedding dimension for {embedding_name}")
        checkpoint_rank = int(match.group("rank"))
        checkpoint_world_size = int(match.group("world"))
        if not 0 <= checkpoint_rank < checkpoint_world_size:
            raise ValueError(f"Invalid checkpoint rank or world_size in {key_file}")
        suffix = f".rank_{checkpoint_rank}.world_size_{checkpoint_world_size}"
        value_file = os.path.join(
            os.path.dirname(key_file), f"{table_name}_emb_values{suffix}"
        )
        score_file = os.path.join(
            os.path.dirname(key_file), f"{table_name}_emb_scores{suffix}"
        )
        key_size = os.path.getsize(key_file)
        value_size = os.path.getsize(value_file)
        score_size = os.path.getsize(score_file)
        if key_size % 8:
            raise ValueError(f"Invalid int64 key file size: {key_file}: {key_size}")
        rows = key_size // 8
        if score_size != key_size:
            raise ValueError(
                f"Dynamic embedding {embedding_name} key/score row mismatch: "
                f"key_bytes={key_size}, score_bytes={score_size}"
            )
        if value_size != rows * embedding_dim * 4:
            raise ValueError(
                f"Dynamic embedding {embedding_name} value row mismatch: "
                f"keys={rows}, value_bytes={value_size}, embedding_dim={embedding_dim}"
            )
        shards_by_table[embedding_name].append(
            _CheckpointShard(
                checkpoint_rank,
                checkpoint_world_size,
                key_file,
                value_file,
                score_file,
                rows,
            )
        )
    for name, shards in shards_by_table.items():
        world_sizes = {shard.world_size for shard in shards}
        if len(world_sizes) != 1:
            raise ValueError(
                f"Dynamic embedding {name} has inconsistent checkpoint "
                f"world_size values: {sorted(world_sizes)}"
            )
        checkpoint_world_size = shards[0].world_size
        ranks = {shard.rank for shard in shards}
        if len(shards) != checkpoint_world_size or len(ranks) != len(shards):
            raise ValueError(
                f"Dynamic embedding {name} has incomplete or duplicate checkpoint "
                f"shard ranks: expected world_size={checkpoint_world_size}, "
                f"found ranks={sorted(shard.rank for shard in shards)}"
            )
        shards.sort(key=lambda shard: shard.rank)
    return dict(shards_by_table)


def get_dynamic_embedding_export(
    checkpoint_path: str,
    embedding_dimensions: Mapping[str, int],
    read_chunk_bytes: int,
    quant_format: str = "",
    on_values: Optional[Callable[[str, np.ndarray, np.ndarray], None]] = None,
    rank: int = 0,
    world_size: int = 1,
) -> Tuple[Dict[str, StreamingArray], Dict[str, Dict[str, Any]]]:
    """Describe dynamic checkpoint arrays and defer their bounded payload reads.

    Keys, values and scores use separate passes to match NPZ entry ordering.
    The values pass rereads aligned keys for the optional upload callback.
    Optimizer and admission counter files are not embedding array inputs.

    Args:
        checkpoint_path: Training checkpoint directory.
        embedding_dimensions: Normalized export table FQN to logical dimension.
        read_chunk_bytes: Maximum combined raw key/value bytes per block.
        quant_format: Distributed quantization format, or empty for FP32.
        on_values: Synchronous callback receiving the exported key/value block.
        rank: Export process rank.
        world_size: Number of export processes.

    Returns:
        NPZ array descriptors and per-table metadata, without feature mappings.
        Descriptors hold single-use iterators and must be consumed exactly once.
    """
    if read_chunk_bytes <= 0:
        raise ValueError("read_chunk_bytes must be positive")
    if not 0 <= rank < world_size:
        raise ValueError("Export rank must be in [0, world_size)")
    arrays: Dict[str, StreamingArray] = {}
    metadata: Dict[str, Dict[str, Any]] = {}
    for name, all_shards in _find_shards(checkpoint_path, embedding_dimensions).items():
        shards = [shard for shard in all_shards if shard.rank % world_size == rank]
        if not shards:
            continue
        dimension = embedding_dimensions[name]
        chunk_rows = read_chunk_bytes // (8 + dimension * 4)
        if chunk_rows == 0:
            raise ValueError(
                f"read_chunk_bytes={read_chunk_bytes} cannot fit one key/value "
                f"row of {name} (requires {8 + dimension * 4} bytes)"
            )
        rows = sum(shard.rows for shard in shards)
        shape = (rows, dimension)
        dtype = np.dtype(np.float32)
        table_meta: Dict[str, Any] = {
            "dimension": dimension,
            "dtype": "float32",
            "shape": list(shape),
            "dense": False,
            "is_dynamic": True,
            "key_dtype": "int64",
            "score_dtype": "int64",
            "key_name": f"{name}.keys",
            "value_name": f"{name}.values",
            "score_name": f"{name}.scores",
        }
        if quant_format:
            empty_values = quant_util.distributed_quantize_embeddings(
                np.empty((0, dimension), dtype=np.float32),
                dimension,
                name,
                quant_format,
            )
            dtype = empty_values.dtype
            shape = (rows, empty_values.shape[1])
            table_meta.update(
                {
                    "dtype": quant_format,
                    "storage_dtype": quant_util._DISTRIBUTED_SPARSE_QUANT_STORAGE_DTYPE,
                    "storage_shape": list(shape),
                    "row_bytes": shape[1],
                    "quant": {
                        "enabled": True,
                        "format": quant_format,
                        "scale_offset_dtype": "float16",
                        "output_dtype": "float16",
                    },
                }
            )
        table_meta["memory"] = rows * shape[1] * dtype.itemsize
        arrays[f"{name}.keys"] = StreamingArray(
            np.dtype(np.int64),
            (rows,),
            _iter_column(shards, "key_file", read_chunk_bytes // 8),
        )
        arrays[f"{name}.values"] = StreamingArray(
            dtype,
            shape,
            _iter_values(shards, name, dimension, chunk_rows, quant_format, on_values),
        )
        arrays[f"{name}.scores"] = StreamingArray(
            np.dtype(np.int64),
            (rows,),
            _iter_column(shards, "score_file", read_chunk_bytes // 8),
        )
        metadata[name] = table_meta
    return arrays, metadata
