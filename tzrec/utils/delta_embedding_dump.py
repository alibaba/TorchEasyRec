# Copyright (c) 2025, Alibaba Group;
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
import re
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Set, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import nn
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torchrec.distributed.embedding import ShardedEmbeddingCollection
from torchrec.distributed.embeddingbag import ShardedEmbeddingBagCollection
from torchrec.distributed.model_tracker.delta_store import DeltaStoreTrec
from torchrec.distributed.model_tracker.model_delta_tracker import (
    ModelDeltaTracker as TorchRecModelDeltaTracker,
)
from torchrec.distributed.model_tracker.types import UniqueRows, UpdateMode
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

from tzrec.protos.train_pb2 import DeltaEmbeddingDumpConfig
from tzrec.utils.logging_util import logger

_CONSUMER = "delta_embedding_dump"
_DELTA_DUMP_SCHEMA = pa.schema(
    [
        ("global_step", pa.int64()),
        ("rank", pa.int32()),
        ("world_size", pa.int32()),
        ("feature_name", pa.string()),
        ("table_fqn", pa.string()),
        ("key_id", pa.int64()),
        ("embedding", pa.list_(pa.float32())),
        ("source", pa.string()),
    ]
)


@dataclass(frozen=True)
class _TableShardInfo:
    row_offset: int = 0
    column_offset: int = 0
    local_rows: int = 0
    local_cols: int = 0
    global_rows: int = 0
    global_cols: int = 0
    has_shard_metadata: bool = False


@dataclass(frozen=True)
class _TableWeight:
    tensor: torch.Tensor
    shard_info: _TableShardInfo


def _distributed_rank_world_size() -> Tuple[int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    return rank, world_size


def validate_delta_embedding_dump_config(
    config: Optional[DeltaEmbeddingDumpConfig], device: torch.device
) -> None:
    """Validate runtime constraints for delta embedding dump.

    Args:
        config: Delta embedding dump configuration, or None to skip validation.
        device: Training device to validate (must be CUDA).
    """
    if config is None:
        return
    if device.type != "cuda":
        raise ValueError(
            "delta_embedding_dump_config only supports CUDA training, "
            f"but got device={device}."
        )
    if config.dump_interval_steps <= 0:
        raise ValueError("delta_embedding_dump_config.dump_interval_steps must be > 0.")


def _has_proto_field(config: Any, field_name: str) -> bool:
    descriptor = getattr(config, "DESCRIPTOR", None)
    if descriptor is None or field_name not in descriptor.fields_by_name:
        return False
    return config.HasField(field_name)


def _feature_config_name(config: Any) -> str:
    return getattr(config, "feature_name", "")


def _zch_feature_names(feature_configs: Iterable[Any]) -> Set[str]:
    zch_feature_names: Set[str] = set()
    for feature_config in feature_configs:
        feature_type = feature_config.WhichOneof("feature")
        if feature_type is None:
            continue
        config = getattr(feature_config, feature_type)
        if _has_proto_field(config, "zch"):
            feature_name = _feature_config_name(config) or feature_type
            zch_feature_names.add(feature_name)
    return zch_feature_names


def validate_delta_embedding_dump_no_zch_features(
    feature_configs: Iterable[Any],
) -> None:
    """Validate that delta embedding dump is not used with MC/ZCH features.

    Args:
        feature_configs: Iterable of feature configuration protos to check.
    """
    zch_feature_names = _zch_feature_names(feature_configs)
    if zch_feature_names:
        raise ValueError(
            "delta_embedding_dump_config does not support MC/ZCH features. "
            "Please convert these zch features to dynamicemb before enabling "
            f"delta embedding dump: {sorted(zch_feature_names)}"
        )


def _feature_name(feature_names: Iterable[str]) -> str:
    names = list(feature_names)
    if len(names) == 1:
        return names[0]
    return ",".join(names)


def _int_attr(value: Any, name: str) -> int:
    attr = getattr(value, name, 0)
    return int(attr) if attr is not None else 0


def _metadata_shard_info(metadata: Any) -> _TableShardInfo:
    if metadata is None or not hasattr(metadata, "shard_offsets"):
        return _TableShardInfo()
    offsets = getattr(metadata, "shard_offsets", [])
    sizes = getattr(metadata, "shard_sizes", [])
    return _TableShardInfo(
        row_offset=int(offsets[0]) if len(offsets) > 0 else 0,
        column_offset=int(offsets[1]) if len(offsets) > 1 else 0,
        local_rows=int(sizes[0]) if len(sizes) > 0 else 0,
        local_cols=int(sizes[1]) if len(sizes) > 1 else 0,
        has_shard_metadata=True,
    )


def _placement_rank(placement: Any) -> Optional[int]:
    if placement is None:
        return None
    rank_fn = getattr(placement, "rank", None)
    if callable(rank_fn):
        rank = rank_fn()
        if rank is not None:
            return int(rank)
    match = re.search(r"rank:(\d+)", str(placement))
    if match is None:
        return None
    return int(match.group(1))


def _table_shard_info_from_parameter_sharding(
    parameter_sharding: Any, rank: int
) -> _TableShardInfo:
    sharding_spec = getattr(parameter_sharding, "sharding_spec", None)
    shards = getattr(sharding_spec, "shards", None)
    if not shards:
        return _TableShardInfo()

    ranks = getattr(parameter_sharding, "ranks", None)
    for idx, shard in enumerate(shards):
        placement_rank = _placement_rank(getattr(shard, "placement", None))
        if placement_rank == rank:
            return _metadata_shard_info(shard)
        if ranks is not None and idx < len(ranks) and ranks[idx] == rank:
            return _metadata_shard_info(shard)

    if ranks is None and 0 <= rank < len(shards):
        return _metadata_shard_info(shards[rank])
    return _TableShardInfo()


def _merge_shard_info(
    primary: _TableShardInfo, fallback: _TableShardInfo
) -> _TableShardInfo:
    primary_has_offsets = primary.has_shard_metadata
    return _TableShardInfo(
        row_offset=primary.row_offset if primary_has_offsets else fallback.row_offset,
        column_offset=(
            primary.column_offset if primary_has_offsets else fallback.column_offset
        ),
        local_rows=primary.local_rows or fallback.local_rows,
        local_cols=primary.local_cols or fallback.local_cols,
        global_rows=primary.global_rows or fallback.global_rows,
        global_cols=primary.global_cols or fallback.global_cols,
        has_shard_metadata=primary.has_shard_metadata or fallback.has_shard_metadata,
    )


def _table_shard_info_from_config(table_config: Any) -> _TableShardInfo:
    metadata_info = _metadata_shard_info(getattr(table_config, "local_metadata", None))
    config_info = _TableShardInfo(
        local_rows=_int_attr(table_config, "local_rows"),
        local_cols=_int_attr(table_config, "local_cols"),
        global_rows=_int_attr(table_config, "num_embeddings"),
        global_cols=_int_attr(table_config, "embedding_dim"),
    )
    return _merge_shard_info(config_info, metadata_info)


def _table_shard_info_from_tensor(
    tensor: torch.Tensor, shard_info: Optional[_TableShardInfo] = None
) -> _TableShardInfo:
    tensor_info = _TableShardInfo(
        local_rows=tensor.size(0) if tensor.dim() > 0 else 0,
        local_cols=tensor.size(1) if tensor.dim() > 1 else 0,
        global_rows=tensor.size(0) if tensor.dim() > 0 else 0,
        global_cols=tensor.size(1) if tensor.dim() > 1 else 0,
    )
    if shard_info is None:
        return tensor_info
    return _merge_shard_info(shard_info, tensor_info)


def _validate_table_shard_info(table_name: str, shard_info: _TableShardInfo) -> None:
    if shard_info.column_offset != 0 or (
        shard_info.local_cols > 0
        and shard_info.global_cols > 0
        and shard_info.local_cols != shard_info.global_cols
    ):
        raise ValueError(
            "delta_embedding_dump_config does not support column-wise "
            "embedding sharding. Please use table-wise, row-wise, or "
            f"data-parallel sharding for table {table_name}. "
            f"local_cols={shard_info.local_cols}, "
            f"global_cols={shard_info.global_cols}, "
            f"column_offset={shard_info.column_offset}."
        )


def _shard_info_quality(shard_info: _TableShardInfo) -> Tuple[bool, bool, bool, bool]:
    return (
        shard_info.has_shard_metadata,
        shard_info.row_offset != 0,
        shard_info.global_rows > 0 and shard_info.global_cols > 0,
        shard_info.local_rows > 0 and shard_info.local_cols > 0,
    )


def _merge_table_shard_info(
    existing: Optional[_TableShardInfo], new_info: _TableShardInfo
) -> _TableShardInfo:
    if existing is None:
        return new_info
    if _shard_info_quality(new_info) >= _shard_info_quality(existing):
        return _merge_shard_info(new_info, existing)
    return _merge_shard_info(existing, new_info)


def _local_table_weight(
    value: Any, shard_info: Optional[_TableShardInfo] = None
) -> _TableWeight:
    if isinstance(value, ShardedTensor):
        shards = value.local_shards()
        if len(shards) != 1:
            raise ValueError(
                "delta embedding dump only supports one local shard per table."
            )
        info = _merge_shard_info(
            shard_info or _TableShardInfo(),
            _metadata_shard_info(getattr(shards[0], "metadata", None)),
        )
        info = _table_shard_info_from_tensor(shards[0].tensor, info)
        return _TableWeight(tensor=shards[0].tensor, shard_info=info)
    if hasattr(value, "to_local"):
        local_value = value.to_local()
        if hasattr(local_value, "local_shards"):
            shards = local_value.local_shards()
            if len(shards) != 1:
                raise ValueError(
                    "delta embedding dump only supports one local shard per table."
                )
            info = _merge_shard_info(
                shard_info or _TableShardInfo(),
                _metadata_shard_info(getattr(shards[0], "metadata", None)),
            )
            info = _table_shard_info_from_tensor(shards[0].tensor, info)
            return _TableWeight(tensor=shards[0].tensor, shard_info=info)
        if isinstance(local_value, torch.Tensor):
            info = _table_shard_info_from_tensor(local_value, shard_info)
            return _TableWeight(tensor=local_value, shard_info=info)
    if isinstance(value, torch.Tensor):
        info = _table_shard_info_from_tensor(value, shard_info)
        return _TableWeight(tensor=value, shard_info=info)
    raise TypeError(f"Unsupported embedding table value type: {type(value)}")


def _embedding_table_fqn(module_fqn: str, module: nn.Module, table_name: str) -> str:
    """Build a state-dict-style table FQN for a sharded sparse module.

    Args:
        module_fqn: FQN of the owning sharded embedding collection.
        module: Owning sharded embedding collection.
        table_name: Raw table name within the collection.

    Returns:
        Owner-qualified table FQN.
    """
    if isinstance(module, ShardedEmbeddingBagCollection):
        table_segment = "embedding_bags"
    elif isinstance(module, ShardedEmbeddingCollection):
        table_segment = "embeddings"
    else:
        raise TypeError(f"Unsupported tracked embedding module: {type(module)}")
    return ".".join(filter(None, (module_fqn, table_segment, table_name)))


class ModelDeltaTracker(TorchRecModelDeltaTracker):
    """Track touched embedding IDs by owner-qualified table FQN.

    This ID-only tracker uses the lookup callback's owning module to keep
    same-named tables in different sharded modules independent.

    Args:
        model: Sharded model whose sparse lookups should be tracked.
        consumers: Independent consumers of the tracked ID stream.
        delete_on_read: Whether to delete IDs after all consumers read them.
        auto_compact: Whether to compact tracked IDs during communication.
    """

    def __init__(
        self,
        model: nn.Module,
        consumers: Optional[List[str]] = None,
        delete_on_read: bool = True,
        auto_compact: bool = False,
    ) -> None:
        consumer_names = consumers or [self.DEFAULT_CONSUMER]
        self._delete_on_read = delete_on_read
        self.per_consumer_batch_idx = {consumer: -1 for consumer in consumer_names}
        self.curr_batch_idx = 0
        self.curr_compact_index = 0
        self.tracked_modules: Dict[str, nn.Module] = {}
        self.fqn_to_feature_names: Dict[str, List[str]] = {}
        self._feature_to_fqn_by_module: Dict[nn.Module, Dict[str, str]] = {}
        self.store = DeltaStoreTrec(UpdateMode.NONE)

        for named_fqn, module in model.named_modules():
            if not isinstance(
                module, (ShardedEmbeddingCollection, ShardedEmbeddingBagCollection)
            ):
                continue

            module_fqn = getattr(module, "_module_fqn", None)
            if not module_fqn:
                module_fqn = self._clean_module_fqn(named_fqn)
            self.tracked_modules[module_fqn] = module

            feature_to_fqn: Dict[str, str] = {}
            for table_name, config in module._table_name_to_config.items():
                table_fqn = _embedding_table_fqn(module_fqn, module, table_name)
                if table_fqn in self.fqn_to_feature_names:
                    raise ValueError(f"Duplicate embedding table FQN: {table_fqn}")
                feature_names = list(config.feature_names)
                self.fqn_to_feature_names[table_fqn] = feature_names
                for feature_name in feature_names:
                    feature_to_fqn[feature_name] = table_fqn
            self._feature_to_fqn_by_module[module] = feature_to_fqn

        for module in self.tracked_modules.values():
            module.register_post_lookup_tracker_fn(self.record_lookup)
            if auto_compact:
                module.register_post_odist_tracker_fn(self.trigger_compaction)

    @staticmethod
    def _clean_module_fqn(fqn: str) -> str:
        """Strip wrapper prefixes from a sharded module FQN."""
        for prefix in ("_dmp_wrapped_module.module.", "module."):
            if fqn.startswith(prefix):
                return fqn[len(prefix) :]
        return fqn

    def record_lookup(
        self,
        kjt: KeyedJaggedTensor,
        states: torch.Tensor,
        emb_module: Optional[nn.Module] = None,
        raw_ids: Optional[torch.Tensor] = None,
    ) -> None:
        """Record touched IDs using the owning module's table FQNs.

        Args:
            kjt: Sparse features used by the lookup.
            states: Lookup output, unused in ID-only tracking mode.
            emb_module: Sharded module that performed the lookup.
            raw_ids: Optional unprocessed IDs, unused by delta embedding dump.
        """
        if emb_module is None:
            raise ValueError("Embedding module is required for FQN delta tracking.")
        feature_to_fqn = self._feature_to_fqn_by_module.get(emb_module)
        if feature_to_fqn is None:
            raise ValueError(
                f"Unrecognized embedding module for FQN delta tracking: {emb_module}"
            )

        ids_by_fqn: Dict[str, List[torch.Tensor]] = {}
        for feature_name in kjt.keys():
            table_fqn = feature_to_fqn[feature_name]
            ids_by_fqn.setdefault(table_fqn, []).append(kjt[feature_name].values())
        for table_fqn, ids in ids_by_fqn.items():
            self.store.append(
                batch_idx=self.curr_batch_idx,
                fqn=table_fqn,
                ids=torch.cat(ids),
                states=None,
            )

    def get_unique_ids(self, consumer: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """Return unique touched IDs keyed by table FQN.

        Args:
            consumer: Consumer whose unread IDs should be returned.

        Returns:
            Unique touched IDs keyed by table FQN.
        """
        return {
            fqn: rows.ids for fqn, rows in self.get_unique(consumer=consumer).items()
        }

    def get_unique(
        self,
        consumer: Optional[str] = None,
        top_percentage: Optional[float] = 1.0,
        per_table_percentage: Optional[Dict[str, Tuple[float, str]]] = None,
        sorted_by_indices: Optional[bool] = True,
    ) -> Dict[str, UniqueRows]:
        """Return unread unique touched IDs keyed by table FQN.

        Args:
            consumer: Consumer whose unread IDs should be returned.
            top_percentage: Unused compatibility argument.
            per_table_percentage: Unused compatibility argument.
            sorted_by_indices: Unused compatibility argument.

        Returns:
            Unique touched rows keyed by table FQN.
        """
        consumer = consumer or self.DEFAULT_CONSUMER
        assert consumer in self.per_consumer_batch_idx, (
            f"consumer {consumer} not present in {self.per_consumer_batch_idx.values()}"
        )

        index_end = self.curr_batch_idx + 1
        index_start = max(self.per_consumer_batch_idx.values())
        if index_start < index_end:
            self.store.compact(index_start, index_end)
        tracker_rows = self.store.get_unique(
            from_idx=self.per_consumer_batch_idx[consumer]
        )
        self.per_consumer_batch_idx[consumer] = index_end
        if self._delete_on_read:
            self.store.delete(up_to_idx=min(self.per_consumer_batch_idx.values()))
        return tracker_rows

    def step(self) -> None:
        """Advance the current tracking batch."""
        self.curr_batch_idx += 1

    def trigger_compaction(self) -> None:
        """Compact newly recorded IDs once per completed batch."""
        if self.curr_compact_index >= self.curr_batch_idx:
            return
        start_idx = max(self.per_consumer_batch_idx.values())
        end_idx = self.curr_batch_idx
        if start_idx < end_idx:
            self.store.compact(start_idx, end_idx)
            self.curr_compact_index = end_idx

    def clear(self, consumer: Optional[str] = None) -> None:
        """Clear tracked IDs using TorchRec's consumer semantics.

        Args:
            consumer: Consumer to clear, or None to clear the whole store.
        """
        if consumer is None:
            self.store.delete()
            return
        assert consumer in self.per_consumer_batch_idx, (
            f"consumer {consumer} not found in {self.per_consumer_batch_idx.values()}"
        )
        if len(self.per_consumer_batch_idx) == 1:
            self.store.delete()


class DeltaEmbeddingDumper:
    """Dump touched embedding ids and latest embedding rows during training.

    Args:
        model: The model containing embedding tables to track.
        config: Configuration for delta embedding dump behavior.
        model_dir: Base directory for model outputs; used as default output location.
        device: Training device; validated to be CUDA.
        feature_configs: Feature configuration protos; validated to be free of
            MC/ZCH features.
    """

    def __init__(
        self,
        model: nn.Module,
        config: DeltaEmbeddingDumpConfig,
        model_dir: str,
        device: torch.device,
        feature_configs: Iterable[Any],
    ) -> None:
        validate_delta_embedding_dump_config(config, device)
        validate_delta_embedding_dump_no_zch_features(feature_configs)
        self._model = model
        self._config = config
        self._interval = config.dump_interval_steps
        self._output_dir = config.output_dir or os.path.join(
            model_dir, "delta_embedding_dump"
        )
        self._file_prefix = config.file_prefix or "delta_embedding"
        self._rank, self._world_size = _distributed_rank_world_size()
        self._tracking_pause_depth = 0
        os.makedirs(self._output_dir, exist_ok=True)

        self._tracker = ModelDeltaTracker(
            model,
            consumers=[_CONSUMER],
            delete_on_read=True,
            auto_compact=True,
        )
        self._table_shard_infos = self._collect_table_shard_infos()
        self._validate_supported_table_sharding(self._table_shard_infos)
        self._install_tracking_pause_guard()
        logger.info(
            "Delta embedding dump enabled: interval=%s output_dir=%s "
            "rank=%s/%s tables=%s",
            self._interval,
            self._output_dir,
            self._rank,
            self._world_size,
            sorted(self._tracker.fqn_to_feature_names),
        )

    def clear(self) -> None:
        """Clear tracked sparse ids, usually after restore-time dummy steps."""
        self._tracker.clear(_CONSUMER)

    @contextmanager
    def pause_tracking(self) -> Iterator[None]:
        """Temporarily skip delta tracking for non-training forward passes."""
        self._tracking_pause_depth += 1
        try:
            yield
        finally:
            self._tracking_pause_depth -= 1

    def maybe_dump(self, global_step: int) -> None:
        """Dump on the configured global-step interval and advance tracker state.

        Args:
            global_step: Current training step.
        """
        if global_step > 0 and global_step % self._interval == 0:
            self.dump(global_step)
        self._tracker.step()

    def final_dump(self, global_step: int) -> Optional[str]:
        """Flush the trailing partial interval at the end of training.

        Boundary steps were already written by ``maybe_dump`` and have no
        remaining delta; re-dumping them would overwrite their shards with an
        empty file under multi-GPU, so skip them here.

        Args:
            global_step: Current training step.

        Returns:
            Path to the dumped parquet file, or None if skipped.
        """
        global_step = self._sync_final_step(global_step)
        if global_step > 0 and global_step % self._interval == 0:
            # Boundary steps were already written (with full delta) by
            # ``maybe_dump``. Re-dumping here has no new delta to flush -- every
            # rank's consumer cursor has already advanced past the boundary's
            # delta -- and torchrec's ``get_unique`` raises
            # ``torch.cat(): expected a non-empty list of Tensors`` on the empty
            # consumer window. Re-dumping would also overwrite the already-written
            # boundary shards (with an empty file under multi-GPU), so skip.
            return None
        return self.dump(global_step)

    def _sync_final_step(self, global_step: int) -> int:
        """Align the final step across ranks before the trailing flush.

        ``maybe_dump`` runs in lockstep so every rank shares ``global_step``,
        but ``final_dump`` is reached with each rank's own last step. With
        ``check_all_workers_data_status=False`` ranks can exhaust the dataloader
        at different steps, so without syncing each would write a lone shard to
        its own ``step_<N>/`` dir -- exactly the ragged shard set the per-rank
        empty-shard logic prevents. Reduce with MAX so the furthest-progressed
        rank's trailing delta is never swallowed by the boundary-step skip, and
        so every rank takes the same skip/dump decision into the same dir.
        """
        if self._world_size <= 1:
            return global_step
        if not (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        ):
            return global_step
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        step_tensor = torch.tensor(global_step, dtype=torch.long, device=device)
        torch.distributed.all_reduce(step_tensor, op=torch.distributed.ReduceOp.MAX)
        return int(step_tensor.item())

    def dump(self, global_step: int) -> Optional[str]:
        """Dump currently tracked sparse ids and embeddings to a parquet file.

        Args:
            global_step: Current training step.

        Returns:
            Path to the dumped parquet file, or None if no data to dump.
        """
        table_weights = self._collect_table_weights()
        dynamic_modules = self._collect_dynamic_modules()
        table_chunks: List[pa.Table] = []
        num_rows = self._append_model_delta_rows(
            table_chunks,
            global_step=global_step,
            table_weights=table_weights,
            dynamic_modules=dynamic_modules,
        )
        if num_rows == 0:
            if self._world_size == 1:
                logger.info("No delta embedding rows to dump at step %s.", global_step)
                return None
            output_path = self._output_path(global_step)
            self._write_table_chunks(table_chunks, output_path)
            logger.info(
                "Dumped empty delta embedding shard to %s at step %s.",
                output_path,
                global_step,
            )
            return output_path
        output_path = self._output_path(global_step)
        self._write_table_chunks(table_chunks, output_path)
        logger.info("Dumped %s delta embedding rows to %s.", num_rows, output_path)
        return output_path

    def _output_path(self, global_step: int) -> str:
        if self._world_size == 1:
            return os.path.join(
                self._output_dir, f"{self._file_prefix}_step_{global_step}.parquet"
            )
        step_dir = os.path.join(self._output_dir, f"step_{global_step}")
        os.makedirs(step_dir, exist_ok=True)
        return os.path.join(
            step_dir,
            (
                f"{self._file_prefix}_step_{global_step}_rank_{self._rank}"
                f"_of_{self._world_size}.parquet"
            ),
        )

    def _install_tracking_pause_guard(self) -> None:
        guarded_modules = getattr(self, "_guarded_tracking_modules", set())
        for module in self._tracker.tracked_modules.values():
            if id(module) in guarded_modules:
                continue
            has_tracker_fn = False
            post_lookup_fn = getattr(module, "post_lookup_tracker_fn", None)
            if post_lookup_fn is not None:
                module.post_lookup_tracker_fn = self._wrap_tracker_fn(post_lookup_fn)
                has_tracker_fn = True
            post_odist_fn = getattr(module, "post_odist_tracker_fn", None)
            if post_odist_fn is not None:
                module.post_odist_tracker_fn = self._wrap_tracker_fn(post_odist_fn)
                has_tracker_fn = True
            if not has_tracker_fn:
                continue
            guarded_modules.add(id(module))
        self._guarded_tracking_modules = guarded_modules

    def _wrap_tracker_fn(self, tracker_fn: Callable[..., Any]) -> Callable[..., Any]:
        def guarded_tracker_fn(*args: Any, **kwargs: Any) -> Any:
            if self._tracking_pause_depth > 0:
                return None
            return tracker_fn(*args, **kwargs)

        return guarded_tracker_fn

    def _append_model_delta_rows(
        self,
        table_chunks: List[pa.Table],
        global_step: int,
        table_weights: Dict[str, _TableWeight],
        dynamic_modules: Dict[str, nn.Module],
    ) -> int:
        num_rows = 0
        # A dynamic module hosting multiple tables is shared across their FQN
        # keys; flush() flushes the whole module, so track which
        # modules were already flushed this dump and skip the redundant repeats.
        flushed_module_ids: Set[int] = set()
        for fqn, unique_rows in self._tracker.get_unique(_CONSUMER).items():
            ids = unique_rows.ids
            if ids.numel() == 0:
                continue
            ids = ids.unique(sorted=True)
            embeddings, key_ids = self._lookup_embeddings(
                fqn,
                ids,
                table_weights=table_weights,
                dynamic_modules=dynamic_modules,
                flushed_module_ids=flushed_module_ids,
            )
            feature_name = _feature_name(
                self._tracker.fqn_to_feature_names.get(fqn, [])
            )
            num_rows += self._append_table_chunk(
                table_chunks,
                global_step=global_step,
                feature_name=feature_name,
                table_fqn=fqn,
                key_ids=key_ids,
                embeddings=embeddings,
                source="model_delta_tracker",
            )
        return num_rows

    def _lookup_embeddings(
        self,
        fqn: str,
        ids: torch.Tensor,
        table_weights: Dict[str, _TableWeight],
        dynamic_modules: Dict[str, nn.Module],
        flushed_module_ids: Optional[Set[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dynamic_module = dynamic_modules.get(fqn)
        if dynamic_module is not None:
            return self._lookup_dynamic_embeddings(
                dynamic_module, fqn, ids, flushed_module_ids
            )
        if fqn not in table_weights:
            raise KeyError(f"Embedding table {fqn} not found in sharded model.")
        table_weight = table_weights[fqn]
        _validate_table_shard_info(fqn, table_weight.shard_info)
        self._validate_row_shard_metadata(fqn, table_weight.shard_info)
        weight = table_weight.tensor
        ids = ids.to(weight.device, dtype=torch.long)
        if ids.numel() == 0:
            return (
                torch.empty(
                    0, weight.size(1), device=weight.device, dtype=weight.dtype
                ),
                torch.empty(0, device=weight.device, dtype=torch.int64),
            )
        valid_mask = (ids >= 0) & (ids < weight.size(0))
        if not bool(valid_mask.all().item()):
            logger.warning(
                "Skip %s ids outside table %s row range [0, %s).",
                int((~valid_mask).sum().item()),
                fqn,
                weight.size(0),
            )
        local_ids = ids[valid_mask]
        key_ids = local_ids + table_weight.shard_info.row_offset
        return weight[local_ids].detach(), key_ids

    def _lookup_dynamic_embeddings(
        self,
        dynamic_module: nn.Module,
        fqn: str,
        ids: torch.Tensor,
        flushed_module_ids: Optional[Set[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            from dynamicemb.types import CopyMode
        except ImportError as exc:
            raise RuntimeError(
                "dynamicemb is required to dump dynamic embedding values."
            ) from exc
        # flush() flushes the whole module; only the first table of a
        # multi-table module needs it within a dump.
        if flushed_module_ids is None or id(dynamic_module) not in flushed_module_ids:
            dynamic_module.flush()
            if flushed_module_ids is not None:
                flushed_module_ids.add(id(dynamic_module))
        table_name = fqn.rsplit(".", maxsplit=1)[-1]
        table_id = dynamic_module.table_names.index(table_name)
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        ids = ids.to(device=device, dtype=torch.int64)
        table_ids = torch.full_like(ids, table_id, dtype=torch.int64)
        _, _, _, _, _, founds, _, values = dynamic_module.tables.find(
            ids, table_ids, CopyMode.EMBEDDING
        )
        emb_dim = dynamic_module._dynamicemb_options[table_id].dim
        founds = founds.to(dtype=torch.bool)
        if not bool(founds.all().item()):
            logger.warning(
                "Skip %s missing dynamic embedding ids for table %s.",
                int((~founds).sum().item()),
                fqn,
            )
        return values[founds, :emb_dim].detach(), ids[founds]

    def _collect_table_shard_infos(self) -> Dict[str, _TableShardInfo]:
        table_shard_infos: Dict[str, _TableShardInfo] = {}
        for module_fqn, module in self._tracker.tracked_modules.items():
            for child_module in module.modules():
                table_name_to_config = getattr(
                    child_module, "_table_name_to_config", None
                )
                if table_name_to_config is not None:
                    for table_name, table_config in table_name_to_config.items():
                        table_fqn = _embedding_table_fqn(module_fqn, module, table_name)
                        table_shard_infos[table_fqn] = _merge_table_shard_info(
                            table_shard_infos.get(table_fqn),
                            _table_shard_info_from_config(table_config),
                        )
                for table_config in self._grouped_embedding_table_configs(child_module):
                    table_name = getattr(table_config, "name", "")
                    if not table_name:
                        continue
                    table_fqn = _embedding_table_fqn(module_fqn, module, table_name)
                    table_shard_infos[table_fqn] = _merge_table_shard_info(
                        table_shard_infos.get(table_fqn),
                        _table_shard_info_from_config(table_config),
                    )
                module_sharding_plan = getattr(
                    child_module, "module_sharding_plan", None
                )
                if module_sharding_plan is None:
                    continue
                for table_name, parameter_sharding in module_sharding_plan.items():
                    table_fqn = _embedding_table_fqn(module_fqn, module, table_name)
                    table_shard_infos[table_fqn] = _merge_table_shard_info(
                        table_shard_infos.get(table_fqn),
                        _table_shard_info_from_parameter_sharding(
                            parameter_sharding, self._rank
                        ),
                    )
        return table_shard_infos

    def _grouped_embedding_table_configs(self, module: nn.Module) -> Iterable[Any]:
        grouped_configs = []
        module_config = getattr(module, "config", None)
        if module_config is not None:
            grouped_configs.append(module_config)
        private_config = getattr(module, "_config", None)
        if private_config is not None and private_config is not module_config:
            grouped_configs.append(private_config)

        for grouped_config in grouped_configs:
            embedding_tables = getattr(grouped_config, "embedding_tables", None)
            if embedding_tables is None:
                continue
            yield from embedding_tables

    def _validate_supported_table_sharding(
        self, table_shard_infos: Dict[str, _TableShardInfo]
    ) -> None:
        for table_fqn, shard_info in table_shard_infos.items():
            _validate_table_shard_info(table_fqn, shard_info)

    def _validate_row_shard_metadata(
        self, table_fqn: str, shard_info: _TableShardInfo
    ) -> None:
        if (
            self._world_size > 1
            and shard_info.local_rows > 0
            and shard_info.global_rows > 0
            and shard_info.local_rows < shard_info.global_rows
            and not shard_info.has_shard_metadata
        ):
            raise ValueError(
                "delta_embedding_dump_config cannot convert local row ids to "
                f"global key ids for row-wise sharded table {table_fqn}, because "
                "TorchRec shard metadata is missing."
            )

    def _collect_table_weights(self) -> Dict[str, _TableWeight]:
        table_weights: Dict[str, _TableWeight] = {}
        table_shard_infos = self._table_shard_infos
        for module_fqn, module in self._tracker.tracked_modules.items():
            lookups = getattr(module, "_lookups", None)
            if lookups is None:
                continue
            for lookup in lookups:
                lookup = getattr(lookup, "module", lookup)
                named_parameters_by_table = getattr(
                    lookup, "named_parameters_by_table", None
                )
                if named_parameters_by_table is None:
                    continue
                for table_name, table_value in named_parameters_by_table():
                    table_fqn = _embedding_table_fqn(module_fqn, module, table_name)
                    table_weights[table_fqn] = _local_table_weight(
                        table_value,
                        table_shard_infos.get(table_fqn),
                    )
        return table_weights

    def _collect_dynamic_modules(self) -> Dict[str, nn.Module]:
        try:
            from dynamicemb.dump_load import get_dynamic_emb_module
        except ImportError:
            return {}
        modules: Dict[str, nn.Module] = {}
        for module_fqn, module in self._tracker.tracked_modules.items():
            for dynamic_module in get_dynamic_emb_module(module):
                for table_name in dynamic_module.table_names:
                    table_fqn = _embedding_table_fqn(module_fqn, module, table_name)
                    modules[table_fqn] = dynamic_module
        return modules

    def _append_table_chunk(
        self,
        table_chunks: List[pa.Table],
        global_step: int,
        feature_name: str,
        table_fqn: str,
        key_ids: torch.Tensor,
        embeddings: torch.Tensor,
        source: str,
    ) -> int:
        key_ids_cpu = key_ids.detach().cpu().to(torch.int64).contiguous()
        embeddings_cpu = embeddings.detach().cpu().to(torch.float32).contiguous()
        if embeddings_cpu.dim() != 2:
            raise ValueError(
                "delta embedding dump expects a 2-D embedding tensor, "
                f"but got shape={tuple(embeddings_cpu.shape)}."
            )
        num_rows = int(key_ids_cpu.numel())
        if num_rows == 0:
            return 0
        if embeddings_cpu.size(0) != num_rows:
            raise ValueError(
                "delta embedding dump key ids and embeddings row count mismatch: "
                f"key_ids={num_rows}, embeddings={embeddings_cpu.size(0)}."
            )

        table_chunks.append(
            pa.Table.from_arrays(
                [
                    pa.repeat(pa.scalar(global_step, pa.int64()), num_rows),
                    pa.repeat(pa.scalar(self._rank, pa.int32()), num_rows),
                    pa.repeat(pa.scalar(self._world_size, pa.int32()), num_rows),
                    pa.repeat(pa.scalar(feature_name, pa.string()), num_rows),
                    pa.repeat(pa.scalar(table_fqn, pa.string()), num_rows),
                    pa.array(key_ids_cpu.numpy(), type=pa.int64()),
                    self._embedding_array(embeddings_cpu),
                    pa.repeat(pa.scalar(source, pa.string()), num_rows),
                ],
                schema=_DELTA_DUMP_SCHEMA,
            )
        )
        return num_rows

    def _embedding_array(self, embeddings: torch.Tensor) -> pa.ListArray:
        num_rows = embeddings.size(0)
        emb_dim = embeddings.size(1)
        if emb_dim == 0:
            offsets = torch.zeros(num_rows + 1, dtype=torch.int32).numpy()
        else:
            offsets = torch.arange(
                0,
                (num_rows + 1) * emb_dim,
                emb_dim,
                dtype=torch.int32,
            ).numpy()
        values = pa.array(embeddings.reshape(-1).numpy(), type=pa.float32())
        return pa.ListArray.from_arrays(pa.array(offsets, type=pa.int32()), values)

    def _write_table_chunks(
        self, table_chunks: List[pa.Table], output_path: str
    ) -> None:
        # Write to a sibling temp file and atomically os.replace() it into place
        # only after the writer closes cleanly. A kill or exception mid-write
        # then leaves at most an orphan .tmp (which the step_*/*.parquet glob
        # ignores), never a truncated shard at the canonical path.
        tmp_path = f"{output_path}.rank{self._rank}.tmp"
        try:
            with pq.ParquetWriter(tmp_path, _DELTA_DUMP_SCHEMA) as writer:
                chunks = table_chunks or [_DELTA_DUMP_SCHEMA.empty_table()]
                for table_chunk in chunks:
                    writer.write_table(table_chunk)
            os.replace(tmp_path, output_path)
        except BaseException:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise
