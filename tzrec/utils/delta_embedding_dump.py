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
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import nn
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torchrec.distributed.model_tracker.model_delta_tracker import (
    ModelDeltaTrackerTrec,
)
from torchrec.distributed.model_tracker.types import TrackingMode

from tzrec.protos.train_pb2 import DeltaEmbeddingDumpConfig
from tzrec.utils.logging_util import logger

_CONSUMER = "delta_embedding_dump"


def _is_enabled(config: DeltaEmbeddingDumpConfig) -> bool:
    return config is not None and config.enable


def validate_delta_embedding_dump_config(
    config: DeltaEmbeddingDumpConfig, device: torch.device
) -> None:
    """Validate runtime constraints for delta embedding dump."""
    if not _is_enabled(config):
        return
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
    if world_size != 1 or device.type != "cuda":
        raise ValueError(
            "delta_embedding_dump_config only supports single GPU training "
            f"for now, but got WORLD_SIZE={world_size}, device={device}."
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


def _zch_feature_info(feature_configs: Iterable[Any]) -> Tuple[Set[str], Set[str]]:
    zch_feature_names: Set[str] = set()
    zch_table_names: Set[str] = set()
    for feature_config in feature_configs:
        feature_type = feature_config.WhichOneof("feature")
        if feature_type is None:
            continue
        config = getattr(feature_config, feature_type)
        if _has_proto_field(config, "zch"):
            feature_name = _feature_config_name(config) or feature_type
            zch_feature_names.add(feature_name)
            zch_table_names.add(
                getattr(config, "embedding_name", "") or f"{feature_name}_emb"
            )
    return zch_feature_names, zch_table_names


def validate_delta_embedding_dump_no_zch_features(
    feature_configs: Iterable[Any],
) -> Tuple[Set[str], Set[str]]:
    """Validate that delta embedding dump is not used with MC/ZCH features."""
    zch_feature_names, zch_table_names = _zch_feature_info(feature_configs)
    if zch_feature_names:
        raise ValueError(
            "delta_embedding_dump_config does not support MC/ZCH features. "
            "Please convert these zch features to dynamicemb before enabling "
            f"delta embedding dump: {sorted(zch_feature_names)}"
        )
    return zch_feature_names, zch_table_names


def _feature_name(feature_names: Iterable[str]) -> str:
    names = list(feature_names)
    if len(names) == 1:
        return names[0]
    return ",".join(names)


def _local_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, ShardedTensor):
        shards = value.local_shards()
        if len(shards) != 1:
            raise ValueError(
                "delta embedding dump only supports one local shard per table."
            )
        return shards[0].tensor
    if hasattr(value, "to_local"):
        local_value = value.to_local()
        if hasattr(local_value, "local_shards"):
            shards = local_value.local_shards()
            if len(shards) != 1:
                raise ValueError(
                    "delta embedding dump only supports one local shard per table."
                )
            return shards[0].tensor
        if isinstance(local_value, torch.Tensor):
            return local_value
    if isinstance(value, torch.Tensor):
        return value
    raise TypeError(f"Unsupported embedding table value type: {type(value)}")


class DeltaEmbeddingDumper:
    """Dump touched embedding ids and latest embedding rows during training."""

    def __init__(
        self,
        model: nn.Module,
        config: DeltaEmbeddingDumpConfig,
        model_dir: str,
        zch_feature_names: Optional[Set[str]] = None,
        zch_table_names: Optional[Set[str]] = None,
    ) -> None:
        self._model = model
        self._config = config
        self._interval = config.dump_interval_steps
        self._zch_feature_names = zch_feature_names or set()
        self._zch_table_names = zch_table_names or set()
        self._batches_seen = 0
        self._output_dir = config.output_dir or os.path.join(
            model_dir, "delta_embedding_dump"
        )
        self._file_prefix = config.file_prefix or "delta_embedding"
        os.makedirs(self._output_dir, exist_ok=True)

        self._tracker = ModelDeltaTrackerTrec(
            model,
            consumers=[_CONSUMER],
            delete_on_read=True,
            mode=TrackingMode.ID_ONLY,
            fqns_to_skip=self._zch_table_names,
        )
        self._table_to_fqn: Dict[str, str] = {}
        self._table_to_fqn.update(self._tracker.table_to_fqn)
        self._fqn_to_table: Dict[str, str] = {
            fqn: table for table, fqn in self._table_to_fqn.items()
        }
        self._fqn_to_feature_names: Dict[str, List[str]] = {}
        self._fqn_to_feature_names.update(self._tracker.fqn_to_feature_names())
        self._skip_fqns = {
            fqn
            for fqn, feature_names in self._fqn_to_feature_names.items()
            if any(name in self._zch_feature_names for name in feature_names)
        }
        if self._skip_fqns:
            logger.warning(
                "Delta embedding dump will skip MC/ZCH embedding tables: %s",
                sorted(self._skip_fqns),
            )

        logger.info(
            "Delta embedding dump enabled: interval=%s output_dir=%s tables=%s",
            self._interval,
            self._output_dir,
            sorted(self._table_to_fqn.keys()),
        )

    def clear(self) -> None:
        """Clear tracked sparse ids, usually after restore-time dummy steps."""
        self._tracker.clear(_CONSUMER)

    def maybe_dump(self, global_step: int) -> None:
        """Dump on the configured batch interval and advance the tracker window."""
        self._batches_seen += 1
        if self._batches_seen % self._interval == 0:
            self.dump(global_step)
        self._tracker.step()

    def dump(self, global_step: int) -> Optional[str]:
        """Dump currently tracked sparse ids and embeddings to a parquet file."""
        table_weights = self._collect_table_weights()
        dynamic_modules = self._collect_dynamic_modules()
        rows: List[Dict[str, Any]] = []
        self._append_model_delta_rows(
            rows,
            global_step=global_step,
            table_weights=table_weights,
            dynamic_modules=dynamic_modules,
        )
        if not rows:
            logger.info("No delta embedding rows to dump at step %s.", global_step)
            return None
        output_path = os.path.join(
            self._output_dir, f"{self._file_prefix}_step_{global_step}.parquet"
        )
        self._write_rows(rows, output_path)
        logger.info("Dumped %s delta embedding rows to %s.", len(rows), output_path)
        return output_path

    def _append_model_delta_rows(
        self,
        rows: List[Dict[str, Any]],
        global_step: int,
        table_weights: Dict[str, torch.Tensor],
        dynamic_modules: Dict[str, nn.Module],
    ) -> None:
        for fqn, unique_rows in self._tracker.get_unique(_CONSUMER).items():
            if fqn in self._skip_fqns:
                continue
            ids = unique_rows.ids
            if ids.numel() == 0:
                continue
            table_name = self._fqn_to_table.get(fqn)
            if table_name is None:
                logger.warning("Skip delta rows for unknown table fqn: %s", fqn)
                continue
            ids = ids.unique(sorted=True)
            embeddings, found_mask = self._lookup_embeddings(
                table_name,
                ids,
                table_weights=table_weights,
                dynamic_modules=dynamic_modules,
            )
            ids = ids[found_mask]
            feature_name = _feature_name(self._fqn_to_feature_names.get(fqn, []))
            self._extend_rows(
                rows,
                global_step=global_step,
                feature_name=feature_name,
                table_fqn=fqn,
                key_ids=ids,
                embeddings=embeddings,
                source="model_delta_tracker",
            )

    def _lookup_embeddings(
        self,
        table_name: str,
        ids: torch.Tensor,
        table_weights: Dict[str, torch.Tensor],
        dynamic_modules: Dict[str, nn.Module],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dynamic_module = dynamic_modules.get(table_name)
        if dynamic_module is not None:
            return self._lookup_dynamic_embeddings(dynamic_module, table_name, ids)
        if table_name not in table_weights:
            raise KeyError(f"Embedding table {table_name} not found in sharded model.")
        weight = table_weights[table_name]
        ids = ids.to(weight.device, dtype=torch.long)
        if ids.numel() == 0:
            return (
                torch.empty(
                    0, weight.size(1), device=weight.device, dtype=weight.dtype
                ),
                torch.empty(0, device=weight.device, dtype=torch.bool),
            )
        valid_mask = (ids >= 0) & (ids < weight.size(0))
        if not bool(valid_mask.all().item()):
            logger.warning(
                "Skip %s ids outside table %s row range [0, %s).",
                int((~valid_mask).sum().item()),
                table_name,
                weight.size(0),
            )
        return weight[ids[valid_mask]].detach(), valid_mask

    def _lookup_dynamic_embeddings(
        self, dynamic_module: nn.Module, table_name: str, ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            from dynamicemb.types import CopyMode
        except ImportError as exc:
            raise RuntimeError(
                "dynamicemb is required to dump dynamic embedding values."
            ) from exc
        dynamic_module.flush()
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
                table_name,
            )
        return values[founds, :emb_dim].detach(), founds

    def _collect_table_weights(self) -> Dict[str, torch.Tensor]:
        table_weights: Dict[str, torch.Tensor] = {}
        for module in self._model.modules():
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
                    table_weights[table_name] = _local_tensor(table_value)
        return table_weights

    def _collect_dynamic_modules(self) -> Dict[str, nn.Module]:
        try:
            from dynamicemb.dump_load import get_dynamic_emb_module
        except ImportError:
            return {}
        modules: Dict[str, nn.Module] = {}
        seen = set()
        for dynamic_module in get_dynamic_emb_module(self._model):
            if id(dynamic_module) in seen:
                continue
            seen.add(id(dynamic_module))
            for table_name in dynamic_module.table_names:
                modules[table_name] = dynamic_module
        return modules

    def _extend_rows(
        self,
        rows: List[Dict[str, Any]],
        global_step: int,
        feature_name: str,
        table_fqn: str,
        key_ids: torch.Tensor,
        embeddings: torch.Tensor,
        source: str,
    ) -> None:
        key_ids_cpu = key_ids.detach().cpu().to(torch.int64).tolist()
        embeddings_cpu = embeddings.detach().cpu().to(torch.float32).tolist()
        for key_id, embedding in zip(key_ids_cpu, embeddings_cpu):
            rows.append(
                {
                    "global_step": global_step,
                    "feature_name": feature_name,
                    "table_fqn": table_fqn,
                    "key_id": key_id,
                    "embedding": embedding,
                    "source": source,
                }
            )

    def _write_rows(self, rows: List[Dict[str, Any]], output_path: str) -> None:
        table = pa.Table.from_arrays(
            [
                pa.array([r["global_step"] for r in rows], type=pa.int64()),
                pa.array([r["feature_name"] for r in rows], type=pa.string()),
                pa.array([r["table_fqn"] for r in rows], type=pa.string()),
                pa.array([r["key_id"] for r in rows], type=pa.int64()),
                pa.array([r["embedding"] for r in rows], type=pa.list_(pa.float32())),
                pa.array([r["source"] for r in rows], type=pa.string()),
            ],
            names=[
                "global_step",
                "feature_name",
                "table_fqn",
                "key_id",
                "embedding",
                "source",
            ],
        )
        pq.write_table(table, output_path)
