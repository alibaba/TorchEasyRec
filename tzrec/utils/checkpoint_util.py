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
import json
import os
import re
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import nn, optim
from torch.distributed.checkpoint import (
    FileSystemReader,
    TensorStorageMetadata,
    load,
    save,
)
from torch.distributed.checkpoint.default_planner import (
    DefaultLoadPlanner,
    DTensor,
    LoadPlan,
    _create_read_items,
)
from torchrec.modules.mc_modules import MCHManagedCollisionModule

from tzrec.constant import TRAIN_EVAL_RESULT_FILENAME
from tzrec.protos import export_pb2
from tzrec.utils.dynamicemb_util import has_dynamicemb
from tzrec.utils.logging_util import logger
from tzrec.utils.state_dict_util import fix_mch_state


class PartialLoadPlanner(DefaultLoadPlanner):
    """Support restore partial states.

    Args:
        flatten_state_dict (bool): Handle state_dict with nested dicts.
        flatten_sharded_tensors (bool): For FSDP in 2D parallel mode.
        ckpt_param_map_path (str): parameter mapping for checkpoint.
        skip_mc_module_state (bool): If True, skip loading MCH (ZCH)
            ManagedCollisionModule state buffers (`_mch_*`,
            `_output_segments_tensor`, ...). Set this when restoring a
            checkpoint that was saved with a different world size, because
            torchrec's `ShardedTensor` resharding for these buffers does a
            byte-level position slice that does not preserve the per-rank
            slot-index value semantics.
    """

    def __init__(
        self,
        flatten_state_dict: bool = True,
        flatten_sharded_tensors: bool = True,
        ckpt_param_map_path: Optional[str] = None,
        skip_mc_module_state: bool = False,
    ) -> None:
        super().__init__(flatten_state_dict, flatten_sharded_tensors)
        self._ckpt_param_map = dict()
        self._skip_mc_module_state = skip_mc_module_state
        if ckpt_param_map_path:
            with open(ckpt_param_map_path) as f:
                for line in f.readlines():
                    cur_param_name, old_param_name = line.strip().split()
                    self._ckpt_param_map[cur_param_name] = old_param_name

    def create_local_plan(self) -> LoadPlan:
        """Create local load plan."""
        requests = []

        # mapping old __BASE__.ec_list.0 to new __BASE__.ec_dict.{dim}
        ec_compat_map = {}
        # pyre-ignore [16]
        for k, v in self.metadata.state_dict_metadata.items():
            if k.endswith(".weight") and isinstance(v, TensorStorageMetadata):
                for old_pattern, new_pattern in [
                    ("mc_ec_list", "mc_ec_dict"),
                    ("ec_list", "ec_dict"),
                ]:
                    if f".{old_pattern}." in k:
                        parts = k.split(".")
                        pattern_idx = parts.index(old_pattern)
                        dim = v.size[1]
                        ec_compat_map[
                            f"{parts[pattern_idx - 1]}.{new_pattern}.{dim}"
                        ] = f"{parts[pattern_idx - 1]}.{old_pattern}.{parts[pattern_idx + 1]}"  # NOQA

        # `_output_segments_tensor` is a fixed-shape [1025] *replicated*
        # buffer in MCH (ZCH) ManagedCollisionModule whose contents describe
        # this rank's partition boundaries. Its shape is identical across
        # ranks and across world sizes, so dist.checkpoint would silently
        # overwrite the freshly-initialized local value with the saved one.
        # When the saved world size differs from the current one, the
        # `_load_state_dict_post_hook` then fails in `validate_state()`
        # because `_output_global_offset` (a Python int that is *not* in the
        # state dict) no longer appears in the loaded segments tensor. The
        # local buffer is rebuilt by `fix_mch_state(model)` in
        # `restore_model` for the current world size, so it is always safe
        # to skip loading this buffer.
        def _is_output_segments_tensor(fqn: str) -> bool:
            return fqn.endswith("._output_segments_tensor")

        # When loading a checkpoint saved with a *smaller* world size, the
        # other MCH state buffers (`_mch_sorted_raw_ids`,
        # `_mch_remapped_ids_mapping`, `_mch_<metadata>`) cannot be loaded
        # via the default `ShardedTensor` position-based path. They are
        # wrapped as `ShardedTensor` by
        # `ShardedManagedCollisionCollection._initialize_torch_state`
        # (torchrec mc_modules.py:262), and dist.checkpoint will happily do
        # a byte-level position slice across world sizes — but the *values*
        # inside `_mch_remapped_ids_mapping` are global slot indices in
        # `arange(output_global_offset, output_global_offset + zch_size)`
        # from the saved per-rank range, which only fit each current local
        # range when `cur_per_rank_zch >= saved_per_rank_zch` (i.e. the
        # current world size is `<=` the saved one). When the current world
        # size is strictly larger we skip these buffers here and rebuild
        # them via `_redistribute_mch_state` in `restore_model`, which
        # routes saved entries to the rank that owns each entry's global
        # value, preserving the (raw_id → embedding row) binding. In the
        # opposite direction (e.g. single-rank export of a multi-rank
        # training checkpoint) position-based loading is correct and must
        # be kept so that user-side MCH modules populated via
        # `ckpt_param_map_path` remaps still receive their saved state.
        def _is_mc_state_buffer(fqn: str) -> bool:
            return "._managed_collision_modules." in fqn

        # pyre-ignore [16]
        for fqn, obj in self.state_dict.items():
            if _is_output_segments_tensor(fqn):
                continue
            if self._skip_mc_module_state and _is_mc_state_buffer(fqn):
                continue

            meta_fqn = fqn

            fqn_remap_set = set()
            if fqn in self._ckpt_param_map:
                meta_fqn = self._ckpt_param_map[fqn]
                fqn_remap_set.add(fqn)
                logger.info(f"Remap restore state [{fqn}] from [{meta_fqn}]")

            for ec_new, ec_old in ec_compat_map.items():
                if ec_new in meta_fqn:
                    new_meta_fqn = meta_fqn
                    meta_fqn = new_meta_fqn.replace(ec_new, ec_old)
                    fqn_remap_set.add(fqn)
                    logger.warning(
                        f"Remap EmbeddingCollection state [{new_meta_fqn}] from old "
                        "[{meta_fqn}], will be deprecated when tzrec version >= 1.0.0"
                    )

            if meta_fqn in self.metadata.state_dict_metadata:
                md = self.metadata.state_dict_metadata[meta_fqn]
            else:
                logger.warning(f"Skip restore state [{fqn}]")
                continue

            read_items = []
            if isinstance(obj, DTensor):
                if obj.device_mesh.get_coordinate() is not None:
                    read_items = _create_read_items(meta_fqn, md, obj)
            else:
                read_items = _create_read_items(meta_fqn, md, obj)

            if fqn in fqn_remap_set:
                read_items = [
                    replace(x, dest_index=replace(x.dest_index, fqn=fqn))
                    for x in read_items
                ]
            requests += read_items

        plan = LoadPlan(requests)
        return plan


def _get_checkpoint_step(ckpt_path: str) -> int:
    """Get checkpoint step from ckpt_path.

    Args:
        ckpt_path: checkpoint path, such as xx/model.ckpt-2000.

    Return:
        ckpt_step: checkpoint step, such as 2000.
    """
    _, ckpt_name = os.path.split(ckpt_path)
    ckpt_name, ext = os.path.splitext(ckpt_name)
    if ext.startswith(".ckpt-"):
        ckpt_name = ext
    toks = ckpt_name.split("-")
    try:
        ckpt_step = int(toks[-1])
    except Exception:
        ckpt_step = 0
    return ckpt_step


def latest_checkpoint(model_dir: str) -> Tuple[Optional[str], int]:
    """Find latest checkpoint under a directory.

    Args:
        model_dir: model directory

    Return:
        latest_ckpt_path: latest checkpoint path.
        latest_step: step of the latest checkpoint
    """
    if "model.ckpt-" not in model_dir:
        # fsspec glob need endswith os.path.sep
        ckpt_metas = glob.glob(os.path.join(model_dir, "model.ckpt-*" + os.path.sep))
        ckpt_metas = list(map(lambda x: x.rstrip(os.path.sep), ckpt_metas))
        if len(ckpt_metas) == 0:
            model_ckpt_dir = os.path.join(model_dir, "model")
            optim_ckpt_dir = os.path.join(model_dir, "optimizer")
            if os.path.exists(model_ckpt_dir) or os.path.exists(optim_ckpt_dir):
                return model_dir, 0
            else:
                return None, -1
        if len(ckpt_metas) > 1:
            ckpt_metas.sort(key=lambda x: _get_checkpoint_step(x))
        latest_ckpt_path = ckpt_metas[-1]
    else:
        latest_ckpt_path = model_dir
    return latest_ckpt_path, _get_checkpoint_step(latest_ckpt_path)


def best_checkpoint(
    model_dir: str,
    export_config: export_pb2.ExportConfig,
    eval_result_filename: str = TRAIN_EVAL_RESULT_FILENAME,
) -> Tuple[Optional[str], int]:
    """Find best checkpoint under a directory.

    Args:
        model_dir: model directory
        export_config: export_pb2.ExportConfig
        eval_result_filename: evaluation result filename

    Return:
        latest_ckpt_path: latest checkpoint path.
        latest_step: step of the latest checkpoint
    """
    eval_path = os.path.join(model_dir, eval_result_filename)
    metric_name = None
    if export_config.HasField("best_exporter_metric"):
        metric_name = export_config.best_exporter_metric
    if os.path.isfile(eval_path):
        step_metric = {}
        with open(eval_path, "r") as f:
            for line in f:
                if line:
                    metric = json.loads(line.strip())
                    step = metric["global_step"]
                    del metric["global_step"]
                    if len(metric) == 1 and metric_name is None:
                        step_metric[step] = metric.values()[0]
                    else:
                        if metric_name not in metric:
                            raise ValueError(
                                f"checkpoint {eval_result_filename}"
                                f" not find {metric_name} metric."
                            )
                        step_metric[step] = metric[metric_name]
        if len(step_metric) < 1:
            logger.info(
                f"not find eval result in {eval_result_filename}, "
                f"will search latest checkpoint"
            )
            return latest_checkpoint(model_dir)
        if export_config.metric_larger_is_better:
            sorted_mertic = sorted(
                step_metric.items(), key=lambda x: x[1], reverse=True
            )
        else:
            sorted_mertic = sorted(
                step_metric.items(), key=lambda x: x[1], reverse=False
            )
        max_metric_step = sorted_mertic[0][0]
        best_ckpt_path = os.path.join(model_dir, f"model.ckpt-{max_metric_step}")
        if os.path.exists(best_ckpt_path):
            logger.info(f"find best checkpoint is {best_ckpt_path}")
            return best_ckpt_path, max_metric_step
        else:
            raise ValueError(
                f"find best metric is {max_metric_step} step,"
                f"but not find {best_ckpt_path}."
            )
    else:
        logger.info(f"not find {eval_result_filename}, will search latest checkpoint")
        return latest_checkpoint(model_dir)


_DISTCP_RANK_RE = re.compile(r"__(\d+)_\d+\.distcp$")


def _ckpt_world_size(ckpt_dir: str) -> int:
    """Return the world size that wrote the distributed checkpoint.

    `ckpt_dir` should point at the directory that contains the per-rank
    `__<rank>_<part>.distcp` shard files (typically
    ``<model_dir>/model.ckpt-N/model``). The world size is one more than
    the highest rank found.
    """
    ranks: set = set()
    for name in os.listdir(ckpt_dir):
        m = _DISTCP_RANK_RE.match(name)
        if m:
            ranks.add(int(m.group(1)))
    if not ranks:
        raise RuntimeError(f"No .distcp files under {ckpt_dir}")
    return max(ranks) + 1


def _strip_dmp_prefix(name: str) -> str:
    """Strip TorchRec DMP wrapper prefix from a module name."""
    for prefix in (
        "_dmp_wrapped_module.module.",
        "_dmp_wrapped_module.",
        "module.",
    ):
        if name.startswith(prefix):
            name = name[len(prefix) :]
    return name


def _find_mch_modules(
    model: nn.Module,
) -> Dict[str, MCHManagedCollisionModule]:
    """Return {state_dict_prefix -> MCHManagedCollisionModule} for all MC modules."""
    out: Dict[str, MCHManagedCollisionModule] = {}
    for name, m in model.named_modules():
        if isinstance(m, MCHManagedCollisionModule):
            out[_strip_dmp_prefix(name)] = m
    return out


def _read_full_mch_tensors(
    model_ckpt_path: str,
    mc_modules: Dict[str, MCHManagedCollisionModule],
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Read the full (un-sharded) saved MCH state for each MC module.

    Returns a dict: {mc_module_prefix -> {buffer_name -> full_global_tensor}}.
    Only persistent MCH state buffers (`_mch_*`) are read; rank-specific
    metadata buffers like `_output_segments_tensor` and `_current_iter_tensor`
    are skipped (they are handled by `fix_mch_state` and the load planner).
    """
    reader = FileSystemReader(model_ckpt_path)
    meta = reader.read_metadata()

    # Build the request: for each MCH module, find every saved buffer under
    # its prefix that starts with `_mch_` and request the full global tensor.
    full_state: Dict[str, torch.Tensor] = {}
    per_module: Dict[str, Dict[str, torch.Tensor]] = {p: {} for p in mc_modules}
    for fqn, md in meta.state_dict_metadata.items():
        if not isinstance(md, TensorStorageMetadata):
            continue
        for prefix in mc_modules:
            mod_prefix = f"{prefix}."
            if not fqn.startswith(mod_prefix):
                continue
            buf_name = fqn[len(mod_prefix) :]
            if not buf_name.startswith("_mch_"):
                continue
            # Allocate a full-sized plain tensor on CPU for the redistribution.
            t = torch.empty(tuple(md.size), dtype=md.properties.dtype)
            full_state[fqn] = t
            per_module[prefix][buf_name] = t
            break

    if full_state:
        load(full_state, checkpoint_id=model_ckpt_path)

    return per_module


def _redistribute_mch_state(
    model: nn.Module,
    model_ckpt_path: str,
    saved_world_size: int,
) -> None:
    """Redistribute MCH (ZCH) state from a checkpoint with a different world size.

    The saved checkpoint stores `_mch_sorted_raw_ids`,
    `_mch_remapped_ids_mapping`, and the eviction-policy metadata buffers as
    full global tensors (size = global zch_size). Their values are *global*
    slot indices in `[0, zch_size)` — NOT positions, NOT bytes — and torchrec
    routes inputs to the rank that owns each value's range. Position-based
    `ShardedTensor` resharding therefore does not preserve semantics across
    different world sizes; the saved values must be redistributed by value.

    Algorithm (per MCH table):
      1. Load the full saved tensors for that table (size = global zch_size).
      2. For each saved slot position p with a non-empty raw_id, look at
         `value = saved_remapped_ids_mapping[p]`. The new owning rank is
         `value // local_zch_size`. (`local_zch_size` is the same on every
         rank for clean row-wise sharding.) Importantly, this matches the
         row-wise position-based sharding used by `ShardedTensor` for the
         embedding `weight`, so the (raw_id → embedding_row) binding is
         preserved end-to-end.
      3. Each rank collects the saved entries assigned to it.
      4. Write them into the local module's buffers, padding empty slots
         with `iinfo.max` (raw ids) and the unused values from this rank's
         local range (remapped ids), and sort via `_sort_mch_buffers`.

    The embedding `weight` tensors are still loaded by torch DCP via the
    normal `ShardedTensor` path, which performs the matching position-based
    slice — so `weight[local_pos]` already corresponds to the saved embedding
    row at the global slot index `local_offset + local_pos`.
    """
    cur_rank = dist.get_rank() if dist.is_initialized() else 0

    mc_modules = _find_mch_modules(model)
    if not mc_modules:
        return

    cur_world_size = dist.get_world_size() if dist.is_initialized() else 1
    if cur_world_size == saved_world_size:
        return

    if cur_rank == 0:
        logger.warning(
            f"Redistributing MCH (ZCH) state from saved world size "
            f"{saved_world_size} to current world size {cur_world_size} "
            f"by value-aware bucketing across {len(mc_modules)} MC modules."
        )

    full_state = _read_full_mch_tensors(model_ckpt_path, mc_modules)
    iinfo_max = torch.iinfo(torch.int64).max

    for prefix, m in mc_modules.items():
        bufs = full_state.get(prefix)
        if not bufs:
            continue
        if "_mch_sorted_raw_ids" not in bufs or "_mch_remapped_ids_mapping" not in bufs:
            continue

        saved_raw_ids = bufs["_mch_sorted_raw_ids"]
        saved_remapped = bufs["_mch_remapped_ids_mapping"]
        saved_total = saved_raw_ids.numel()
        local_zch_size: int = m._zch_size
        local_offset: int = m._output_global_offset

        # Sanity check: global zch_size must be invariant across world sizes.
        if saved_total != local_zch_size * cur_world_size:
            logger.warning(
                f"Skipping MCH redistribution for [{prefix}]: saved global "
                f"size {saved_total} != local_zch_size {local_zch_size} * "
                f"world_size {cur_world_size}."
            )
            continue

        # Value-based assignment: this rank owns saved entries whose
        # `_mch_remapped_ids_mapping` value falls in [local_offset,
        # local_offset + local_zch_size).
        nonempty_mask = saved_raw_ids != iinfo_max
        owns_mask = (saved_remapped >= local_offset) & (
            saved_remapped < local_offset + local_zch_size
        )
        keep_mask = nonempty_mask & owns_mask
        my_raw_ids = saved_raw_ids[keep_mask]
        my_remapped = saved_remapped[keep_mask]

        local_dev = m._current_iter_tensor.device

        # Build new local buffers for this rank.
        new_raw_ids = torch.full(
            (local_zch_size,), iinfo_max, dtype=torch.int64, device=local_dev
        )
        new_remapped = torch.arange(
            local_offset,
            local_offset + local_zch_size,
            dtype=torch.int64,
            device=local_dev,
        )

        n = my_raw_ids.numel()
        if n > local_zch_size:
            # Should not happen for clean row-wise sharding, but truncate
            # defensively to avoid overflowing the local table.
            logger.warning(
                f"MCH [{prefix}]: rank {cur_rank} has {n} reassigned entries "
                f"but local_zch_size={local_zch_size}; truncating."
            )
            my_raw_ids = my_raw_ids[:local_zch_size]
            my_remapped = my_remapped[:local_zch_size]
            n = local_zch_size

        if n > 0:
            new_raw_ids[:n] = my_raw_ids.to(local_dev)
            # Place each entry's saved value at the matching position; pad
            # the empty positions with the local-range values that were not
            # taken by any kept entry, so the full local range
            # [local_offset, local_offset + local_zch_size) is still covered
            # exactly once across positions (preserving the invariant of
            # `_mch_remapped_ids_mapping` after `_sort_mch_buffers`).
            kept_values = my_remapped.to(local_dev)
            new_remapped[:n] = kept_values
            all_local = torch.arange(
                local_offset,
                local_offset + local_zch_size,
                dtype=torch.int64,
                device=local_dev,
            )
            taken = torch.zeros(local_zch_size, dtype=torch.bool, device=local_dev)
            taken[kept_values - local_offset] = True
            unused = all_local[~taken]
            new_remapped[n:] = unused[: local_zch_size - n]

        # Copy into the live buffers.
        m._buffers["_mch_sorted_raw_ids"].copy_(new_raw_ids)
        m._buffers["_mch_remapped_ids_mapping"].copy_(new_remapped)

        # Eviction-policy metadata (`_mch_<name>` for LFU/LRU/etc.) is also
        # per-slot. Reset it for the slots we kept (the saved values are
        # tied to the saved world's slot positions and don't carry useful
        # ordering across a different per-rank slot count). For empty slots
        # leave at zero.
        for buf_name in m._buffers:
            if (
                buf_name.startswith("_mch_")
                and buf_name not in ("_mch_sorted_raw_ids", "_mch_remapped_ids_mapping")
                and m._buffers[buf_name] is not None
                and m._buffers[buf_name].dim() > 0
                and m._buffers[buf_name].shape[0] == local_zch_size
            ):
                m._buffers[buf_name].zero_()

        # Sort the local buffers to keep the binary-search invariant.
        m._sort_mch_buffers()


def restore_model(
    checkpoint_dir: str,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    ckpt_param_map_path: Optional[str] = None,
) -> None:
    """Restore model state.

    Args:
        checkpoint_dir (str): easyrec model checkpoint dir.
        model (nn.Module): a EasyRec model.
        optimizer (optim.Optimizer, optional): a optimizer.
        ckpt_param_map_path (str): parameter mapping for checkpoint.
    """
    is_local_rank_zero = int(os.environ.get("LOCAL_RANK", 0)) == 0
    if is_local_rank_zero:
        logger.info(f"Restoring checkpoint from {checkpoint_dir}...")
    if not os.path.exists(checkpoint_dir):
        raise RuntimeError(f"checkpoint_dir[{checkpoint_dir}] not exists.")

    meta_path = os.path.join(checkpoint_dir, "meta")
    model_ckpt_path = os.path.join(checkpoint_dir, "model")
    optim_ckpt_path = os.path.join(checkpoint_dir, "optimizer")

    # Detect whether the current world size is *larger* than the world size
    # the checkpoint was saved with by counting the per-rank `.distcp`
    # shards under the model checkpoint dir. Only this direction needs
    # value-aware redistribution of MCH (ZCH) state: the saved
    # `_mch_remapped_ids_mapping` values live in the saved rank's per-rank
    # range `[offset, offset + saved_per_rank_zch)`, which is *larger* than
    # the current per-rank range when `cur > saved`, so position-based
    # `ShardedTensor` slicing would fill the current local buffer with
    # global values that fall outside the local range and crash the FBGEMM
    # / CUDA gather kernel. When `cur <= saved` (e.g. single-rank export of
    # a multi-rank training checkpoint), each current local range still
    # contains all relevant saved values, so the stock position-based load
    # path is correct and avoids touching user-side MCH modules loaded via
    # `ckpt_param_map_path` remaps.
    saved_world_size: Optional[int] = None
    if os.path.exists(model_ckpt_path):
        try:
            saved_world_size = _ckpt_world_size(model_ckpt_path)
        except Exception as e:
            logger.warning(
                f"Failed to detect saved world size from {model_ckpt_path}: {e}"
            )
    cur_world_size = dist.get_world_size() if dist.is_initialized() else 1
    needs_mch_redistribution = (
        saved_world_size is not None and cur_world_size > saved_world_size
    )
    if needs_mch_redistribution and is_local_rank_zero:
        logger.warning(
            f"Checkpoint world size ({saved_world_size}) is smaller than "
            f"current world size ({cur_world_size}); MCH (ZCH) state will "
            "be redistributed by value to preserve per-position ZCH "
            "semantics across the world size change."
        )

    # `_output_segments_tensor` describes this rank's partition boundaries
    # and must be locally rebuilt for the current world size before any load
    # path that calls `validate_state()` (e.g. nn.Module.load_state_dict's
    # post-hook in MCHManagedCollisionModule). Doing it here is also a
    # safeguard for the export path, where init_parameters() materializes
    # the buffer with zeros and then PartialLoadPlanner skips loading it.
    fix_mch_state(model)

    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)

    if not meta.get("load_model", True):
        pass
    elif os.path.exists(model_ckpt_path):
        if is_local_rank_zero:
            logger.info(f"Restoring model state from {model_ckpt_path}...")
        state_dict = model.state_dict()
        load(
            state_dict,
            checkpoint_id=model_ckpt_path,
            planner=PartialLoadPlanner(
                ckpt_param_map_path=ckpt_param_map_path,
                skip_mc_module_state=needs_mch_redistribution,
            ),
        )
        if needs_mch_redistribution:
            # The MCH state ShardedTensors were not filled by `load` above
            # (PartialLoadPlanner skipped them). Manually redistribute the
            # full saved state into the per-rank local buffers, preserving
            # the (raw_id → embedding row) binding via value-based bucketing.
            _redistribute_mch_state(
                model,
                model_ckpt_path,
                # pyre-ignore [6]
                saved_world_size,
            )
        model.load_state_dict(state_dict)
    else:
        raise RuntimeError(f"model_ckpt_path[{model_ckpt_path}] not exists.")

    if optimizer:
        if not meta.get("load_optim", True):
            pass
        elif os.path.exists(optim_ckpt_path):
            if is_local_rank_zero:
                logger.info(f"Restoring optimizer state from {optim_ckpt_path}...")
            state_dict = optimizer.state_dict()
            # The optimizer state only contains trainable-parameter state
            # (e.g. fused-optimizer accumulators for embedding `weight`),
            # which is sharded by row position and resharded correctly by
            # the default position-based path in both directions. The MCH
            # buffers are registered via `register_buffer`, never appear
            # in the optimizer state dict, and need no special handling on
            # this load.
            load(
                state_dict,
                checkpoint_id=optim_ckpt_path,
                planner=PartialLoadPlanner(
                    ckpt_param_map_path=ckpt_param_map_path,
                ),
            )
            optimizer.load_state_dict(state_dict)
        else:
            if is_local_rank_zero:
                logger.warning(f"optim_ckpt_path[{optim_ckpt_path}] not exists.")

    if has_dynamicemb:
        from dynamicemb.dump_load import DynamicEmbLoad

        dynamicemb_path = os.path.join(checkpoint_dir, "dynamicemb")
        if os.path.exists(dynamicemb_path):
            logger.info(
                f"RANK[{os.environ.get('RANK', 0)}] restoring dynamic embedding..."
            )
            DynamicEmbLoad(
                dynamicemb_path,
                model,
                table_names=meta.get("dynamicemb_load_table_names", None),
                optim=meta.get("dynamicemb_load_optim", optimizer is not None),
                counter=True,
            )
            logger.info(
                f"RANK[{os.environ.get('RANK', 0)}] restore dynamic embedding finished."
            )


def save_model(
    checkpoint_dir: str, model: nn.Module, optimizer: Optional[optim.Optimizer] = None
) -> None:
    """Save model state.

    Args:
        checkpoint_dir (str): easyrec model checkpoint dir.
        model (nn.Module): a EasyRec model.
        optimizer (optim.Optimizer, optional): a optimizer.
    """
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        logger.info(f"Saving checkpoint to {checkpoint_dir}...")
    save(model.state_dict(), checkpoint_id=os.path.join(checkpoint_dir, "model"))
    if optimizer:
        save(
            optimizer.state_dict(),
            checkpoint_id=os.path.join(checkpoint_dir, "optimizer"),
        )
    if has_dynamicemb:
        from dynamicemb.dump_load import DynamicEmbDump

        DynamicEmbDump(
            os.path.join(checkpoint_dir, "dynamicemb"),
            model,
            optim=optimizer is not None,
            counter=True,
        )
    # save model plan
    if hasattr(model, "_plan") and model._plan is not None:
        if int(os.environ.get("RANK", 0)) == 0:
            plan = {}
            for module_path, module_plan in model._plan.plan.items():
                plan[module_path] = {}
                for param_name, param_sharding in module_plan.items():
                    plan[module_path][param_name] = {
                        "sharding_type": param_sharding.sharding_type,
                        "compute_kernel": param_sharding.compute_kernel,
                        "ranks": param_sharding.ranks,
                    }
            with open(os.path.join(checkpoint_dir, "plan"), "w") as f:
                json.dump(plan, f)


DATALOADER_CKPT_FILENAME = "dataloader_state.json"


def save_dataloader_state(
    checkpoint_dir: str,
    dataloader_state: Dict[str, int],
) -> None:
    """Save dataloader state, aggregating from all ranks first.

    This function aggregates checkpoint states from all ranks by taking
    the maximum consumed row for each source, then rank 0 writes the
    merged state to a JSON file.

    Args:
        checkpoint_dir: Directory to save the checkpoint state.
        dataloader_state: Local checkpoint state {source_key: max_consumed_row}.
    """
    # All-gather states from all ranks
    if dist.is_initialized():
        world_size = dist.get_world_size()
        all_states = [None] * world_size
        dist.all_gather_object(all_states, dataloader_state)

        # Merge by taking max for each key
        merged_state: Dict[str, int] = {}
        for state in all_states:
            if state:
                for key, value in state.items():
                    if key in merged_state:
                        merged_state[key] = max(merged_state[key], value)
                    else:
                        merged_state[key] = value
    else:
        merged_state = dataloader_state

    # Only rank 0 writes to file
    if int(os.environ.get("RANK", 0)) == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        ckpt_path = os.path.join(checkpoint_dir, DATALOADER_CKPT_FILENAME)
        with open(ckpt_path, "w") as f:
            json.dump(merged_state, f, indent=2)
        logger.info(f"Saved dataloader state to {ckpt_path}")


def restore_dataloader_state(checkpoint_dir: str) -> Optional[Dict[str, int]]:
    """Restore dataloader checkpoint state from file.

    Args:
        checkpoint_dir: Directory containing the checkpoint state file.

    Returns:
        dataloader state dict {source_key: max_consumed_row}, or None if not found.
    """
    ckpt_path = os.path.join(checkpoint_dir, DATALOADER_CKPT_FILENAME)
    if not os.path.exists(ckpt_path):
        logger.info(f"No dataloader state found at {ckpt_path}")
        return None

    with open(ckpt_path, "r") as f:
        state = json.load(f)

    is_local_rank_zero = int(os.environ.get("LOCAL_RANK", 0)) == 0
    if is_local_rank_zero:
        logger.info(f"Restored dataloader state from {ckpt_path}")
    return state


def update_dataloder_state(
    dataloader_state: Dict[str, int],
    checkpoint_info: Optional[Dict[str, int]],
) -> None:
    """Merge batch's checkpoint_info into dataloader_state by taking max per key.

    This updates dataloader_state in-place.

    Args:
        dataloader_state: Accumulated dataloader state to update.
        checkpoint_info: Checkpoint info from current batch, or None.
    """
    if checkpoint_info is None:
        return

    for key, value in checkpoint_info.items():
        if key in dataloader_state:
            dataloader_state[key] = max(dataloader_state[key], value)
        else:
            dataloader_state[key] = value


def list_distcp_param(checkpoint_dir: str) -> List[str]:
    """List distributed checkpoint parameter names."""
    meta_paths = []
    if os.path.exists(os.path.join(checkpoint_dir, ".metadata")):
        meta_paths.append(checkpoint_dir)
    else:
        if os.path.exists(os.path.join(checkpoint_dir, "model", ".metadata")):
            meta_paths.append(os.path.join(checkpoint_dir, "model"))
        if os.path.exists(os.path.join(checkpoint_dir, "optimizer", ".metadata")):
            meta_paths.append(os.path.join(checkpoint_dir, "optimizer"))
    if len(meta_paths) == 0:
        raise RuntimeError(f"Can't find distribute checkpoint in {checkpoint_dir}")

    param_names = []
    for meta_path in meta_paths:
        reader = FileSystemReader(path=meta_path)
        meta = reader.read_metadata()
        logger.info(f"Params in {meta_path}:")
        for k, v in meta.state_dict_metadata.items():
            if isinstance(v, TensorStorageMetadata):
                param_names.append(k)
                logger.info(f"{k}: {v.size}")
    return param_names
