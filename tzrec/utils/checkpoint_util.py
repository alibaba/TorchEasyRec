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
import math
import os
import queue
import re
import shutil
import threading
import weakref
from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

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

# queue token meaning "run a prune pass"; ``None`` means "stop the worker".
_PRUNE_REQUEST = object()


class PartialLoadPlanner(DefaultLoadPlanner):
    """Support restore partial states.

    Args:
        flatten_state_dict (bool): Handle state_dict with nested dicts.
        flatten_sharded_tensors (bool): For FSDP in 2D parallel mode.
        ckpt_param_map_path (str): parameter mapping for checkpoint.
        skip_output_segments_tensor (bool): If True, skip loading the
            MCH (ZCH) ``_output_segments_tensor`` buffer. It is replicated
            but rank-specific, and loading a saved value whose boundaries
            do not include the current rank's boundaries would crash
            ``validate_state()``. Set this when the saved world size is
            not a multiple divisor of the current world size.
    """

    def __init__(
        self,
        flatten_state_dict: bool = True,
        flatten_sharded_tensors: bool = True,
        ckpt_param_map_path: Optional[str] = None,
        skip_output_segments_tensor: bool = False,
    ) -> None:
        super().__init__(flatten_state_dict, flatten_sharded_tensors)
        self._ckpt_param_map = dict()
        self._skip_output_segments_tensor = skip_output_segments_tensor
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

        # pyre-ignore [16]
        for fqn, obj in self.state_dict.items():
            if self._skip_output_segments_tensor and fqn.endswith(
                "._output_segments_tensor"
            ):
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


class CheckpointManager:
    """Saves training checkpoints and prunes old ones asynchronously.

    ``prune`` never deletes on the calling thread: it enqueues a coalesced
    request to a single daemon worker that does all filesystem work (listing,
    best-checkpoint lookup, deletion), so the training loop never blocks on
    ``model_dir`` I/O. ``glob`` of ``model_dir`` is the source of truth on every
    pass, so the manager holds no checkpoint registry. Load/discovery methods
    delegate to the module-level free functions.

    Args:
        model_dir: directory holding ``model.ckpt-<step>`` checkpoints.
        keep_checkpoint_max: max number of recent checkpoints to keep; 0 keeps all.
        export_config: when ``exporter_type == "best"``, the current best checkpoint
            (by eval metric) is always retained even if older than the kept window.
        eval_result_filename: eval result file (relative to ``model_dir``) used to
            locate the best checkpoint.
    """

    def __init__(
        self,
        model_dir: str,
        keep_checkpoint_max: int = 0,
        export_config: Optional[export_pb2.ExportConfig] = None,
        eval_result_filename: str = TRAIN_EVAL_RESULT_FILENAME,
    ) -> None:
        self._model_dir = model_dir
        self._keep_checkpoint_max = keep_checkpoint_max
        self._export_config = export_config
        self._eval_result_filename = eval_result_filename
        self._prune_queue: "queue.Queue[object]" = queue.Queue()
        self._prune_pending = False
        self._lock = threading.Lock()
        self._prune_worker: Optional[threading.Thread] = None
        self._finalizer: Optional[weakref.finalize] = None
        # save-cadence policy (set via set_save_policy on the train path only;
        # defaults disable all triggers so eval/predict/export paths are unaffected)
        self._save_steps = 0
        self._save_epochs = 0
        self._ts_interval = 0
        self._ts_targets: List[int] = []
        self._ts_quorum = 0.5
        # cadence state owned here so dedupe is centralized across all save sites
        self._last_ckpt_step = -1
        self._last_data_ts: Optional[float] = None

    def save(
        self,
        step: int,
        model: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
        dataloader_state: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save a checkpoint at the given step, then request an async prune."""
        ckpt_dir = os.path.join(self._model_dir, f"model.ckpt-{step}")
        save_model(ckpt_dir, model, optimizer)
        if dataloader_state is not None:
            save_dataloader_state(ckpt_dir, dataloader_state)
        self.prune()
        return ckpt_dir

    def set_save_policy(
        self,
        save_steps: int,
        save_epochs: int,
        ts_interval_s: int,
        ts_targets: List[int],
        ts_quorum: float,
    ) -> None:
        """Configure when ``maybe_save`` fires (train path only).

        Sets cadence config only (never the ``_last_*`` state), so a watermark
        seeded on resume survives.

        Args:
            save_steps: step interval; 0 disables the step trigger.
            save_epochs: epoch interval; 0 disables the epoch trigger.
            ts_interval_s: event-time interval in seconds; 0 disables.
            ts_targets: absolute event-time targets (Unix-epoch seconds).
            ts_quorum: fraction of workers (0, 1] past a boundary to trigger a save.
        """
        if (ts_interval_s > 0 or len(ts_targets) > 0) and not (0.0 < ts_quorum <= 1.0):
            raise ValueError(
                f"save_checkpoints_timestamp_quorum must be in (0, 1], got {ts_quorum}."
            )
        self._save_steps = save_steps
        self._save_epochs = save_epochs
        self._ts_interval = ts_interval_s
        self._ts_targets = ts_targets
        self._ts_quorum = ts_quorum

    def needs_worker_timestamps(self) -> bool:
        """Return whether the policy needs per-worker event-times gathered."""
        return self._ts_interval > 0 or len(self._ts_targets) > 0

    def maybe_save(
        self,
        step: int,
        model: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
        dataloader_state: Optional[Dict[str, Any]] = None,
        *,
        epoch: Optional[int] = None,
        worker_ts_list: Optional[List[float]] = None,
        final: bool = False,
    ) -> bool:
        """Save a checkpoint if a configured trigger fires; return whether it did.

        Centralizes the step / epoch / event-time decisions and the single
        per-step dedupe so no call site can re-save a step another already saved.
        The decision is deterministic and identical across ranks, so the
        collective ``save`` is entered in lockstep and nothing collective runs
        when not saving.

        Args:
            step: current global step.
            model: model to save.
            optimizer: optimizer to save, if any.
            dataloader_state: dataloader resume state; the event-time watermark is
                stamped into it on save.
            epoch: current epoch; enables the epoch trigger when not None.
            worker_ts_list: per-rank consumed event-times (seconds) gathered by the
                caller; reconciled here via the worker quorum.
            final: force a save (still subject to the dedupe), e.g. at train end.

        Returns:
            True if a checkpoint was saved.
        """
        data_ts = (
            quorum_event_time(worker_ts_list, self._ts_quorum)
            if worker_ts_list
            else None
        )
        # a -1.0-dominated quorum (too few workers with a timestamp) -> no event-time
        if data_ts is not None and data_ts < 0:
            data_ts = None

        want = final
        if self._save_steps > 0 and step > 0 and step % self._save_steps == 0:
            want = True
        if (
            epoch is not None
            and self._save_epochs > 0
            and step > 0
            and (epoch + 1) % self._save_epochs == 0
        ):
            want = True
        if data_ts is not None:
            if self._last_data_ts is None:
                # first event-time seen: set the reference, do not save
                self._last_data_ts = data_ts
            elif should_save_on_timestamp(
                data_ts, self._last_data_ts, self._ts_interval, self._ts_targets
            ):
                want = True

        if not want or step == self._last_ckpt_step:
            return False

        self._last_ckpt_step = step
        if data_ts is not None:
            # advance + persist the watermark on every save so resume is exact
            self._last_data_ts = data_ts
            if dataloader_state is not None:
                dataloader_state[DATA_TS_WATERMARK] = data_ts
        self.save(step, model, optimizer, dataloader_state)
        return True

    def prune(self) -> None:
        """Request an async prune pass (keep recent N + best). Rank 0 only.

        Cheap and non-blocking: only flips a flag and enqueues one coalesced
        request. The flag drops the request when a pass is already queued or
        in-flight, so the queue stays bounded. All filesystem work happens on
        the worker thread.
        """
        if self._keep_checkpoint_max <= 0:
            return
        if int(os.environ.get("RANK", 0)) != 0:
            return
        with self._lock:
            if self._prune_pending:
                return
            self._prune_pending = True
        self._ensure_prune_worker()
        self._prune_queue.put(_PRUNE_REQUEST)

    def close(self) -> None:
        """Drain queued prune passes and stop the worker (idempotent).

        Called at the end of training for a deterministic flush. A weakref.finalize
        safety net (registered when the worker starts) runs the same drain at
        interpreter exit if close() is skipped (e.g. training raised), so the worker
        is never leaked and pending deletions still complete.
        """
        if self._prune_worker is None:
            return
        self._finalizer.detach()
        self._drain(self._prune_queue, self._prune_worker)
        self._prune_worker = None

    # --- load / discovery (delegate to the module-level free functions) ---

    def latest_checkpoint(self) -> Tuple[Optional[str], int]:
        """Latest checkpoint under this manager's model_dir."""
        return latest_checkpoint(self._model_dir)

    def best_checkpoint(self) -> Tuple[Optional[str], int]:
        """Best checkpoint under model_dir per the configured export metric."""
        return best_checkpoint(
            self._model_dir, self._export_config, self._eval_result_filename
        )

    def restore(
        self,
        ckpt_path: str,
        model: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
        ckpt_param_map_path: Optional[str] = None,
    ) -> None:
        """Restore model/optimizer state from a checkpoint dir."""
        restore_model(ckpt_path, model, optimizer, ckpt_param_map_path)

    def restore_dataloader_state(self, ckpt_path: str) -> Optional[Dict[str, Any]]:
        """Restore dataloader state saved alongside a checkpoint.

        Also seeds the event-time watermark so ``maybe_save`` resumes the
        timestamp trigger; absent -> initialize from the first batch.
        """
        state = restore_dataloader_state(ckpt_path)
        if state is not None:
            self._last_data_ts = state.get(DATA_TS_WATERMARK)
        return state

    @staticmethod
    def _drain(prune_queue: "queue.Queue[object]", worker: threading.Thread) -> None:
        """Stop the prune worker and wait for it to finish (no ``self`` ref).

        Used by both ``close()`` and the weakref.finalize safety net. FIFO ensures
        any pending prune request runs before the stop sentinel.
        """
        prune_queue.put(None)  # stop sentinel
        worker.join()

    def _ensure_prune_worker(self) -> None:
        if self._prune_worker is None:
            self._prune_worker = threading.Thread(
                target=self._prune_worker_loop, name="ckpt-prune", daemon=True
            )
            self._prune_worker.start()
            # Safety net: if close() is never reached (e.g. training raised), drain
            # the worker at interpreter exit so it is not leaked. The callback must
            # not reference self, or it would pin the manager.
            self._finalizer = weakref.finalize(
                self, self._drain, self._prune_queue, self._prune_worker
            )

    def _prune_worker_loop(self) -> None:
        while True:
            item = self._prune_queue.get()
            try:
                if item is None:  # stop sentinel
                    return
                # allow a fresh request to be enqueued while this pass runs.
                with self._lock:
                    self._prune_pending = False
                try:
                    self._run_prune()
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"Checkpoint prune pass failed: {e}")
            finally:
                self._prune_queue.task_done()

    def _run_prune(self) -> None:
        """All filesystem work for one prune pass -- runs only on the worker."""
        ckpt_metas = glob.glob(
            os.path.join(self._model_dir, "model.ckpt-*" + os.path.sep)
        )
        ckpt_metas = [x.rstrip(os.path.sep) for x in ckpt_metas]
        if len(ckpt_metas) <= self._keep_checkpoint_max:
            return
        ckpt_metas.sort(key=_get_checkpoint_step)
        protected = set(ckpt_metas[-self._keep_checkpoint_max :])
        if (
            self._export_config is not None
            and self._export_config.exporter_type == "best"
        ):
            best_ckpt_path, _ = best_checkpoint(
                self._model_dir, self._export_config, self._eval_result_filename
            )
            if best_ckpt_path is not None:
                protected.add(best_ckpt_path.rstrip(os.path.sep))
        for ckpt_path in ckpt_metas:
            if ckpt_path not in protected:
                logger.info(f"Removing old checkpoint {ckpt_path}...")
                try:
                    shutil.rmtree(ckpt_path)
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"Failed to remove checkpoint {ckpt_path}: {e}")


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


def _needs_mch_redistribution(model_ckpt_path: str, cur_world_size: int) -> bool:
    """Whether MCH (ZCH) state needs value-aware redistribution on load.

    Position-based ShardedTensor slicing is only safe when each saved
    per-rank value range lies entirely inside one current per-rank value
    range. For uniform row-wise sharding this holds iff
    ``saved_world_size % cur_world_size == 0``; otherwise the saved
    `_mch_remapped_ids_mapping` values fall outside the current local
    range and must be reassigned by `_redistribute_mch_state`.
    """
    if not os.path.exists(model_ckpt_path):
        return False
    try:
        saved_world_size = _ckpt_world_size(model_ckpt_path)
    except Exception as e:
        logger.warning(f"Failed to detect saved world size from {model_ckpt_path}: {e}")
        return False
    return saved_world_size != cur_world_size and saved_world_size % cur_world_size != 0


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


def _redistribute_mch_state(model: nn.Module) -> None:
    """Value-aware all-to-all redistribution of MCH (ZCH) state.

    Called between the normal DCP load and ``model.load_state_dict``, once
    each rank already holds a position-based slice of the saved global MCH
    buffers. Those entries may belong to any rank under the current world
    size's per-rank value ranges; we permute them via ``all_to_all_single``
    so every rank ends up owning exactly the entries whose remapped global
    value falls in its local range, preserving the (raw_id → embedding row)
    binding end-to-end with O(local_zch_size) extra memory per rank.

    ``model.load_state_dict``'s post-hook then sorts the new local buffers
    via ``MCHManagedCollisionModule.validate_state``/``_sort_mch_buffers``,
    so we do not need to sort here.
    """
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return
    mc_modules = _find_mch_modules(model)
    if not mc_modules:
        return

    cur_rank = dist.get_rank()
    world_size = dist.get_world_size()
    iinfo_max = torch.iinfo(torch.int64).max

    if cur_rank == 0:
        logger.warning(
            f"Redistributing MCH (ZCH) state via all_to_all_single across "
            f"{len(mc_modules)} MC modules."
        )

    for prefix, m in mc_modules.items():
        cps: int = m._zch_size
        local_offset: int = m._output_global_offset
        raw_ids_buf = m._buffers["_mch_sorted_raw_ids"]
        remapped_buf = m._buffers["_mch_remapped_ids_mapping"]
        local_dev = raw_ids_buf.device

        # Sharded per-slot eviction metadata (_mch_counts,
        # _mch_last_access_iter, ...). Non-persistent helpers like
        # _mch_slots and _delimiter have shape != [cps] so they are
        # filtered out.
        metadata_names = [
            name
            for name, buf in m._buffers.items()
            if name.startswith("_mch_")
            and name not in ("_mch_sorted_raw_ids", "_mch_remapped_ids_mapping")
            and name not in m._non_persistent_buffers_set
            and buf is not None
            and buf.dim() == 1
            and buf.shape[0] == cps
        ]

        # Filter valid entries.
        valid_mask = raw_ids_buf != iinfo_max
        raw_ids = raw_ids_buf[valid_mask]
        remapped = remapped_buf[valid_mask]
        metadata = {name: m._buffers[name][valid_mask] for name in metadata_names}

        # Destination rank for each valid entry = value-based bucket.
        dest_rank = (remapped // cps).to(torch.int64).clamp_(0, world_size - 1)

        # Sort by dest_rank so sends are contiguous per destination.
        perm = torch.argsort(dest_rank, stable=True)
        raw_ids = raw_ids[perm]
        remapped = remapped[perm]
        metadata = {name: t[perm] for name, t in metadata.items()}
        dest_rank = dest_rank[perm]

        # Exchange split sizes.
        send_counts = torch.bincount(dest_rank, minlength=world_size).to(torch.int64)
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts)
        send_split = send_counts.tolist()
        recv_split = recv_counts.tolist()
        n = int(recv_counts.sum().item())
        assert n <= cps, (
            f"MCH [{prefix}] rank {cur_rank}: received {n} > local zch_size {cps}"
        )

        def _a2a(
            send_tensor: torch.Tensor,
            n: int = n,
            recv_split: List[int] = recv_split,
            send_split: List[int] = send_split,
            local_dev: torch.device = local_dev,
        ) -> torch.Tensor:
            recv = torch.empty(n, dtype=send_tensor.dtype, device=local_dev)
            dist.all_to_all_single(
                recv,
                send_tensor.contiguous(),
                output_split_sizes=recv_split,
                input_split_sizes=send_split,
            )
            return recv

        raw_ids_recv = _a2a(raw_ids)
        remapped_recv = _a2a(remapped)
        metadata_recv = {name: _a2a(t) for name, t in metadata.items()}

        # Build new local buffer contents.
        new_raw = torch.full((cps,), iinfo_max, dtype=torch.int64, device=local_dev)
        new_raw[:n] = raw_ids_recv

        all_local = torch.arange(
            local_offset, local_offset + cps, dtype=torch.int64, device=local_dev
        )
        taken = torch.zeros(cps, dtype=torch.bool, device=local_dev)
        if n > 0:
            taken[remapped_recv - local_offset] = True
        unused = all_local[~taken]
        new_remapped = torch.empty(cps, dtype=torch.int64, device=local_dev)
        new_remapped[:n] = remapped_recv
        new_remapped[n:] = unused[: cps - n]

        # In-place copy so state_dict ShardedTensor wrappers pick up the
        # new values before the upcoming model.load_state_dict.
        raw_ids_buf.copy_(new_raw)
        remapped_buf.copy_(new_remapped)
        for name, recv in metadata_recv.items():
            new_meta = torch.zeros(cps, dtype=recv.dtype, device=local_dev)
            new_meta[:n] = recv
            m._buffers[name].copy_(new_meta)


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

    cur_world_size = dist.get_world_size() if dist.is_initialized() else 1
    needs_mch_redistribution = _needs_mch_redistribution(
        model_ckpt_path, cur_world_size
    )

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
                skip_output_segments_tensor=needs_mch_redistribution,
            ),
        )
        if needs_mch_redistribution:
            _redistribute_mch_state(model)
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
# reserved key in dataloader_state holding the event-time watermark (Unix-epoch
# seconds) of the last checkpoint; no ":" so per-source consumers skip it.
DATA_TS_WATERMARK = "__data_ts_watermark__"


def save_dataloader_state(
    checkpoint_dir: str,
    dataloader_state: Dict[str, Any],
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
        merged_state: Dict[str, Any] = {}
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


def restore_dataloader_state(
    checkpoint_dir: str,
) -> Optional[Dict[str, Any]]:
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
    dataloader_state: Dict[str, Any],
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


def should_save_on_timestamp(
    data_ts_s: float,
    last_ckpt_ts_s: Optional[float],
    interval_s: int,
    target_ts_list: List[int],
) -> bool:
    """Decide whether to save a checkpoint based on consumed event-time.

    Drives event-time checkpointing from the data timestamp (e.g. the kafka
    message timestamp surfaced on ``Batch.data_timestamp``). Two triggers:

    * interval: a boundary ``floor(ts / interval_s)`` -- aligned to the Unix
      epoch (wall-clock), NOT to training epochs -- has been crossed since the
      last save.
    * targets: the consumed event-time has crossed an absolute target.

    Args:
        data_ts_s: current consumed event-time (Unix-epoch seconds), already
            reconciled across ranks.
        last_ckpt_ts_s: event-time at the last save, or None when no reference
            has been established yet (first batch / just restored). When None
            this returns False so the caller only initializes the reference
            instead of saving.
        interval_s: event-time interval in seconds, boundaries aligned to the
            Unix epoch; 0 disables the interval trigger.
        target_ts_list: absolute event-time targets (Unix-epoch seconds); a save
            fires once when the consumed event-time crosses each target.

    Returns:
        True if a checkpoint should be saved for this event-time.
    """
    # No reference yet: only initialize, never save on the first observed batch.
    if last_ckpt_ts_s is None:
        return False
    # Unix-epoch-aligned interval boundary crossed since the last save.
    if interval_s > 0 and int(data_ts_s // interval_s) > int(
        last_ckpt_ts_s // interval_s
    ):
        return True
    # Any absolute target crossed within (last_ckpt_ts_s, data_ts_s].
    for target in target_ts_list:
        if last_ckpt_ts_s < target <= data_ts_s:
            return True
    return False


def quorum_event_time(local_ts_list: List[float], quorum: float) -> Optional[float]:
    """Reconcile per-worker consumed event-times into one global event-time.

    Returns the event-time that at least ``quorum`` fraction of the workers have
    reached -- i.e. the ``(1 - quorum)`` upper quantile of the per-worker values:
    the largest ``T`` such that at least ``ceil(quorum * m)`` of the ``m`` values
    are ``>= T``. ``quorum=1.0`` -> min (all workers past ``T``); ``quorum`` near
    0 -> max (any one worker). Using a quantile makes the default (0.5) robust to
    a single outlier/garbage timestamp.

    A worker without a timestamp carries the -1.0 sentinel, which sorts low and so
    counts as "not past" -- the quorum is over all workers; the result is negative
    when too few have a real timestamp.

    Args:
        local_ts_list: per-worker event-times (seconds), -1.0 for workers without
            one.
        quorum: fraction of workers in (0, 1].

    Returns:
        The reconciled event-time, or None when ``local_ts_list`` is empty.
    """
    vals = sorted(local_ts_list)
    m = len(vals)
    if m == 0:
        return None
    # epsilon guards float overshoot (e.g. 0.1 * 10 -> 1.0000000001 -> ceil 2)
    k = max(1, min(m, math.ceil(quorum * m - 1e-9)))
    return vals[m - k]


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
