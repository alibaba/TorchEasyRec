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

# Copyright (c) 2026, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");

r"""Utilities for exporting and atomically publishing online dense models.

Serving publish contract:

<ONLINE_DENSE_EXPORT_DIR>/dense_hot_export/
├── current.json                  # Atomic pointer to the latest version
└── versions/
    └── <yyyyMMddHHmmss>/         # Immutable version directory
        ├── scripted_model.pt     # TorchScript dense model
        ├── dense_meta.json       # Placeholder -> serving embedding mapping
        ├── graph/                # Graph dump for debugging
        └── READY                 # Completion marker written before the switch

current.json (manifest v2):
{
  "version": "20260724052000",
  "checkpoint_step": 1200,
  "sparse_step": 1250,
  "data_timestamp": 1782365432.0,
  "created_at": "2026-07-24T05:20:00.000000+00:00",
  "publish_interval_minutes": 10,
  "sparse_probes": [
    {"pk": "<remapped_table_fqn>", "sk": 123456,
     "crc32": "9a3e17b0", "encoding": "int8"}
  ]
}

- Build each version under a temporary directory, write ``READY``, then rename
  it atomically into place.
- ``current.json`` flips only when the version's dense pairing step
  (``checkpoint_step``) is <= the cross-rank sparse upload watermark, and
  ``sparse_step`` records that watermark at flip time, so
  ``sparse_step >= checkpoint_step`` always holds: sparse ahead of dense is
  the only allowed skew, and the processor never serves a dense version whose
  paired embeddings have not finished uploading.
- ``created_at`` is the flip time; ``publish_interval_minutes`` is present
  only when a timed publish cadence is configured, and the processor derives
  its "no new version" alert threshold from it.
- ``sparse_probes`` samples keys from the completed sparse uploads (capped at
  64 in total, may be empty); the processor verifies each (pk, sk, crc32)
  through its serving read path before hot-swapping the dense model, and an
  empty list (offline bootstrap) skips verification.
- The processor reads only the version named by ``current.json``; a version
  superseded by a newer build before it was published is removed.
"""

import datetime
import json
import os
import shutil
import tempfile
import time
import weakref
from threading import Condition, Event, Lock, Thread
from typing import Any, Dict, List, Optional, Tuple, cast

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed._tensor import DTensor

from tzrec.acc import utils as acc_utils
from tzrec.optim.ema import DenseEMA
from tzrec.utils import config_util
from tzrec.utils.checkpoint_util import remap_input_tile_user_key
from tzrec.utils.export_util import (
    build_dense_graph_module,
    create_dense_export_warmup_data,
    finalize_dense_export,
)
from tzrec.utils.logging_util import logger

VERSIONS_DIR = "versions"
CURRENT_JSON = "current.json"
_VERSION_TIME_FORMAT = "%Y%m%d%H%M%S"


def _utc_now() -> str:
    """Current UTC time as an ISO-8601 string."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _atomic_write_json(path: str, data: Dict[str, Any]) -> None:
    """Write JSON atomically: tmp file in the same dir, then os.replace."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp_path, path)


def _publish_current(current_path: str, payload: Dict[str, Any]) -> None:
    """Atomically publish the service-facing current.json pointer."""
    _atomic_write_json(current_path, payload)


def _read_current_version(current_path: str) -> Optional[str]:
    """Return the version current.json points at, or None if absent/unreadable.

    Best-effort: a missing or corrupt current.json means there is no live
    pointer to spare, so pruning falls back to pure newest-K retention.
    """
    try:
        with open(current_path) as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        logger.warning("could not read current version from %s: %s", current_path, e)
        return None
    return data.get("version") if isinstance(data, dict) else None


def _max_kept_versions() -> int:
    """Max published dense versions to retain (0 = keep all).

    Reads ``ONLINE_DENSE_EXPORT_KEEP_VERSIONS`` (default 0, keep every
    exported version). A positive value retains the newest K versions and
    must be at least 3: serving reads current.json and needs the previous
    version for an atomic swap, so a smaller K buys no retention safety.
    """
    keep = int(os.environ.get("ONLINE_DENSE_EXPORT_KEEP_VERSIONS", "0"))
    if keep < 0 or 0 < keep < 3:
        raise ValueError(
            "ONLINE_DENSE_EXPORT_KEEP_VERSIONS must be 0 (keep all) or "
            f">= 3, got {keep}."
        )
    return keep


def _prune_old_dense_versions(export_root: str, versions_root: str) -> None:
    """Best-effort retention: keep the newest K versions, sweep stale tmp artifacts.

    K comes from ``ONLINE_DENSE_EXPORT_KEEP_VERSIONS`` and defaults to 0
    (keep every exported version); a positive K retains the newest K versions.
    The version current.json points at is always spared even when it sorts
    outside the newest K: an explicit --version or clock rollback after a
    restart can publish an older timestamp, and deleting it would leave the
    serving pointer referencing a missing directory. Stale ``*.tmp.<pid>``
    dirs and current.json.tmp.<pid> files left by crashed exports are swept
    so they don't accumulate under the serving-facing tree; this process's
    own ``.tmp.<pid>`` artifacts are spared because the flip-time prune runs
    on the training thread while the export worker may be building the next
    version's tmp directory.
    """
    max_versions = _max_kept_versions()
    own_tmp_suffix = f".tmp.{os.getpid()}"
    for base in (versions_root, export_root):
        try:
            entries = os.listdir(base)
        except FileNotFoundError:
            continue
        for name in entries:
            if ".tmp." not in name and not name.endswith(".tmp"):
                continue
            if name.endswith(own_tmp_suffix):
                continue
            path = os.path.join(base, name)
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
                logger.info("removed stale dense export tmp: %s", path)
            except OSError as e:
                logger.warning("failed to remove stale tmp %s: %s", path, e)
    if max_versions <= 0:
        return
    try:
        entries = os.listdir(versions_root)
    except FileNotFoundError:
        return
    current_version = _read_current_version(os.path.join(export_root, CURRENT_JSON))
    version_dirs = sorted(
        os.path.join(versions_root, name)
        for name in entries
        # tmp build dirs are the sweep's business; the spared own-pid one may
        # be an in-flight build and must never be retention-pruned.
        if os.path.isdir(os.path.join(versions_root, name)) and ".tmp." not in name
    )
    for path in version_dirs[:-max_versions]:
        if os.path.basename(path) == current_version:
            continue
        try:
            shutil.rmtree(path)
            logger.info("removed old dense export version: %s", path)
        except OSError as e:
            logger.warning("failed to remove old version %s: %s", path, e)


def _format_version(now: datetime.datetime) -> str:
    return now.strftime(_VERSION_TIME_FORMAT)


def make_version(now: Optional[datetime.datetime] = None) -> str:
    """Build a yyyyMMddHHmmss dense export version name."""
    now = now or datetime.datetime.now()
    return _format_version(now)


def _make_monotonic_version(
    last_version: str, now: Optional[datetime.datetime] = None
) -> str:
    version = make_version(now)
    if not last_version or version > last_version:
        return version
    last_version_dt = datetime.datetime.strptime(last_version, _VERSION_TIME_FORMAT)
    return _format_version(last_version_dt + datetime.timedelta(seconds=1))


def _online_dense_export_enabled() -> bool:
    return os.environ.get("ONLINE_DENSE_EXPORT", "0") == "1"


def resolve_dense_export_root(model_dir: str) -> str:
    """Resolve the online dense export publish root.

    The publish tree always lives at ``<root>/dense_hot_export``, where
    ``<root>`` is the ``ONLINE_DENSE_EXPORT_DIR`` serving root when set, else
    the training ``model_dir``. The inference processor reads from the
    ``dense_hot_export`` leaf, so decoupling the serving root from
    ``model_dir`` (which may be remote or hold checkpoints) keeps the layout
    identical in both cases. The raw, pre-abspath value is returned so callers
    can detect fsspec-URL remotes before ``os.path.abspath`` mangles them.
    """
    root = os.environ.get("ONLINE_DENSE_EXPORT_DIR") or model_dir
    return os.path.join(root, "dense_hot_export")


def _is_remote_path(path: str) -> bool:
    """Whether path has an fsspec protocol such as oss:// or dfs://."""
    from fsspec.core import split_protocol

    return split_protocol(path)[0] is not None


class OnlineDenseExportManager:
    """In-process online-learning dense model export.

    Rank zero builds the serving dense graph once at construction time. On
    each trigger (the train loop's delta-dump boundary decision) all ranks
    gather the DMP model's dense weights in memory -- scoped to exactly the
    state keys the dense graph carries, so sparse / dynamicemb / MCH state is
    never materialized -- and rank zero hot-swaps them into the resident
    graph from a background thread. The finished version is held as "ready"
    and ``current.json`` flips only once ``poll_publish`` observes a
    cross-rank sparse upload watermark covering its pairing step, so serving
    never sees dense ahead of sparse.

    Every collective the manager enters (group creation, the startup
    key-list broadcast, the per-export weight gather, the per-step publish
    poll and its flip-time probe gather) is called identically on all ranks:
    the export trigger is the rank-synchronized dump decision passed in as
    ``force``, and ``poll_publish`` is called unconditionally on every step.
    Like the pipeline's own per-step collectives this assumes lockstep
    stepping; a rank that stops stepping early (ragged dataloader exhaustion
    with ``check_all_workers_data_status=False``) desyncs the poll.
    """

    def __init__(
        self,
        model_dir: str,
        pipeline_config_path: str,
        model: nn.Module,
        publish_interval_minutes: Optional[int] = None,
    ) -> None:
        self._enabled = _online_dense_export_enabled()
        self._rank = int(os.environ.get("RANK", 0))
        self._cond = Condition()
        self._pending: Optional[Dict[str, Any]] = None
        self._drain_event = Event()
        self._worker: Optional[Thread] = None
        self._finalizer: Optional[weakref.finalize] = None
        self._last_version = ""
        self._export_timeout = float(
            os.environ.get("ONLINE_DENSE_EXPORT_TIMEOUT", "3600")
        )
        # Covers an in-flight plus one pending task timeout during close() drain.
        self._close_timeout = 2 * self._export_timeout + 120.0
        self._publish_interval_minutes = publish_interval_minutes
        self._last_export_step = -1
        self._group: Optional[dist.ProcessGroup] = None
        # rank-zero publish gating state, guarded by _publish_lock: the newest
        # ready-but-unpublished version, written by the export worker and
        # published by the training thread's poll_publish.
        self._publish_lock = Lock()
        self._ready: Optional[Dict[str, Any]] = None
        # (gm state key, DMP state_dict source key) pairs, sorted; identical
        # on all ranks after the startup broadcast. The gather iterates them
        # in this order on every rank so collectives stay in lockstep.
        self._state_pairs: List[Tuple[str, str]] = []
        # rank-zero resident export state, built once at construction
        self._twin_model: Optional[nn.Module] = None
        self._gm: Optional[torch.fx.GraphModule] = None
        self._full_graph: Optional[torch.fx.Graph] = None
        self._warmup_data: Optional[Dict[str, Any]] = None
        self._dense_graph_config: Optional[Dict[str, Any]] = None
        # FX-traced gm, traced once at construction and reused by every export
        # so the worker never fx-traces concurrent with training forwards.
        self._dense_model_traced: Optional[torch.fx.GraphModule] = None

        override = os.environ.get("ONLINE_DENSE_EXPORT_DIR")
        export_root = resolve_dense_export_root(model_dir)
        self._export_root = os.path.abspath(export_root)
        self._serving_root = os.path.abspath(override) if override else None
        self._model_dir = os.path.abspath(model_dir)
        self._pipeline_config_path = os.path.abspath(pipeline_config_path)

        if not self._enabled:
            return
        if not override:
            raise RuntimeError(
                "ONLINE_DENSE_EXPORT=1 requires ONLINE_DENSE_EXPORT_DIR to be set "
                "to the serving root the inference processor reads from; refusing "
                "to default the publish tree to the training model_dir."
            )
        if not acc_utils.use_distributed_embedding():
            raise RuntimeError(
                "ONLINE_DENSE_EXPORT=1 requires USE_DISTRIBUTED_EMBEDDING=1."
            )
        # fail fast on a misconfigured retention K instead of at prune time,
        # where the worker's exception guard would swallow it and silently
        # disable retention.
        _max_kept_versions()
        # The publish tree (os.rename / current.json) is local-FS only;
        # fsspec URLs break both. Check the actual export root --
        # <serving_root>/dense_hot_export -- so a local override decouples
        # the publish tree from a remote model_dir.
        for label, path in (
            ("export_root", export_root),
            ("pipeline_config_path", pipeline_config_path),
        ):
            if _is_remote_path(path):
                raise RuntimeError(
                    f"ONLINE_DENSE_EXPORT requires a local {label}, got remote: {path}"
                )
        if dist.is_initialized() and dist.get_world_size() > 1:
            # collective; ONLINE_DENSE_EXPORT is job-uniform so all ranks enter.
            # finalize_publish drains the export worker on rank zero (up to
            # _close_timeout) before its gather while the other ranks already
            # wait inside it, so the group timeout must cover that drain
            # instead of the 30-minute gloo default.
            self._group = cast(
                Optional[dist.ProcessGroup],
                dist.new_group(
                    backend="gloo",
                    timeout=datetime.timedelta(seconds=self._close_timeout),
                ),
            )

        state_pairs: List[Tuple[str, str]] = []
        if self._rank == 0:
            state_pairs = self._build_export_graph(model)
        if self._group is not None:
            pair_box: List[List[Tuple[str, str]]] = [state_pairs]
            dist.broadcast_object_list(pair_box, src=0, group=self._group)
            self._state_pairs = pair_box[0]
        else:
            self._state_pairs = state_pairs
        self._verify_state_pairs(model)

        if self._rank == 0:
            self._worker = Thread(
                target=self._worker_loop,
                name="online-dense-export",
                daemon=True,
            )
            self._worker.start()
            self._finalizer = weakref.finalize(
                self,
                type(self)._drain_worker,
                self._worker,
                self._cond,
                self._drain_event,
                self._close_timeout,
            )
            logger.info(
                "ONLINE_DENSE_EXPORT enabled; dense versions will be exported under %s",
                self._export_root,
            )

    @property
    def enabled(self) -> bool:
        """Whether ONLINE_DENSE_EXPORT is enabled for this process."""
        return self._enabled

    def _build_export_graph(self, model: nn.Module) -> List[Tuple[str, str]]:
        """Build the resident dense export graph once, before training starts.

        Runs inside a scoped ``INPUT_TILE=3`` / ``WORLD_SIZE=1`` env window:
        the export-side model has user-side twin modules the training process
        never builds, and INPUT_TILE is read at model construction and
        batch-parse time. WORLD_SIZE gives the warm-up dataloader the same
        single-rank view the full export runs under: the Parquet and ODPS
        readers run startup collectives (file-metadata all-gather, scan
        session broadcast) on the default process group when WORLD_SIZE > 1,
        and this rank-zero-only build has no peer to join, which would
        deadlock the collective. The training model and its dataloader
        workers are already constructed by the time this runs, so the window
        cannot affect them.

        Fails fast (before training) on any trace/script error via a dry-run
        finalize, and on any dense-graph state key with no gatherable source
        in the live model's state dict.

        Args:
            model: the live DMP training model, used to resolve and validate
                the dense graph's state keys against the real state_dict.

        Returns:
            Sorted (gm_key, dmp_source_key) pairs for the lockstep gather.
        """
        # lazy import: tzrec.main imports this module
        from tzrec.main import _create_features, _create_model
        from tzrec.models.match_model import MatchModel
        from tzrec.models.model import ScriptWrapper
        from tzrec.models.tdm import TDM

        device = torch.device("cpu")
        prev_input_tile = os.environ.get("INPUT_TILE")
        prev_world_size = os.environ.get("WORLD_SIZE")
        os.environ["INPUT_TILE"] = "3"
        os.environ["WORLD_SIZE"] = "1"
        try:
            pipeline_config = config_util.load_pipeline_config(
                self._pipeline_config_path
            )
            features = _create_features(
                list(pipeline_config.feature_configs), pipeline_config.data_config
            )
            twin_model = _create_model(
                pipeline_config.model_config,
                features,
                list(pipeline_config.data_config.label_fields),
                sampler_type=None,
            )
            if isinstance(twin_model, (MatchModel, TDM)):
                # The full export emits per-tower (MatchModel) or per-module
                # (TDM) artifacts; a single monolithic dense export cannot
                # mirror that layout, so a hot swap would load an
                # incompatible artifact.
                raise RuntimeError(
                    f"ONLINE_DENSE_EXPORT does not support "
                    f"{type(twin_model).__name__} models; use the full export "
                    "(export_model) instead."
                )
            twin_model = ScriptWrapper(twin_model)
            warmup_data = create_dense_export_warmup_data(
                pipeline_config, twin_model, device
            )
            gm, full_graph, dense_graph_config = build_dense_graph_module(
                twin_model, warmup_data, device
            )
            # Fail fast on trace/script errors instead of at the first export.
            # Tracing happens here on the main thread and the traced module is
            # reused by every export; fx-tracing on the worker would patch
            # nn.Module.__call__ process-wide and race training forwards.
            with tempfile.TemporaryDirectory(
                prefix="online_dense_export_dryrun_"
            ) as dry_run_dir:
                dense_model_traced = finalize_dense_export(
                    twin_model,
                    full_graph,
                    gm,
                    warmup_data,
                    device,
                    dry_run_dir,
                    dense_graph_config,
                )
        finally:
            if prev_input_tile is None:
                os.environ.pop("INPUT_TILE", None)
            else:
                os.environ["INPUT_TILE"] = prev_input_tile
            if prev_world_size is None:
                os.environ.pop("WORLD_SIZE", None)
            else:
                os.environ["WORLD_SIZE"] = prev_world_size

        source_keys = model.state_dict()
        pairs: List[Tuple[str, str]] = []
        missing: List[str] = []
        # gm keys match the DMP state_dict namespace directly (both wrappers
        # name the model `model`); user-side twin keys added by INPUT_TILE=3
        # fall back to their non-user sources, as on checkpoint restore.
        for gm_key in sorted(gm.state_dict().keys()):
            source = (
                gm_key
                if gm_key in source_keys
                else remap_input_tile_user_key(gm_key, source_keys)
            )
            if source not in source_keys:
                missing.append(gm_key)
                continue
            value = source_keys[source]
            if isinstance(value, ShardedTensor):
                raise RuntimeError(
                    f"ONLINE_DENSE_EXPORT cannot gather dense state [{gm_key}]: "
                    "dense graph must not carry sharded (sparse/embedding) state"
                )
            if not isinstance(value, torch.Tensor):
                raise RuntimeError(
                    f"ONLINE_DENSE_EXPORT cannot gather dense state [{gm_key}]: "
                    f"unsupported state type {type(value).__name__}"
                )
            pairs.append((gm_key, source))
        if missing:
            raise RuntimeError(
                "ONLINE_DENSE_EXPORT cannot gather "
                f"{len(missing)} dense model states from the live model: "
                + ", ".join(missing)
            )

        self._twin_model = twin_model
        self._gm = gm
        self._full_graph = full_graph
        self._warmup_data = warmup_data
        self._dense_graph_config = dense_graph_config
        self._dense_model_traced = dense_model_traced
        return pairs

    def _verify_state_pairs(self, model: nn.Module) -> None:
        """Fail fast if any gather source key is absent from this rank's model.

        Guards against rank-skewed state_dict structure; the pairs were
        resolved against rank zero's model (or broadcast from it).
        """
        source_keys = set(model.state_dict().keys())
        missing = [
            source for _, source in self._state_pairs if source not in source_keys
        ]
        if missing:
            raise RuntimeError(
                f"ONLINE_DENSE_EXPORT rank {self._rank} model is missing "
                f"{len(missing)} dense export source states: "
                + ", ".join(sorted(set(missing)))
            )

    def maybe_export(
        self,
        step: int,
        data_timestamp: float,
        model: nn.Module,
        force: bool = False,
        final: bool = False,
        dense_ema: Optional[DenseEMA] = None,
    ) -> None:
        """Export a dense version now if the delta-dump boundary fired.

        All ranks must call this in lockstep from the train loop: the trigger
        is the delta-dump boundary decision passed in as ``force``, which the
        dumper already synchronizes across ranks, so a firing decision enters
        the collective weight gather on every rank.

        Args:
            step: current global step.
            data_timestamp: this rank's consumed event-time (seconds), -1.0
                if none; recorded in the published manifest.
            model: the live DMP training model to gather weights from.
            force: the rank-synchronized delta-dump decision; True pairs a
                dense version with the sparse delta dumped at this step.
            final: force an export at train end (still subject to the
                per-step dedupe).
            dense_ema: Dense EMA state to use for exported parameters.
        """
        if not self._enabled:
            return
        if not (force or final) or step == self._last_export_step:
            return
        self._last_export_step = step
        self._gather_and_submit(step, data_timestamp, model, dense_ema)

    def poll_publish(
        self, sparse_state: Optional[Tuple[int, List[Dict[str, Any]]]]
    ) -> None:
        """Reconcile the sparse upload watermark and flip current.json if covered.

        The train loop calls this on every step on all ranks; it is the only
        publisher of ``current.json``. When enabled with world size > 1 the
        per-step cost is one gloo ``all_gather`` of a two-int64 tensor: each
        rank's completed upload step plus rank zero's ready-version step
        (-1 when nothing is ready). The pickled probe dicts are only
        object-gathered when that gather shows the watermark covers the
        ready version, i.e. once per flip rather than per step. Rank zero
        takes the minimum completed step across ranks as the global
        watermark and publishes the ready version with the concatenated
        probes (capped at 64).

        Args:
            sparse_state: this rank's ``(last completed upload step, probes)``
                snapshot, or None when unavailable (treated as ``(0, [])``).
        """
        if not self._enabled:
            return
        step, probes = sparse_state or (0, [])
        if self._group is None:
            watermark = step
        else:
            # Only rank zero's ready slot is ever set; its step travels in the
            # second slot so every rank takes the same probe-gather decision.
            with self._publish_lock:
                ready_step = self._ready["step"] if self._ready is not None else -1
            world_size = dist.get_world_size(self._group)
            local = torch.tensor([step, ready_step], dtype=torch.int64)
            gathered = [torch.empty(2, dtype=torch.int64) for _ in range(world_size)]
            dist.all_gather(gathered, local, group=self._group)
            watermark = min(int(state[0]) for state in gathered)
            ready_step = int(gathered[0][1])
            if ready_step < 0 or ready_step > watermark:
                return
            rank_probes: List[List[Dict[str, Any]]] = [[] for _ in range(world_size)]
            dist.all_gather_object(rank_probes, probes, group=self._group)
            probes = [probe for one_rank in rank_probes for probe in one_rank]
        if self._rank != 0:
            return
        with self._publish_lock:
            self._publish_locked(watermark, probes[:64])

    def finalize_publish(
        self, sparse_state: Optional[Tuple[int, List[Dict[str, Any]]]]
    ) -> None:
        """Drain the export worker and run one final gated publish.

        A collective: all ranks must call it, and only on the normal shutdown
        path after the delta uploads have drained -- never from exception
        paths, where missing ranks would hang the gather. Rank zero first
        drains the export worker via close() so the final dense version is
        recorded as ready (instant on other ranks, and idempotent for the
        finally-block close()); then all ranks gather the final upload state
        and rank zero flips current.json if the pairing is complete.

        Args:
            sparse_state: this rank's final ``(completed upload step, probes)``
                snapshot, or None when unavailable (treated as ``(0, [])``).
        """
        if not self._enabled:
            return
        self.close()
        self.poll_publish(sparse_state)

    def _gather_and_submit(
        self,
        step: int,
        data_timestamp: float,
        model: nn.Module,
        dense_ema: Optional[DenseEMA] = None,
    ) -> None:
        """Gather the dense graph's weights from the DMP model (all ranks).

        Scoped to exactly the state keys the resident dense graph carries
        (resolved at construction), so sparse / dynamicemb / MCH state is
        never materialized. Plain tensors are DDP-replicated and need no
        communication; DTensors are all-gathered on their mesh. Rank zero
        then copies the gathered weights into pinned CPU buffers (a single
        copy-stream sync) and enqueues the snapshot on the latest-wins
        worker queue.
        """
        is_rank_zero = self._rank == 0
        gathered: Dict[str, torch.Tensor] = {}
        if self._state_pairs:
            source_state = model.state_dict()
            ema_state = dense_ema.state_dict() if dense_ema is not None else {}
            for gm_key, source_key in self._state_pairs:
                value = ema_state.get(source_key, source_state[source_key])
                if isinstance(value, DTensor):
                    # collective on the DTensor's mesh; all ranks participate
                    tensor = value.full_tensor()
                elif isinstance(value, torch.Tensor):
                    tensor = value
                else:
                    raise RuntimeError(
                        f"ONLINE_DENSE_EXPORT cannot gather dense state "
                        f"[{gm_key}]: unsupported state type "
                        f"{type(value).__name__}"
                    )
                if is_rank_zero:
                    gathered[gm_key] = tensor
        if not is_rank_zero:
            return
        self._enqueue(step, data_timestamp, self._copy_snapshot_to_cpu(gathered))

    def _copy_snapshot_to_cpu(
        self, gathered: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Copy gathered weights to pinned CPU buffers with a single sync.

        A per-tensor ``.cpu()`` synchronizes the training thread for every
        CUDA weight, adding per-weight step latency on large models. Instead
        each CUDA tensor is issued a non-blocking copy into a pinned buffer
        on a dedicated stream, and that stream is synchronized once before
        the snapshot is handed to the worker. CUDA stream synchronization --
        not a distributed barrier -- is what lets the worker read completed
        copies; tensors already on CPU are passed through detached.
        """
        if not any(tensor.is_cuda for tensor in gathered.values()):
            return {key: tensor.detach().cpu() for key, tensor in gathered.items()}
        copy_stream = torch.cuda.Stream()
        # The gather ran on the current stream; the copy stream must wait on
        # it so the non-blocking copies read fully-written source tensors.
        copy_stream.wait_stream(torch.cuda.current_stream())
        snapshot: Dict[str, torch.Tensor] = {}
        with torch.cuda.stream(copy_stream):
            for key, tensor in gathered.items():
                if tensor.is_cuda:
                    buffer = torch.empty(
                        tensor.shape, dtype=tensor.dtype, pin_memory=True
                    )
                    buffer.copy_(tensor.detach(), non_blocking=True)
                    snapshot[key] = buffer
                else:
                    snapshot[key] = tensor.detach().cpu()
        copy_stream.synchronize()
        return snapshot

    def _enqueue(
        self, step: int, data_timestamp: float, snapshot: Dict[str, torch.Tensor]
    ) -> None:
        """Queue a dense export task; a not-yet-started task is superseded.

        Latest-wins: online serving only consumes the freshest dense version,
        so a backlog is collapsed to the newest snapshot instead of pinning
        worker time (and its memory) per queued task.
        """
        version = _make_monotonic_version(self._last_version)
        self._last_version = version
        superseded: Optional[str] = None
        with self._cond:
            if self._drain_event.is_set():
                logger.warning("online dense export draining; skip step %s", step)
                return
            if self._pending is not None:
                superseded = self._pending["version"]
            self._pending = {
                "step": step,
                "data_timestamp": data_timestamp,
                "version": version,
                "snapshot": snapshot,
            }
            self._cond.notify()
        if superseded is not None:
            logger.info(
                "online dense export version %s superseded by %s before it started",
                superseded,
                version,
            )

    def close(self) -> None:
        """Wait for in-flight and pending dense export tasks to finish.

        Detach the finalizer only after the worker actually stops, so a worker
        that outlives the close timeout keeps the atexit drain backstop
        instead of leaking a live publisher.
        """
        if self._worker is None:
            return
        self._drain_event.set()
        with self._cond:
            self._cond.notify_all()
        self._worker.join(timeout=self._close_timeout)
        if self._worker.is_alive():
            logger.warning(
                "online dense export worker did not finish within %ss; "
                "leaving finalizer attached as a drain backstop",
                self._close_timeout,
            )
        elif self._finalizer is not None:
            self._finalizer.detach()

    @staticmethod
    def _drain_worker(
        worker: Thread,
        cond: Condition,
        drain_event: Event,
        close_timeout: float,
    ) -> None:
        """Drain the export worker if close() was never called.

        Registered via weakref.finalize so that if training raises before
        close() (the manager local goes out of scope), the worker is still
        stopped instead of leaking as a daemon thread with an in-flight
        publish that could advance current.json unattended.
        """
        drain_event.set()
        with cond:
            cond.notify_all()
        worker.join(timeout=close_timeout)

    def _worker_loop(self) -> None:
        while True:
            with self._cond:
                while self._pending is None and not self._drain_event.is_set():
                    self._cond.wait()
                if self._pending is None:
                    # draining and nothing left to run
                    return
                task = self._pending
                self._pending = None
            try:
                self._run_task(task)
            except Exception:
                # Keep the worker alive across unexpected task failures (e.g.
                # OSError from makedirs/open); without this guard a single
                # transient I/O error would permanently disable exports.
                logger.exception("online dense export task failed; continuing")

    def _run_task(self, task: Dict[str, Any]) -> None:
        """Load the snapshot into the resident graph, script it, mark it ready.

        Builds the immutable version directory (READY marker + atomic
        rename) but does not flip current.json: the finished version is
        recorded as the newest ready one and the training thread's next
        ``poll_publish`` publishes it once the sparse upload watermark covers
        its step. A previously ready but still unpublished version is
        superseded and its directory removed, so unpublished builds neither
        pile up while the watermark lags nor eat into the newest-K retention
        quota of published versions.
        """
        version = task["version"]
        versions_root = os.path.join(self._export_root, VERSIONS_DIR)
        version_dir = os.path.join(versions_root, version)
        tmp_dir = f"{version_dir}.tmp.{os.getpid()}"
        device = torch.device("cpu")
        start_time = time.monotonic()
        logger.info(
            "start online dense export version %s (step %s)", version, task["step"]
        )
        try:
            if os.path.exists(version_dir):
                raise RuntimeError(f"dense version already exists: {version_dir}")
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)
            os.makedirs(tmp_dir, exist_ok=True)
            assert self._gm is not None
            assert self._twin_model is not None
            assert self._full_graph is not None
            assert self._warmup_data is not None
            assert self._dense_graph_config is not None
            assert self._dense_model_traced is not None
            # load_state_dict copies in place and the traced module shares these
            # parameters, so it carries the reloaded weights without re-tracing.
            self._gm.load_state_dict(task["snapshot"])
            finalize_dense_export(
                self._twin_model,
                self._full_graph,
                self._gm,
                self._warmup_data,
                device,
                tmp_dir,
                self._dense_graph_config,
                dense_model_traced=self._dense_model_traced,
            )
            ready_path = os.path.join(tmp_dir, "READY")
            with open(ready_path, "w") as f:
                f.write(_utc_now())
                f.write("\n")
            os.rename(tmp_dir, version_dir)
        except BaseException:
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)
            raise

        with self._publish_lock:
            superseded = self._ready
            if superseded is not None:
                # Never named by current.json, so nothing can reference it.
                superseded_dir = os.path.join(versions_root, superseded["version"])
                logger.info(
                    "online dense export version %s (step %s) superseded by %s "
                    "before publish; removing %s",
                    superseded["version"],
                    superseded["step"],
                    version,
                    superseded_dir,
                )
                try:
                    shutil.rmtree(superseded_dir)
                except OSError as e:
                    logger.warning(
                        "failed to remove superseded version %s: %s", superseded_dir, e
                    )
            self._ready = {
                "step": task["step"],
                "version": version,
                "data_timestamp": task["data_timestamp"],
            }
        elapsed = time.monotonic() - start_time
        if elapsed > self._export_timeout:
            logger.warning(
                "online dense export version %s took %.1fs, exceeding "
                "ONLINE_DENSE_EXPORT_TIMEOUT=%.1fs",
                version,
                elapsed,
                self._export_timeout,
            )
        logger.info(
            "built online dense export version %s at %s (%.1fs)",
            version,
            version_dir,
            elapsed,
        )

    def _publish_locked(
        self, sparse_watermark: int, sparse_probes: List[Dict[str, Any]]
    ) -> None:
        """Flip current.json if the sparse watermark covers the ready version.

        Caller must hold ``_publish_lock``. Publishes the newest ready dense
        version only when its pairing step is <= the cross-rank sparse upload
        watermark, so the flipped manifest always satisfies
        ``sparse_step >= checkpoint_step`` and sparse-ahead-of-dense stays the
        only allowed skew. The ready slot is cleared as soon as the pointer
        has flipped, before the best-effort prune: a prune failure must not
        leave the published version marked ready, or the next build would
        "supersede" it and remove the directory current.json points at.

        Args:
            sparse_watermark: minimum completed upload step across ranks.
            sparse_probes: gathered probe rows to record in the manifest.
        """
        ready = self._ready
        if ready is None or ready["step"] > sparse_watermark:
            return
        payload: Dict[str, Any] = {
            "version": ready["version"],
            "checkpoint_step": ready["step"],
            "sparse_step": sparse_watermark,
            "data_timestamp": ready["data_timestamp"],
            "created_at": _utc_now(),
            "sparse_probes": sparse_probes,
        }
        if self._publish_interval_minutes is not None:
            payload["publish_interval_minutes"] = self._publish_interval_minutes
        # Keep the service-facing pointer beside the immutable dense versions.
        _publish_current(os.path.join(self._export_root, CURRENT_JSON), payload)
        self._ready = None
        logger.info(
            "published online dense export version %s (checkpoint_step %s, "
            "sparse_step %s)",
            ready["version"],
            ready["step"],
            sparse_watermark,
        )
        _prune_old_dense_versions(
            self._export_root, os.path.join(self._export_root, VERSIONS_DIR)
        )
