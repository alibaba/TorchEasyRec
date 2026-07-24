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

import datetime
import json
import os
import shutil
import tempfile
import time
import weakref
from threading import Condition, Event, Thread
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed._tensor import DTensor

from tzrec.acc import utils as acc_utils
from tzrec.utils import config_util
from tzrec.utils.checkpoint_util import (
    quorum_event_time,
    remap_input_tile_user_key,
    should_save_on_timestamp,
)
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
    """Max published dense versions to retain (0 = keep all)."""
    return int(os.environ.get("ONLINE_DENSE_EXPORT_KEEP_VERSIONS", "3"))


def _prune_old_dense_versions(export_root: str, versions_root: str) -> None:
    """Best-effort retention: keep the newest K versions, sweep stale tmp artifacts.

    Serving reads current.json (the newest pointer) and needs the previous
    version for an atomic swap, so K defaults to 3. The version current.json
    points at is always spared even when it sorts outside the newest K: an
    explicit --version or clock rollback after a restart can publish an older
    timestamp, and deleting it would leave the serving pointer referencing a
    missing directory. Stale ``*.tmp.<pid>`` dirs and current.json.tmp.<pid>
    files left by crashed exports are swept so they don't accumulate under the
    serving-facing tree.
    """
    max_versions = _max_kept_versions()
    for base in (versions_root, export_root):
        try:
            entries = os.listdir(base)
        except FileNotFoundError:
            continue
        for name in entries:
            if ".tmp." not in name and not name.endswith(".tmp"):
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
        if os.path.isdir(os.path.join(versions_root, name))
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
    each trigger (independent of checkpoint saving) all ranks gather the DMP
    model's dense weights in memory -- scoped to exactly the state keys the
    dense graph carries, so sparse / dynamicemb / MCH state is never
    materialized -- and rank zero hot-swaps them into the resident graph from
    a background thread, then publishes a version. No checkpoint write is
    needed per export.

    Every collective the manager enters (group creation, the per-step
    event-time reconcile, the startup key-list broadcast, the per-export
    gather) is called identically on all ranks: the trigger decision is a
    deterministic function of job-uniform env config, the step counter and
    the quorum-reconciled event-time, mirroring CheckpointManager.maybe_save.
    """

    def __init__(
        self,
        model_dir: str,
        pipeline_config_path: str,
        model: nn.Module,
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
        # trigger config; identical on all ranks (env is job-uniform), so the
        # trigger decision needs no consensus beyond the event-time reconcile
        self._export_steps = int(os.environ.get("ONLINE_DENSE_EXPORT_STEPS", "0"))
        self._ts_interval = int(os.environ.get("ONLINE_DENSE_EXPORT_INTERVAL", "0"))
        self._ts_quorum = float(os.environ.get("ONLINE_DENSE_EXPORT_QUORUM", "0.5"))
        self._last_export_step = -1
        self._last_data_ts: Optional[float] = None
        self._group: Optional[dist.ProcessGroup] = None
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
        if self._export_steps <= 0 and self._ts_interval <= 0:
            raise RuntimeError(
                "ONLINE_DENSE_EXPORT=1 requires ONLINE_DENSE_EXPORT_STEPS or "
                "ONLINE_DENSE_EXPORT_INTERVAL to configure the export cadence."
            )
        if self._ts_interval > 0 and not (0.0 < self._ts_quorum <= 1.0):
            raise RuntimeError(
                f"ONLINE_DENSE_EXPORT_QUORUM must be in (0, 1], got {self._ts_quorum}."
            )
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
            # collective; ONLINE_DENSE_EXPORT is job-uniform so all ranks enter
            self._group = dist.new_group(backend="gloo")

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

    def _build_export_graph(self, model: nn.Module) -> List[Tuple[str, str]]:
        """Build the resident dense export graph once, before training starts.

        Runs inside a scoped ``INPUT_TILE=3`` env window: the export-side
        model has user-side twin modules the training process never builds,
        and INPUT_TILE is read at model construction and batch-parse time.
        The training model and its dataloader workers are already constructed
        by the time this runs, so the window cannot affect them.

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
        os.environ["INPUT_TILE"] = "3"
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
            with tempfile.TemporaryDirectory(
                prefix="online_dense_export_dryrun_"
            ) as dry_run_dir:
                finalize_dense_export(
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
            if not isinstance(value, (torch.Tensor, ShardedTensor)):
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

    def _reconcile_event_time(self, data_timestamp: float) -> Optional[float]:
        """Quorum-reconcile this rank's event-time across workers.

        A collective over the exporter's gloo group; every rank calls it on
        the same steps (on every maybe_export call when the event-time
        trigger is configured). Workers without a timestamp contribute the
        -1.0 sentinel, which sorts low and counts as "not past".
        """
        if self._ts_interval <= 0:
            return None
        if self._group is not None:
            worker_ts: List[float] = [0.0] * dist.get_world_size(self._group)
            dist.all_gather_object(worker_ts, data_timestamp, group=self._group)
        else:
            worker_ts = [data_timestamp]
        data_ts = quorum_event_time(worker_ts, self._ts_quorum)
        return data_ts if data_ts is not None and data_ts >= 0 else None

    def maybe_export(
        self,
        step: int,
        data_timestamp: float,
        model: nn.Module,
        final: bool = False,
    ) -> None:
        """Export a dense version now if a configured trigger fires.

        All ranks must call this in lockstep from the train loop (it is
        invoked unconditionally, not gated on a checkpoint save): the trigger
        decision is deterministic and identical across ranks, and a firing
        decision enters the collective weight gather on every rank.

        Args:
            step: current global step.
            data_timestamp: this rank's consumed event-time (seconds), -1.0
                if none; quorum-reconciled across workers for the event-time
                trigger.
            model: the live DMP training model to gather weights from.
            final: force an export (still subject to the per-step dedupe),
                e.g. at train end.
        """
        if not self._enabled:
            return
        want = final
        if self._export_steps > 0 and step > 0 and step % self._export_steps == 0:
            want = True
        data_ts = self._reconcile_event_time(data_timestamp)
        if data_ts is not None:
            if self._last_data_ts is None:
                # first event-time seen: set the reference, do not export
                self._last_data_ts = data_ts
            elif should_save_on_timestamp(
                data_ts, self._last_data_ts, self._ts_interval, []
            ):
                want = True
        if not want or step == self._last_export_step:
            return
        self._last_export_step = step
        if data_ts is not None:
            # advance the watermark on every export
            self._last_data_ts = data_ts
        self._gather_and_submit(step, data_timestamp, model)

    def _gather_and_submit(
        self, step: int, data_timestamp: float, model: nn.Module
    ) -> None:
        """Gather the dense graph's weights from the DMP model (all ranks).

        Scoped to exactly the state keys the resident dense graph carries
        (resolved at construction), so sparse / dynamicemb / MCH state is
        never materialized. Plain tensors are DDP-replicated and need no
        communication; DTensors are all-gathered on their mesh;
        ShardedTensors are staged into a full-size CPU tensor and summed over
        the exporter's gloo group (position-based shards never overlap).
        Rank zero then enqueues the snapshot on the latest-wins worker queue.
        """
        is_rank_zero = self._rank == 0
        snapshot: Dict[str, torch.Tensor] = {}
        if self._state_pairs:
            source_state = model.state_dict()
            for gm_key, source_key in self._state_pairs:
                value = source_state[source_key]
                if isinstance(value, DTensor):
                    # collective on the DTensor's mesh; all ranks participate
                    gathered = value.full_tensor()
                elif isinstance(value, ShardedTensor):
                    gathered = self._gather_sharded_tensor(value)
                elif isinstance(value, torch.Tensor):
                    gathered = value
                else:
                    raise RuntimeError(
                        f"ONLINE_DENSE_EXPORT cannot gather dense state "
                        f"[{gm_key}]: unsupported state type "
                        f"{type(value).__name__}"
                    )
                if is_rank_zero:
                    snapshot[gm_key] = gathered.detach().cpu()
        if not is_rank_zero:
            return
        self._enqueue(step, data_timestamp, snapshot)

    def _gather_sharded_tensor(self, value: ShardedTensor) -> torch.Tensor:
        """Reconstruct a full CPU tensor from position-based ShardedTensor shards.

        Every rank stages its local shards into a zeroed full-size tensor at
        their global offsets, then the exporter's gloo group sums them: each
        element is written by exactly one rank, so the sum is the union.
        """
        metadata = value.metadata()
        gathered = torch.zeros(metadata.size, dtype=metadata.tensor_properties.dtype)
        for shard in value.local_shards():
            region = gathered
            for dim, (offset, size) in enumerate(
                zip(shard.metadata.shard_offsets, shard.metadata.shard_sizes)
            ):
                region = region.narrow(dim, offset, size)
            region.copy_(shard.tensor.detach().cpu())
        if self._group is not None:
            dist.all_reduce(gathered, group=self._group)
        return gathered

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
        """Load the snapshot into the resident graph, script it and publish."""
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
            self._gm.load_state_dict(task["snapshot"])
            finalize_dense_export(
                self._twin_model,
                self._full_graph,
                self._gm,
                self._warmup_data,
                device,
                tmp_dir,
                self._dense_graph_config,
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

        payload: Dict[str, Any] = {
            "version": version,
            "checkpoint_step": task["step"],
            "data_timestamp": task["data_timestamp"],
            "created_at": _utc_now(),
        }
        # Keep the service-facing pointer beside the immutable dense versions.
        _publish_current(os.path.join(self._export_root, CURRENT_JSON), payload)
        _prune_old_dense_versions(self._export_root, versions_root)
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
            "published online dense export version %s to %s (%.1fs)",
            version,
            version_dir,
            elapsed,
        )
