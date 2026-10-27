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

"""TorchEasyRec lifecycle and checkpoint coordination for RocksDB dynamicemb tables.

Layout under ``rocksdb_config.local_path``::

    <run_id>/rank_<r>/<mode>-<ms>-<pid>/   writable working DB of one process

``run_id`` is derived from the model_dir, so restarts of one job share it and
different jobs never collide. The working DB is never a checkpoint: saves
write RocksDB tables through ``DynamicEmbDump`` into the checkpoint's
``dynamicemb/`` files like every other dynamicemb table, and a resume always
rebuilds a fresh working DB from those files.

Checkpoints written before this format carry ``model.ckpt-<step>/rocksdb/``
manifests that point at native RocksDB checkpoints
(``<run_id>/rank_<r>/ckpt-<step>/``); they are still restored by cloning.
"""

import contextlib
import dataclasses
import glob
import hashlib
import json
import math
import os
import shutil
import time
from typing import Any, Dict, Iterator, List, Optional, cast

import torch.distributed as dist
from torch import nn

from tzrec.protos import feature_pb2, train_pb2
from tzrec.utils.logging_util import logger

_rocksdb_config: Optional[train_pb2.RocksDBConfig] = None
# dynamicemb tables AUTO placement may move to RocksDB, by table name.
_auto_candidates: Dict[str, feature_pb2.DynamicEmbedding] = {}
# Host memory a rank needs beyond the configured RocksDB budgets (table
# readers, iterators, pinned blocks, allocator slack).
_HOST_OVERHEAD_BYTES = 2 << 30
# DynamicEmb sizes GPU caches in buckets of this many rows.
CACHE_BUCKET_ROWS = 1024
MANIFEST_DIR = "rocksdb"
FORMAT_VERSION = 1


def set_rocksdb_config(config: Optional[train_pb2.RocksDBConfig]) -> None:
    """Set the RocksDB config used by tables planned and built afterwards.

    Constraint building and planning cannot see the train config, so it is
    handed over here (same pattern as ``set_auto_retain_evicted_keys``).
    """
    global _rocksdb_config
    _rocksdb_config = config
    _auto_candidates.clear()


def get_rocksdb_config() -> Optional[train_pb2.RocksDBConfig]:
    """Return the RocksDB config set for this process, if any."""
    return _rocksdb_config


def is_auto() -> bool:
    """Whether AUTO placement is enabled."""
    return (
        _rocksdb_config is not None
        and _rocksdb_config.placement == train_pb2.DynamicEmbPlacement.AUTO
    )


def _import_backend():
    try:
        import tzrec_rocksdb
        import tzrec_rocksdb.storage
    except ImportError as e:
        raise RuntimeError(
            "dynamicemb tables placed in RocksDB need the tzrec-rocksdb wheel "
            "from TorchEasyRec-addons (pip install tzrec_rocksdb-*.whl)."
        ) from e
    return tzrec_rocksdb


def backend_config():
    """Backend config shared by every RocksDB table factory of this run."""
    cfg = _rocksdb_config
    if cfg is None:
        raise ValueError(
            "dynamicemb `rocksdb {}` requires train_config.rocksdb_config."
        )
    return _import_backend().BackendConfig(
        local_path=cfg.local_path,
        block_cache_bytes=cfg.block_cache_bytes_per_rank,
        write_buffer_bytes=cfg.write_buffer_bytes_per_rank,
        staging_buffer_bytes=cfg.staging_buffer_bytes_per_rank,
        io_threads=cfg.io_threads,
        max_background_jobs=cfg.max_background_jobs,
    )


def is_rocksdb_options(options: Any) -> bool:
    """Whether DynamicEmbTableOptions are backed by a RocksDB factory."""
    storage = getattr(options, "external_storage", None)
    if storage is None:
        return False
    from tzrec_rocksdb.storage import RocksDBStorageFactory

    return isinstance(storage, RocksDBStorageFactory)


def auto_eligible(dynamicemb_cfg: feature_pb2.DynamicEmbedding) -> bool:
    """Whether AUTO placement may move a table without ``rocksdb {}`` to RocksDB.

    ``cache_load_factor`` pins a built-in placement, and CUSTOMIZED scores
    have no producer in TorchEasyRec.
    """
    return (
        is_auto()
        and not dynamicemb_cfg.HasField("cache_load_factor")
        and dynamicemb_cfg.score_strategy != "CUSTOMIZED"
    )


def register_auto_candidate(
    dynamicemb_cfg: feature_pb2.DynamicEmbedding, table_name: str
) -> None:
    """Record a table AUTO placement may move to RocksDB."""
    _auto_candidates[table_name] = dynamicemb_cfg


def auto_candidate(table_name: str) -> Optional[feature_pb2.DynamicEmbedding]:
    """The config of an AUTO candidate table, or None."""
    return _auto_candidates.get(table_name)


def table_factory(dynamicemb_cfg: feature_pb2.DynamicEmbedding, table_name: str):
    """Validate a RocksDB table config and return its storage factory.

    The namespace is the table name here; planning replaces it with the
    table's module-qualified name once that is known.
    """
    if dynamicemb_cfg.score_strategy == "CUSTOMIZED":
        raise ValueError(
            f"dynamicemb table {table_name} uses rocksdb, which does not support "
            "score_strategy CUSTOMIZED (TorchEasyRec has no score producer for it)."
        )
    if dynamicemb_cfg.HasField("cache_load_factor"):
        raise ValueError(
            f"dynamicemb table {table_name} uses rocksdb; its GPU cache comes from "
            "train_config.rocksdb_config.gpu_cache_value_bytes_per_rank, so "
            "cache_load_factor must be unset."
        )
    factory_cls = _import_backend().storage.RocksDBStorageFactory
    return factory_cls(backend_config(), table_name)


def with_namespace(options: Any, namespace: str) -> None:
    """Give a planned table's factory its persistent namespace."""
    options.external_storage = dataclasses.replace(
        options.external_storage, table_namespace=namespace
    )


def host_reserve_bytes() -> int:
    """Host memory a rank reserves once for its RocksDB store (0 if unused)."""
    cfg = _rocksdb_config
    if cfg is None:
        return 0
    return (
        cfg.block_cache_bytes_per_rank
        + cfg.write_buffer_bytes_per_rank
        + cfg.staging_buffer_bytes_per_rank
        + _HOST_OVERHEAD_BYTES
    )


def cache_ratio(value_bytes: List[int]) -> float:
    """GPU cache fraction shared by all RocksDB tables of a rank.

    The rank-wide value budget is split in proportion to each table's
    rank-local value bytes, which is one common ratio. RocksDB tables always
    run behind the GPU cache, so the budget must be positive; DynamicEmb
    ignores external storage when a group's budget covers the whole group
    (HBM_ONLY), so a budget at or above the total is rejected.
    """
    cfg = _rocksdb_config
    budget = cfg.gpu_cache_value_bytes_per_rank if cfg is not None else 0
    total = sum(value_bytes)
    if budget == 0:
        raise ValueError(
            "RocksDB tables run behind DynamicEmb's GPU cache; set "
            "rocksdb_config.gpu_cache_value_bytes_per_rank > 0."
        )
    if budget >= total:
        raise ValueError(
            f"rocksdb_config.gpu_cache_value_bytes_per_rank={budget} must be smaller "
            f"than the rank-local value bytes of all rocksdb tables ({total}); "
            "otherwise DynamicEmb keeps the tables in HBM and ignores RocksDB."
        )
    return budget / total


def cache_rows(rows: int, ratio: float) -> int:
    """GPU cache rows DynamicEmb allocates for a table at ``ratio``."""
    if ratio <= 0:
        return 0
    rows = max(1, int(rows * ratio))
    return int(math.ceil(rows / CACHE_BUCKET_ROWS) * CACHE_BUCKET_ROWS)


def _rank() -> int:
    return dist.get_rank() if dist.is_initialized() else int(os.environ.get("RANK", 0))


def _world_size() -> int:
    if dist.is_initialized():
        return dist.get_world_size()
    return int(os.environ.get("WORLD_SIZE", 1))


def rank_dir(model_dir: str, rank: int) -> str:
    """Rank directory of a run under rocksdb_config.local_path."""
    cfg = _rocksdb_config
    assert cfg is not None
    run_id = hashlib.sha1(os.path.realpath(model_dir).encode()).hexdigest()[:12]
    return os.path.join(cfg.local_path, run_id, f"rank_{rank}")


_session = None
_model_dir: Optional[str] = None
_mode = ""
_groups: List[Any] = []
_monitor = None
# Session metrics at the previous save: save_stats logs per-save deltas.
_saved_metrics: Dict[str, float] = {}


def open_session(model_dir: str, mode: str = "train"):
    """Register this rank's RocksDB session before model construction.

    The database itself opens on first use, so a run whose plan keeps every
    table in memory touches no disk. Every process start gets a fresh working
    directory: a working DB left by an earlier attempt is never reopened,
    because only a complete checkpoint may be resumed (and on network
    filesystems its lock cannot stop a stale writer). A training start
    removes earlier ``train-*`` directories of its rank; other modes remove
    only their own directory on close.

    Returns:
        the session, or None when the run has no RocksDB tables.
    """
    global _session, _model_dir, _mode
    if _rocksdb_config is None or not _has_rocksdb_backend():
        return None
    backend = _import_backend()
    root = rank_dir(model_dir, _rank())
    if mode == "train":
        for stale in glob.glob(os.path.join(root, "train-*")):
            logger.info(f"removing stale RocksDB working directory {stale}")
            shutil.rmtree(stale, ignore_errors=True)
    work = os.path.join(root, f"{mode}-{int(time.time() * 1000)}-{os.getpid()}")
    _session = backend.open_session(backend_config(), work)
    _model_dir, _mode = model_dir, mode
    return _session


def _has_rocksdb_backend() -> bool:
    """Whether the backend is importable; AUTO without it keeps tables in memory."""
    if not is_auto():
        return True
    try:
        _import_backend()
    except RuntimeError:
        logger.warning(
            "rocksdb_config placement AUTO without the tzrec-rocksdb wheel: every "
            "dynamicemb table stays in GPU/host memory."
        )
        return False
    return True


def session():
    """This rank's session, or None."""
    return _session


def close_session() -> None:
    """Close this rank's session (working data of non-train modes is removed)."""
    global _session
    if _session is None:
        return
    global _monitor
    if _monitor is not None:
        _monitor.close()
    work = _session.work_dir
    _session.close()
    if _mode != "train":
        shutil.rmtree(work, ignore_errors=True)
    _session = None
    _groups.clear()
    _saved_metrics.clear()
    _monitor = None


def rocksdb_modules(model: nn.Module) -> List[Any]:
    """Return the fused DynamicEmb modules of ``model`` stored in RocksDB."""
    if _rocksdb_config is None:
        return []
    from dynamicemb.dump_load import get_dynamic_emb_module
    from tzrec_rocksdb.storage import RocksDBStorage

    return [
        m for m in get_dynamic_emb_module(model) if isinstance(m.tables, RocksDBStorage)
    ]


def builtin_dynamicemb_tables(model: nn.Module) -> Dict[str, List[str]]:
    """``DynamicEmbLoad`` table selection without RocksDB tables.

    Used after a legacy native restore. DynamicEmb treats a collection missing
    from the dict as "all tables", so every collection is listed, with an
    empty list if needed.
    """
    from dynamicemb.dump_load import find_sharded_modules, get_dynamic_emb_module
    from tzrec_rocksdb.storage import RocksDBStorage

    selection: Dict[str, List[str]] = {}
    for path, _, collection in find_sharded_modules(model, ""):
        names = selection.setdefault(path, [])
        for m in get_dynamic_emb_module(collection):
            if not isinstance(m.tables, RocksDBStorage):
                names.extend(m.table_names)
    return selection


def check_fused_opt_args(dynamicemb_dir: str, model: nn.Module) -> None:
    """Reject a load whose fused groups would mix different optimizer state.

    A fused group shares one optimizer, whose scalar arguments (e.g. Adam's
    iteration count) come from each table's ``*_opt_args.json``. A plan can
    place tables in other groups than the checkpointing run (another backend
    or budget); members of one new group must then agree on those arguments.
    """
    from dynamicemb.dump_load import find_sharded_modules, get_dynamic_emb_module

    for path, _, collection in find_sharded_modules(model, ""):
        for m in cast(List[Any], get_dynamic_emb_module(collection)):
            keys = set(m.optimizer.get_opt_args())
            seen: Dict[str, Any] = {}
            for name in m.table_names:
                meta_path = os.path.join(dynamicemb_dir, path, f"{name}_opt_args.json")
                if not os.path.exists(meta_path):
                    continue
                with open(meta_path) as f:
                    meta = json.load(f)
                args = {k: meta[k] for k in sorted(keys) if k in meta}
                for other, other_args in seen.items():
                    if other_args != args:
                        raise ValueError(
                            f"dynamicemb tables {other} and {name} now share one "
                            f"optimizer but were saved with different optimizer "
                            f"arguments ({other_args} vs {args}); keep them in "
                            "separate groups or restart their optimizer state."
                        )
                seen[name] = args


def validate_model(model: nn.Module) -> Dict[str, Any]:
    """Check that constructed modules really use RocksDB; fail before training.

    Every table planned for RocksDB must sit in a CACHING_PS module whose
    storage is a RocksDBStorage with a complete namespace map. Also starts the
    disk headroom monitor of a training run.

    Returns:
        a fusion report: per group its tables, namespaces, layout, cache rows.
    """
    global _monitor
    from dynamicemb.dump_load import get_dynamic_emb_module
    from tzrec_rocksdb.storage import RocksDBStorage

    groups = []
    for m in cast(List[Any], get_dynamic_emb_module(model)):
        rocksdb_tables = [
            o.external_storage.table_namespace
            for o in m._dynamicemb_options
            if is_rocksdb_options(o)
        ]
        if not rocksdb_tables:
            continue
        storage = m.tables
        layout = m._storage_layout.name
        if not isinstance(storage, RocksDBStorage) or layout != "CACHING_PS":
            raise RuntimeError(
                f"dynamicemb tables {m.table_names} are planned for RocksDB but "
                f"were built with {type(storage).__name__} in layout {layout}; "
                "DynamicEmb ignores external storage in HBM_ONLY/HYBRID layouts."
            )
        if storage.namespaces != rocksdb_tables or len(
            set(storage.namespace_ids)
        ) != len(rocksdb_tables):
            raise RuntimeError(f"incomplete RocksDB namespace map for {m.table_names}")
        storage.bind_cache(m.cache)
        cache_rows = m.cache._state.capacity if m.cache is not None else 0
        groups.append(
            {
                "tables": list(m.table_names),
                "namespaces": dict(zip(storage.namespaces, storage.namespace_ids)),
                "layout": layout,
                "cache_rows": int(cache_rows),
                "dims": [storage.embedding_dim(t) for t in range(len(m.table_names))],
                "score_strategies": [
                    o.score_strategy.name for o in m._dynamicemb_options
                ],
                "capacities": [int(o.max_capacity) for o in m._dynamicemb_options],
            }
        )
    _groups[:] = rocksdb_modules(model)
    if groups and int(os.environ.get("LOCAL_RANK", 0)) == 0:
        logger.info(f"RocksDB dynamicemb groups: {json.dumps(groups)}")
    if groups and _mode == "train" and _rocksdb_config is not None:
        from tzrec.utils.rocksdb_capacity import CapacityMonitor

        assert _model_dir is not None
        _monitor = CapacityMonitor(
            _rocksdb_config, rank_dir(_model_dir, _rank()), _session
        )
    return {"groups": groups}


@contextlib.contextmanager
def writeback_scope(scope: str) -> Iterator[None]:
    """Attribute RocksDB writes to ``scope`` ("checkpoint" or "publish").

    GPU-cache flushes for a checkpoint or a publication leave the rows cached,
    so their writes must not create eviction-index entries.
    """
    if _session is None:
        yield
        return
    with _session.writeback_scope(scope):
        yield


def _save_need() -> int:
    """Disk bytes a save may add on this rank: overwritten cached rows."""
    need = 0
    for m in _groups:
        storage = m.tables
        rows = m.cache._state.capacity if m.cache is not None else 0
        need += rows * max(storage.record_bytes(t) for t in range(len(m.table_names)))
    return need


def step_boundary(step: int) -> Optional[str]:
    """Training step boundary: logical eviction, then the disk headroom check.

    Call after an update completed and before any checkpoint or publication
    of that step; no lookup is in flight then, so no cached or in-use row can
    be a victim.

    Returns:
        the headroom action of :class:`CapacityMonitor` (None if healthy).
    """
    if not _groups:
        return None
    evict()
    if _monitor is None:
        return None
    return _monitor.check(step, _save_need())


def evict() -> int:
    """Logically evict every RocksDB table above its capacity; returns the rows.

    Call only with no lookup in flight (a completed update, or the end of a pass).
    """
    t0 = time.perf_counter()
    evicted = 0
    for m in _groups:
        evicted += sum(m.tables.evict_to_capacity(m.cache).values())
    if evicted and _session is not None:
        _session.metrics.record("evict", time.perf_counter() - t0, rows=evicted)
    return evicted


def save_stats(seconds: float) -> None:
    """Log this rank's RocksDB costs of a checkpoint save."""
    if _session is None or not _groups:
        return
    stats = _session.stats()
    m = stats["metrics"]
    props = stats["properties"]
    tick = stats["tickers"]
    d = {
        k: m.get(k, 0.0) - _saved_metrics.get(k, 0.0)
        for k in (
            "checkpoint_insert_seconds",
            "checkpoint_insert_rows",
            "dump_seconds",
            "dump_rows",
            "dump_bytes",
        )
    }
    _saved_metrics.update(m)
    logger.info(
        f"RANK[{_rank()}] RocksDB checkpoint: {seconds:.2f}s total, cache flush "
        f"{d['checkpoint_insert_seconds']:.2f}s "
        f"({int(d['checkpoint_insert_rows'])} rows), logical export "
        f"{d['dump_seconds']:.2f}s ({int(d['dump_rows'])} rows, "
        f"{d['dump_bytes'] / (1 << 30):.2f}GiB written); cumulative "
        f"train writeback {int(m.get('train_insert_rows', 0))} rows, publish "
        f"writeback {int(m.get('publish_insert_rows', 0))} rows, evicted "
        f"{int(m.get('evict_rows', 0))} rows, index {stats['index']}; RocksDB "
        f"cumulative flush {tick.get('rocksdb.flush.write.bytes', 0)} bytes, "
        f"compaction {tick.get('rocksdb.compact.write.bytes', 0)} bytes written, "
        f"stalls {tick.get('rocksdb.stall.micros', 0) / 1e6:.1f}s; after "
        f"release: live SST {props['rocksdb.live-sst-files-size']}, all SST "
        f"{props['rocksdb.total-sst-files-size']}, obsolete SST "
        f"{props['rocksdb.obsolete-sst-files-size']}, snapshots "
        f"{props['rocksdb.num-snapshots']}"
    )
    prune_native_checkpoints()


def _dynamicemb_version() -> str:
    from importlib.metadata import version

    return version("dynamicemb")


def has_native_checkpoint(ckpt_dir: str) -> bool:
    """Whether a checkpoint carries legacy native RocksDB manifests."""
    return os.path.isdir(os.path.join(ckpt_dir, MANIFEST_DIR))


def _validate_groups(
    manifest: Dict[str, Any], groups: List[Any]
) -> List[Dict[str, Any]]:
    """Match each RocksDB module to its checkpoint state.

    A training checkpoint must keep its fusion membership (order may change).
    An imported checkpoint records one entry per table, since the importer
    cannot know the runtime fusion; its tables must then agree on optimizer
    arguments within each runtime group.
    """
    if manifest.get("imported"):
        by_ns = {s["namespaces"][0]: s for s in manifest["groups"]}
        matched: List[Dict[str, Any]] = []
        for m in groups:
            members = [by_ns.get(ns) for ns in m.tables.namespaces]
            if any(x is None for x in members):
                raise ValueError(f"imported checkpoint lacks tables of {m.table_names}")
            if any(x["opt_args"] != members[0]["opt_args"] for x in members):
                raise ValueError(
                    f"imported optimizer args differ within {m.table_names}"
                )
            matched.append(
                {
                    "namespaces": list(m.tables.namespaces),
                    "layouts": {k: v for x in members for k, v in x["layouts"].items()},
                    "dist_types": [x["dist_types"][0] for x in members],
                    "opt_args": members[0]["opt_args"],
                    "scores": {k: v for x in members for k, v in x["scores"].items()},
                    "counter": all(x.get("counter") for x in members),
                    "num_shards": members[0]["num_shards"],
                    "shard_index": members[0]["shard_index"],
                }
            )
        if sum(len(x["namespaces"]) for x in matched) != len(by_ns):
            raise ValueError("the model lacks RocksDB tables present in the checkpoint")
    else:
        by_members = {frozenset(s["namespaces"]): s for s in manifest["groups"]}
        matched: List[Dict[str, Any]] = []
        for m in groups:
            state = by_members.get(frozenset(m.tables.namespaces))
            if state is None:
                known = [s["namespaces"] for s in manifest["groups"]]
                raise ValueError(
                    f"RocksDB group {m.tables.namespaces} is not a fusion group of "
                    f"the checkpoint ({known}); changing fusion membership "
                    "requires a migration."
                )
            matched.append(state)
        if len(matched) != len(manifest["groups"]):
            raise ValueError("the model lacks RocksDB groups present in the checkpoint")
    for m, state in zip(groups, matched):
        storage = m.tables
        training = all(o.training for o in m._dynamicemb_options)
        for t, ns in enumerate(storage.namespaces):
            want, have = state["layouts"][ns], storage.table_layout(t)
            if not training:
                # Without an optimizer only the embedding part is read.
                want = {k: want[k] for k in ("dtype", "dim")}
                have = {k: have[k] for k in ("dtype", "dim")}
            if want != have:
                raise ValueError(
                    f"table {ns} layout {have} differs from checkpoint {want}"
                )
            dist_type = state["dist_types"][state["namespaces"].index(ns)]
            if m._dynamicemb_options[t].dist_type != dist_type:
                raise ValueError(f"key routing of {ns} changed ({dist_type})")
    return matched


def restore_checkpoint(ckpt_dir: str, model: nn.Module, load_optim: bool) -> None:
    """Resume RocksDB tables from a legacy native-checkpoint ``ckpt_dir``.

    Validates the manifest against the model, empties every RocksDB GPU cache
    (rows touched before restore must not be written back), swaps the session
    to a fresh writable clone of the immutable native checkpoint, re-resolves
    namespaces, and restores optimizer arguments, scores and counters. Native
    checkpoints predate the eviction index, so only NO_EVICTION tables can
    resume from them; re-save such a run once to move it to the logical format.
    """
    from tzrec.utils.checkpoint_util import CKPT_SUCCESS_FILENAME

    groups = rocksdb_modules(model)
    if not groups:
        return
    from tzrec_rocksdb.checkpoint import clone_checkpoint

    if not os.path.exists(os.path.join(ckpt_dir, CKPT_SUCCESS_FILENAME)):
        raise RuntimeError(
            f"{ckpt_dir} has no {CKPT_SUCCESS_FILENAME}; refusing to resume RocksDB "
            "tables from an incomplete checkpoint."
        )
    rank = _rank()
    with open(os.path.join(ckpt_dir, MANIFEST_DIR, f"rank_{rank}.json")) as f:
        manifest = json.load(f)
    if manifest["format_version"] != FORMAT_VERSION:
        raise ValueError(f"unsupported RocksDB manifest format in {ckpt_dir}")
    if manifest["world_size"] != _world_size() or manifest["rank"] != rank:
        raise ValueError(
            f"RocksDB checkpoint was written by rank {manifest['rank']} of "
            f"{manifest['world_size']}; resharding to {rank} of {_world_size()} is "
            "not supported."
        )
    states = _validate_groups(manifest, groups)
    for m in groups:
        evicting = [
            ns
            for ns, o in zip(m.tables.namespaces, m._dynamicemb_options)
            if o.score_strategy.name != "NO_EVICTION"
        ]
        if evicting:
            raise ValueError(
                f"legacy native checkpoint {ckpt_dir} has no eviction index for "
                f"{evicting}; resume it with score_strategy NO_EVICTION and save "
                "once to convert it to the logical format."
            )
    logger.info(f"restoring RocksDB tables from legacy native checkpoint {ckpt_dir}")
    if not manifest.get("exact_resume", True):
        logger.warning(
            f"{ckpt_dir} was imported without optimizer state; RocksDB tables "
            "start with reset optimizer state (not an exact resume)."
        )
    assert _session is not None and _model_dir is not None
    t0 = time.perf_counter()
    for m in groups:
        m.reset_cache_states()
    clone = os.path.join(
        rank_dir(_model_dir, rank), f"{_mode}-{int(time.time() * 1000)}-{os.getpid()}"
    )
    stats = clone_checkpoint(manifest["native_checkpoint"], clone)
    _session.swap_working_db(clone)
    for m, state in zip(groups, states):
        m.tables.refresh_namespaces()
        if load_optim:
            m.optimizer.set_opt_args(state["opt_args"])
        if getattr(m, "_scores", None) is not None:
            for name, score in state["scores"].items():
                if name in m._scores:
                    m._scores[name] = score
        counter = getattr(m, "_admission_counter", None)
        if counter is not None and state.get("counter"):
            for t, ns in enumerate(m.tables.namespaces):
                counter.load(
                    *_legacy_counter_paths(ckpt_dir, ns, rank),
                    t,
                    m._dynamicemb_options[t].dist_type,
                    shard_index=state["shard_index"],
                    num_shards=state["num_shards"],
                )
    logger.info(
        f"RANK[{rank}] restored RocksDB tables from {manifest['native_checkpoint']} "
        f"in {time.perf_counter() - t0:.2f}s ({stats})"
    )


def _legacy_counter_paths(ckpt_dir: str, namespace: str, rank: int) -> List[str]:
    base = os.path.join(ckpt_dir, MANIFEST_DIR, "counters")
    return [
        os.path.join(base, f"{namespace}.keys.rank_{rank}"),
        os.path.join(base, f"{namespace}.frequencies.rank_{rank}"),
    ]


def reset_for_import(model: nn.Module) -> None:
    """Start RocksDB tables from a fresh empty DB before ``DynamicEmbLoad``.

    Empties the GPU caches first: rows touched before the restore (the
    optimizer warm-up step) must not be written back into the new DB.
    """
    groups = rocksdb_modules(model)
    if not groups:
        return
    assert _session is not None and _model_dir is not None
    for m in groups:
        m.reset_cache_states()
    _session.swap_working_db(
        os.path.join(
            rank_dir(_model_dir, _rank()),
            f"{_mode}-{int(time.time() * 1000)}-{os.getpid()}",
        )
    )
    for m in groups:
        m.tables.refresh_namespaces()
    logger.info(
        f"RANK[{_rank()}] loading RocksDB tables into fresh working DB "
        f"{_session.work_dir}"
    )


def prune_native_checkpoints() -> None:
    """Delete this rank's legacy native checkpoints whose model.ckpt dir is gone.

    Retention (keep_checkpoint_max, best and protected checkpoints) is decided
    for ``model.ckpt-*`` on rank 0; every rank follows it for its own disk.
    """
    if _session is None or _model_dir is None:
        return
    for native in glob.glob(os.path.join(rank_dir(_model_dir, _rank()), "ckpt-*")):
        step = native.rsplit("-", 1)[-1]
        if not os.path.isdir(os.path.join(_model_dir, f"model.ckpt-{step}")):
            shutil.rmtree(native, ignore_errors=True)


def remove_incomplete_checkpoints(model_dir: str, after_step: int) -> None:
    """Delete checkpoints newer than the resumed one that lack ``_SUCCESS``.

    They can never be selected, the resumed run re-saves those steps (which
    needs an empty directory), and on an OSS mount a rename would copy every
    object. The deletion is logged with its size. Rank 0 only.
    """
    from tzrec.utils.checkpoint_util import CKPT_SUCCESS_FILENAME

    if _rank() != 0:
        return
    for ckpt in glob.glob(os.path.join(model_dir, "model.ckpt-*")):
        try:
            step = int(ckpt.rsplit("-", 1)[-1])
        except ValueError:
            continue
        if step > after_step and not os.path.exists(
            os.path.join(ckpt, CKPT_SUCCESS_FILENAME)
        ):
            files = [os.path.join(d, f) for d, _, fs in os.walk(ckpt) for f in fs]
            size = sum(os.path.getsize(f) for f in files if os.path.isfile(f))
            logger.warning(
                f"deleting incomplete checkpoint {ckpt} ({len(files)} files, "
                f"{size} bytes): it has no {CKPT_SUCCESS_FILENAME}"
            )
            shutil.rmtree(ckpt)


def start(train_config: train_pb2.TrainConfig, model_dir: str, mode: str) -> bool:
    """Configure RocksDB tables for this process and register its session.

    Must run before features build their parameter constraints and before
    the model is constructed. Returns whether RocksDB may be used.
    """
    set_rocksdb_config(
        train_config.rocksdb_config if train_config.HasField("rocksdb_config") else None
    )
    return open_session(model_dir, mode) is not None


def stop() -> None:
    """Close this process's RocksDB session, if any."""
    close_session()
