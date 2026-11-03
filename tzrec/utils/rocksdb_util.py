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

"""TorchEasyRec lifecycle of dynamicemb tables offloaded to RocksDB.

Layout under ``rocksdb_config.root_dir``::

    <run_id>/rank_<r>/<mode>-<ms>-<pid>/   working DB of one process

``run_id`` is derived from the model_dir, so restarts of one job share it and
different jobs never collide; ``rank_<r>`` separates ranks on a shared disk and
``<mode>-<ms>-<pid>`` concurrent processes. The working DB is never a
checkpoint: saves write RocksDB tables through ``DynamicEmbDump`` into the
checkpoint's ``dynamicemb/`` files like every other dynamicemb table, and a
restore always imports them into a fresh working DB.
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

from tzrec.protos import train_pb2
from tzrec.utils.logging_util import logger

# Host memory a rank needs beyond the planned RocksDB budgets (table readers,
# iterators, pinned blocks, allocator slack).
_HOST_OVERHEAD_BYTES = 2 << 30
# Planned host budgets that do not depend on the tables: the memtables, the
# pinned staging of one storage call (larger calls are split) and the data
# share of the block cache (other data blocks are left to the page cache).
WRITE_BUFFER_BYTES = 1 << 30
STAGING_BUFFER_BYTES = 256 << 20
BLOCK_CACHE_DATA_BYTES = 1 << 30
# DynamicEmb sizes GPU caches in buckets of this many rows, beside an overflow
# area of three buckets per table.
CACHE_BUCKET_ROWS = 1024
CACHE_OVERFLOW_ROWS = 3 * CACHE_BUCKET_ROWS
# GPU cache shares of its value bytes the planner offers a RocksDB table.
CACHE_RATIOS = (0.0001, 0.001, 0.01) + tuple(i / 10 for i in range(1, 10))


def fixed_host_bytes() -> int:
    """Host memory a rank spends on RocksDB once any table is offloaded."""
    return (
        WRITE_BUFFER_BYTES
        + STAGING_BUFFER_BYTES
        + BLOCK_CACHE_DATA_BYTES
        + _HOST_OVERHEAD_BYTES
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


def is_rocksdb_options(options: Any) -> bool:
    """Whether DynamicEmbTableOptions are backed by a RocksDB factory."""
    storage = getattr(options, "external_storage", None)
    if storage is None:
        return False
    from tzrec_rocksdb.storage import RocksDBStorageFactory

    return isinstance(storage, RocksDBStorageFactory)


def cache_rows(rows: int, ratio: float) -> int:
    """GPU cache rows DynamicEmb allocates for a table at ``ratio``."""
    if ratio <= 0:
        return 0
    rows = max(1, int(rows * ratio))
    return int(math.ceil(rows / CACHE_BUCKET_ROWS) * CACHE_BUCKET_ROWS)


def _rank() -> int:
    return dist.get_rank() if dist.is_initialized() else int(os.environ.get("RANK", 0))


def rocksdb_modules(model: nn.Module) -> List[Any]:
    """Return the fused DynamicEmb modules of ``model`` stored in RocksDB."""
    from dynamicemb.dump_load import get_dynamic_emb_module
    from tzrec_rocksdb.storage import RocksDBStorage

    return [
        m for m in get_dynamic_emb_module(model) if isinstance(m.tables, RocksDBStorage)
    ]


class RocksDBRuntime:
    """RocksDB state of one train/eval/export/predict process on this rank.

    Owns the working DB session, opened once the sharding plan is known
    (:meth:`bind`), the fused RocksDB groups of the built model and, in
    training, the disk headroom monitor (:meth:`attach`). Created by
    :func:`open_runtime`, which closes it however the run ends.

    Args:
        config (RocksDBConfig): the job's RocksDB config.
        model_dir (str): the job's model_dir; it names the run's directory.
        mode (str): "train", "eval", "export" or "predict".
    """

    def __init__(
        self, config: train_pb2.RocksDBConfig, model_dir: str, mode: str
    ) -> None:
        self.config = config
        self.mode = mode
        run_id = hashlib.sha1(os.path.realpath(model_dir).encode()).hexdigest()[:12]
        self.rank_dir = os.path.join(config.root_dir, run_id, f"rank_{_rank()}")
        if mode == "train":
            # Only a complete checkpoint is ever resumed, so a working DB left
            # by an earlier attempt of this rank is stale.
            for stale in glob.glob(os.path.join(self.rank_dir, "train-*")):
                logger.info(f"removing stale RocksDB working directory {stale}")
                shutil.rmtree(stale, ignore_errors=True)
        self.session: Any = None
        self.groups: List[Any] = []
        self._monitor: Any = None
        # Session metrics at the previous save: save_stats logs per-save deltas.
        self._saved_metrics: Dict[str, float] = {}

    def _new_work_dir(self) -> str:
        return os.path.join(
            self.rank_dir, f"{self.mode}-{int(time.time() * 1000)}-{os.getpid()}"
        )

    def bind(self, plan: Any) -> None:
        """Open this rank's working DB and bind the plan's RocksDB tables to it.

        The plan crosses ranks without sessions; every rank opens its own with
        the backend config the planner chose, before the model is sharded.
        """
        options = [
            o
            for module_plan in plan.plan.values()
            for sharding in module_plan.values()
            if is_rocksdb_options(o := getattr(sharding, "dynamicemb_options", None))
        ]
        if not options:
            return
        config = options[0].external_storage.backend_config
        if self.session is None:
            backend = _import_backend()
            self.session = backend.RocksDBSession(config, self._new_work_dir())
        elif self.session.config != config:
            raise RuntimeError("a second plan changed this run's RocksDB config")
        for o in options:
            o.external_storage = dataclasses.replace(
                o.external_storage, session=self.session
            )

    def attach(self, model: nn.Module) -> None:
        """Check the built RocksDB groups and start a training run's monitor.

        Every table planned for RocksDB must sit in a CACHING_PS module whose
        storage is a RocksDBStorage with a complete namespace map.
        """
        from dynamicemb.dump_load import get_dynamic_emb_module
        from tzrec_rocksdb.storage import RocksDBStorage

        report = []
        for m in cast(List[Any], get_dynamic_emb_module(model)):
            tables = [
                o.external_storage.table_namespace
                for o in m._dynamicemb_options
                if is_rocksdb_options(o)
            ]
            if not tables:
                continue
            storage = m.tables
            layout = m._storage_layout.name
            if not isinstance(storage, RocksDBStorage) or layout != "CACHING_PS":
                raise RuntimeError(
                    f"dynamicemb tables {m.table_names} are planned for RocksDB but "
                    f"were built with {type(storage).__name__} in layout {layout}; "
                    "DynamicEmb ignores external storage in HBM_ONLY/HYBRID layouts."
                )
            if storage.namespaces != tables or len(set(storage.namespace_ids)) != len(
                tables
            ):
                raise RuntimeError(
                    f"incomplete RocksDB namespace map for {m.table_names}"
                )
            storage.bind_cache(m.cache)
            self.groups.append(m)
            report.append(
                {
                    "tables": list(m.table_names),
                    "namespaces": dict(zip(storage.namespaces, storage.namespace_ids)),
                    "cache_rows": list(
                        m.cache._state.key_index_map.per_table_capacity_
                    ),
                    "dims": [storage.embedding_dim(t) for t in range(len(tables))],
                    "score_strategies": [
                        o.score_strategy.name for o in m._dynamicemb_options
                    ],
                    "capacities": [int(o.max_capacity) for o in m._dynamicemb_options],
                }
            )
        if report and int(os.environ.get("LOCAL_RANK", 0)) == 0:
            logger.info(f"RocksDB dynamicemb groups: {json.dumps(report)}")
        if self.groups and self.mode == "train":
            from tzrec.utils.rocksdb_capacity import CapacityMonitor

            self._monitor = CapacityMonitor(self.config, self.rank_dir, self.session)

    @contextlib.contextmanager
    def writeback_scope(self, scope: str) -> Iterator[None]:
        """Attribute RocksDB writes to ``scope`` ("checkpoint" or "publish").

        GPU-cache flushes for a checkpoint or a publication leave the rows
        cached, so their writes must not create eviction-index entries.
        """
        if self.session is None:
            yield
            return
        with self.session.writeback_scope(scope):
            yield

    def _save_need(self) -> int:
        """Disk bytes a save may add on this rank: overwritten cached rows."""
        need = 0
        for m in self.groups:
            storage = m.tables
            need += m.cache._state.capacity * max(
                storage.record_bytes(t) for t in range(len(m.table_names))
            )
        return need

    def step_boundary(self, step: int) -> Optional[str]:
        """Training step boundary: logical eviction, then the disk headroom check.

        Call after an update completed and before any checkpoint or publication
        of that step; no lookup is in flight then, so no cached or in-use row
        can be a victim.

        Returns:
            the headroom action of :class:`CapacityMonitor` (None if healthy).
        """
        if not self.groups:
            return None
        self.evict()
        if self._monitor is None:
            return None
        return self._monitor.check(step, self._save_need())

    def evict(self) -> int:
        """Evict every RocksDB table above its capacity; returns the rows.

        Call only with no lookup in flight (a completed update, or the end of
        a pass).
        """
        t0 = time.perf_counter()
        evicted = 0
        for m in self.groups:
            evicted += sum(m.tables.evict_to_capacity(m.cache).values())
        if evicted:
            self.session.metrics.record("evict", time.perf_counter() - t0, rows=evicted)
        return evicted

    def save_stats(self, seconds: float) -> None:
        """Log this rank's RocksDB costs of a checkpoint save."""
        if not self.groups:
            return
        stats = self.session.stats()
        m = stats["metrics"]
        props = stats["properties"]
        tick = stats["tickers"]
        d = {
            k: m.get(k, 0.0) - self._saved_metrics.get(k, 0.0)
            for k in (
                "checkpoint_insert_seconds",
                "checkpoint_insert_rows",
                "dump_seconds",
                "dump_rows",
                "dump_bytes",
                "export_pause_seconds",
                "export_pause_memtable_bytes",
            )
        }
        self._saved_metrics.update(m)
        logger.info(
            f"RANK[{_rank()}] RocksDB checkpoint: {seconds:.2f}s total, cache flush "
            f"{d['checkpoint_insert_seconds']:.2f}s "
            f"({int(d['checkpoint_insert_rows'])} rows), logical export "
            f"{d['dump_seconds']:.2f}s ({int(d['dump_rows'])} rows, "
            f"{d['dump_bytes'] / (1 << 30):.2f}GiB written, background-work pause "
            f"wait {d['export_pause_seconds']:.2f}s, read from "
            f"{d['export_pause_memtable_bytes'] / (1 << 20):.1f}MiB unflushed "
            "memtables); cumulative "
            f"train writeback {int(m.get('train_insert_rows', 0))} rows, publish "
            f"writeback {int(m.get('publish_insert_rows', 0))} rows, evicted "
            f"{int(m.get('evict_rows', 0))} rows, index {stats['index']}; RocksDB "
            f"cumulative flush {tick.get('rocksdb.flush.write.bytes', 0)} bytes, "
            f"compaction {tick.get('rocksdb.compact.write.bytes', 0)} bytes "
            f"written, stalls {tick.get('rocksdb.stall.micros', 0) / 1e6:.1f}s; "
            f"after release: live SST {props['rocksdb.live-sst-files-size']}, all "
            f"SST {props['rocksdb.total-sst-files-size']}, obsolete SST "
            f"{props['rocksdb.obsolete-sst-files-size']}, snapshots "
            f"{props['rocksdb.num-snapshots']}"
        )

    def reset_for_import(self, model: nn.Module) -> None:
        """Start RocksDB tables from a fresh empty DB before ``DynamicEmbLoad``.

        Empties the GPU caches first: rows touched before the restore (the
        optimizer warm-up step) must not be written back into the new DB.
        """
        groups = rocksdb_modules(model)
        if not groups:
            return
        for m in groups:
            m.reset_cache_states()
        self.session.swap_working_db(self._new_work_dir())
        for m in groups:
            m.tables.refresh_namespaces()
        logger.info(
            f"RANK[{_rank()}] loading RocksDB tables into fresh working DB "
            f"{self.session.work_dir}"
        )

    def close(self) -> None:
        """Stop the monitor, close the working DB and delete its directory."""
        if self._monitor is not None:
            self._monitor.close()
            self._monitor = None
        if self.session is not None:
            work = self.session.work_dir
            self.session.close()
            self.session = None
            shutil.rmtree(work, ignore_errors=True)
        self.groups = []


@contextlib.contextmanager
def open_runtime(
    train_config: train_pb2.TrainConfig, model_dir: str, mode: str
) -> Iterator[Optional[RocksDBRuntime]]:
    """The run's RocksDB runtime, or None when RocksDB is not configured.

    Open it before the model is planned and keep it until the process is done
    with the model; it is closed when the block exits, normally or not.
    """
    if not train_config.HasField("rocksdb_config"):
        yield None
        return
    try:
        _import_backend()
    except RuntimeError:
        logger.warning(
            "rocksdb_config without the tzrec-rocksdb wheel: every dynamicemb "
            "table stays in GPU/host memory."
        )
        yield None
        return
    runtime = RocksDBRuntime(train_config.rocksdb_config, model_dir, mode)
    try:
        yield runtime
    finally:
        runtime.close()
