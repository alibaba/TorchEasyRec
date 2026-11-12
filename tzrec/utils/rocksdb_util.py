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

Working databases live under ``rocksdb_config.root_dir``::

    <run_id>/rank_<r>/<mode>-<ms>-<pid>/

``run_id`` is derived from the model_dir, so restarts of one job share it and
different jobs never collide; ``rank_<r>`` separates ranks on a shared disk and
``<mode>-<ms>-<pid>`` concurrent processes. The working DB is never a
checkpoint: saves write RocksDB tables through ``DynamicEmbDump`` into the
checkpoint's ``dynamicemb/`` files like every other dynamicemb table, and a
restore always imports them into a fresh working DB.

Startup leaves existing working directories untouched. Closing a runtime
removes its own working directory; leftovers from abrupt termination require
manual cleanup.

Planning-time capacities come from ``resource_util``: homogeneous scalar
budgets per rank, with the sharing scope set by ``shared_across_nodes``. This
module keeps the shared backend reservation, charged whenever RocksDB is
configured, and the runtime lifecycle.
"""

import dataclasses
import hashlib
import os
import shutil
import time
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, cast

from torch import nn

from tzrec.protos import train_pb2
from tzrec.utils.logging_util import logger

if TYPE_CHECKING:
    from dynamicemb import DynamicEmbTableOptions
    from torchrec.distributed.planner.types import ShardingOption

# Host memory a rank needs beyond the planned RocksDB budgets (table readers,
# iterators, pinned blocks, allocator slack).
_HOST_OVERHEAD_BYTES = 2 << 30
# Planned host budgets that do not depend on the tables: the memtables, the
# pinned staging of one storage call (larger calls are split) and the data
# share of the block cache (other data blocks are left to the page cache).
WRITE_BUFFER_BYTES = 1 << 30
STAGING_BUFFER_BYTES = 256 << 20
BLOCK_CACHE_DATA_BYTES = 1 << 30

# Empirical SSD allowance for SST overhead and restore spill, not a peak bound.
SST_FACTOR = 1.1
# Shared free-space margin for transient growth, including rows awaiting eviction.
SSD_RESERVE_BYTES = 16 << 30


def backend_config(
    local_world_size: int,
    *,
    block_cache_bytes: int,
    import_budget_bytes: int,
) -> Any:
    """The common backend configuration a plan derives its budgets from.

    Threads follow the CPUs one rank may use; the fixed buffers are planned
    constants. Block-cache and import budgets come from the selected plan.

    Args:
        local_world_size: Number of ranks sharing the host's CPUs.
        block_cache_bytes: Block-cache capacity per rank, including metadata.
        import_budget_bytes: Temporary host-memory budget for checkpoint import.
    """
    from tzrec_ext.rocksdb import BackendConfig

    cpus = max(1, len(os.sched_getaffinity(0)) // max(local_world_size, 1))
    return BackendConfig(
        block_cache_bytes=block_cache_bytes,
        write_buffer_bytes=WRITE_BUFFER_BYTES,
        staging_buffer_bytes=STAGING_BUFFER_BYTES,
        import_budget_bytes=import_budget_bytes,
        io_threads=max(4, cpus),
        max_background_jobs=max(2, cpus // 4),
    )


def configure_backend(
    selected: List["ShardingOption"],
    *,
    ddr_caps: Tuple[int, ...],
    local_world_size: int,
) -> List["ShardingOption"]:
    """Configure the shared RocksDB backend from selected shard budgets.

    Args:
        selected: Selected options with ranks assigned to their shards.
        ddr_caps: Per-rank host capacities before the shared RocksDB reserve.
        local_world_size: Number of ranks sharing one host.

    Returns:
        The selected options with backend configuration attached to their
        RocksDB factories. Other table options are unchanged.
    """
    rocksdb_options = []
    metadata_bytes_per_rank = [0] * len(ddr_caps)
    import_available_per_rank = list(ddr_caps)
    for option in selected:
        is_rocksdb = is_rocksdb_options(option.dynamicemb_options)
        if is_rocksdb:
            rocksdb_options.append(option)
        for shard in option.shards:
            rank = cast(int, shard.rank)
            ddr_bytes = shard.storage.ddr or 0
            if is_rocksdb:
                metadata_bytes_per_rank[rank] += ddr_bytes
            else:
                import_available_per_rank[rank] -= ddr_bytes

    if not rocksdb_options:
        return selected

    block_cache_bytes = BLOCK_CACHE_DATA_BYTES + max(metadata_bytes_per_rank)
    # Import reuses the empty memtable/cache budgets; built-in tables remain resident.
    import_budget_bytes = (
        min(import_available_per_rank) - STAGING_BUFFER_BYTES - _HOST_OVERHEAD_BYTES
    )
    config = backend_config(
        local_world_size=local_world_size,
        block_cache_bytes=block_cache_bytes,
        import_budget_bytes=import_budget_bytes,
    )
    for option in rocksdb_options:
        options = option.dynamicemb_options
        options.external_storage = dataclasses.replace(
            options.external_storage, backend_config=config
        )
    logger.info(
        "dynamicemb tables offloaded to RocksDB (per rank): "
        + ", ".join(
            f"{option.fqn} cache ratio {option.cache_load_factor}"
            for option in rocksdb_options
        )
        + f"; backend {config}"
    )
    return selected


def shared_storage(local_world_size: int) -> Any:
    """Shared per-rank resources reserved before RocksDB candidate enumeration.

    DDR covers fixed buffers and host headroom; SSD covers compaction and
    other DB files. Table SSD estimates use SST_FACTOR as an empirical
    allowance for restore spill. Filesystem reserve is deducted separately.
    """
    from torchrec.distributed.planner.types import Storage
    from tzrec_ext.rocksdb import disk_overheads

    overheads = disk_overheads(
        backend_config(local_world_size, block_cache_bytes=0, import_budget_bytes=0)
    )
    return Storage(
        hbm=0,
        ddr=WRITE_BUFFER_BYTES
        + STAGING_BUFFER_BYTES
        + BLOCK_CACHE_DATA_BYTES
        + _HOST_OVERHEAD_BYTES,
        ssd=overheads["compaction_headroom"] + overheads["other_files"],
    )


def is_rocksdb_options(options: Optional["DynamicEmbTableOptions"]) -> bool:
    """Whether DynamicEmbTableOptions are backed by a RocksDB factory."""
    if options is None or options.external_storage is None:
        return False
    from tzrec_ext.rocksdb.storage import RocksDBStorageFactory

    return isinstance(options.external_storage, RocksDBStorageFactory)


def rocksdb_modules(model: nn.Module) -> List[Any]:
    """Return the fused DynamicEmb modules of ``model`` stored in RocksDB."""
    from dynamicemb.dump_load import get_dynamic_emb_module
    from tzrec_ext.rocksdb.storage import RocksDBStorage

    return [
        m for m in get_dynamic_emb_module(model) if isinstance(m.tables, RocksDBStorage)
    ]


class RocksDBRuntime:
    """RocksDB state of one train/eval/export/predict process on this rank.

    Owns the working DB session, opened once the sharding plan is known
    (:meth:`bind`), and the fused RocksDB groups of the built model
    (:meth:`attach`). Constructed by :func:`create_rocksdb_runtime`;
    ``ShardedModelRuntime.open`` owns the instance and closes it however
    the run ends.

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
        self._rank = int(os.environ.get("RANK", 0))
        run_id = hashlib.sha1(os.path.realpath(model_dir).encode()).hexdigest()[:12]
        self.rank_dir = os.path.join(config.root_dir, run_id, f"rank_{self._rank}")
        self.session: Any = None
        self.groups: List[Any] = []

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
            from tzrec_ext.rocksdb.session import RocksDBSession

            self.session = RocksDBSession(config, self._new_work_dir())
        elif self.session.config != config:
            raise RuntimeError("a second plan changed this run's RocksDB config")
        for o in options:
            o.external_storage = dataclasses.replace(
                o.external_storage, session=self.session
            )

    def attach(self, model: nn.Module) -> None:
        """Check the built RocksDB groups and bind each storage to its cache.

        Every table planned for RocksDB must sit in a CACHING_PS module whose
        storage is a RocksDBStorage with a complete namespace map.
        """
        from dynamicemb.dump_load import get_dynamic_emb_module
        from tzrec_ext.rocksdb.storage import RocksDBStorage

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

    def evict(self) -> None:
        """Evict every RocksDB table above its capacity.

        Call only with no lookup in flight (a completed update, or the end of
        a pass).
        """
        for m in self.groups:
            m.tables.evict_to_capacity()

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
            m.tables.reset_for_new_working_db()

    def close(self) -> None:
        """Close the working DB and delete its directory."""
        if self.session is not None:
            work = self.session.work_dir
            self.session.close()
            self.session = None
            shutil.rmtree(work, ignore_errors=True)
        self.groups = []


def create_rocksdb_runtime(
    config: train_pb2.RocksDBConfig, model_dir: str, mode: str
) -> Optional[RocksDBRuntime]:
    """The run's RocksDB runtime, or None while the backend is unavailable.

    Ownership stays with the caller -- ``ShardedModelRuntime.open``
    acquires here and closes the backend after the checkpoint manager on
    every exit path. A missing top-level tzrec-ext wheel keeps every
    dynamicemb table in GPU/host memory with a warning; anything broken
    inside an installed backend surfaces its real error.

    Args:
        config: the job's selected RocksDB config.
        model_dir: the job's model_dir; it names the run's directory.
        mode: "train", "eval", "predict_checkpoint" or "export".
    """
    try:
        import tzrec_ext  # noqa: F401
    except ModuleNotFoundError as e:
        if e.name != "tzrec_ext":
            raise
        logger.warning(
            "rocksdb_config without the tzrec-ext wheel: every dynamicemb "
            "table stays in GPU/host memory."
        )
        return None
    return RocksDBRuntime(config, model_dir, mode)
