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

Ranks whose ``root_dir`` lies on the same filesystem draw from one disk pool:
its free space, and the optional ``disk_budget_bytes`` that caps the job's use
of it, are shared (one NFS server, or one local device of a host). The planner
projects each pool's peak occupancy from the logical row bytes of the RocksDB
tables.
"""

import contextlib
import dataclasses
import glob
import hashlib
import math
import os
import shutil
import socket
import time
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Tuple, cast

import torch.distributed as dist
from torch import nn

from tzrec.protos import train_pb2
from tzrec.utils.logging_util import logger

if TYPE_CHECKING:
    from dynamicemb import DynamicEmbTableOptions

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

# SST bytes per logical byte (uncompressed rows; 1.02-1.05 measured on a
# 0.6 TB table set after sorted ingest and a day of training).
SST_FACTOR = 1.1
# Space a pool keeps free beyond the projected peak.
DISK_RESERVE_BYTES = 16 << 30
_NETWORK_FS = ("nfs", "nfs4", "cpfs", "cpfs-nfs", "lustre", "cephfs", "glusterfs")


def fixed_host_bytes() -> int:
    """Host memory a rank spends on RocksDB once any table is offloaded."""
    return (
        WRITE_BUFFER_BYTES
        + STAGING_BUFFER_BYTES
        + BLOCK_CACHE_DATA_BYTES
        + _HOST_OVERHEAD_BYTES
    )


class DiskCapacityError(RuntimeError):
    """Raised when RocksDB tables do not fit their disk budget."""


@dataclasses.dataclass
class DiskView:
    """One rank's view of the disk pool behind ``root_dir``.

    Attributes:
        rank (int): global rank.
        pool (str): identity of the shared free space (see :func:`pool_key`).
        free_bytes (int): bytes available to unprivileged writers (statvfs).
        job_bytes (int): bytes already allocated by this job's rank directory.
    """

    rank: int
    pool: str
    free_bytes: int
    job_bytes: int


def _existing_ancestor(path: str) -> str:
    path = os.path.realpath(path)
    while not os.path.exists(path):
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    return path


def _mount_entry(path: str) -> Tuple[str, str, str]:
    """(mount_point, fstype, source) of the mount containing ``path``."""
    best = ("/", "unknown", "unknown")
    try:
        with open("/proc/self/mountinfo") as f:
            for line in f:
                left, _, right = line.partition(" - ")
                fields = left.split()
                mount_point = fields[4].replace("\\040", " ")
                if path == mount_point or path.startswith(
                    mount_point.rstrip("/") + "/"
                ):
                    if len(mount_point) >= len(best[0]):
                        rest = right.split()
                        best = (mount_point, rest[0], rest[1] if len(rest) > 1 else "")
    except OSError:
        pass
    return best


def allocated_bytes(root: str) -> int:
    """Filesystem allocation of the files under ``root``, each inode once.

    Hard links are counted once; the allocation is ``st_blocks * 512``, which
    on NFS is the server's report and may differ from quota accounting.
    """
    seen = set()
    total = 0
    for dirpath, _, files in os.walk(root):
        for name in files:
            try:
                st = os.lstat(os.path.join(dirpath, name))
            except OSError:
                continue
            key = (st.st_dev, st.st_ino)
            if key in seen:
                continue
            seen.add(key)
            total += st.st_blocks * 512
    return total


def pool_key(path: str, fstype: str, source: str, host: str, st_dev: int) -> str:
    """Identity of the free space a filesystem shares across ranks.

    The filesystem type decides only this: a network filesystem is one pool
    per server, whichever host or export the ranks mount it from; a local
    device is one pool per host and device. FUSE and unknown filesystems are
    rejected, since nothing tells which ranks share them.
    """
    if fstype in _NETWORK_FS or fstype.startswith("nfs"):
        return f"net:{source.split(':', 1)[0]}"
    if source.startswith("/dev/") or fstype in ("overlay", "tmpfs"):
        return f"local:{host}:{st_dev}"
    raise DiskCapacityError(
        f"rocksdb_config.root_dir {path} is on {fstype} from {source}; RocksDB "
        "tables need a local disk or an NFS-like filesystem."
    )


def observe(root_dir: str, job_root: str, rank: int) -> DiskView:
    """Describe the disk pool behind ``root_dir`` from this rank."""
    path = _existing_ancestor(root_dir)
    _, fstype, source = _mount_entry(path)
    st = os.statvfs(path)
    return DiskView(
        rank=rank,
        pool=pool_key(path, fstype, source, socket.gethostname(), os.stat(path).st_dev),
        free_bytes=st.f_bavail * st.f_frsize,
        job_bytes=allocated_bytes(job_root) if os.path.isdir(job_root) else 0,
    )


def gather_views(root_dir: str, job_root: str) -> List[DiskView]:
    """Disk views of every rank (collective when distributed)."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    mine = observe(root_dir, job_root, rank)
    if not dist.is_initialized():
        return [mine]
    out: List[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(out, mine)
    return out


@dataclasses.dataclass
class TableDisk:
    """Disk-relevant facts of one RocksDB table on one rank.

    Attributes:
        name (str): table namespace.
        rows (int): peak live rows on the rank.
        row_bytes (int): stored bytes of one row (key, score, embedding,
            optimizer state), from the backend's record layout.
        index_bytes (int): eviction-index bytes per row (0 if not evicting).
        bound (str): where ``rows`` comes from and whether it is enforced.
        cache_rows (int): rows of its GPU cache on the rank.
        overshoot_rows (int): new rows one step can add before eviction.
    """

    name: str
    rows: int
    row_bytes: int
    index_bytes: int
    bound: str
    cache_rows: int
    overshoot_rows: int

    @property
    def logical_bytes(self) -> int:
        """Stored row plus index bytes at the peak row count."""
        return self.rows * (self.row_bytes + self.index_bytes)

    @property
    def growth_bytes(self) -> int:
        """Disk the table adds to a pool per rank: SSTs, overshoot, save flush.

        Additive over tables; :func:`pool_fixed_bytes` holds the rest.
        """
        return (
            int(
                (
                    self.logical_bytes
                    + self.overshoot_rows * (self.row_bytes + self.index_bytes)
                )
                * SST_FACTOR
            )
            + self.cache_rows * self.row_bytes
        )


def pool_fixed_bytes(overheads: Dict[str, int], spill: int) -> int:
    """Disk a rank's DB needs whatever its tables hold.

    Compaction output, MANIFEST and logs, and the local spill of a restore
    (bounded by the largest table).
    """
    return overheads["compaction_headroom"] + overheads["other_files"] + spill


def pool_capacity(views: List[DiskView], budget: Optional[int], fixed: int) -> int:
    """Disk each rank may give its tables' growth, in the tightest pool.

    A pool's room is the budget (if set, else its free space plus what the job
    already holds) less the reserve and every member's fixed bytes, split
    evenly: row-wise shards give every rank the same rows.
    """
    pools: Dict[str, List[DiskView]] = {}
    for v in views:
        pools.setdefault(v.pool, []).append(v)
    room = []
    for members in pools.values():
        n = len(members)
        avail = min(v.free_bytes for v in members) + sum(v.job_bytes for v in members)
        if budget is not None:
            avail = min(avail, budget)
        room.append((avail - DISK_RESERVE_BYTES - n * fixed) // n)
    return max(0, min(room))


@dataclasses.dataclass
class DiskPoolPlan:
    """Projected disk use of one disk pool; byte fields cover all its ranks."""

    pool: str
    ranks: List[int]
    logical_bytes: int
    sst_bytes: int
    compaction_headroom: int
    other_files: int
    overshoot_bytes: int
    save_extra: int
    training_peak: int
    save_peak: int
    restore_peak: int
    projected_peak: int
    existing_job_bytes: int
    free_bytes: int
    budget: Optional[int]
    problems: List[str]

    @property
    def feasible(self) -> bool:
        """Whether the projection fits both the budget and the free space."""
        return not self.problems

    def describe(self) -> str:
        """One-line breakdown for logs and errors."""

        def g(b: Optional[int]) -> str:
            return "unset" if b is None else f"{b / (1 << 30):.1f}GiB"

        text = (
            f"disk pool {self.pool} ranks {self.ranks}: logical "
            f"{g(self.logical_bytes)} x SST factor {SST_FACTOR:.2f} = "
            f"{g(self.sst_bytes)}, compaction headroom "
            f"{g(self.compaction_headroom)}, other files {g(self.other_files)}, "
            f"overshoot {g(self.overshoot_bytes)}, save +{g(self.save_extra)}, "
            f"reserve {g(DISK_RESERVE_BYTES)} -> peak {g(self.projected_peak)} "
            f"(train {g(self.training_peak)}, save {g(self.save_peak)}, restore "
            f"{g(self.restore_peak)}); budget {g(self.budget)}, existing job "
            f"{g(self.existing_job_bytes)}, free {g(self.free_bytes)}"
        )
        if self.problems:
            text += "; PROBLEMS: " + "; ".join(self.problems)
        return text


def plan_disk_pools(
    tables: List[TableDisk],
    views: List[DiskView],
    budget: Optional[int],
    overheads: Dict[str, int],
) -> List[DiskPoolPlan]:
    """Project the peak disk use of every disk pool.

    ``tables`` are one rank's RocksDB tables; row-wise sharding gives every
    rank the same per-rank rows. Peaks are compared per scenario, not summed:
    training (live SSTs, compaction output, other files, one step of
    overshoot), save (training plus the overwritten versions a full cache
    flush leaves until compaction; background work is paused while the export
    reads, so a snapshot pins nothing) and restore (a fresh DB ingested from
    the logical checkpoint as sorted files, plus the local spill of a table
    too large to import at once, bounded by the largest table).
    """
    logical = sum(t.logical_bytes for t in tables)
    overshoot = sum(t.overshoot_rows * (t.row_bytes + t.index_bytes) for t in tables)
    flush = sum(t.cache_rows * t.row_bytes for t in tables)
    spill = max((t.rows * t.row_bytes for t in tables), default=0)
    pools: Dict[str, List[DiskView]] = {}
    for v in views:
        pools.setdefault(v.pool, []).append(v)
    plans = []
    for key, members in sorted(pools.items()):
        n = len(members)
        sst = int(logical * SST_FACTOR) * n
        headroom = overheads["compaction_headroom"] * n
        other = overheads["other_files"] * n
        over = int(overshoot * SST_FACTOR) * n
        training = sst + headroom + other + over
        save = training + flush * n
        restore = sst + headroom + other + spill * n
        peak = max(training, save, restore) + DISK_RESERVE_BYTES
        existing = sum(v.job_bytes for v in members)
        free = min(v.free_bytes for v in members)
        problems = []
        if budget is not None and peak > budget:
            problems.append(
                f"projected peak exceeds the budget by {peak - budget} bytes"
            )
        # Bytes the job already holds are part of the peak and already taken
        # from the free space; only the growth must still fit.
        if peak - existing > free:
            problems.append(
                f"growth needs {peak - existing - free} bytes more than are free"
            )
        plans.append(
            DiskPoolPlan(
                pool=key,
                ranks=sorted(v.rank for v in members),
                logical_bytes=logical * n,
                sst_bytes=sst,
                compaction_headroom=headroom,
                other_files=other,
                overshoot_bytes=over,
                save_extra=flush * n,
                training_peak=training,
                save_peak=save,
                restore_peak=restore,
                projected_peak=peak,
                existing_job_bytes=existing,
                free_bytes=free,
                budget=budget,
                problems=problems,
            )
        )
    return plans


def is_rocksdb_options(options: Optional["DynamicEmbTableOptions"]) -> bool:
    """Whether DynamicEmbTableOptions are backed by a RocksDB factory."""
    if options is None or options.external_storage is None:
        return False
    from tzrec_ext.rocksdb.storage import RocksDBStorageFactory

    return isinstance(options.external_storage, RocksDBStorageFactory)


def cache_rows(rows: int, ratio: float) -> int:
    """GPU cache rows DynamicEmb allocates for a table at ``ratio``."""
    if ratio <= 0:
        return 0
    rows = max(1, int(rows * ratio))
    return int(math.ceil(rows / CACHE_BUCKET_ROWS) * CACHE_BUCKET_ROWS)


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
    (:meth:`attach`). Created by :func:`open_runtime`, which closes it however
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
        if mode == "train":
            # Only a complete checkpoint is ever resumed, so a working DB left
            # by an earlier attempt of this rank is stale.
            for stale in glob.glob(os.path.join(self.rank_dir, "train-*")):
                logger.info(f"removing stale RocksDB working directory {stale}")
                shutil.rmtree(stale, ignore_errors=True)
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

    def evict(self) -> int:
        """Evict every RocksDB table above its capacity; returns the rows.

        Call only with no lookup in flight (a completed update, or the end of
        a pass).
        """
        evicted = 0
        for m in self.groups:
            evicted += sum(m.tables.evict_to_capacity().values())
        return evicted

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
        import tzrec_ext  # noqa: F401
    except ModuleNotFoundError as e:
        if e.name != "tzrec_ext":
            raise
        logger.warning(
            "rocksdb_config without the tzrec-ext wheel: every dynamicemb "
            "table stays in GPU/host memory."
        )
        yield None
        return
    runtime = RocksDBRuntime(train_config.rocksdb_config, model_dir, mode)
    try:
        yield runtime
    finally:
        runtime.close()
