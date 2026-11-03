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

"""Disk capacity planning and headroom monitoring for RocksDB tables.

Ranks whose ``rocksdb_config.root_dir`` lies on the same filesystem draw from
one disk pool: its free space, and the optional ``disk_budget_bytes`` that
caps the job's use of it, are shared (one NFS server, or one local device of a
host). The planner projects each pool's peak occupancy from the logical row
bytes of the RocksDB tables; the monitor re-checks the real occupancy and free
space while training.
"""

import dataclasses
import os
import socket
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import torch.distributed as dist

from tzrec.protos import train_pb2
from tzrec.utils.logging_util import logger

# SST bytes per logical byte (uncompressed rows; 1.02-1.05 measured on a
# 0.6 TB table set after sorted ingest and a day of training).
SST_FACTOR = 1.1
# Space a pool keeps free beyond the projected peak.
DISK_RESERVE_BYTES = 16 << 30
# Training headroom below this share of the pool's capacity saves and stops.
CRITICAL_HEADROOM = 0.05
CHECK_INTERVAL_STEPS = 100
_NETWORK_FS = ("nfs", "nfs4", "cpfs", "cpfs-nfs", "lustre", "cephfs", "glusterfs")
_TREND_SAMPLES = 8


class DiskCapacityError(RuntimeError):
    """Raised when RocksDB tables do not fit, or no longer fit, their disk budget."""


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


class CapacityMonitor:
    """Bounded-cadence headroom checks of the live RocksDB storage.

    Every ``CHECK_INTERVAL_STEPS`` steps the ranks exchange their latest
    sample (free space, this job's allocation, DB space properties) and agree
    on the worst headroom of any disk pool: the free space above the reserve,
    or the budget left, whichever is smaller. The allocation scan runs on a
    background thread, so a check never walks the database directory on the
    training path.

    Args:
        cfg (RocksDBConfig): the RocksDB config.
        job_root (str): this rank's directory under root_dir.
        session: this rank's RocksDB session.
    """

    def __init__(self, cfg: train_pb2.RocksDBConfig, job_root: str, session: Any):
        self._cfg = cfg
        self._budget = (
            cfg.disk_budget_bytes if cfg.HasField("disk_budget_bytes") else None
        )
        self._job_root = job_root
        self._session = session
        self._lock = threading.Lock()
        self._sample: Optional[Dict[str, Any]] = None
        self._thread: Optional[threading.Thread] = None
        self._history: List[Tuple[float, int, int]] = []
        self.last: Optional[Dict[str, Any]] = None

    def _take_sample(self) -> None:
        rank = dist.get_rank() if dist.is_initialized() else 0
        obs = observe(self._cfg.root_dir, self._job_root, rank)
        props: Dict[str, Any] = {}
        if self._session is not None and self._session.opened:
            db = self._session.db
            for name in (
                "rocksdb.live-sst-files-size",
                "rocksdb.total-sst-files-size",
                "rocksdb.estimate-live-data-size",
                "rocksdb.estimate-pending-compaction-bytes",
                "rocksdb.num-snapshots",
            ):
                props[name] = db.get_int_property(name)
        with self._lock:
            self._sample = {"obs": obs, "props": props, "time": time.time()}

    def close(self) -> None:
        """Wait for a running sample; call before the session's DB closes."""
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def _refresh(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._take_sample, daemon=True)
            self._thread.start()

    def check(self, step: int, save_need: int) -> Optional[str]:
        """Collective headroom check at a training step boundary.

        Args:
            step (int): the step just completed.
            save_need (int): disk bytes a checkpoint save may add on this rank.

        Returns:
            None, "save_and_stop" (critical, and a save still fits) or "stop"
            (critical, no room for a save).
        """
        if step % CHECK_INTERVAL_STEPS != 0:
            return None
        if self._sample is None:
            self._take_sample()
        with self._lock:
            sample = dict(self._sample or {})
        sample["save_need"] = save_need
        samples: List[Any] = [sample]
        if dist.is_initialized():
            samples = [None] * dist.get_world_size()
            dist.all_gather_object(samples, sample)
        self._refresh()
        pools: Dict[str, List[Dict[str, Any]]] = {}
        for s in samples:
            pools.setdefault(s["obs"].pool, []).append(s)
        action = None
        report = []
        for key, members in sorted(pools.items()):
            occupancy = sum(m["obs"].job_bytes for m in members)
            free = min(m["obs"].free_bytes for m in members)
            need = sum(m["save_need"] for m in members)
            capacity = free + occupancy
            headroom = free - DISK_RESERVE_BYTES
            if self._budget is not None:
                capacity = min(capacity, self._budget)
                headroom = min(headroom, self._budget - occupancy)
            pending = sum(
                m["props"].get("rocksdb.estimate-pending-compaction-bytes") or 0
                for m in members
            )
            report.append(
                {
                    "pool": key,
                    "occupancy": occupancy,
                    "free": free,
                    "headroom": headroom,
                    "save_need": need,
                    "live_sst": sum(
                        m["props"].get("rocksdb.live-sst-files-size") or 0
                        for m in members
                    ),
                    "total_sst": sum(
                        m["props"].get("rocksdb.total-sst-files-size") or 0
                        for m in members
                    ),
                    "pending_compaction": pending,
                    "snapshots": sum(
                        m["props"].get("rocksdb.num-snapshots") or 0 for m in members
                    ),
                }
            )
            if headroom < CRITICAL_HEADROOM * capacity:
                act = "save_and_stop" if headroom - need >= 0 else "stop"
                action = "stop" if action == "stop" else act
        self._log(step, report)
        self.last = {"step": step, "pools": report, "action": action}
        return action

    def _log(self, step: int, report: List[Dict[str, Any]]) -> None:
        gib = 1 << 30
        worst = min(report, key=lambda r: r["headroom"])
        now = time.time()
        self._history.append((now, worst["occupancy"], worst["pending_compaction"]))
        self._history = self._history[-_TREND_SAMPLES:]
        eta = ""
        if len(self._history) >= 2:
            (t0, o0, _), (t1, o1, _) = self._history[0], self._history[-1]
            slope = (o1 - o0) / max(t1 - t0, 1e-6)
            if slope > 0:
                eta = f", ~{worst['headroom'] / slope / 3600:.1f}h to the reserve"
            if len(self._history) == _TREND_SAMPLES and all(
                b[2] > a[2] for a, b in zip(self._history, self._history[1:])
            ):
                eta += ", compaction backlog growing at every check (unstable)"
        if int(os.environ.get("RANK", 0)) == 0:
            for r in report:
                logger.info(
                    f"RocksDB disk at step {step}: {r['pool']} occupancy "
                    f"{r['occupancy'] / gib:.1f}GiB, free {r['free'] / gib:.1f}GiB, "
                    f"headroom {r['headroom'] / gib:.1f}GiB, live SST "
                    f"{r['live_sst'] / gib:.1f}GiB, all SST {r['total_sst'] / gib:.1f}"
                    f"GiB, pending compaction {r['pending_compaction'] / gib:.1f}GiB"
                    f" (rewrite work, not space), snapshots {r['snapshots']}"
                    + (eta if r is worst else "")
                )
