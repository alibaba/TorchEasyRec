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

"""Live-storage capacity planning and headroom monitoring for RocksDB tables.

Ranks whose ``rocksdb_config.local_path`` resolves to the same storage domain
(one NFS server, one local block device, or an explicit
``storage_domain``) share one ``disk_budget_bytes``. The planner projects each
domain's peak occupancy from the logical row bytes of the RocksDB tables and
checks it against both the job budget and the free space the ranks observe;
the monitor re-checks the real occupancy and free space while training.
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

# Used until a workload calibrates sst_space_factor; not a measured value.
PROVISIONAL_SST_FACTOR = 1.5
_NETWORK_FS = ("nfs", "nfs4", "cpfs", "cpfs-nfs", "lustre", "cephfs", "glusterfs")
_TREND_SAMPLES = 8


class DiskCapacityError(RuntimeError):
    """Raised when RocksDB tables do not fit, or no longer fit, their disk budget."""


@dataclasses.dataclass
class DomainObservation:
    """One rank's view of the filesystem behind ``local_path``.

    Attributes:
        rank (int): global rank.
        host (str): hostname.
        path (str): resolved local_path (or its nearest existing ancestor).
        fstype (str): filesystem type from mountinfo.
        source (str): mount source (device or ``server:/export``).
        mount_point (str): mount point.
        st_dev (int): device id of the path.
        free_bytes (int): bytes available to unprivileged writers (statvfs).
        total_bytes (int): filesystem size; reported only, never used as budget.
        job_bytes (int): bytes already allocated by this job's rank directory.
    """

    rank: int
    host: str
    path: str
    fstype: str
    source: str
    mount_point: str
    st_dev: int
    free_bytes: int
    total_bytes: int
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

    Hard links (for example SSTs shared with a native checkpoint) are counted
    once; the allocation is ``st_blocks * 512``, which on NFS is the server's
    report and may differ from quota accounting.
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


def observe(local_path: str, job_root: str, rank: int) -> DomainObservation:
    """Describe the storage behind ``local_path`` from this rank."""
    path = _existing_ancestor(local_path)
    mount_point, fstype, source = _mount_entry(path)
    st = os.statvfs(path)
    return DomainObservation(
        rank=rank,
        host=socket.gethostname(),
        path=path,
        fstype=fstype,
        source=source,
        mount_point=mount_point,
        st_dev=os.stat(path).st_dev,
        free_bytes=st.f_bavail * st.f_frsize,
        total_bytes=st.f_blocks * st.f_frsize,
        job_bytes=allocated_bytes(job_root) if os.path.isdir(job_root) else 0,
    )


def domain_key(obs: DomainObservation, explicit: str = "") -> str:
    """Identity of the capacity pool behind an observation.

    Network filesystems are keyed by server, so different exports or mount
    points of one server share a pool; local block devices by host and
    device. FUSE and unknown filesystems need an explicit ``storage_domain``.
    """
    if explicit:
        return f"explicit:{explicit}"
    if obs.fstype in _NETWORK_FS or obs.fstype.startswith("nfs"):
        return f"net:{obs.source.split(':', 1)[0]}"
    if obs.source.startswith("/dev/") or obs.fstype in ("overlay", "tmpfs"):
        return f"local:{obs.host}:{obs.st_dev}"
    raise DiskCapacityError(
        f"cannot tell which storage {obs.path} ({obs.fstype} from {obs.source}) "
        "shares capacity with; set rocksdb_config.storage_domain."
    )


def gather_observations(local_path: str, job_root: str) -> List[DomainObservation]:
    """Observations of every rank (collective when distributed)."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    mine = observe(local_path, job_root, rank)
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
        value_file_bytes (int): largest logical checkpoint file of the table.
    """

    name: str
    rows: int
    row_bytes: int
    index_bytes: int
    bound: str
    cache_rows: int
    overshoot_rows: int
    value_file_bytes: int

    @property
    def logical_bytes(self) -> int:
        """Stored row plus index bytes at the peak row count."""
        return self.rows * (self.row_bytes + self.index_bytes)


@dataclasses.dataclass
class DomainPlan:
    """Projected disk use of one storage domain.

    All byte fields cover every rank of the domain.
    """

    domain: str
    ranks: List[int]
    logical_bytes: int
    sst_factor: float
    sst_factor_source: str
    sst_bytes: int
    compaction_headroom: int
    other_files: int
    overshoot_bytes: int
    save_extra: int
    reserve: int
    training_peak: int
    save_peak: int
    restore_peak: int
    projected_peak: int
    existing_job_bytes: int
    free_bytes: int
    budget: int
    quota_observable: bool
    problems: List[str]

    @property
    def feasible(self) -> bool:
        """Whether the projection fits both the budget and the free space."""
        return not self.problems

    @property
    def remaining(self) -> int:
        """Capacity left at the projected peak (budget or free space)."""
        return min(
            self.budget - self.projected_peak,
            self.free_bytes + self.existing_job_bytes - self.projected_peak,
        )

    def describe(self) -> str:
        """One-line breakdown for logs and errors."""

        def g(b: int) -> str:
            return f"{b / (1 << 30):.1f}GiB"

        quota = (
            "observable" if self.quota_observable else "not observable, budget governs"
        )
        text = (
            f"domain {self.domain} ranks {self.ranks}: logical "
            f"{g(self.logical_bytes)} x SST factor {self.sst_factor:.2f} "
            f"({self.sst_factor_source}) = {g(self.sst_bytes)}, compaction "
            f"headroom {g(self.compaction_headroom)}, other files "
            f"{g(self.other_files)}, overshoot {g(self.overshoot_bytes)}, save "
            f"+{g(self.save_extra)}, reserve {g(self.reserve)} -> peak "
            f"{g(self.projected_peak)} (train {g(self.training_peak)}, save "
            f"{g(self.save_peak)}, restore {g(self.restore_peak)}); budget "
            f"{g(self.budget)}, existing job {g(self.existing_job_bytes)}, free "
            f"{g(self.free_bytes)}, remaining {g(self.remaining)}; quota {quota}"
        )
        if self.problems:
            text += "; PROBLEMS: " + "; ".join(self.problems)
        return text


def sst_factor(cfg: train_pb2.RocksDBConfig) -> Tuple[float, str]:
    """The SST space factor and where it comes from."""
    if cfg.HasField("sst_space_factor"):
        return cfg.sst_space_factor, "configured"
    return PROVISIONAL_SST_FACTOR, "provisional, not measured"


def plan_domains(
    tables: List[TableDisk],
    observations: List[DomainObservation],
    cfg: train_pb2.RocksDBConfig,
    overheads: Dict[str, int],
) -> List[DomainPlan]:
    """Project the peak disk use of every storage domain.

    ``tables`` are one rank's RocksDB tables; row-wise sharding gives every
    rank the same per-rank rows. Peaks are compared per scenario, not summed:
    training (live SSTs, compaction output, other files, one step of
    overshoot), save (training plus the overwritten versions a full cache
    flush leaves until compaction; background work is paused while the export
    reads, so a snapshot pins nothing) and restore (a fresh DB bulk-loaded from
    the logical checkpoint, compacting while it loads).
    """
    factor, source = sst_factor(cfg)
    logical = sum(t.logical_bytes for t in tables)
    overshoot = sum(t.overshoot_rows * (t.row_bytes + t.index_bytes) for t in tables)
    flush = sum(t.cache_rows * t.row_bytes for t in tables)
    by_domain: Dict[str, List[DomainObservation]] = {}
    for obs in observations:
        by_domain.setdefault(domain_key(obs, cfg.storage_domain), []).append(obs)
    plans = []
    for key, members in sorted(by_domain.items()):
        n = len(members)
        sst = int(logical * factor) * n
        headroom = overheads["compaction_headroom"] * n
        other = overheads["other_files"] * n
        over = int(overshoot * factor) * n
        training = sst + headroom + other + over
        save_extra = flush * n
        reserve = cfg.disk_reserve_bytes
        save = training + save_extra
        restore = sst + headroom + other
        peak = max(training, save, restore) + reserve
        existing = sum(o.job_bytes for o in members)
        free = min(o.free_bytes for o in members)
        budget = cfg.disk_budget_bytes
        problems = []
        if not cfg.HasField("disk_budget_bytes"):
            problems.append("rocksdb_config.disk_budget_bytes is not set")
        elif peak > budget:
            problems.append(
                f"projected peak exceeds the budget by {peak - budget} bytes"
            )
        # Bytes the job already holds are part of the peak and already taken
        # from the free space; only the growth must still fit.
        growth = peak - existing
        if growth > free:
            problems.append(f"growth needs {growth - free} bytes more than are free")
        plans.append(
            DomainPlan(
                domain=key,
                ranks=sorted(o.rank for o in members),
                logical_bytes=logical * n,
                sst_factor=factor,
                sst_factor_source=source,
                sst_bytes=sst,
                compaction_headroom=headroom,
                other_files=other,
                overshoot_bytes=over,
                save_extra=save_extra,
                reserve=reserve,
                training_peak=training,
                save_peak=save,
                restore_peak=restore,
                projected_peak=peak,
                existing_job_bytes=existing,
                free_bytes=free,
                budget=budget,
                quota_observable=not key.startswith("net:")
                and not key.startswith("explicit:"),
                problems=problems,
            )
        )
    return plans


class CapacityMonitor:
    """Bounded-cadence headroom checks of the live RocksDB storage.

    Every ``interval_steps`` steps the ranks exchange their latest sample
    (free space, this job's allocation, DB space properties) and agree on the
    worst headroom of any domain: ``min(free - reserve, budget - occupancy)``.
    The allocation scan runs on a background thread, so a check never walks
    the database directory on the training path.

    Args:
        cfg (RocksDBConfig): the RocksDB config.
        job_root (str): this rank's directory under local_path.
        session: this rank's RocksDB session.
    """

    def __init__(self, cfg: train_pb2.RocksDBConfig, job_root: str, session: Any):
        self._cfg = cfg
        self._job_root = job_root
        self._session = session
        budget = cfg.disk_budget_bytes
        self.warn = (
            cfg.disk_warn_headroom_bytes
            if cfg.HasField("disk_warn_headroom_bytes")
            else budget // 10
        )
        self.critical = (
            cfg.disk_critical_headroom_bytes
            if cfg.HasField("disk_critical_headroom_bytes")
            else budget // 20
        )
        self._lock = threading.Lock()
        self._sample: Optional[Dict[str, Any]] = None
        self._thread: Optional[threading.Thread] = None
        self._history: List[Tuple[float, int, int]] = []
        self.last: Optional[Dict[str, Any]] = None

    def _take_sample(self) -> None:
        rank = dist.get_rank() if dist.is_initialized() else 0
        obs = observe(self._cfg.local_path, self._job_root, rank)
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
            None, "warn", "save_and_stop" (critical, and a save still fits) or
            "stop" (critical, no room for a save).
        """
        interval = self._cfg.capacity_check_interval_steps
        if interval <= 0 or step % interval != 0:
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
        domains: Dict[str, List[Dict[str, Any]]] = {}
        for s in samples:
            domains.setdefault(
                domain_key(s["obs"], self._cfg.storage_domain), []
            ).append(s)
        action = None
        report = []
        for key, members in sorted(domains.items()):
            occupancy = sum(m["obs"].job_bytes for m in members)
            free = min(m["obs"].free_bytes for m in members)
            need = sum(m["save_need"] for m in members)
            headroom = min(
                free - self._cfg.disk_reserve_bytes,
                self._cfg.disk_budget_bytes - occupancy,
            )
            pending = sum(
                m["props"].get("rocksdb.estimate-pending-compaction-bytes") or 0
                for m in members
            )
            report.append(
                {
                    "domain": key,
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
            if headroom < self.critical:
                act = "save_and_stop" if headroom - need >= 0 else "stop"
                action = "stop" if action == "stop" else act
            elif headroom < self.warn and action is None:
                action = "warn"
        self._log(step, report)
        self.last = {"step": step, "domains": report, "action": action}
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
                    f"RocksDB disk at step {step}: {r['domain']} occupancy "
                    f"{r['occupancy'] / gib:.1f}GiB, free {r['free'] / gib:.1f}GiB, "
                    f"headroom {r['headroom'] / gib:.1f}GiB, live SST "
                    f"{r['live_sst'] / gib:.1f}GiB, all SST {r['total_sst'] / gib:.1f}"
                    f"GiB, pending compaction {r['pending_compaction'] / gib:.1f}GiB"
                    f" (rewrite work, not space), snapshots {r['snapshots']}"
                    + (eta if r is worst else "")
                )
