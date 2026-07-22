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
import os
import socket
import subprocess
import sys
import weakref
from threading import Condition, Event, Thread
from typing import Any, Dict, Optional

from tzrec.utils import checkpoint_util, env_util
from tzrec.utils.logging_util import logger

_DISTRIBUTED_ENV_KEYS = {
    "GROUP_RANK",
    "GROUP_WORLD_SIZE",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "RANK",
    "ROLE_NAME",
    "ROLE_RANK",
    "ROLE_WORLD_SIZE",
    "WORLD_SIZE",
}
_DISTRIBUTED_ENV_PREFIXES = ("TORCHELASTIC_",)
_VERSION_TIME_FORMAT = "%Y%m%d%H%M%S"


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


def _get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _build_export_subprocess_env(repo_root: str) -> Dict[str, str]:
    env = os.environ.copy()
    for key in list(env):
        if key in _DISTRIBUTED_ENV_KEYS or key.startswith(_DISTRIBUTED_ENV_PREFIXES):
            del env[key]
    env.update(
        {
            "USE_DISTRIBUTED_EMBEDDING": "1",
            "INPUT_TILE": "3",
            "CUDA_VISIBLE_DEVICES": "",
            "RANK": "0",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": "1",
            "LOCAL_WORLD_SIZE": "1",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(_get_free_port()),
        }
    )
    env["PYTHONPATH"] = (
        repo_root
        if not env.get("PYTHONPATH")
        else repo_root + os.pathsep + env["PYTHONPATH"]
    )
    return env


class OnlineDenseExportManager:
    """Background launcher for online-learning dense model export."""

    def __init__(
        self,
        model_dir: str,
        pipeline_config_path: str,
        ckpt_manager: checkpoint_util.CheckpointManager,
    ) -> None:
        self._enabled = _online_dense_export_enabled()
        self._rank = int(os.environ.get("RANK", 0))
        self._ckpt_manager = ckpt_manager
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
        self._keep_logs = int(os.environ.get("ONLINE_DENSE_EXPORT_KEEP_LOGS", "3"))

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
        if not env_util.use_distributed_embedding():
            raise RuntimeError(
                "ONLINE_DENSE_EXPORT=1 requires USE_DISTRIBUTED_EMBEDDING=1."
            )
        # The publish tree (os.rename / current.json / protect-key glob) is
        # local-FS only; fsspec URLs break all three. Check the actual export
        # root -- <serving_root>/dense_hot_export -- so a local override
        # decouples the publish tree from a remote model_dir.
        for label, path in (
            ("export_root", export_root),
            ("pipeline_config_path", pipeline_config_path),
        ):
            if _is_remote_path(path):
                raise RuntimeError(
                    f"ONLINE_DENSE_EXPORT requires a local {label}, got remote: {path}"
                )
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

    def submit(
        self,
        step: int,
        checkpoint_path: str,
        data_timestamp: float,
    ) -> None:
        """Queue a dense export task for the freshest saved checkpoint."""
        if not self._enabled or self._rank != 0:
            return
        checkpoint_path = os.path.abspath(checkpoint_path)
        version = _make_monotonic_version(self._last_version)
        self._last_version = version
        with self._cond:
            if self._drain_event.is_set():
                logger.warning("online dense export draining; skip step %s", step)
                return
            # latest-wins: a not-yet-started pending task is superseded and
            # its checkpoint unprotected immediately so prune can reclaim it.
            # Online serving only consumes the freshest dense version, so a
            # backlog cannot pin one protected checkpoint per queued task.
            superseded = self._pending["checkpoint_path"] if self._pending else None
            self._pending = {
                "step": step,
                "checkpoint_path": checkpoint_path,
                "data_timestamp": data_timestamp,
                "version": version,
            }
            self._ckpt_manager.protect_checkpoint(checkpoint_path)
            self._cond.notify()
        if superseded is not None:
            self._ckpt_manager.unprotect_checkpoint(superseded)
            self._ckpt_manager.prune()

    def close(self) -> None:
        """Wait for in-flight and pending dense export tasks to finish.

        Detach the finalizer only after the worker actually stops, so a worker
        that outlives the close timeout keeps the atexit drain backstop
        instead of leaking a live subprocess publisher.
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
        subprocess that could publish current.json unattended.
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
                # OSError from makedirs/open/socket). _run_task's finally
                # still unprotects the failing checkpoint; without this guard a
                # single transient I/O error would permanently disable exports
                # and silently void keep_checkpoint_max for all future submits.
                logger.exception("online dense export task failed; continuing")

    def _run_task(self, task: Dict[str, Any]) -> None:
        checkpoint_path = task["checkpoint_path"]
        log_dir = os.path.join(self._export_root, "logs")
        try:
            if not os.path.exists(checkpoint_path):
                logger.error(
                    "skip online dense export version %s: checkpoint missing: %s",
                    task["version"],
                    checkpoint_path,
                )
                return

            repo_root = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            env = _build_export_subprocess_env(repo_root)
            # Pass the pre-suffix serving root (abspath'd) so the subprocess
            # re-resolves it via resolve_dense_export_root to the same
            # <serving_root>/dense_hot_export tree the manager logs into,
            # independent of subprocess cwd or a relative ONLINE_DENSE_EXPORT_DIR.
            env["ONLINE_DENSE_EXPORT_DIR"] = self._serving_root
            cmd = [
                sys.executable,
                "-m",
                "tzrec.tools.online_dense_export",
                "--pipeline_config_path",
                self._pipeline_config_path,
                "--checkpoint_path",
                checkpoint_path,
                "--model_dir",
                self._model_dir,
                "--version",
                task["version"],
                "--checkpoint_step",
                str(task["step"]),
                "--data_timestamp",
                str(task["data_timestamp"]),
            ]
            logger.info(
                "start online dense export version %s from %s",
                task["version"],
                checkpoint_path,
            )
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, f"{task['version']}.log")
            try:
                with open(log_path, "w") as log_file:
                    subprocess.run(
                        cmd,
                        check=True,
                        env=env,
                        cwd=repo_root,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        timeout=self._export_timeout,
                    )
            except subprocess.TimeoutExpired:
                logger.error(
                    "online dense export version %s timed out after %ss, see %s",
                    task["version"],
                    self._export_timeout,
                    log_path,
                )
                return
            except subprocess.CalledProcessError as e:
                logger.error(
                    "online dense export version %s failed with return code %s, see %s",
                    task["version"],
                    e.returncode,
                    log_path,
                )
                return
            logger.info("online dense export version %s finished", task["version"])
        finally:
            self._ckpt_manager.unprotect_checkpoint(checkpoint_path)
            self._ckpt_manager.prune()
            self._prune_export_logs(log_dir)

    def _prune_export_logs(self, log_dir: str) -> None:
        """Best-effort: keep the newest K export logs, drop the rest.

        Each export attempt (success, timeout, or failure) writes one
        ``<version>.log``; version retention never touches this directory, so
        without this a long-running job accumulates one file/inode per
        checkpoint including failed exports. Version names are timestamps, so
        name order is chronological -- keep the newest K and remove the rest.

        Args:
            log_dir: Directory holding ``<version>.log`` files.
        """
        if self._keep_logs <= 0:
            return
        try:
            entries = os.listdir(log_dir)
        except FileNotFoundError:
            return
        logs = sorted(name for name in entries if name.endswith(".log"))
        for name in logs[: -self._keep_logs]:
            path = os.path.join(log_dir, name)
            try:
                os.remove(path)
                logger.info("removed old dense export log: %s", path)
            except OSError as e:
                logger.warning("failed to remove old log %s: %s", path, e)
