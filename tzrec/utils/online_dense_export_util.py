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
from queue import Queue
from threading import Thread
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
        self._model_dir = os.path.abspath(model_dir)
        self._pipeline_config_path = os.path.abspath(pipeline_config_path)
        self._ckpt_manager = ckpt_manager
        self._queue: "Queue[Optional[Dict[str, Any]]]" = Queue()
        self._worker: Optional[Thread] = None
        self._last_version = ""

        if not self._enabled:
            return
        if not env_util.use_distributed_embedding():
            raise RuntimeError(
                "ONLINE_DENSE_EXPORT=1 requires USE_DISTRIBUTED_EMBEDDING=1."
            )
        if self._rank == 0:
            self._worker = Thread(
                target=self._worker_loop,
                name="online-dense-export",
                daemon=True,
            )
            self._worker.start()
            logger.info(
                "ONLINE_DENSE_EXPORT enabled; dense versions will be exported under %s",
                os.path.join(model_dir, "dense_hot_export"),
            )

    def submit(
        self,
        step: int,
        checkpoint_path: str,
        data_timestamp: float,
    ) -> None:
        """Queue a dense export task for one saved checkpoint."""
        if not self._enabled or self._rank != 0:
            return
        checkpoint_path = os.path.abspath(checkpoint_path)
        version = _make_monotonic_version(self._last_version)
        self._last_version = version
        self._ckpt_manager.protect_checkpoint(checkpoint_path)
        self._queue.put(
            {
                "step": step,
                "checkpoint_path": checkpoint_path,
                "data_timestamp": data_timestamp,
                "version": version,
            }
        )

    def close(self) -> None:
        """Wait for queued dense export tasks to finish."""
        if self._worker is None:
            return
        self._queue.put(None)
        self._worker.join()

    def _worker_loop(self) -> None:
        while True:
            task = self._queue.get()
            try:
                if task is None:
                    return
                self._run_task(task)
            finally:
                self._queue.task_done()

    def _run_task(self, task: Dict[str, Any]) -> None:
        checkpoint_path = task["checkpoint_path"]
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
            log_dir = os.path.join(self._model_dir, "dense_hot_export", "logs")
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
                    )
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
