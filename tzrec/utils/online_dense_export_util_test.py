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
import subprocess
import tempfile
import threading
import unittest
from unittest import mock

from tzrec.utils.online_dense_export_util import (
    OnlineDenseExportManager,
    _build_export_subprocess_env,
    _make_monotonic_version,
    make_version,
    resolve_dense_export_root,
)


class OnlineDenseExportUtilTest(unittest.TestCase):
    """Tests for online dense export utilities."""

    def test_make_version_uses_yyyymmddhhmmss(self) -> None:
        version = make_version(datetime.datetime(2026, 6, 23, 17, 47, 3))

        self.assertEqual(version, "20260623174703")

    def test_make_monotonic_version_keeps_timestamp_format(self) -> None:
        version = _make_monotonic_version(
            "20260623174703", datetime.datetime(2026, 6, 23, 17, 47, 3)
        )

        self.assertEqual(version, "20260623174704")

    def test_resolve_dense_export_root_defaults_when_unset(self) -> None:
        """Without ONLINE_DENSE_EXPORT_DIR, root is <model_dir>/dense_hot_export."""
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ONLINE_DENSE_EXPORT_DIR", None)
            self.assertEqual(
                resolve_dense_export_root("/model"), "/model/dense_hot_export"
            )

    def test_resolve_dense_export_root_honors_env(self) -> None:
        """ONLINE_DENSE_EXPORT_DIR overrides the default root."""
        with mock.patch.dict(os.environ, {"ONLINE_DENSE_EXPORT_DIR": "/serving/dense"}):
            self.assertEqual(resolve_dense_export_root("/model"), "/serving/dense")

    def test_build_export_subprocess_env_removes_torchelastic_env(self) -> None:
        with (
            mock.patch.dict(
                os.environ,
                {
                    "GROUP_RANK": "3",
                    "LOCAL_RANK": "2",
                    "MASTER_ADDR": "elastic-master",
                    "MASTER_PORT": "123",
                    "PATH": "/usr/bin",
                    "PYTHONPATH": "/old/path",
                    "RANK": "2",
                    "TORCHELASTIC_RUN_ID": "job",
                    "TORCHELASTIC_USE_AGENT_STORE": "True",
                    "WORLD_SIZE": "4",
                },
                clear=True,
            ),
            mock.patch(
                "tzrec.utils.online_dense_export_util._get_free_port",
                return_value=45678,
            ),
        ):
            env = _build_export_subprocess_env("/repo")

        self.assertNotIn("GROUP_RANK", env)
        self.assertNotIn("TORCHELASTIC_RUN_ID", env)
        self.assertNotIn("TORCHELASTIC_USE_AGENT_STORE", env)
        self.assertEqual(env["RANK"], "0")
        self.assertEqual(env["LOCAL_RANK"], "0")
        self.assertEqual(env["WORLD_SIZE"], "1")
        self.assertEqual(env["LOCAL_WORLD_SIZE"], "1")
        self.assertEqual(env["MASTER_ADDR"], "127.0.0.1")
        self.assertEqual(env["MASTER_PORT"], "45678")
        self.assertEqual(env["USE_DISTRIBUTED_EMBEDDING"], "1")
        self.assertEqual(env["INPUT_TILE"], "3")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(env["PYTHONPATH"], "/repo:/old/path")

    def test_worker_survives_task_failure(self) -> None:
        """A raising _run_task must not kill the worker or skip later tasks."""
        ckpt = mock.Mock()
        calls = []
        done = threading.Event()

        def fake_run(cmd, **kwargs):
            calls.append(cmd[cmd.index("--version") + 1])
            done.set()
            if len(calls) == 1:
                raise OSError("boom")

        env = {
            "ONLINE_DENSE_EXPORT": "1",
            "USE_DISTRIBUTED_EMBEDDING": "1",
            "RANK": "0",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": "1",
            "LOCAL_WORLD_SIZE": "1",
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = os.path.join(tmp_dir, "model.ckpt-1")
            os.makedirs(ckpt_path)
            with (
                mock.patch.dict(os.environ, env),
                mock.patch.dict(
                    os.environ,
                    {
                        "ONLINE_DENSE_EXPORT_DIR": os.path.join(
                            tmp_dir, "dense_hot_export"
                        )
                    },
                ),
                mock.patch(
                    "tzrec.utils.online_dense_export_util.subprocess.run",
                    side_effect=fake_run,
                ),
            ):
                mgr = OnlineDenseExportManager(
                    model_dir=tmp_dir,
                    pipeline_config_path=os.path.join(tmp_dir, "pipeline.config"),
                    ckpt_manager=ckpt,
                )
                try:
                    mgr.submit(1, ckpt_path, 1.0)
                    self.assertTrue(done.wait(timeout=10))
                    # worker must still be alive: a second submit is processed
                    done.clear()
                    mgr.submit(2, ckpt_path, 2.0)
                    self.assertTrue(done.wait(timeout=10))
                finally:
                    mgr.close()
        self.assertEqual(len(calls), 2)
        # both tasks' checkpoints were unprotected in _run_task's finally
        self.assertEqual(ckpt.unprotect_checkpoint.call_count, 2)

    def test_submit_coalesces_to_latest_pending(self) -> None:
        """A not-yet-started pending task is superseded and unprotected."""
        ckpt = mock.Mock()
        unprotected = []
        ckpt.unprotect_checkpoint.side_effect = unprotected.append
        calls = []
        started = threading.Event()
        proceed = threading.Event()

        def fake_run(cmd, **kwargs):
            calls.append(cmd[cmd.index("--version") + 1])
            if len(calls) == 1:
                started.set()
                proceed.wait(timeout=10)

        env = {
            "ONLINE_DENSE_EXPORT": "1",
            "USE_DISTRIBUTED_EMBEDDING": "1",
            "RANK": "0",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": "1",
            "LOCAL_WORLD_SIZE": "1",
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt1 = os.path.join(tmp_dir, "model.ckpt-1")
            ckpt2 = os.path.join(tmp_dir, "model.ckpt-2")
            ckpt3 = os.path.join(tmp_dir, "model.ckpt-3")
            for path in (ckpt1, ckpt2, ckpt3):
                os.makedirs(path)
            with (
                mock.patch.dict(os.environ, env),
                mock.patch.dict(
                    os.environ,
                    {
                        "ONLINE_DENSE_EXPORT_DIR": os.path.join(
                            tmp_dir, "dense_hot_export"
                        )
                    },
                ),
                mock.patch(
                    "tzrec.utils.online_dense_export_util.subprocess.run",
                    side_effect=fake_run,
                ),
            ):
                mgr = OnlineDenseExportManager(
                    model_dir=tmp_dir,
                    pipeline_config_path=os.path.join(tmp_dir, "pipeline.config"),
                    ckpt_manager=ckpt,
                )
                try:
                    mgr.submit(1, ckpt1, 1.0)
                    self.assertTrue(started.wait(timeout=10))
                    mgr.submit(2, ckpt2, 2.0)  # pending, not yet started
                    mgr.submit(3, ckpt3, 3.0)  # supersedes task 2
                    proceed.set()  # release task 1
                finally:
                    mgr.close()
        # task 2 was superseded (never run); tasks 1 and 3 were processed
        self.assertEqual(len(calls), 2)
        self.assertIn(ckpt2, unprotected)  # unprotected at enqueue

    def test_subprocess_run_uses_timeout_and_survives_timeout_expired(
        self,
    ) -> None:
        """subprocess.run gets the timeout kwarg; TimeoutExpired is handled."""
        ckpt = mock.Mock()
        calls = []
        done = threading.Event()

        def fake_run(cmd, **kwargs):
            calls.append(kwargs.get("timeout"))
            done.set()
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs.get("timeout"))

        env = {
            "ONLINE_DENSE_EXPORT": "1",
            "USE_DISTRIBUTED_EMBEDDING": "1",
            "ONLINE_DENSE_EXPORT_TIMEOUT": "2",
            "RANK": "0",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": "1",
            "LOCAL_WORLD_SIZE": "1",
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = os.path.join(tmp_dir, "model.ckpt-1")
            os.makedirs(ckpt_path)
            with (
                mock.patch.dict(os.environ, env),
                mock.patch.dict(
                    os.environ,
                    {
                        "ONLINE_DENSE_EXPORT_DIR": os.path.join(
                            tmp_dir, "dense_hot_export"
                        )
                    },
                ),
                mock.patch(
                    "tzrec.utils.online_dense_export_util.subprocess.run",
                    side_effect=fake_run,
                ),
            ):
                mgr = OnlineDenseExportManager(
                    model_dir=tmp_dir,
                    pipeline_config_path=os.path.join(tmp_dir, "pipeline.config"),
                    ckpt_manager=ckpt,
                )
                try:
                    mgr.submit(1, ckpt_path, 1.0)
                    self.assertTrue(done.wait(timeout=10))
                finally:
                    mgr.close()
        self.assertEqual(calls, [2.0])
        ckpt.unprotect_checkpoint.assert_called_once()

    def test_finalizer_drains_worker_when_close_not_called(self) -> None:
        """If training raises before close(), the finalizer still stops the worker.

        A live worker keeps the manager reachable (threading._active), so the
        finalizer does not fire via GC; it fires via atexit at interpreter exit.
        Invoke it directly to simulate that path.
        """
        ckpt = mock.Mock()
        done = threading.Event()

        def fake_run(cmd, **kwargs):
            done.set()

        env = {
            "ONLINE_DENSE_EXPORT": "1",
            "USE_DISTRIBUTED_EMBEDDING": "1",
            "RANK": "0",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": "1",
            "LOCAL_WORLD_SIZE": "1",
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = os.path.join(tmp_dir, "model.ckpt-1")
            os.makedirs(ckpt_path)
            with (
                mock.patch.dict(os.environ, env),
                mock.patch.dict(
                    os.environ,
                    {
                        "ONLINE_DENSE_EXPORT_DIR": os.path.join(
                            tmp_dir, "dense_hot_export"
                        )
                    },
                ),
                mock.patch(
                    "tzrec.utils.online_dense_export_util.subprocess.run",
                    side_effect=fake_run,
                ),
            ):
                mgr = OnlineDenseExportManager(
                    model_dir=tmp_dir,
                    pipeline_config_path=os.path.join(tmp_dir, "pipeline.config"),
                    ckpt_manager=ckpt,
                )
                self.assertTrue(mgr._finalizer.alive)
                mgr.submit(1, ckpt_path, 1.0)
                self.assertTrue(done.wait(timeout=10))
                worker = mgr._worker
                # simulate the atexit finalizer firing (training raised, close skipped)
                mgr._finalizer()
            self.assertFalse(worker.is_alive())
            self.assertFalse(mgr._finalizer.alive)

    def test_close_timeout_covers_two_task_timeouts(self) -> None:
        """_close_timeout must cover an in-flight plus a pending task."""
        with mock.patch.dict(os.environ, {"ONLINE_DENSE_EXPORT_TIMEOUT": "200"}):
            mgr = OnlineDenseExportManager(
                model_dir="/tmp/tzrec_unused_model_dir",
                pipeline_config_path="/tmp/tzrec_unused_pipeline.config",
                ckpt_manager=mock.Mock(),
            )
        self.assertGreaterEqual(mgr._close_timeout, 2 * mgr._export_timeout)

    def test_close_keeps_finalizer_when_worker_outlives_join(self) -> None:
        """close() detaches the finalizer only after the worker stops."""
        ckpt = mock.Mock()
        started = threading.Event()
        proceed = threading.Event()

        def fake_run(cmd, **kwargs):
            started.set()
            proceed.wait(timeout=30)

        env = {
            "ONLINE_DENSE_EXPORT": "1",
            "USE_DISTRIBUTED_EMBEDDING": "1",
            "RANK": "0",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": "1",
            "LOCAL_WORLD_SIZE": "1",
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = os.path.join(tmp_dir, "model.ckpt-1")
            os.makedirs(ckpt_path)
            with (
                mock.patch.dict(os.environ, env),
                mock.patch.dict(
                    os.environ,
                    {
                        "ONLINE_DENSE_EXPORT_DIR": os.path.join(
                            tmp_dir, "dense_hot_export"
                        )
                    },
                ),
                mock.patch(
                    "tzrec.utils.online_dense_export_util.subprocess.run",
                    side_effect=fake_run,
                ),
            ):
                mgr = OnlineDenseExportManager(
                    model_dir=tmp_dir,
                    pipeline_config_path=os.path.join(tmp_dir, "pipeline.config"),
                    ckpt_manager=ckpt,
                )
                try:
                    mgr._close_timeout = 0.05  # force close to time out mid-task
                    mgr.submit(1, ckpt_path, 1.0)
                    self.assertTrue(started.wait(timeout=10))
                    mgr.close()  # worker still in-flight; join times out
                    self.assertTrue(mgr._worker.is_alive())
                    # backstop must stay attached so atexit can still drain
                    self.assertTrue(mgr._finalizer.alive)
                finally:
                    proceed.set()
                    mgr._finalizer()
            self.assertFalse(mgr._worker.is_alive())
            self.assertFalse(mgr._finalizer.alive)

    def test_prune_export_logs_keeps_newest_k(self) -> None:
        """Retain the newest K export logs; leave non-log files alone."""
        with tempfile.TemporaryDirectory() as log_dir:
            for v in (
                "20260101000001",
                "20260101000002",
                "20260101000003",
                "20260101000004",
                "20260101000005",
            ):
                with open(os.path.join(log_dir, f"{v}.log"), "w") as f:
                    f.write("x")
            open(os.path.join(log_dir, "notes.txt"), "w").close()
            with mock.patch.dict(os.environ, {"ONLINE_DENSE_EXPORT_KEEP_LOGS": "3"}):
                mgr = OnlineDenseExportManager(
                    model_dir="/tmp/tzrec_unused_model_dir",
                    pipeline_config_path="/tmp/tzrec_unused_pipeline.config",
                    ckpt_manager=mock.Mock(),
                )
            mgr._prune_export_logs(log_dir)
            self.assertEqual(
                sorted(os.listdir(log_dir)),
                [
                    "20260101000003.log",
                    "20260101000004.log",
                    "20260101000005.log",
                    "notes.txt",
                ],
            )

    def test_run_task_prunes_old_logs(self) -> None:
        """_run_task prunes old export logs after each attempt."""
        ckpt = mock.Mock()
        done = threading.Event()

        def fake_run(cmd, **kwargs):
            done.set()

        env = {
            "ONLINE_DENSE_EXPORT": "1",
            "USE_DISTRIBUTED_EMBEDDING": "1",
            "ONLINE_DENSE_EXPORT_KEEP_LOGS": "2",
            "RANK": "0",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": "1",
            "LOCAL_WORLD_SIZE": "1",
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_dir = os.path.join(tmp_dir, "dense_hot_export", "logs")
            os.makedirs(log_dir)
            for v in (
                "20260101000001",
                "20260101000002",
                "20260101000003",
            ):
                open(os.path.join(log_dir, f"{v}.log"), "w").close()
            ckpt_path = os.path.join(tmp_dir, "model.ckpt-1")
            os.makedirs(ckpt_path)
            with (
                mock.patch.dict(os.environ, env),
                mock.patch.dict(
                    os.environ,
                    {
                        "ONLINE_DENSE_EXPORT_DIR": os.path.join(
                            tmp_dir, "dense_hot_export"
                        )
                    },
                ),
                mock.patch(
                    "tzrec.utils.online_dense_export_util.subprocess.run",
                    side_effect=fake_run,
                ),
            ):
                mgr = OnlineDenseExportManager(
                    model_dir=tmp_dir,
                    pipeline_config_path=os.path.join(tmp_dir, "pipeline.config"),
                    ckpt_manager=ckpt,
                )
                try:
                    mgr.submit(1, ckpt_path, 1.0)
                    self.assertTrue(done.wait(timeout=10))
                finally:
                    mgr.close()
            remaining = sorted(f for f in os.listdir(log_dir) if f.endswith(".log"))
            # the new attempt's log plus the single newest old log survive
            self.assertEqual(len(remaining), 2)
            self.assertIn("20260101000003.log", remaining)
            self.assertNotIn("20260101000001.log", remaining)
            self.assertNotIn("20260101000002.log", remaining)

    def test_init_requires_export_dir_when_enabled(self) -> None:
        """ONLINE_DENSE_EXPORT=1 without ONLINE_DENSE_EXPORT_DIR must fail fast."""
        ckpt = mock.Mock()
        env = {
            "ONLINE_DENSE_EXPORT": "1",
            "USE_DISTRIBUTED_EMBEDDING": "1",
            "RANK": "0",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": "1",
            "LOCAL_WORLD_SIZE": "1",
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            with mock.patch.dict(os.environ, env, clear=False):
                os.environ.pop("ONLINE_DENSE_EXPORT_DIR", None)
                # a remote model_dir surfaces the missing-dir error, not a
                # remote-path error: the dir requirement fires first.
                with self.assertRaisesRegex(RuntimeError, "ONLINE_DENSE_EXPORT_DIR"):
                    OnlineDenseExportManager(
                        model_dir="oss://bucket/m",
                        pipeline_config_path=os.path.join(tmp_dir, "pipeline.config"),
                        ckpt_manager=ckpt,
                    )
                # a local model_dir is not a substitute for the explicit dir
                with self.assertRaisesRegex(RuntimeError, "ONLINE_DENSE_EXPORT_DIR"):
                    OnlineDenseExportManager(
                        model_dir=tmp_dir,
                        pipeline_config_path=os.path.join(tmp_dir, "pipeline.config"),
                        ckpt_manager=ckpt,
                    )

    def test_init_rejects_remote_export_root_and_pipeline_config(self) -> None:
        """With the dir set, remote export_root/pipeline_config fail fast."""
        ckpt = mock.Mock()
        env = {
            "ONLINE_DENSE_EXPORT": "1",
            "USE_DISTRIBUTED_EMBEDDING": "1",
            "RANK": "0",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": "1",
            "LOCAL_WORLD_SIZE": "1",
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            local_dir = os.path.join(tmp_dir, "dhx")
            with mock.patch.dict(
                os.environ, {**env, "ONLINE_DENSE_EXPORT_DIR": local_dir}
            ):
                with self.assertRaisesRegex(RuntimeError, "local pipeline_config_path"):
                    OnlineDenseExportManager(
                        model_dir=tmp_dir,
                        pipeline_config_path="dfs://bucket/pipeline.config",
                        ckpt_manager=ckpt,
                    )
                with mock.patch.dict(
                    os.environ, {"ONLINE_DENSE_EXPORT_DIR": "oss://bucket/export"}
                ):
                    with self.assertRaisesRegex(RuntimeError, "local export_root"):
                        OnlineDenseExportManager(
                            model_dir=tmp_dir,
                            pipeline_config_path=os.path.join(
                                tmp_dir, "pipeline.config"
                            ),
                            ckpt_manager=ckpt,
                        )

    def test_init_allows_remote_model_dir_when_export_root_overridden(self) -> None:
        """A local ONLINE_DENSE_EXPORT_DIR decouples the publish tree from model_dir."""
        ckpt = mock.Mock()
        env = {
            "ONLINE_DENSE_EXPORT": "1",
            "USE_DISTRIBUTED_EMBEDDING": "1",
            "RANK": "0",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": "1",
            "LOCAL_WORLD_SIZE": "1",
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            override = os.path.join(tmp_dir, "serving_root")
            with mock.patch.dict(
                os.environ, {**env, "ONLINE_DENSE_EXPORT_DIR": override}
            ):
                mgr = OnlineDenseExportManager(
                    model_dir="oss://bucket/m",
                    pipeline_config_path=os.path.join(tmp_dir, "pipeline.config"),
                    ckpt_manager=ckpt,
                )
                try:
                    self.assertEqual(mgr._export_root, os.path.abspath(override))
                finally:
                    mgr.close()

    def test_run_task_uses_overridden_export_root(self) -> None:
        """ONLINE_DENSE_EXPORT_DIR redirects the log dir and is propagated abspath'd."""
        ckpt = mock.Mock()
        captured = {}
        done = threading.Event()

        def fake_run(cmd, **kwargs):
            captured["env"] = kwargs.get("env")
            done.set()

        env = {
            "ONLINE_DENSE_EXPORT": "1",
            "USE_DISTRIBUTED_EMBEDDING": "1",
            "RANK": "0",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": "1",
            "LOCAL_WORLD_SIZE": "1",
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            override = os.path.join(tmp_dir, "serving_dense")
            ckpt_path = os.path.join(tmp_dir, "model.ckpt-1")
            os.makedirs(ckpt_path)
            with (
                mock.patch.dict(
                    os.environ, {**env, "ONLINE_DENSE_EXPORT_DIR": override}
                ),
                mock.patch(
                    "tzrec.utils.online_dense_export_util.subprocess.run",
                    side_effect=fake_run,
                ),
            ):
                mgr = OnlineDenseExportManager(
                    model_dir=tmp_dir,
                    pipeline_config_path=os.path.join(tmp_dir, "pipeline.config"),
                    ckpt_manager=ckpt,
                )
                try:
                    self.assertEqual(mgr._export_root, os.path.abspath(override))
                    mgr.submit(1, ckpt_path, 1.0)
                    self.assertTrue(done.wait(timeout=10))
                finally:
                    mgr.close()
            self.assertEqual(
                captured["env"]["ONLINE_DENSE_EXPORT_DIR"],
                os.path.abspath(override),
            )
            logs_dir = os.path.join(override, "logs")
            self.assertTrue(os.path.isdir(logs_dir))
            self.assertEqual(
                len([f for f in os.listdir(logs_dir) if f.endswith(".log")]), 1
            )
            self.assertFalse(os.path.exists(os.path.join(tmp_dir, "dense_hot_export")))


if __name__ == "__main__":
    unittest.main()
