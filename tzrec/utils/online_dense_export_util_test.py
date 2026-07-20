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
import tempfile
import threading
import unittest
from unittest import mock

from tzrec.utils.online_dense_export_util import (
    OnlineDenseExportManager,
    _build_export_subprocess_env,
    _make_monotonic_version,
    make_version,
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


if __name__ == "__main__":
    unittest.main()
