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

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tzrec.tests.ci_job import JobRunner, artifact_json, artifact_text, capacity


class JobTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)
        config = json.loads(Path(__file__).with_name("ci_config.json").read_text())
        self.config = dict(config, **config["hardware"]["h20"])
        self.config.update(name="node-a", hardware="h20")
        self.config["devices"] = {
            "verified_pairs": [[str(i), str(i + 1)] for i in range(0, 8, 2)]
        }

    def tearDown(self):
        self.temp.cleanup()

    def test_cpu_and_container_reservations_bound_capacity(self):
        self.assertEqual(capacity(self.config, 184, 1800, 0), 4)
        self.assertEqual(capacity(self.config, 184, 1800, 13), 1)
        self.assertEqual(capacity(self.config, 20, 100, 0), 0)

    def test_gpu_container_mounts_no_socket_or_sibling_and_allowlists_environment(self):
        runner = JobRunner(self.config, "gpu", self.path, self.path / "job")
        runner.root.mkdir()
        runner.image = "sha256:known"
        calls = []
        with (
            mock.patch.dict(
                os.environ,
                {"GITHUB_TOKEN": "private", "CI_ALIKAFKA_INSTANCE_ID": "private-data"},
            ),
            mock.patch(
                "tzrec.tests.ci_job.docker",
                side_effect=lambda *a, **k: (
                    calls.append(a) or SimpleNamespace(stdout="id")
                ),
            ),
        ):
            runner.start(
                "worker-0",
                ["python", "tzrec/tests/worker.py"],
                source=self.path / "checkout",
                devices=["GPU-a", "GPU-b"],
                cores=2,
                memory=4,
                shm=1,
            )
        argv = list(calls[0])
        text = " ".join(argv)
        self.assertNotIn("private", text)
        self.assertNotIn("docker.sock", text)
        self.assertNotIn("--privileged", text)
        self.assertIn('"device=GPU-a,GPU-b"', argv)
        self.assertIn(str(runner.root / "env") + ":/job/env:ro", argv)
        self.assertIn(str(runner.root / "passwd") + ":/etc/passwd:ro", argv)
        entries = (runner.root / "passwd").read_text().splitlines()
        own = [
            entry.split(":")
            for entry in entries
            if int(entry.split(":")[2]) == os.getuid()
        ]
        self.assertEqual(len(own), 1)
        self.assertEqual(own[0][3], str(os.getgid()))
        self.assertEqual(own[0][5], "/job/home")

    def test_artifacts_do_not_follow_symlinks_or_read_special_files(self):
        secret = self.path / "secret.json"
        secret.write_text('{"private":"value"}')
        root = self.path / "session"
        root.mkdir()
        (root / "phase.json").symlink_to(secret)
        with self.assertRaises(OSError):
            artifact_json(root, "phase.json")
        (root / "phase.json").unlink()
        os.mkfifo(root / "phase.json")
        with self.assertRaises(ValueError):
            artifact_json(root, "phase.json")

    def test_cpu_never_passes_accelerators(self):
        runner = JobRunner(self.config, "cpu", self.path, self.path / "job")
        runner.root.mkdir()
        runner.image = "cpu"
        with mock.patch(
            "tzrec.tests.ci_job.docker", return_value=SimpleNamespace(stdout="id")
        ) as d:
            runner.start(
                "baseline",
                ["python"],
                source=self.path / "source",
                devices=[],
                cores=2,
                memory=4,
                shm=1,
            )
        argv = d.call_args.args
        self.assertNotIn("--gpus", argv)
        self.assertIn("NVIDIA_VISIBLE_DEVICES=void", argv)

    def test_receipt_directory_exists_before_persisting_job_identity(self):
        runner = JobRunner(self.config, "cpu", self.path, self.path / "job")
        observed = []

        def persist(path, state):
            observed.append((runner.root / "status").is_dir())

        with (
            mock.patch("tzrec.tests.ci_job.atomic", side_effect=persist),
            mock.patch(
                "tzrec.tests.ci_job.shutil.copytree",
                side_effect=OSError("stop before containers"),
            ),
        ):
            with self.assertRaises(OSError):
                runner.run()
        self.assertEqual(observed, [True])

    def test_probe_failure_cleans_all_owned_pair_containers(self):
        runner = JobRunner(self.config, "gpu", self.path, self.path / "job")
        runner.root.mkdir()
        runner.image = "gpu"
        rows = [dict(index=i, id=f"GPU-{i}", bus=f"{i}:00.0") for i in range(8)]
        with (
            mock.patch("tzrec.tests.ci_job.devices.inventory", return_value=rows),
            mock.patch(
                "tzrec.tests.ci_job.docker", return_value=SimpleNamespace(stdout="id")
            ) as calls,
            mock.patch("tzrec.tests.ci_job.subprocess.run"),
            mock.patch.object(
                runner, "wait", side_effect=TimeoutError("probe timeout")
            ),
        ):
            with self.assertRaises(TimeoutError):
                runner.probe(self.path)
            self.assertEqual(len(runner.containers), 4)
            runner.cleanup(2)
        self.assertFalse(runner.containers)
        starts = [c.args for c in calls.call_args_list if c.args[0] == "run"]
        self.assertEqual(len(starts), 4)
        self.assertTrue(
            all("tzci.node=node-a" in c and "tzci.role=accel" in c for c in starts)
        )
        removed = [c.args for c in calls.call_args_list if c.args[0] == "rm"]
        self.assertEqual(len(removed), 4)

    def test_cpu_probe_never_enumerates_accelerators(self):
        runner = JobRunner(self.config, "cpu", self.path, self.path / "job")
        with (
            mock.patch.object(runner, "start", return_value="cpu-probe") as start,
            mock.patch.object(runner, "wait", return_value=0),
            mock.patch.object(runner, "remove"),
            mock.patch("tzrec.tests.ci_job.devices.inventory") as inventory,
        ):
            self.assertEqual(
                runner.probe(self.path), dict(baseline=[], verified_pairs=[])
            )
        inventory.assert_not_called()
        self.assertEqual(start.call_args.kwargs["devices"], [])

    def test_cleanup_attempts_every_container_after_one_removal_fails(self):
        runner = JobRunner(self.config, "gpu", self.path, self.path / "job")
        runner.root.mkdir()
        runner.containers = ["owned-a", "owned-b"]
        with (
            mock.patch("tzrec.tests.ci_job.docker"),
            mock.patch.object(
                runner, "remove", side_effect=[RuntimeError("lost"), None]
            ) as remove,
        ):
            with self.assertRaisesRegex(RuntimeError, "job cleanup failed"):
                runner.cleanup(0)
        self.assertEqual(
            remove.call_args_list, [mock.call("owned-a"), mock.call("owned-b")]
        )
        self.assertEqual(
            json.loads((runner.root / "state.json").read_text())["code"], 2
        )

    def test_failure_receipts_and_bounded_log_tails_are_retained(self):
        runner = JobRunner(self.config, "gpu", self.path, self.path / "job")
        key = "a" * 24
        results = runner.root / "session/results"
        logs = runner.root / "session/logs"
        results.mkdir(parents=True)
        logs.mkdir()
        (results / (key + ".json")).write_text('{"code":1,"cases":[{"id":"failed"}]}')
        (logs / (key + ".log")).write_text("x" * (2 * 1024 * 1024) + "failure detail\n")
        runner.cleanup(1)
        self.assertTrue((runner.root / "report/results" / (key + ".json")).exists())
        text = artifact_text(runner.root / "report/logs", key + ".log")
        self.assertTrue(text.endswith("failure detail\n"))
        self.assertIn("Earlier log output omitted", text)
        self.assertLess(len(text), 1024 * 1024 + 100)


if __name__ == "__main__":
    unittest.main()
