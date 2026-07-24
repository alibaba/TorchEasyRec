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
import json
import os
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from typing import Any, Callable, Dict, List
from unittest import mock

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed._shard.metadata import ShardMetadata
from torch.distributed._shard.sharded_tensor import Shard, ShardedTensor
from torch.distributed._shard.sharded_tensor.metadata import (
    ShardedTensorMetadata,
    TensorProperties,
)
from torch.distributed._tensor import DTensor, Replicate
from torch.distributed.device_mesh import init_device_mesh

from tzrec.utils import misc_util
from tzrec.utils.online_dense_export_util import (
    OnlineDenseExportManager,
    _atomic_write_json,
    _make_monotonic_version,
    _read_current_version,
    make_version,
    resolve_dense_export_root,
)


def _base_env(tmp_dir: str, **extra: str) -> Dict[str, str]:
    env = {
        "ONLINE_DENSE_EXPORT": "1",
        "USE_DISTRIBUTED_EMBEDDING": "1",
        "ONLINE_DENSE_EXPORT_DIR": os.path.join(tmp_dir, "serving_root"),
        "ONLINE_DENSE_EXPORT_STEPS": "5",
        "RANK": "0",
        "LOCAL_RANK": "0",
        "WORLD_SIZE": "1",
        "LOCAL_WORLD_SIZE": "1",
    }
    env.update(extra)
    return env


def _mock_model(state: Dict[str, Any]) -> mock.Mock:
    model = mock.Mock()
    model.state_dict.return_value = state
    return model


def _wait_for(cond: Callable[[], bool], timeout: float = 10.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cond():
            return True
        time.sleep(0.01)
    return cond()


class _TinyModel(nn.Module):
    """Stand-in dense graph module with a single loadable parameter."""

    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.zeros(2))


_BUILD_PATCHES = "tzrec.utils.online_dense_export_util.{}"


def _dummy_pipeline_config() -> SimpleNamespace:
    return SimpleNamespace(
        feature_configs=[],
        data_config=SimpleNamespace(label_fields=[]),
        model_config=SimpleNamespace(),
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
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                resolve_dense_export_root("/model"), "/model/dense_hot_export"
            )

    def test_resolve_dense_export_root_honors_env(self) -> None:
        """ONLINE_DENSE_EXPORT_DIR names the serving root; dense_hot_export appended."""
        with mock.patch.dict(
            os.environ, {"ONLINE_DENSE_EXPORT_DIR": "/serving/dense"}, clear=True
        ):
            self.assertEqual(
                resolve_dense_export_root("/model"), "/serving/dense/dense_hot_export"
            )

    # --- init-time validation ---

    def test_init_requires_export_dir_when_enabled(self) -> None:
        """ONLINE_DENSE_EXPORT=1 without ONLINE_DENSE_EXPORT_DIR must fail fast."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            env = _base_env(tmp_dir)
            del env["ONLINE_DENSE_EXPORT_DIR"]
            with mock.patch.dict(os.environ, env, clear=True):
                # a remote model_dir surfaces the missing-dir error, not a
                # remote-path error: the dir requirement fires first.
                with self.assertRaisesRegex(RuntimeError, "ONLINE_DENSE_EXPORT_DIR"):
                    OnlineDenseExportManager(
                        model_dir="oss://bucket/m",
                        pipeline_config_path=os.path.join(tmp_dir, "pipeline.config"),
                        model=_mock_model({}),
                    )
                # a local model_dir is not a substitute for the explicit dir
                with self.assertRaisesRegex(RuntimeError, "ONLINE_DENSE_EXPORT_DIR"):
                    OnlineDenseExportManager(
                        model_dir=tmp_dir,
                        pipeline_config_path=os.path.join(tmp_dir, "pipeline.config"),
                        model=_mock_model({}),
                    )

    def test_init_requires_trigger_config_when_enabled(self) -> None:
        """Without STEPS or INTERVAL the cadence is undefined: fail fast."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            env = _base_env(tmp_dir)
            del env["ONLINE_DENSE_EXPORT_STEPS"]
            with mock.patch.dict(os.environ, env, clear=True):
                with self.assertRaisesRegex(RuntimeError, "ONLINE_DENSE_EXPORT_STEPS"):
                    OnlineDenseExportManager(
                        model_dir=tmp_dir,
                        pipeline_config_path=os.path.join(tmp_dir, "pipeline.config"),
                        model=_mock_model({}),
                    )

    def test_init_rejects_bad_quorum(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            env = _base_env(
                tmp_dir,
                **{
                    "ONLINE_DENSE_EXPORT_STEPS": "0",
                    "ONLINE_DENSE_EXPORT_INTERVAL": "60",
                    "ONLINE_DENSE_EXPORT_QUORUM": "1.5",
                },
            )
            with mock.patch.dict(os.environ, env, clear=True):
                with self.assertRaisesRegex(RuntimeError, "ONLINE_DENSE_EXPORT_QUORUM"):
                    OnlineDenseExportManager(
                        model_dir=tmp_dir,
                        pipeline_config_path=os.path.join(tmp_dir, "pipeline.config"),
                        model=_mock_model({}),
                    )

    def test_init_rejects_remote_export_root_and_pipeline_config(self) -> None:
        """With the dir set, remote export_root/pipeline_config fail fast."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            local_dir = os.path.join(tmp_dir, "dhx")
            env = _base_env(tmp_dir, **{"ONLINE_DENSE_EXPORT_DIR": local_dir})
            with mock.patch.dict(os.environ, env, clear=True):
                with self.assertRaisesRegex(RuntimeError, "local pipeline_config_path"):
                    OnlineDenseExportManager(
                        model_dir=tmp_dir,
                        pipeline_config_path="dfs://bucket/pipeline.config",
                        model=_mock_model({}),
                    )
            env = _base_env(tmp_dir, **{"ONLINE_DENSE_EXPORT_DIR": "oss://bucket/x"})
            with mock.patch.dict(os.environ, env, clear=True):
                with self.assertRaisesRegex(RuntimeError, "local export_root"):
                    OnlineDenseExportManager(
                        model_dir=tmp_dir,
                        pipeline_config_path=os.path.join(tmp_dir, "pipeline.config"),
                        model=_mock_model({}),
                    )

    def test_init_allows_remote_model_dir_when_export_root_overridden(self) -> None:
        """A local ONLINE_DENSE_EXPORT_DIR decouples the publish tree from model_dir."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            override = os.path.join(tmp_dir, "serving_root")
            env = _base_env(tmp_dir, **{"ONLINE_DENSE_EXPORT_DIR": override})
            with (
                mock.patch.dict(os.environ, env, clear=True),
                mock.patch.object(
                    OnlineDenseExportManager, "_build_export_graph", return_value=[]
                ),
            ):
                mgr = OnlineDenseExportManager(
                    model_dir="oss://bucket/m",
                    pipeline_config_path=os.path.join(tmp_dir, "pipeline.config"),
                    model=_mock_model({}),
                )
                try:
                    self.assertEqual(
                        mgr._export_root,
                        os.path.abspath(os.path.join(override, "dense_hot_export")),
                    )
                finally:
                    mgr.close()

    # --- one-time build phase ---

    def test_build_scopes_input_tile_env_and_restores_it(self) -> None:
        """INPUT_TILE=3 holds only during the twin build, even on failure."""
        seen: Dict[str, Any] = {}

        def fake_finalize(*args: Any, **kwargs: Any) -> None:
            seen["input_tile"] = os.environ.get("INPUT_TILE")

        fake_gm = mock.Mock()
        fake_gm.state_dict.return_value = {}
        with tempfile.TemporaryDirectory() as tmp_dir:
            env = _base_env(tmp_dir)
            with mock.patch.dict(os.environ, env, clear=True):
                with (
                    mock.patch(
                        _BUILD_PATCHES.format("config_util.load_pipeline_config"),
                        return_value=_dummy_pipeline_config(),
                    ),
                    mock.patch("tzrec.main._create_features", return_value=[]),
                    mock.patch("tzrec.main._create_model", return_value=mock.Mock()),
                    mock.patch(
                        "tzrec.models.model.ScriptWrapper",
                        side_effect=lambda m: m,
                    ),
                    mock.patch(
                        _BUILD_PATCHES.format("create_dense_export_warmup_data"),
                        return_value={},
                    ),
                    mock.patch(
                        _BUILD_PATCHES.format("build_dense_graph_module"),
                        return_value=(fake_gm, mock.Mock(), {}),
                    ),
                    mock.patch(
                        _BUILD_PATCHES.format("finalize_dense_export"),
                        side_effect=fake_finalize,
                    ),
                ):
                    mgr = OnlineDenseExportManager(
                        model_dir=tmp_dir,
                        pipeline_config_path=os.path.join(tmp_dir, "pipeline.config"),
                        model=_mock_model({}),
                    )
                    mgr.close()
                self.assertEqual(seen["input_tile"], "3")
                self.assertNotIn("INPUT_TILE", os.environ)

                # a failing build must also restore the env
                with (
                    mock.patch(
                        _BUILD_PATCHES.format("config_util.load_pipeline_config"),
                        side_effect=RuntimeError("boom"),
                    ),
                ):
                    with self.assertRaisesRegex(RuntimeError, "boom"):
                        OnlineDenseExportManager(
                            model_dir=tmp_dir,
                            pipeline_config_path=os.path.join(
                                tmp_dir, "pipeline.config"
                            ),
                            model=_mock_model({}),
                        )
                self.assertNotIn("INPUT_TILE", os.environ)

    def test_build_maps_user_twin_keys_to_non_user_sources(self) -> None:
        """INPUT_TILE=3 user-side keys resolve to their non-user training twins."""
        gm_state = {
            "model.mlp.weight": None,
            "model.eg.ebc_user.embedding_bags.t.weight": None,
        }
        dmp_state = {
            "model.mlp.weight": torch.zeros(2),
            "model.eg.ebc.embedding_bags.t.weight": torch.zeros(2),
        }
        fake_gm = mock.Mock()
        fake_gm.state_dict.return_value = gm_state
        with tempfile.TemporaryDirectory() as tmp_dir:
            env = _base_env(tmp_dir)
            with (
                mock.patch.dict(os.environ, env, clear=True),
                mock.patch(
                    _BUILD_PATCHES.format("config_util.load_pipeline_config"),
                    return_value=_dummy_pipeline_config(),
                ),
                mock.patch("tzrec.main._create_features", return_value=[]),
                mock.patch("tzrec.main._create_model", return_value=mock.Mock()),
                mock.patch("tzrec.models.model.ScriptWrapper", side_effect=lambda m: m),
                mock.patch(
                    _BUILD_PATCHES.format("create_dense_export_warmup_data"),
                    return_value={},
                ),
                mock.patch(
                    _BUILD_PATCHES.format("build_dense_graph_module"),
                    return_value=(fake_gm, mock.Mock(), {}),
                ),
                mock.patch(_BUILD_PATCHES.format("finalize_dense_export")),
            ):
                mgr = OnlineDenseExportManager(
                    model_dir=tmp_dir,
                    pipeline_config_path=os.path.join(tmp_dir, "pipeline.config"),
                    model=_mock_model(dmp_state),
                )
                try:
                    self.assertEqual(
                        mgr._state_pairs,
                        [
                            (
                                "model.eg.ebc_user.embedding_bags.t.weight",
                                "model.eg.ebc.embedding_bags.t.weight",
                            ),
                            ("model.mlp.weight", "model.mlp.weight"),
                        ],
                    )
                finally:
                    mgr.close()

    def test_build_fails_on_ungatherable_state(self) -> None:
        """A dense-graph key with no training source must abort startup."""
        fake_gm = mock.Mock()
        fake_gm.state_dict.return_value = {"model.missing.weight": None}
        with tempfile.TemporaryDirectory() as tmp_dir:
            env = _base_env(tmp_dir)
            with (
                mock.patch.dict(os.environ, env, clear=True),
                mock.patch(
                    _BUILD_PATCHES.format("config_util.load_pipeline_config"),
                    return_value=_dummy_pipeline_config(),
                ),
                mock.patch("tzrec.main._create_features", return_value=[]),
                mock.patch("tzrec.main._create_model", return_value=mock.Mock()),
                mock.patch("tzrec.models.model.ScriptWrapper", side_effect=lambda m: m),
                mock.patch(
                    _BUILD_PATCHES.format("create_dense_export_warmup_data"),
                    return_value={},
                ),
                mock.patch(
                    _BUILD_PATCHES.format("build_dense_graph_module"),
                    return_value=(fake_gm, mock.Mock(), {}),
                ),
                mock.patch(_BUILD_PATCHES.format("finalize_dense_export")),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "cannot gather 1 dense model states"
                ):
                    OnlineDenseExportManager(
                        model_dir=tmp_dir,
                        pipeline_config_path=os.path.join(tmp_dir, "pipeline.config"),
                        model=_mock_model({}),
                    )

    def test_build_rejects_unsupported_state_type(self) -> None:
        fake_gm = mock.Mock()
        fake_gm.state_dict.return_value = {"model.w": None}
        with tempfile.TemporaryDirectory() as tmp_dir:
            env = _base_env(tmp_dir)
            with (
                mock.patch.dict(os.environ, env, clear=True),
                mock.patch(
                    _BUILD_PATCHES.format("config_util.load_pipeline_config"),
                    return_value=_dummy_pipeline_config(),
                ),
                mock.patch("tzrec.main._create_features", return_value=[]),
                mock.patch("tzrec.main._create_model", return_value=mock.Mock()),
                mock.patch("tzrec.models.model.ScriptWrapper", side_effect=lambda m: m),
                mock.patch(
                    _BUILD_PATCHES.format("create_dense_export_warmup_data"),
                    return_value={},
                ),
                mock.patch(
                    _BUILD_PATCHES.format("build_dense_graph_module"),
                    return_value=(fake_gm, mock.Mock(), {}),
                ),
                mock.patch(_BUILD_PATCHES.format("finalize_dense_export")),
            ):
                with self.assertRaisesRegex(RuntimeError, "unsupported state type"):
                    OnlineDenseExportManager(
                        model_dir=tmp_dir,
                        pipeline_config_path=os.path.join(tmp_dir, "pipeline.config"),
                        model=_mock_model({"model.w": "not-a-tensor"}),
                    )

    # --- worker / thread semantics (drive _enqueue, patch _run_task) ---

    def test_worker_survives_task_failure(self) -> None:
        """A raising _run_task must not kill the worker or skip later tasks."""
        calls: List[str] = []
        done = threading.Event()

        def fake_run_task(task: Dict[str, Any]) -> None:
            calls.append(task["version"])
            done.set()
            if len(calls) == 1:
                raise OSError("boom")

        with tempfile.TemporaryDirectory() as tmp_dir:
            with (
                mock.patch.dict(os.environ, _base_env(tmp_dir), clear=True),
                mock.patch.object(
                    OnlineDenseExportManager, "_build_export_graph", return_value=[]
                ),
            ):
                mgr = OnlineDenseExportManager(
                    model_dir=tmp_dir,
                    pipeline_config_path=os.path.join(tmp_dir, "pipeline.config"),
                    model=_mock_model({}),
                )
                try:
                    with mock.patch.object(mgr, "_run_task", side_effect=fake_run_task):
                        mgr._enqueue(1, 1.0, {})
                        self.assertTrue(done.wait(timeout=10))
                        # worker must still be alive: a second task is processed
                        done.clear()
                        mgr._enqueue(2, 2.0, {})
                        self.assertTrue(done.wait(timeout=10))
                finally:
                    mgr.close()
        self.assertEqual(len(calls), 2)

    def test_enqueue_coalesces_to_latest_pending(self) -> None:
        """A not-yet-started pending task is superseded by the newest one."""
        snapshots: List[Dict[str, str]] = []
        started = threading.Event()
        proceed = threading.Event()

        def fake_run_task(task: Dict[str, Any]) -> None:
            snapshots.append(task["snapshot"])
            if len(snapshots) == 1:
                started.set()
                proceed.wait(timeout=10)

        with tempfile.TemporaryDirectory() as tmp_dir:
            with (
                mock.patch.dict(os.environ, _base_env(tmp_dir), clear=True),
                mock.patch.object(
                    OnlineDenseExportManager, "_build_export_graph", return_value=[]
                ),
            ):
                mgr = OnlineDenseExportManager(
                    model_dir=tmp_dir,
                    pipeline_config_path=os.path.join(tmp_dir, "pipeline.config"),
                    model=_mock_model({}),
                )
                try:
                    # close() must drain inside the patch: after the fake
                    # releases, the worker still picks up the pending task.
                    with mock.patch.object(mgr, "_run_task", side_effect=fake_run_task):
                        mgr._enqueue(1, 1.0, {"m": "a"})
                        self.assertTrue(started.wait(timeout=10))
                        mgr._enqueue(2, 2.0, {"m": "b"})  # pending, not yet started
                        mgr._enqueue(3, 3.0, {"m": "c"})  # supersedes b
                        proceed.set()
                        mgr.close()
                finally:
                    mgr.close()
        # b was superseded (never run); a and c were processed
        self.assertEqual(snapshots, [{"m": "a"}, {"m": "c"}])

    def test_finalizer_drains_worker_when_close_not_called(self) -> None:
        """If training raises before close(), the finalizer still stops the worker."""
        done = threading.Event()

        def fake_run_task(task: Dict[str, Any]) -> None:
            done.set()

        with tempfile.TemporaryDirectory() as tmp_dir:
            with (
                mock.patch.dict(os.environ, _base_env(tmp_dir), clear=True),
                mock.patch.object(
                    OnlineDenseExportManager, "_build_export_graph", return_value=[]
                ),
            ):
                mgr = OnlineDenseExportManager(
                    model_dir=tmp_dir,
                    pipeline_config_path=os.path.join(tmp_dir, "pipeline.config"),
                    model=_mock_model({}),
                )
                self.assertTrue(mgr._finalizer.alive)
                with mock.patch.object(mgr, "_run_task", side_effect=fake_run_task):
                    mgr._enqueue(1, 1.0, {})
                    self.assertTrue(done.wait(timeout=10))
                worker = mgr._worker
                # simulate the atexit finalizer firing (training raised, close skipped)
                mgr._finalizer()
            self.assertFalse(worker.is_alive())
            self.assertFalse(mgr._finalizer.alive)

    def test_close_timeout_covers_two_task_timeouts(self) -> None:
        """_close_timeout must cover an in-flight plus a pending task."""
        with mock.patch.dict(
            os.environ, {"ONLINE_DENSE_EXPORT_TIMEOUT": "200"}, clear=True
        ):
            mgr = OnlineDenseExportManager(
                model_dir="/tmp/tzrec_unused_model_dir",
                pipeline_config_path="/tmp/tzrec_unused_pipeline.config",
                model=_mock_model({}),
            )
        self.assertGreaterEqual(mgr._close_timeout, 2 * mgr._export_timeout)

    def test_close_keeps_finalizer_when_worker_outlives_join(self) -> None:
        """close() detaches the finalizer only after the worker stops."""
        started = threading.Event()
        proceed = threading.Event()

        def fake_run_task(task: Dict[str, Any]) -> None:
            started.set()
            proceed.wait(timeout=30)

        with tempfile.TemporaryDirectory() as tmp_dir:
            with (
                mock.patch.dict(os.environ, _base_env(tmp_dir), clear=True),
                mock.patch.object(
                    OnlineDenseExportManager, "_build_export_graph", return_value=[]
                ),
            ):
                mgr = OnlineDenseExportManager(
                    model_dir=tmp_dir,
                    pipeline_config_path=os.path.join(tmp_dir, "pipeline.config"),
                    model=_mock_model({}),
                )
                try:
                    with mock.patch.object(mgr, "_run_task", side_effect=fake_run_task):
                        mgr._close_timeout = 0.05  # force close to time out mid-task
                        mgr._enqueue(1, 1.0, {})
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

    # --- trigger decisions ---

    def test_maybe_export_fires_on_step_interval_and_dedupes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with (
                mock.patch.dict(os.environ, _base_env(tmp_dir), clear=True),
                mock.patch.object(
                    OnlineDenseExportManager, "_build_export_graph", return_value=[]
                ),
            ):
                mgr = OnlineDenseExportManager(
                    model_dir=tmp_dir,
                    pipeline_config_path=os.path.join(tmp_dir, "pipeline.config"),
                    model=_mock_model({}),
                )
                try:
                    model = _mock_model({})
                    with mock.patch.object(mgr, "_gather_and_submit") as gather:
                        for step in (0, 1, 4):
                            mgr.maybe_export(step, -1.0, model)
                        gather.assert_not_called()
                        mgr.maybe_export(5, -1.0, model)
                        gather.assert_called_once_with(5, -1.0, model)
                        # same step (even forced) is deduped
                        mgr.maybe_export(5, -1.0, model, final=True)
                        gather.assert_called_once()
                        mgr.maybe_export(6, -1.0, model)
                        gather.assert_called_once()
                        mgr.maybe_export(10, -1.0, model)
                        self.assertEqual(gather.call_count, 2)
                finally:
                    mgr.close()

    def test_maybe_export_fires_on_event_time_interval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            env = _base_env(
                tmp_dir,
                **{
                    "ONLINE_DENSE_EXPORT_STEPS": "0",
                    "ONLINE_DENSE_EXPORT_INTERVAL": "60",
                },
            )
            with (
                mock.patch.dict(os.environ, env, clear=True),
                mock.patch.object(
                    OnlineDenseExportManager, "_build_export_graph", return_value=[]
                ),
            ):
                mgr = OnlineDenseExportManager(
                    model_dir=tmp_dir,
                    pipeline_config_path=os.path.join(tmp_dir, "pipeline.config"),
                    model=_mock_model({}),
                )
                try:
                    model = _mock_model({})
                    with mock.patch.object(mgr, "_gather_and_submit") as gather:
                        # first event-time seen only sets the reference
                        mgr.maybe_export(1, 1000.0, model)
                        gather.assert_not_called()
                        # 1000/60 -> 16, 1050/60 -> 17: boundary crossed
                        mgr.maybe_export(2, 1050.0, model)
                        gather.assert_called_once()
                        # still bucket 17: no fire
                        mgr.maybe_export(3, 1070.0, model)
                        gather.assert_called_once()
                        # 1120/60 -> 18: fires
                        mgr.maybe_export(4, 1120.0, model)
                        self.assertEqual(gather.call_count, 2)
                finally:
                    mgr.close()

    def test_maybe_export_final_fires_and_disabled_is_noop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with (
                mock.patch.dict(os.environ, _base_env(tmp_dir), clear=True),
                mock.patch.object(
                    OnlineDenseExportManager, "_build_export_graph", return_value=[]
                ),
            ):
                mgr = OnlineDenseExportManager(
                    model_dir=tmp_dir,
                    pipeline_config_path=os.path.join(tmp_dir, "pipeline.config"),
                    model=_mock_model({}),
                )
                try:
                    model = _mock_model({})
                    with mock.patch.object(mgr, "_gather_and_submit") as gather:
                        mgr.maybe_export(3, -1.0, model, final=True)
                        gather.assert_called_once()
                finally:
                    mgr.close()
            with mock.patch.dict(os.environ, {}, clear=True):
                mgr = OnlineDenseExportManager(
                    model_dir=tmp_dir,
                    pipeline_config_path=os.path.join(tmp_dir, "pipeline.config"),
                    model=_mock_model({}),
                )
                self.assertFalse(mgr._enabled)
                # no gather state at all; must be a pure no-op
                mgr.maybe_export(1, -1.0, _mock_model({}), final=True)
                mgr.close()

    # --- gather + publish, with the real worker ---

    def _run_one_export(self, tmp_dir: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """Build a manager over a tiny gm, fire one export, return the payload."""
        box: Dict[str, Any] = {}

        def fake_build(self: OnlineDenseExportManager, model: nn.Module) -> List:
            gm = _TinyModel()
            box["gm"] = gm
            self._gm = gm
            self._twin_model = mock.Mock()
            self._full_graph = mock.Mock()
            self._warmup_data = {}
            self._dense_graph_config = {}
            return [("w", "model.w")]

        with (
            mock.patch.dict(os.environ, _base_env(tmp_dir), clear=True),
            mock.patch.object(
                OnlineDenseExportManager, "_build_export_graph", fake_build
            ),
            mock.patch(_BUILD_PATCHES.format("finalize_dense_export")),
        ):
            mgr = OnlineDenseExportManager(
                model_dir=tmp_dir,
                pipeline_config_path=os.path.join(tmp_dir, "pipeline.config"),
                model=_mock_model(state),
            )
            try:
                mgr.maybe_export(5, 42.0, _mock_model(state))
                current_path = os.path.join(
                    tmp_dir, "serving_root", "dense_hot_export", "current.json"
                )
                self.assertTrue(_wait_for(lambda: os.path.exists(current_path)))
            finally:
                mgr.close()
        with open(current_path) as f:
            payload = json.load(f)
        box["payload"] = payload
        return box

    def test_maybe_export_gathers_plain_tensor_and_publishes(self) -> None:
        """Replicated tensors need no collective; the version is published."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            box = self._run_one_export(tmp_dir, {"model.w": torch.full((2,), 7.0)})
            payload = box["payload"]
            self.assertEqual(
                set(payload.keys()),
                {"version", "checkpoint_step", "data_timestamp", "created_at"},
            )
            self.assertEqual(payload["checkpoint_step"], 5)
            self.assertEqual(payload["data_timestamp"], 42.0)
            self.assertTrue(payload["version"])
            versions_root = os.path.join(
                tmp_dir, "serving_root", "dense_hot_export", "versions"
            )
            self.assertEqual(os.listdir(versions_root), [payload["version"]])
            self.assertTrue(
                os.path.exists(os.path.join(versions_root, payload["version"], "READY"))
            )
        torch.testing.assert_close(box["gm"].w, torch.full((2,), 7.0))

    def test_maybe_export_gathers_dtensor(self) -> None:
        """Sharded-as-DTensor state is all-gathered via full_tensor()."""
        port = misc_util.get_free_port()
        dist.init_process_group(
            backend="gloo",
            init_method=f"tcp://127.0.0.1:{port}",
            world_size=1,
            rank=0,
        )
        try:
            mesh = init_device_mesh("cpu", (1,))
            value = DTensor.from_local(torch.full((2,), 9.0), mesh, [Replicate()])
            with tempfile.TemporaryDirectory() as tmp_dir:
                box = self._run_one_export(tmp_dir, {"model.w": value})
        finally:
            dist.destroy_process_group()
        torch.testing.assert_close(box["gm"].w, torch.full((2,), 9.0))

    def test_maybe_export_gathers_sharded_tensor(self) -> None:
        """ShardedTensor local shards are staged at their global offsets."""
        port = misc_util.get_free_port()
        dist.init_process_group(
            backend="gloo",
            init_method=f"tcp://127.0.0.1:{port}",
            world_size=1,
            rank=0,
        )
        try:
            placement = "rank:0/cpu"
            meta = ShardedTensorMetadata(
                shards_metadata=[
                    ShardMetadata(
                        shard_offsets=[0], shard_sizes=[1], placement=placement
                    ),
                    ShardMetadata(
                        shard_offsets=[1], shard_sizes=[1], placement=placement
                    ),
                ],
                size=torch.Size([2]),
                tensor_properties=TensorProperties(dtype=torch.float32),
            )
            value = ShardedTensor._init_from_local_shards_and_global_metadata(
                [
                    Shard(
                        torch.full((1,), 3.0),
                        ShardMetadata(
                            shard_offsets=[0], shard_sizes=[1], placement=placement
                        ),
                    ),
                    Shard(
                        torch.full((1,), 5.0),
                        ShardMetadata(
                            shard_offsets=[1], shard_sizes=[1], placement=placement
                        ),
                    ),
                ],
                meta,
            )
            with tempfile.TemporaryDirectory() as tmp_dir:
                box = self._run_one_export(tmp_dir, {"model.w": value})
        finally:
            dist.destroy_process_group()
        torch.testing.assert_close(box["gm"].w, torch.tensor([3.0, 5.0]))

    # --- publish helpers ---

    def test_atomic_write_json_creates_dirs_and_replaces(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "nested", "current.json")
            _atomic_write_json(path, {"a": 1})
            with open(path) as f:
                self.assertEqual(json.load(f), {"a": 1})
            _atomic_write_json(path, {"a": 2})
            with open(path) as f:
                self.assertEqual(json.load(f), {"a": 2})
            self.assertEqual(
                os.listdir(os.path.join(tmp_dir, "nested")), ["current.json"]
            )

    def test_read_current_version_best_effort(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            missing = os.path.join(tmp_dir, "current.json")
            self.assertIsNone(_read_current_version(missing))
            with open(missing, "w") as f:
                f.write("{not json")
            self.assertIsNone(_read_current_version(missing))
            with open(missing, "w") as f:
                json.dump({"version": "20260101000001"}, f)
            self.assertEqual(_read_current_version(missing), "20260101000001")


if __name__ == "__main__":
    unittest.main()
