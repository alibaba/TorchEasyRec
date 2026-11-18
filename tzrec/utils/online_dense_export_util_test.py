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

import datetime
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple
from unittest import mock

import pyarrow as pa
import pyarrow.parquet as pq
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
from torch.utils._pytree import treespec_loads

from tzrec.acc import utils as acc_utils
from tzrec.features.feature import create_features
from tzrec.models.model import ScriptWrapper
from tzrec.models.multi_tower import MultiTower
from tzrec.optim.ema import DenseEMA
from tzrec.protos import (
    data_pb2,
    export_pb2,
    feature_pb2,
    loss_pb2,
    model_pb2,
    module_pb2,
    tower_pb2,
    train_pb2,
)
from tzrec.protos.models import rank_model_pb2
from tzrec.protos.pipeline_pb2 import EasyRecConfig
from tzrec.tools.online_dense_export import export_online_dense_model
from tzrec.tools.online_dense_export import main as online_dense_export_main
from tzrec.utils import checkpoint_util, config_util, misc_util
from tzrec.utils.online_dense_export_util import (
    OnlineDenseExportManager,
    _atomic_write_json,
    _make_monotonic_version,
    _read_current_version,
    make_version,
    resolve_dense_export_root,
)
from tzrec.utils.state_dict_util import init_parameters
from tzrec.utils.test_util import gpu_unavailable, mark_ci_scope


def _base_env(tmp_dir: str, **extra: str) -> Dict[str, str]:
    env = {
        "ONLINE_DENSE_EXPORT": "1",
        "USE_DISTRIBUTED_EMBEDDING": "1",
        "ONLINE_DENSE_EXPORT_DIR": os.path.join(tmp_dir, "serving_root"),
        "RANK": "0",
        "LOCAL_RANK": "0",
        "WORLD_SIZE": "1",
        "LOCAL_WORLD_SIZE": "1",
    }
    env.update(extra)
    return env


def _current_path(tmp_dir: str) -> str:
    return os.path.join(tmp_dir, "serving_root", "dense_hot_export", "current.json")


def _versions_root(tmp_dir: str) -> str:
    return os.path.join(tmp_dir, "serving_root", "dense_hot_export", "versions")


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

    def test_build_scopes_export_env_and_restores_it(self) -> None:
        """INPUT_TILE/WORLD_SIZE hold only during the twin build, even on failure."""
        seen: Dict[str, Any] = {}

        def fake_warmup(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            seen["world_size"] = os.environ.get("WORLD_SIZE")
            seen["data_world_size"] = kwargs.get("world_size")
            return {}

        def fake_finalize(*args: Any, **kwargs: Any) -> None:
            seen["input_tile"] = acc_utils.get_input_tile()
            seen["env_input_tile"] = os.environ.get("INPUT_TILE")

        fake_gm = mock.Mock()
        fake_gm.state_dict.return_value = {}
        with tempfile.TemporaryDirectory() as tmp_dir:
            env = _base_env(tmp_dir, WORLD_SIZE="4")
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
                        side_effect=fake_warmup,
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
                self.assertIsNone(seen["env_input_tile"])
                self.assertEqual(seen["data_world_size"], 1)
                self.assertEqual(seen["world_size"], "4")
                self.assertNotIn("INPUT_TILE", os.environ)
                self.assertEqual(os.environ["WORLD_SIZE"], "4")

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
                self.assertEqual(os.environ["WORLD_SIZE"], "4")

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

    def test_build_rejects_sharded_tensor(self) -> None:
        """Dense-graph state backed by a ShardedTensor must abort startup.

        Dense export gathers only replicated or DTensor dense weights; a
        ShardedTensor in the dense graph means sparse/embedding state leaked
        in, so construction fails fast instead of trying to gather it.
        """
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
        finally:
            dist.destroy_process_group()
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
                with self.assertRaisesRegex(RuntimeError, "must not carry sharded"):
                    OnlineDenseExportManager(
                        model_dir=tmp_dir,
                        pipeline_config_path=os.path.join(tmp_dir, "pipeline.config"),
                        model=_mock_model({"model.w": value}),
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
                        # pending, not yet started
                        mgr._enqueue(2, 2.0, {"m": "b"})
                        # supersedes b
                        mgr._enqueue(3, 3.0, {"m": "c"})
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

    def test_init_failure_after_worker_start_drains_worker(self) -> None:
        """A post-start setup failure must not leak the live export worker.

        No caller-side guard can clean up a constructor that fails after
        starting the worker, so __init__ drains it locally before re-raising.
        """
        real_close = OnlineDenseExportManager.close
        closed_workers: List[Any] = []

        def recording_close(self: OnlineDenseExportManager) -> None:
            closed_workers.append(self._worker)
            real_close(self)

        with tempfile.TemporaryDirectory() as tmp_dir:
            with (
                mock.patch.dict(os.environ, _base_env(tmp_dir), clear=True),
                mock.patch.object(
                    OnlineDenseExportManager, "_build_export_graph", return_value=[]
                ),
                mock.patch.object(OnlineDenseExportManager, "close", recording_close),
                mock.patch(
                    "tzrec.utils.online_dense_export_util.weakref.finalize",
                    side_effect=RuntimeError("finalize boom"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "finalize boom"):
                    OnlineDenseExportManager(
                        model_dir=tmp_dir,
                        pipeline_config_path=os.path.join(tmp_dir, "pipeline.config"),
                        model=_mock_model({}),
                    )
        self.assertEqual(len(closed_workers), 1)
        self.assertIsNotNone(closed_workers[0])
        self.assertFalse(closed_workers[0].is_alive())

    # --- trigger decisions ---

    def test_maybe_export_fires_on_force_and_dedupes(self) -> None:
        """Only the delta-dump decision (force) or final fires; steps dedupe."""
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
                    self.assertTrue(mgr.enabled)
                    model = _mock_model({})
                    with mock.patch.object(mgr, "_gather_and_submit") as gather:
                        for step in (0, 1, 4, 5):
                            mgr.maybe_export(step, -1.0, model)
                        gather.assert_not_called()
                        mgr.maybe_export(5, -1.0, model, force=True)
                        gather.assert_called_once_with(5, -1.0, model, None)
                        # same step (even forced/final) is deduped
                        mgr.maybe_export(5, -1.0, model, force=True)
                        mgr.maybe_export(5, -1.0, model, final=True)
                        gather.assert_called_once()
                        mgr.maybe_export(6, -1.0, model)
                        gather.assert_called_once()
                        mgr.maybe_export(7, -1.0, model, final=True)
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
                self.assertFalse(mgr.enabled)
                # no gather/publish state at all; all must be pure no-ops
                mgr.maybe_export(1, -1.0, _mock_model({}), final=True)
                mgr.poll_publish((1, []))
                mgr.poll_publish(None)
                mgr.finalize_publish(None)
                mgr.close()

    # --- gather + publish, with the real worker ---

    @contextmanager
    def _tiny_manager(
        self,
        tmp_dir: str,
        state: Dict[str, Any],
        publish_interval_minutes: Optional[int] = None,
    ) -> Iterator[OnlineDenseExportManager]:
        """Manager over a tiny gm with the heavy graph build patched out."""

        def fake_build(self_: OnlineDenseExportManager, model: nn.Module) -> List[Any]:
            self_._gm = _TinyModel()
            self_._twin_model = mock.Mock()
            self_._full_graph = mock.Mock()
            self_._warmup_data = {}
            self_._dense_graph_config = {}
            self_._dense_model_traced = mock.Mock()
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
                publish_interval_minutes=publish_interval_minutes,
            )
            try:
                yield mgr
            finally:
                mgr.close()

    def _run_one_export(self, tmp_dir: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """Fire one forced export, flip via poll_publish, return gm and payload."""
        box: Dict[str, Any] = {}
        with self._tiny_manager(tmp_dir, state) as mgr:
            box["gm"] = mgr._gm
            mgr.maybe_export(5, 42.0, _mock_model(state), force=True)
            current_path = _current_path(tmp_dir)
            # the worker only records the version as ready; the next poll
            # with a watermark past the step is what flips the pointer
            self.assertTrue(_wait_for(lambda: mgr._ready is not None))
            self.assertFalse(os.path.exists(current_path))
            mgr.poll_publish((9, []))
            self.assertTrue(os.path.exists(current_path))
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
                {
                    "version",
                    "checkpoint_step",
                    "sparse_step",
                    "data_timestamp",
                    "created_at",
                    "sparse_probes",
                },
            )
            self.assertEqual(payload["checkpoint_step"], 5)
            self.assertEqual(payload["sparse_step"], 9)
            self.assertGreaterEqual(payload["sparse_step"], payload["checkpoint_step"])
            self.assertEqual(payload["data_timestamp"], 42.0)
            self.assertEqual(payload["sparse_probes"], [])
            self.assertTrue(payload["version"])
            versions_root = _versions_root(tmp_dir)
            self.assertEqual(os.listdir(versions_root), [payload["version"]])
            self.assertTrue(
                os.path.exists(os.path.join(versions_root, payload["version"], "READY"))
            )
        torch.testing.assert_close(box["gm"].w, torch.full((2,), 7.0))

    def test_ready_version_not_published_until_watermark(self) -> None:
        """A finished dense version must not flip current.json ahead of sparse."""
        state = {"model.w": torch.zeros(2)}
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self._tiny_manager(tmp_dir, state) as mgr:
                mgr.maybe_export(5, 42.0, _mock_model(state), force=True)
                self.assertTrue(_wait_for(lambda: mgr._ready is not None))
                current_path = _current_path(tmp_dir)
                # the version dir is ready on disk, but the pointer stays put
                self.assertEqual(len(os.listdir(_versions_root(tmp_dir))), 1)
                self.assertFalse(os.path.exists(current_path))
                mgr.poll_publish(None)
                self.assertFalse(os.path.exists(current_path))
                mgr.poll_publish((4, []))
                self.assertFalse(os.path.exists(current_path))
                mgr.poll_publish((5, []))
                self.assertTrue(os.path.exists(current_path))
                with open(current_path) as f:
                    payload = json.load(f)
                self.assertEqual(payload["checkpoint_step"], 5)
                self.assertEqual(payload["sparse_step"], 5)

    def test_poll_publish_payload_carries_v2_fields(self) -> None:
        """The flip carries capped probes and publish_interval_minutes."""
        probes = [
            {"pk": "emb.t", "sk": i, "crc32": "0a" * 4, "encoding": "fp32"}
            for i in range(70)
        ]
        state = {"model.w": torch.zeros(2)}
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self._tiny_manager(tmp_dir, state, publish_interval_minutes=10) as mgr:
                mgr.maybe_export(5, 42.0, _mock_model(state), force=True)
                self.assertTrue(_wait_for(lambda: mgr._ready is not None))
                mgr.poll_publish((6, probes))
                with open(_current_path(tmp_dir)) as f:
                    payload = json.load(f)
        self.assertEqual(
            set(payload.keys()),
            {
                "version",
                "checkpoint_step",
                "sparse_step",
                "data_timestamp",
                "created_at",
                "publish_interval_minutes",
                "sparse_probes",
            },
        )
        self.assertEqual(payload["publish_interval_minutes"], 10)
        self.assertEqual(payload["checkpoint_step"], 5)
        self.assertEqual(payload["sparse_step"], 6)
        self.assertGreaterEqual(payload["sparse_step"], payload["checkpoint_step"])
        self.assertEqual(len(payload["sparse_probes"]), 64)
        self.assertEqual(payload["sparse_probes"], probes[:64])

    def test_latest_wins_publishes_only_newest_ready_version(self) -> None:
        """Only the newest ready version flips; the superseded one is removed."""
        state = {"model.w": torch.zeros(2)}
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self._tiny_manager(tmp_dir, state) as mgr:
                mgr.maybe_export(5, 42.0, _mock_model(state), force=True)
                self.assertTrue(_wait_for(lambda: (mgr._ready or {}).get("step") == 5))
                superseded_version = mgr._ready["version"]
                mgr.maybe_export(7, 43.0, _mock_model(state), force=True)
                self.assertTrue(_wait_for(lambda: (mgr._ready or {}).get("step") == 7))
                # the superseded step-5 version was never published and is
                # removed, so it cannot pile up or count against retention
                self.assertFalse(
                    os.path.exists(
                        os.path.join(_versions_root(tmp_dir), superseded_version)
                    )
                )
                current_path = _current_path(tmp_dir)
                mgr.poll_publish((5, []))
                self.assertFalse(os.path.exists(current_path))
                mgr.poll_publish((7, []))
                with open(current_path) as f:
                    payload = json.load(f)
                self.assertEqual(payload["checkpoint_step"], 7)
                self.assertEqual(payload["data_timestamp"], 43.0)
                versions = os.listdir(_versions_root(tmp_dir))
                self.assertEqual(versions, [payload["version"]])
                # nothing left to publish: a later poll leaves the pointer alone
                mgr.poll_publish((9, []))
                with open(current_path) as f:
                    self.assertEqual(json.load(f), payload)

    def test_finalize_publish_drains_worker_and_flips(self) -> None:
        """finalize_publish drains the in-flight build, then publishes."""
        state = {"model.w": torch.zeros(2)}
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self._tiny_manager(tmp_dir, state) as mgr:
                mgr.maybe_export(5, 42.0, _mock_model(state), force=True)
                # no wait: finalize must drain the in-flight build itself
                mgr.finalize_publish((5, []))
                self.assertFalse(mgr._worker.is_alive())
                with open(_current_path(tmp_dir)) as f:
                    payload = json.load(f)
                self.assertEqual(payload["checkpoint_step"], 5)
                self.assertEqual(payload["sparse_step"], 5)
                # the finally-block close after finalize stays idempotent
                mgr.close()

    def test_publish_clears_ready_before_prune(self) -> None:
        """A prune failure after the flip must not leave the live version 'ready'.

        Otherwise the next build would treat the published version as
        superseded and remove the directory current.json points at.
        """
        state = {"model.w": torch.zeros(2)}
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self._tiny_manager(tmp_dir, state) as mgr:
                mgr.maybe_export(5, 42.0, _mock_model(state), force=True)
                self.assertTrue(_wait_for(lambda: mgr._ready is not None))
                with mock.patch(
                    _BUILD_PATCHES.format("_prune_old_dense_versions"),
                    side_effect=OSError("boom"),
                ):
                    with self.assertRaises(OSError):
                        mgr.poll_publish((5, []))
                with open(_current_path(tmp_dir)) as f:
                    published = json.load(f)["version"]
                self.assertIsNone(mgr._ready)
                mgr.maybe_export(7, 43.0, _mock_model(state), force=True)
                self.assertTrue(_wait_for(lambda: (mgr._ready or {}).get("step") == 7))
                self.assertTrue(
                    os.path.isdir(os.path.join(_versions_root(tmp_dir), published))
                )

    # --- multi-rank publish gate (two simulated ranks over the module dist) ---

    def test_poll_publish_multi_rank_gates_on_min_watermark(self) -> None:
        """The lagging rank holds the flip; probes are gathered only at the flip.

        Simulates rank one through the module-level ``dist`` so the
        world-size > 1 branch runs: the per-step tensor gather carries
        (completed step, rank-zero ready step), the minimum completed step is
        the watermark, and the object gather of probes happens once, when
        that watermark covers the ready version.
        """
        state = {"model.w": torch.zeros(2)}
        other = {"step": 0}
        own_probes = [
            {"pk": "emb.t", "sk": i, "crc32": "aa" * 4, "encoding": "fp32"}
            for i in range(40)
        ]
        other_probes = [
            {"pk": "emb.t", "sk": 100 + i, "crc32": "bb" * 4, "encoding": "fp32"}
            for i in range(40)
        ]

        def fake_all_gather(gathered: List[Any], local: Any, group: Any = None) -> None:
            gathered[0].copy_(local)
            gathered[1].copy_(torch.tensor([other["step"], -1], dtype=torch.int64))

        def fake_all_gather_object(box: List[Any], obj: Any, group: Any = None) -> None:
            box[0] = obj
            box[1] = other_probes

        with tempfile.TemporaryDirectory() as tmp_dir:
            with (
                self._tiny_manager(tmp_dir, state) as mgr,
                mock.patch(
                    _BUILD_PATCHES.format("dist.get_world_size"), return_value=2
                ),
                mock.patch(
                    _BUILD_PATCHES.format("dist.all_gather"),
                    side_effect=fake_all_gather,
                ) as all_gather,
                mock.patch(
                    _BUILD_PATCHES.format("dist.all_gather_object"),
                    side_effect=fake_all_gather_object,
                ) as gather_probes,
            ):
                mgr._group = mock.Mock()
                current_path = _current_path(tmp_dir)
                # nothing ready: the cheap tensor gather still runs on every
                # step, the probe object gather does not
                mgr.poll_publish((5, own_probes))
                torch.testing.assert_close(
                    all_gather.call_args.args[1], torch.tensor([5, -1])
                )
                gather_probes.assert_not_called()

                mgr.maybe_export(5, 42.0, _mock_model(state), force=True)
                self.assertTrue(_wait_for(lambda: mgr._ready is not None))
                # own rank covers step 5 but rank one lags: min() holds the flip
                other["step"] = 4
                mgr.poll_publish((5, own_probes))
                torch.testing.assert_close(
                    all_gather.call_args.args[1], torch.tensor([5, 5])
                )
                self.assertFalse(os.path.exists(current_path))
                gather_probes.assert_not_called()

                other["step"] = 6
                mgr.poll_publish((5, own_probes))
                self.assertEqual(all_gather.call_count, 3)
                gather_probes.assert_called_once()
                with open(current_path) as f:
                    payload = json.load(f)
                self.assertEqual(payload["checkpoint_step"], 5)
                self.assertEqual(payload["sparse_step"], 5)
                self.assertEqual(
                    payload["sparse_probes"], (own_probes + other_probes)[:64]
                )
                # published: later polls gather nothing more
                mgr.poll_publish((7, own_probes))
                gather_probes.assert_called_once()

    def test_poll_publish_non_zero_rank_joins_collectives_without_publishing(
        self,
    ) -> None:
        """A non-zero rank enters both gathers in lockstep but never flips."""
        probe = {"pk": "emb.t", "sk": 1, "crc32": "aa" * 4, "encoding": "fp32"}

        def fake_all_gather(gathered: List[Any], local: Any, group: Any = None) -> None:
            # rank zero: completed step 7, ready version at step 5
            gathered[0].copy_(torch.tensor([7, 5], dtype=torch.int64))
            gathered[1].copy_(local)

        def fake_all_gather_object(box: List[Any], obj: Any, group: Any = None) -> None:
            box[0] = []
            box[1] = obj

        with tempfile.TemporaryDirectory() as tmp_dir:
            with (
                mock.patch.dict(
                    os.environ,
                    _base_env(tmp_dir, RANK="1", LOCAL_RANK="1"),
                    clear=True,
                ),
                mock.patch.object(
                    OnlineDenseExportManager, "_build_export_graph", return_value=[]
                ),
                mock.patch(
                    _BUILD_PATCHES.format("dist.get_world_size"), return_value=2
                ),
                mock.patch(
                    _BUILD_PATCHES.format("dist.all_gather"),
                    side_effect=fake_all_gather,
                ) as all_gather,
                mock.patch(
                    _BUILD_PATCHES.format("dist.all_gather_object"),
                    side_effect=fake_all_gather_object,
                ) as gather_probes,
            ):
                mgr = OnlineDenseExportManager(
                    model_dir=tmp_dir,
                    pipeline_config_path=os.path.join(tmp_dir, "pipeline.config"),
                    model=_mock_model({}),
                )
                self.assertIsNone(mgr._worker)
                mgr._group = mock.Mock()
                current_path = _current_path(tmp_dir)
                mgr.poll_publish((6, [probe]))
                torch.testing.assert_close(
                    all_gather.call_args.args[1], torch.tensor([6, -1])
                )
                # rank zero's ready step 5 <= min(7, 6): this rank must join
                # the probe gather too, yet only rank zero writes the pointer
                gather_probes.assert_called_once()
                self.assertEqual(gather_probes.call_args.args[1], [probe])
                self.assertFalse(os.path.exists(current_path))
                mgr.finalize_publish((6, [probe]))
                self.assertEqual(all_gather.call_count, 2)
                self.assertFalse(os.path.exists(current_path))
                mgr.close()

    def test_gather_uses_ema_parameters_and_live_buffers(self) -> None:
        manager = OnlineDenseExportManager.__new__(OnlineDenseExportManager)
        manager._rank = 0
        manager._state_pairs = [
            ("w", "model.w"),
            ("running", "model.running"),
        ]
        parameter = nn.Parameter(torch.full((2,), 2.0))
        dense_ema = DenseEMA({"model.w": parameter}, decay=0.5)
        dense_ema.update()
        parameter.data.fill_(4.0)
        dense_ema.update()
        model = _mock_model(
            {
                "model.w": parameter,
                "model.running": torch.full((2,), 7.0),
            }
        )

        with mock.patch.object(manager, "_enqueue") as enqueue:
            manager._gather_and_submit(5, 42.0, model, dense_ema)

        snapshot = enqueue.call_args.args[2]
        torch.testing.assert_close(snapshot["w"], torch.full((2,), 3.0))
        torch.testing.assert_close(snapshot["running"], torch.full((2,), 7.0))

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

    def test_worker_reuses_construction_time_traced_module(self) -> None:
        """The worker scripts the construction-time traced module, not a re-trace.

        The manager traces once during the construction-time dry run and must
        forward that exact traced module to every worker export. A regression
        that re-traces on the worker would re-open the fx-trace race this fix
        targets, yet still pass an output-equivalence check (re-tracing is
        weight-correct in a single-threaded test), so assert object identity.
        """
        sentinel = mock.Mock(name="traced_sentinel")
        seen: List[Dict[str, Any]] = []

        def fake_finalize(*args: Any, **kwargs: Any) -> Any:
            seen.append(kwargs)
            # construction dry run returns the cached traced module; the worker
            # call (with dense_model_traced=) must forward it verbatim.
            return sentinel

        fake_gm = mock.Mock()
        fake_gm.state_dict.return_value = {}
        with tempfile.TemporaryDirectory() as tmp_dir:
            with (
                mock.patch.dict(os.environ, _base_env(tmp_dir), clear=True),
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
                try:
                    # the real build stored the construction-time finalize
                    # return as the cached traced module
                    self.assertIs(mgr._dense_model_traced, sentinel)
                    mgr.maybe_export(5, 42.0, _mock_model({}), force=True)
                    self.assertTrue(_wait_for(lambda: mgr._ready is not None))
                    mgr.poll_publish((5, []))
                    self.assertTrue(os.path.exists(_current_path(tmp_dir)))
                finally:
                    mgr.close()
            # one construction call (no cached module) then one worker call
            # forwarding the construction-time sentinel verbatim
            self.assertEqual(len(seen), 2)
            self.assertNotIn("dense_model_traced", seen[0])
            self.assertIn("dense_model_traced", seen[1])
            self.assertIs(seen[1]["dense_model_traced"], sentinel)

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

    # --- AOTI online export (CPU validation + GPU end-to-end) ---

    def test_init_aot_requires_cuda(self) -> None:
        """ENABLE_AOT with ONLINE_DENSE_EXPORT or bootstrap fails fast without CUDA."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            with (
                mock.patch.dict(
                    os.environ, _base_env(tmp_dir, ENABLE_AOT="2"), clear=True
                ),
                mock.patch("torch.cuda.is_available", return_value=False),
            ):
                with self.assertRaisesRegex(RuntimeError, "CUDA"):
                    OnlineDenseExportManager(
                        model_dir=tmp_dir,
                        pipeline_config_path=os.path.join(tmp_dir, "pipeline.config"),
                        model=_mock_model({}),
                    )
                with self.assertRaisesRegex(RuntimeError, "CUDA"):
                    export_online_dense_model(
                        pipeline_config_path=os.path.join(tmp_dir, "pipeline.config"),
                        checkpoint_path=os.path.join(tmp_dir, "model.ckpt-1"),
                        model_dir=tmp_dir,
                    )

    def test_aot_build_and_worker_preserve_tf32_and_mixed_precision(self) -> None:
        """AOTI build and worker apply export TF32/AMP and restore globals."""
        cfg = EasyRecConfig(
            train_config=train_pb2.TrainConfig(
                cudnn_allow_tf32=True, cuda_matmul_allow_tf32=True
            ),
            export_config=export_pb2.ExportConfig(
                mixed_precision="BF16",
                cudnn_allow_tf32=False,
                cuda_matmul_allow_tf32=False,
            ),
        )
        fake_gm = _TinyModel()
        observed: List[Dict[str, Any]] = []

        def fake_finalize(*args: Any, **kwargs: Any) -> Any:
            observed.append(
                {
                    "device": args[4],
                    "mixed_precision": kwargs.get("mixed_precision"),
                    "has_dense_model_traced": "dense_model_traced" in kwargs,
                    "dense_model_traced": kwargs.get("dense_model_traced"),
                    "input_tile": kwargs.get("input_tile"),
                    "input_tile_3_online": kwargs.get("input_tile_3_online"),
                    "hard_search_features": kwargs.get("hard_search_features"),
                    "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                    "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                }
            )
            return fake_gm

        @contextmanager
        def fake_stream_ctx(_stream: Any) -> Iterator[None]:
            yield

        fake_stream = mock.Mock()
        prev_cudnn = torch.backends.cudnn.allow_tf32
        prev_matmul = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                env = _base_env(tmp_dir, ENABLE_AOT="2")
                with (
                    mock.patch.dict(os.environ, env, clear=True),
                    mock.patch("torch.cuda.is_available", return_value=True),
                    mock.patch("torch.cuda.set_device") as set_device,
                    mock.patch("torch.cuda.Stream", return_value=fake_stream),
                    mock.patch("torch.cuda.stream", side_effect=fake_stream_ctx),
                    mock.patch(
                        _BUILD_PATCHES.format("config_util.load_pipeline_config"),
                        return_value=cfg,
                    ),
                    mock.patch("tzrec.main._create_features", return_value=[]),
                    mock.patch("tzrec.main._create_model", return_value=mock.Mock()),
                    mock.patch(
                        "tzrec.models.model.ScriptWrapper", side_effect=lambda m: m
                    ),
                    mock.patch(
                        _BUILD_PATCHES.format("create_dense_export_warmup_data"),
                        return_value={},
                    ),
                    mock.patch(
                        _BUILD_PATCHES.format("build_dense_graph_module"),
                        return_value=(fake_gm, mock.Mock(), {}),
                    ) as mock_build,
                    mock.patch(
                        _BUILD_PATCHES.format("finalize_dense_export"),
                        side_effect=fake_finalize,
                    ),
                ):
                    state = {"w": torch.ones(2)}
                    mgr = OnlineDenseExportManager(
                        model_dir=tmp_dir,
                        pipeline_config_path=os.path.join(tmp_dir, "pipeline.config"),
                        model=_mock_model(state),
                    )
                    try:
                        self.assertTrue(mgr._is_aot)
                        self.assertIs(mgr._dense_model_traced, fake_gm)
                        self.assertTrue(torch.backends.cudnn.allow_tf32)
                        self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
                        self.assertEqual(
                            mock_build.call_args.kwargs["mixed_precision"], "BF16"
                        )
                        mgr.maybe_export(5, 42.0, _mock_model(state), force=True)
                        self.assertTrue(_wait_for(lambda: mgr._ready is not None))
                        self.assertTrue(torch.backends.cudnn.allow_tf32)
                        self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
                    finally:
                        mgr.close()
                self.assertEqual(len(observed), 2)
                for call in observed:
                    self.assertEqual(call["device"], torch.device("cuda:0"))
                    self.assertEqual(call["mixed_precision"], "BF16")
                    self.assertTrue(call["cudnn_tf32"])
                    self.assertTrue(call["matmul_tf32"])
                self.assertFalse(observed[0]["has_dense_model_traced"])
                self.assertIs(observed[1]["dense_model_traced"], fake_gm)
                self.assertIs(observed[0]["input_tile"], True)
                self.assertIs(observed[1]["input_tile"], True)
                self.assertIs(observed[0]["input_tile_3_online"], False)
                self.assertIs(observed[1]["input_tile_3_online"], False)
                self.assertEqual(observed[0]["hard_search_features"], {})
                self.assertEqual(observed[1]["hard_search_features"], {})
                self.assertGreaterEqual(set_device.call_count, 2)
                fake_stream.synchronize.assert_called_once()
        finally:
            torch.backends.cudnn.allow_tf32 = prev_cudnn
            torch.backends.cuda.matmul.allow_tf32 = prev_matmul

    def _create_aot_test_fixture(
        self, tmp_dir: str
    ) -> Tuple[EasyRecConfig, str, List[Any]]:
        """Create a real pipeline.config and parquet data for GPU AOTI tests."""
        data_dir = os.path.join(tmp_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        pq.write_table(
            pa.table(
                {
                    "u_id": pa.array([1, 2, 1, 2], type=pa.int64()),
                    "i_id": pa.array([3, 4, 5, 6], type=pa.int64()),
                    "u_dense": pa.array([[0.2, -0.1]] * 4, type=pa.list_(pa.float32())),
                    "i_dense": pa.array(
                        [[0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0]],
                        type=pa.list_(pa.float32()),
                    ),
                }
            ),
            os.path.join(data_dir, "part-0.parquet"),
        )
        cfgs = [
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="u_id",
                    expression="user:u_id",
                    embedding_dim=4,
                    num_buckets=8,
                )
            ),
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="i_id",
                    expression="item:i_id",
                    embedding_dim=4,
                    num_buckets=8,
                )
            ),
            feature_pb2.FeatureConfig(
                raw_feature=feature_pb2.RawFeature(
                    feature_name="u_dense", expression="user:u_dense", value_dim=2
                )
            ),
            feature_pb2.FeatureConfig(
                raw_feature=feature_pb2.RawFeature(
                    feature_name="i_dense", expression="item:i_dense", value_dim=2
                )
            ),
        ]
        fgs = [
            model_pb2.FeatureGroupConfig(
                group_name="deep",
                feature_names=["u_id", "i_id", "u_dense", "i_dense"],
                group_type=model_pb2.DEEP,
            )
        ]
        pipeline_config = EasyRecConfig(
            train_input_path=os.path.join(data_dir, "*.parquet"),
            data_config=data_pb2.DataConfig(
                batch_size=2,
                dataset_type=data_pb2.DatasetType.ParquetDataset,
                fg_mode=data_pb2.FgMode.FG_DAG,
                num_workers=1,
            ),
            feature_configs=cfgs,
            model_config=model_pb2.ModelConfig(
                feature_groups=fgs,
                multi_tower=rank_model_pb2.MultiTower(
                    towers=[
                        tower_pb2.Tower(
                            input="deep", mlp=module_pb2.MLP(hidden_units=[8])
                        )
                    ],
                    final=module_pb2.MLP(hidden_units=[4]),
                ),
                losses=[
                    loss_pb2.LossConfig(
                        binary_cross_entropy=loss_pb2.BinaryCrossEntropy()
                    )
                ],
            ),
            train_config=train_pb2.TrainConfig(
                cudnn_allow_tf32=True,
                cuda_matmul_allow_tf32=True,
            ),
            export_config=export_pb2.ExportConfig(
                cudnn_allow_tf32=False,
                cuda_matmul_allow_tf32=False,
            ),
        )
        pipeline_config_path = os.path.join(tmp_dir, "pipeline.config")
        config_util.save_message(pipeline_config, pipeline_config_path)
        features = create_features(
            list(pipeline_config.feature_configs), pipeline_config.data_config
        )
        return pipeline_config, pipeline_config_path, features

    def _build_serving_inputs(
        self, version_dir: str, pt2_model: Any, device: torch.device
    ) -> Dict[str, torch.Tensor]:
        with open(os.path.join(version_dir, "dense_meta.json")) as f:
            dense_meta = json.load(f)
        in_spec = treespec_loads(pt2_model.loader.get_call_spec()[0])
        key_order = list(in_spec.child(0).child(0).context)
        for raw_key in ("u_id.values", "u_id.lengths", "i_id.values", "i_id.lengths"):
            self.assertNotIn(raw_key, key_order)
        ebc_item_key = next(
            k for k in dense_meta if k != "sequence__ec" and not k.endswith("_user")
        )
        ebc_user_key = next(k for k in dense_meta if k.endswith("_user"))
        user_bs, item_bs = 1, 3
        torch.manual_seed(7)
        serving_inputs = {
            ebc_item_key: torch.randn(
                item_bs, 4 * len(dense_meta[ebc_item_key]), device=device
            ),
            ebc_user_key: torch.randn(
                user_bs, 4 * len(dense_meta[ebc_user_key]), device=device
            ),
            "u_dense.values": torch.randn(user_bs, 2, device=device),
            "i_dense.values": torch.randn(item_bs, 2, device=device),
            "batch_size": torch.tensor(
                item_bs // user_bs, dtype=torch.int64, device=device
            ),
        }
        return {k: serving_inputs[k] for k in key_order}

    @mark_ci_scope("gpu")
    @unittest.skipIf(*gpu_unavailable)
    def test_online_dense_export_manager_aot_v1_v2_publish(self) -> None:
        """Manager builds, gates, and publishes v1->v2 .pt2 versions."""
        prev_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        prev_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                pipeline_config, pipeline_config_path, features = (
                    self._create_aot_test_fixture(tmp_dir)
                )
                device = torch.device("cuda:0")
                torch.cuda.set_device(device)
                inner = MultiTower(pipeline_config.model_config, features, labels=[])
                init_parameters(inner, device)
                live_model = ScriptWrapper(inner).to(device).eval()

                with torch.no_grad():
                    for p in live_model.parameters():
                        p.fill_(0.05)
                    live_model.model.output_mlp.bias.fill_(-2.0)

                model_dir = os.path.join(tmp_dir, "train_dir")
                os.makedirs(model_dir, exist_ok=True)
                env = _base_env(tmp_dir, ENABLE_AOT="2")
                with mock.patch.dict(os.environ, env, clear=False):
                    mgr = OnlineDenseExportManager(
                        model_dir=model_dir,
                        pipeline_config_path=pipeline_config_path,
                        model=live_model,
                    )
                    try:
                        self.assertTrue(mgr.enabled)
                        self.assertTrue(mgr._is_aot)
                        self.assertIsNotNone(mgr._dense_model_traced)
                        self.assertTrue(torch.backends.cudnn.allow_tf32)
                        self.assertTrue(torch.backends.cuda.matmul.allow_tf32)

                        versions_root = _versions_root(tmp_dir)
                        current_path = _current_path(tmp_dir)

                        # Export v1 and verify watermark gating before poll_publish.
                        mgr.maybe_export(5, 100.0, live_model, force=True)
                        self.assertTrue(
                            _wait_for(lambda: mgr._ready is not None, timeout=120.0)
                        )
                        assert mgr._ready is not None
                        v1 = mgr._ready["version"]
                        v1_dir = os.path.join(versions_root, v1)
                        v1_pt2 = os.path.join(v1_dir, "aoti", "aoti_model.pt2")
                        self.assertTrue(os.path.exists(v1_pt2))
                        self.assertTrue(
                            os.path.exists(os.path.join(v1_dir, "dense_meta.json"))
                        )
                        self.assertTrue(os.path.exists(os.path.join(v1_dir, "READY")))
                        self.assertFalse(
                            os.path.exists(os.path.join(v1_dir, "scripted_model.pt"))
                        )
                        self.assertFalse(os.path.exists(current_path))

                        mgr.poll_publish((4, []))
                        self.assertFalse(os.path.exists(current_path))
                        mgr.poll_publish((5, []))
                        self.assertTrue(os.path.exists(current_path))
                        with open(current_path) as f:
                            payload_v1 = json.load(f)
                        self.assertEqual(payload_v1["version"], v1)
                        self.assertEqual(payload_v1["checkpoint_step"], 5)
                        self.assertEqual(payload_v1["sparse_step"], 5)

                        assert mgr._gm is not None
                        pt2_v1 = torch._inductor.aoti_load_package(v1_pt2)
                        serving_inputs = self._build_serving_inputs(
                            v1_dir, pt2_v1, device
                        )
                        self.assertTrue(torch.backends.cudnn.allow_tf32)
                        self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
                        torch.backends.cudnn.allow_tf32 = False
                        torch.backends.cuda.matmul.allow_tf32 = False
                        try:
                            with torch.no_grad():
                                eager_v1 = mgr._gm(serving_inputs, device)
                        finally:
                            torch.backends.cudnn.allow_tf32 = True
                            torch.backends.cuda.matmul.allow_tf32 = True
                        with torch.no_grad():
                            out_v1 = pt2_v1(serving_inputs)
                        torch.testing.assert_close(
                            out_v1["probs"], eager_v1["probs"], atol=1e-5, rtol=1e-5
                        )

                        # Update live weights for v2 and verify published .pt2 changes.
                        with torch.no_grad():
                            live_model.model.output_mlp.bias.fill_(2.0)
                        mgr.maybe_export(10, 200.0, live_model, force=True)
                        self.assertTrue(
                            _wait_for(lambda: mgr._ready is not None, timeout=120.0)
                        )
                        assert mgr._ready is not None
                        v2 = mgr._ready["version"]
                        self.assertNotEqual(v2, v1)
                        v2_dir = os.path.join(versions_root, v2)
                        v2_pt2 = os.path.join(v2_dir, "aoti", "aoti_model.pt2")
                        self.assertTrue(os.path.exists(v2_pt2))
                        self.assertTrue(
                            os.path.exists(os.path.join(v2_dir, "dense_meta.json"))
                        )
                        self.assertTrue(os.path.exists(os.path.join(v2_dir, "READY")))
                        self.assertFalse(
                            os.path.exists(os.path.join(v2_dir, "scripted_model.pt"))
                        )

                        mgr.poll_publish((9, []))
                        with open(current_path) as f:
                            self.assertEqual(json.load(f)["version"], v1)
                        mgr.poll_publish((10, []))
                        with open(current_path) as f:
                            payload_v2 = json.load(f)
                        self.assertEqual(payload_v2["version"], v2)
                        self.assertEqual(payload_v2["checkpoint_step"], 10)
                        self.assertEqual(payload_v2["sparse_step"], 10)

                        pt2_v2 = torch._inductor.aoti_load_package(v2_pt2)
                        self.assertTrue(torch.backends.cudnn.allow_tf32)
                        self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
                        torch.backends.cudnn.allow_tf32 = False
                        torch.backends.cuda.matmul.allow_tf32 = False
                        try:
                            with torch.no_grad():
                                eager_v2 = mgr._gm(serving_inputs, device)
                        finally:
                            torch.backends.cudnn.allow_tf32 = True
                            torch.backends.cuda.matmul.allow_tf32 = True
                        with torch.no_grad():
                            out_v2 = pt2_v2(serving_inputs)
                        torch.testing.assert_close(
                            out_v2["probs"], eager_v2["probs"], atol=1e-5, rtol=1e-5
                        )
                        self.assertFalse(
                            torch.allclose(out_v1["probs"], out_v2["probs"], atol=1e-2)
                        )
                        self.assertTrue(torch.backends.cudnn.allow_tf32)
                        self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
                    finally:
                        mgr.close()
        finally:
            torch.backends.cudnn.allow_tf32 = prev_cudnn_tf32
            torch.backends.cuda.matmul.allow_tf32 = prev_matmul_tf32

    @mark_ci_scope("gpu")
    @unittest.skipIf(*gpu_unavailable)
    def test_export_online_dense_model_bootstrap_aot(self) -> None:
        """Bootstrap CLI exports aoti/aoti_model.pt2 and flips current.json."""
        prev_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        prev_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                pipeline_config, pipeline_config_path, features = (
                    self._create_aot_test_fixture(tmp_dir)
                )
                inner = MultiTower(pipeline_config.model_config, features, labels=[])
                init_parameters(inner, torch.device("cpu"))
                ckpt_model = ScriptWrapper(inner)
                with torch.no_grad():
                    for p in ckpt_model.parameters():
                        p.fill_(0.05)
                    ckpt_model.model.output_mlp.bias.fill_(1.5)

                ckpt_dir = os.path.join(tmp_dir, "model.ckpt-12")
                with mock.patch("tzrec.utils.checkpoint_util.has_dynamicemb", False):
                    checkpoint_util.save_model(ckpt_dir, ckpt_model)

                model_dir = os.path.join(tmp_dir, "train_dir")
                serving_root = os.path.join(tmp_dir, "serving_root")
                version = "20261117010203"
                env = {
                    "USE_DISTRIBUTED_EMBEDDING": "1",
                    "ENABLE_AOT": "2",
                    "ONLINE_DENSE_EXPORT_DIR": serving_root,
                }
                argv = [
                    "online_dense_export",
                    "--pipeline_config_path",
                    pipeline_config_path,
                    "--checkpoint_path",
                    ckpt_dir,
                    "--model_dir",
                    model_dir,
                    "--version",
                    version,
                    "--data_timestamp",
                    "123.0",
                ]
                with (
                    mock.patch.dict(os.environ, env, clear=False),
                    mock.patch.object(sys, "argv", argv),
                ):
                    online_dense_export_main()

                version_dir = os.path.join(
                    serving_root, "dense_hot_export", "versions", version
                )
                pt2_path = os.path.join(version_dir, "aoti", "aoti_model.pt2")
                self.assertTrue(os.path.exists(pt2_path))
                self.assertTrue(
                    os.path.exists(os.path.join(version_dir, "dense_meta.json"))
                )
                self.assertTrue(os.path.exists(os.path.join(version_dir, "READY")))
                self.assertFalse(
                    os.path.exists(os.path.join(version_dir, "scripted_model.pt"))
                )

                current_path = os.path.join(
                    serving_root, "dense_hot_export", "current.json"
                )
                with open(current_path) as f:
                    payload = json.load(f)
                self.assertEqual(payload["version"], version)
                self.assertEqual(payload["checkpoint_step"], 12)
                self.assertEqual(payload["sparse_step"], 12)
                self.assertEqual(payload["data_timestamp"], 123.0)
                self.assertEqual(payload["sparse_probes"], [])

                device = torch.device("cuda:0")
                pt2_model = torch._inductor.aoti_load_package(pt2_path)
                serving_inputs = self._build_serving_inputs(
                    version_dir, pt2_model, device
                )
                with torch.no_grad():
                    out = pt2_model(serving_inputs)
                self.assertEqual(out["probs"].shape, torch.Size([3]))
                self.assertTrue(torch.all(out["probs"] > 0.7))
        finally:
            torch.backends.cudnn.allow_tf32 = prev_cudnn_tf32
            torch.backends.cuda.matmul.allow_tf32 = prev_matmul_tf32

    def test_rejects_genrec_before_create_model(self) -> None:
        from tzrec.protos import pipeline_pb2, prompt_pb2
        from tzrec.protos.models import genrec_model_pb2
        from tzrec.tools import online_dense_export as online_dense_export_tool

        prompt_cfg = pipeline_pb2.EasyRecConfig(prompt_config=prompt_pb2.PromptConfig())
        genrec_model_cfg = pipeline_pb2.EasyRecConfig(
            model_config=model_pb2.ModelConfig(
                genrec_causal_lm_model=genrec_model_pb2.GenRecCausalLMModel()
            )
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            env = _base_env(tmp_dir)
            for name, cfg in (
                ("prompt_config", prompt_cfg),
                ("genrec_causal_lm_model", genrec_model_cfg),
            ):
                with self.subTest(case=name):
                    with (
                        mock.patch.dict(os.environ, env, clear=True),
                        mock.patch(
                            _BUILD_PATCHES.format("config_util.load_pipeline_config"),
                            return_value=cfg,
                        ),
                        mock.patch(
                            "tzrec.tools.online_dense_export."
                            "config_util.load_pipeline_config",
                            return_value=cfg,
                        ),
                        mock.patch("tzrec.main._create_model") as create_model_mock,
                        mock.patch(
                            "tzrec.tools.online_dense_export._create_model"
                        ) as cli_create_model_mock,
                    ):
                        with self.assertRaisesRegex(
                            RuntimeError, "does not support BaseGenRecModel"
                        ):
                            OnlineDenseExportManager(
                                model_dir=tmp_dir,
                                pipeline_config_path=os.path.join(
                                    tmp_dir, "pipeline.config"
                                ),
                                model=_mock_model({}),
                            )
                        create_model_mock.assert_not_called()

                        with self.assertRaisesRegex(
                            RuntimeError, "does not support BaseGenRecModel"
                        ):
                            online_dense_export_tool.export_online_dense_model(
                                pipeline_config_path=os.path.join(
                                    tmp_dir, "pipeline.config"
                                ),
                                checkpoint_path=tmp_dir,
                                model_dir=tmp_dir,
                            )
                        cli_create_model_mock.assert_not_called()

    def test_export_online_dense_model_version_validation(self) -> None:
        from tzrec.tools import online_dense_export as online_dense_export_tool

        invalid_versions = ["../bad", ".bad", "bad/ver", "bad ver", "a" * 256]
        with tempfile.TemporaryDirectory() as tmp_dir:
            for bad in invalid_versions:
                with self.subTest(version=bad):
                    with (
                        mock.patch.dict(os.environ, _base_env(tmp_dir), clear=True),
                        self.assertRaisesRegex(
                            ValueError, "Invalid dense export version"
                        ),
                    ):
                        online_dense_export_tool.export_online_dense_model(
                            pipeline_config_path="unused",
                            checkpoint_path="unused",
                            model_dir=tmp_dir,
                            version=bad,
                        )

            # A valid version passes validation before reaching config load.
            with (
                mock.patch.dict(os.environ, _base_env(tmp_dir), clear=True),
                mock.patch(
                    "tzrec.tools.online_dense_export.config_util.load_pipeline_config",
                    side_effect=RuntimeError("reached_config_load"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "reached_config_load"):
                    online_dense_export_tool.export_online_dense_model(
                        pipeline_config_path="unused",
                        checkpoint_path="unused",
                        model_dir=tmp_dir,
                        version="v1.0_alpha-2",
                    )


if __name__ == "__main__":
    unittest.main()
