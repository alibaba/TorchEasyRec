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

import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from tzrec.models.match_model import MatchModel
from tzrec.models.tdm import TDM
from tzrec.tools.online_dense_export import (
    CURRENT_JSON,
    _prune_old_dense_versions,
    export_online_dense_model,
)
from tzrec.utils.online_dense_export_util import _max_kept_versions


class OnlineDenseExportTest(unittest.TestCase):
    def test_export_online_dense_model_publishes_ready_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = os.path.join(tmp_dir, "model.ckpt-10")
            os.makedirs(checkpoint_path)
            pipeline_config_path = os.path.join(tmp_dir, "pipeline.config")
            open(pipeline_config_path, "w").close()

            def fake_export_dense_model_cpu(**kwargs):
                self.assertNotIn("dense_only", kwargs)
                save_dir = kwargs["save_dir"]
                os.makedirs(save_dir, exist_ok=True)
                with open(os.path.join(save_dir, "scripted_model.pt"), "w") as f:
                    f.write("pt")
                with open(os.path.join(save_dir, "dense_meta.json"), "w") as f:
                    json.dump({"group": ["feature__ebc"]}, f)

            dummy_config = SimpleNamespace(
                feature_configs=[],
                data_config=SimpleNamespace(label_fields=[]),
                model_config=SimpleNamespace(),
            )
            dummy_model = mock.Mock()

            with (
                mock.patch.dict(
                    os.environ,
                    {"USE_DISTRIBUTED_EMBEDDING": "1"},
                    clear=False,
                ),
                mock.patch(
                    "tzrec.tools.online_dense_export.config_util.load_pipeline_config",
                    return_value=dummy_config,
                ),
                mock.patch(
                    "tzrec.tools.online_dense_export._create_features",
                    return_value=[],
                ),
                mock.patch(
                    "tzrec.tools.online_dense_export._create_model",
                    return_value=dummy_model,
                ),
                mock.patch(
                    "tzrec.tools.online_dense_export.ScriptWrapper",
                    side_effect=lambda model: model,
                ),
                mock.patch(
                    "tzrec.tools.online_dense_export.export_dense_model_cpu",
                    side_effect=fake_export_dense_model_cpu,
                ),
            ):
                payload = export_online_dense_model(
                    pipeline_config_path=pipeline_config_path,
                    checkpoint_path=checkpoint_path,
                    model_dir=tmp_dir,
                    version="20260623174703",
                    checkpoint_step=10,
                    data_timestamp=42.0,
                )

            version_dir = os.path.join(
                tmp_dir, "dense_hot_export", "versions", "20260623174703"
            )
            self.assertTrue(os.path.exists(os.path.join(version_dir, "READY")))
            self.assertTrue(
                os.path.exists(os.path.join(version_dir, "scripted_model.pt"))
            )
            self.assertTrue(
                os.path.exists(os.path.join(version_dir, "dense_meta.json"))
            )

            current_path = os.path.join(tmp_dir, "dense_hot_export", "current.json")
            with open(current_path) as f:
                current = json.load(f)
            self.assertEqual(
                set(current.keys()),
                {
                    "checkpoint_path",
                    "checkpoint_step",
                    "created_at",
                    "data_timestamp",
                    "sparse_probes",
                    "sparse_step",
                    "version",
                },
            )
            self.assertEqual(current["version"], "20260623174703")
            self.assertEqual(
                current["checkpoint_path"], os.path.abspath(checkpoint_path)
            )
            self.assertEqual(current["checkpoint_step"], 10)
            # bootstrap pairing: FS full load and checkpoint are same-source
            self.assertEqual(current["sparse_step"], 10)
            self.assertEqual(current["sparse_probes"], [])
            self.assertEqual(current["data_timestamp"], 42.0)
            self.assertTrue(current["created_at"])
            self.assertFalse(os.path.exists(os.path.join(tmp_dir, "dense_hot_update")))

            self.assertEqual(payload, current)
            self.assertEqual(payload["version"], "20260623174703")

    def test_prune_old_versions_and_stale_tmp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = os.path.join(tmp_dir, "model.ckpt-10")
            os.makedirs(checkpoint_path)
            pipeline_config_path = os.path.join(tmp_dir, "pipeline.config")
            open(pipeline_config_path, "w").close()

            export_root = os.path.join(tmp_dir, "dense_hot_export")
            versions_root = os.path.join(export_root, "versions")
            os.makedirs(versions_root)
            for v in (
                "20260101000001",
                "20260101000002",
                "20260101000003",
                "20260101000004",
            ):
                vd = os.path.join(versions_root, v)
                os.makedirs(vd)
                with open(os.path.join(vd, "READY"), "w") as f:
                    f.write("x")
            # stale per-PID tmp dir + current.json.tmp left by a crashed export
            stale_pid = os.getpid() + 1
            os.makedirs(os.path.join(versions_root, f"20260101000001.tmp.{stale_pid}"))
            with open(
                os.path.join(export_root, f"current.json.tmp.{stale_pid}"), "w"
            ) as f:
                f.write("x")
            # this process's own tmp dir mimics an in-flight worker build and
            # must survive both the sweep and the newest-K retention
            own_tmp = f"20260101000002.tmp.{os.getpid()}"
            os.makedirs(os.path.join(versions_root, own_tmp))

            def fake_export_dense_model_cpu(**kwargs):
                save_dir = kwargs["save_dir"]
                os.makedirs(save_dir, exist_ok=True)
                with open(os.path.join(save_dir, "scripted_model.pt"), "w") as f:
                    f.write("pt")
                with open(os.path.join(save_dir, "dense_meta.json"), "w") as f:
                    json.dump({}, f)

            dummy_config = SimpleNamespace(
                feature_configs=[],
                data_config=SimpleNamespace(label_fields=[]),
                model_config=SimpleNamespace(),
            )

            with (
                mock.patch.dict(
                    os.environ,
                    {
                        "USE_DISTRIBUTED_EMBEDDING": "1",
                        "ONLINE_DENSE_EXPORT_KEEP_VERSIONS": "3",
                    },
                    clear=False,
                ),
                mock.patch(
                    "tzrec.tools.online_dense_export.config_util.load_pipeline_config",
                    return_value=dummy_config,
                ),
                mock.patch(
                    "tzrec.tools.online_dense_export._create_features",
                    return_value=[],
                ),
                mock.patch(
                    "tzrec.tools.online_dense_export._create_model",
                    return_value=mock.Mock(),
                ),
                mock.patch(
                    "tzrec.tools.online_dense_export.ScriptWrapper",
                    side_effect=lambda model: model,
                ),
                mock.patch(
                    "tzrec.tools.online_dense_export.export_dense_model_cpu",
                    side_effect=fake_export_dense_model_cpu,
                ),
            ):
                export_online_dense_model(
                    pipeline_config_path=pipeline_config_path,
                    checkpoint_path=checkpoint_path,
                    model_dir=tmp_dir,
                    version="20260623174704",
                    checkpoint_step=10,
                    data_timestamp=42.0,
                )

            remaining = sorted(os.listdir(versions_root))
            self.assertEqual(
                remaining,
                [own_tmp, "20260101000003", "20260101000004", "20260623174704"],
            )
            self.assertFalse(
                os.path.exists(
                    os.path.join(versions_root, f"20260101000001.tmp.{stale_pid}")
                )
            )
            self.assertFalse(
                os.path.exists(
                    os.path.join(export_root, f"current.json.tmp.{stale_pid}")
                )
            )

    def test_prune_spare_current_version_when_oldest(self) -> None:
        """current.json's version is never pruned, even when it sorts oldest."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            export_root = os.path.join(tmp_dir, "dense_hot_export")
            versions_root = os.path.join(export_root, "versions")
            os.makedirs(versions_root)
            for v in (
                "20260101000001",
                "20260101000002",
                "20260101000003",
                "20260101000004",
                "20260101000005",
                "20260101000006",
            ):
                os.makedirs(os.path.join(versions_root, v))
            # explicit --version (or clock rollback) publishes an OLDER
            # timestamp than every on-disk version
            published = "20200101000000"
            os.makedirs(os.path.join(versions_root, published))
            with open(os.path.join(export_root, CURRENT_JSON), "w") as f:
                json.dump({"version": published}, f)

            with mock.patch.dict(
                os.environ,
                {"ONLINE_DENSE_EXPORT_KEEP_VERSIONS": "3"},
                clear=False,
            ):
                _prune_old_dense_versions(export_root, versions_root)

            # the published (oldest) version survives despite KEEP=3
            self.assertTrue(os.path.isdir(os.path.join(versions_root, published)))
            # the three newest on-disk versions survive
            for v in ("20260101000004", "20260101000005", "20260101000006"):
                self.assertTrue(os.path.isdir(os.path.join(versions_root, v)))
            # older non-current versions are pruned
            for v in ("20260101000001", "20260101000002", "20260101000003"):
                self.assertFalse(os.path.isdir(os.path.join(versions_root, v)))

    def test_prune_keep_all_by_default(self) -> None:
        """Unset KEEP_VERSIONS defaults to 0 and prunes no dense versions."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            export_root = os.path.join(tmp_dir, "dense_hot_export")
            versions_root = os.path.join(export_root, "versions")
            os.makedirs(versions_root)
            versions = [f"2026010100000{i}" for i in range(1, 7)]
            for v in versions:
                os.makedirs(os.path.join(versions_root, v))

            env = os.environ.copy()
            env.pop("ONLINE_DENSE_EXPORT_KEEP_VERSIONS", None)
            with mock.patch.dict(os.environ, env, clear=True):
                _prune_old_dense_versions(export_root, versions_root)

            self.assertEqual(sorted(os.listdir(versions_root)), versions)

    def test_keep_versions_validation(self) -> None:
        """A positive KEEP_VERSIONS below 3 is rejected; 0 and >=3 pass."""
        for value in ("-1", "1", "2"):
            with mock.patch.dict(
                os.environ,
                {"ONLINE_DENSE_EXPORT_KEEP_VERSIONS": value},
                clear=False,
            ):
                with self.assertRaises(ValueError):
                    _max_kept_versions()
        for value, expected in (("0", 0), ("3", 3), ("10", 10)):
            with mock.patch.dict(
                os.environ,
                {"ONLINE_DENSE_EXPORT_KEEP_VERSIONS": value},
                clear=False,
            ):
                self.assertEqual(_max_kept_versions(), expected)

    def test_rejects_match_model_and_tdm(self) -> None:
        """MatchModel/TDM need per-tower/per-module export; reject them."""
        for spec in (MatchModel, TDM):
            with tempfile.TemporaryDirectory() as tmp_dir:
                checkpoint_path = os.path.join(tmp_dir, "model.ckpt-10")
                os.makedirs(checkpoint_path)
                pipeline_config_path = os.path.join(tmp_dir, "pipeline.config")
                open(pipeline_config_path, "w").close()
                dummy_config = SimpleNamespace(
                    feature_configs=[],
                    data_config=SimpleNamespace(label_fields=[]),
                    model_config=SimpleNamespace(),
                )
                with (
                    mock.patch.dict(
                        os.environ,
                        {"USE_DISTRIBUTED_EMBEDDING": "1"},
                        clear=False,
                    ),
                    mock.patch(
                        "tzrec.tools.online_dense_export.config_util.load_pipeline_config",
                        return_value=dummy_config,
                    ),
                    mock.patch(
                        "tzrec.tools.online_dense_export._create_features",
                        return_value=[],
                    ),
                    mock.patch(
                        "tzrec.tools.online_dense_export._create_model",
                        return_value=mock.Mock(spec=spec),
                    ),
                ):
                    with self.assertRaisesRegex(RuntimeError, "does not support"):
                        export_online_dense_model(
                            pipeline_config_path=pipeline_config_path,
                            checkpoint_path=checkpoint_path,
                            model_dir=tmp_dir,
                            version="20260623174705",
                        )

    def test_export_online_dense_model_honors_online_dense_export_dir(self) -> None:
        """ONLINE_DENSE_EXPORT_DIR redirects the publish root away from model_dir."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = os.path.join(tmp_dir, "model.ckpt-10")
            os.makedirs(checkpoint_path)
            pipeline_config_path = os.path.join(tmp_dir, "pipeline.config")
            open(pipeline_config_path, "w").close()
            override_root = os.path.join(tmp_dir, "serving_root")

            def fake_export_dense_model_cpu(**kwargs):
                save_dir = kwargs["save_dir"]
                os.makedirs(save_dir, exist_ok=True)
                with open(os.path.join(save_dir, "scripted_model.pt"), "w") as f:
                    f.write("pt")
                with open(os.path.join(save_dir, "dense_meta.json"), "w") as f:
                    json.dump({}, f)

            dummy_config = SimpleNamespace(
                feature_configs=[],
                data_config=SimpleNamespace(label_fields=[]),
                model_config=SimpleNamespace(),
            )
            with (
                mock.patch.dict(
                    os.environ,
                    {
                        "USE_DISTRIBUTED_EMBEDDING": "1",
                        "ONLINE_DENSE_EXPORT_DIR": override_root,
                    },
                    clear=False,
                ),
                mock.patch(
                    "tzrec.tools.online_dense_export.config_util.load_pipeline_config",
                    return_value=dummy_config,
                ),
                mock.patch(
                    "tzrec.tools.online_dense_export._create_features",
                    return_value=[],
                ),
                mock.patch(
                    "tzrec.tools.online_dense_export._create_model",
                    return_value=mock.Mock(),
                ),
                mock.patch(
                    "tzrec.tools.online_dense_export.ScriptWrapper",
                    side_effect=lambda model: model,
                ),
                mock.patch(
                    "tzrec.tools.online_dense_export.export_dense_model_cpu",
                    side_effect=fake_export_dense_model_cpu,
                ),
            ):
                export_online_dense_model(
                    pipeline_config_path=pipeline_config_path,
                    checkpoint_path=checkpoint_path,
                    model_dir=tmp_dir,
                    version="20260623174706",
                    checkpoint_step=10,
                    data_timestamp=42.0,
                )

            version_dir = os.path.join(
                override_root, "dense_hot_export", "versions", "20260623174706"
            )
            self.assertTrue(os.path.exists(os.path.join(version_dir, "READY")))
            self.assertTrue(
                os.path.exists(os.path.join(version_dir, "scripted_model.pt"))
            )
            self.assertTrue(
                os.path.exists(
                    os.path.join(override_root, "dense_hot_export", "current.json")
                )
            )
            self.assertFalse(os.path.exists(os.path.join(tmp_dir, "dense_hot_export")))


if __name__ == "__main__":
    unittest.main()
