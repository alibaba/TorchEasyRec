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

from tzrec.tools.online_dense_export import export_online_dense_model


class OnlineDenseExportTest(unittest.TestCase):
    def test_export_online_dense_model_publishes_ready_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = os.path.join(tmp_dir, "model.ckpt-10")
            os.makedirs(checkpoint_path)
            pipeline_config_path = os.path.join(tmp_dir, "pipeline.config")
            open(pipeline_config_path, "w").close()

            def fake_export_distributed_embedding(**kwargs):
                self.assertTrue(kwargs["dense_only"])
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
                    "tzrec.tools.online_dense_export.export_distributed_embedding",
                    side_effect=fake_export_distributed_embedding,
                ),
            ):
                payload = export_online_dense_model(
                    pipeline_config_path=pipeline_config_path,
                    checkpoint_path=checkpoint_path,
                    model_dir=tmp_dir,
                    version="1234567890",
                    checkpoint_step=10,
                    data_timestamp=42.0,
                )

            version_dir = os.path.join(
                tmp_dir, "dense_hot_export", "versions", "1234567890"
            )
            self.assertTrue(os.path.exists(os.path.join(version_dir, "READY")))
            self.assertTrue(
                os.path.exists(os.path.join(version_dir, "scripted_model.pt"))
            )
            self.assertTrue(
                os.path.exists(os.path.join(version_dir, "dense_meta.json"))
            )

            for current_path in (
                os.path.join(tmp_dir, "dense_hot_update", "current.json"),
                os.path.join(tmp_dir, "dense_hot_export", "current.json"),
            ):
                with open(current_path) as f:
                    current = json.load(f)
                self.assertEqual(current["version"], "1234567890")
                self.assertEqual(current["checkpoint_step"], 10)
                self.assertEqual(current["data_timestamp"], 42.0)
                self.assertEqual(current["export_path"], os.path.abspath(version_dir))

            self.assertEqual(payload["version"], "1234567890")


if __name__ == "__main__":
    unittest.main()
