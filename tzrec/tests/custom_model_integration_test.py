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

import glob
import os
import shutil
import subprocess
import sys
import unittest

import tzrec
from tzrec.tests import utils
from tzrec.utils import checkpoint_util
from tzrec.utils.test_util import make_test_dir

_PROTO = """syntax = "proto2";
package my_models.protos;

import "tzrec/protos/module.proto";

message CustomRankModelConfig {
    required tzrec.protos.MLP mlp = 1;
}
"""

_MODEL = """from typing import Any, Dict, List, Optional

import torch
from torch import nn

from tzrec.datasets.utils import Batch
from tzrec.features.feature import BaseFeature
from tzrec.models.rank_model import RankModel
from tzrec.modules.mlp import MLP
from tzrec.protos.model_pb2 import ModelConfig
from tzrec.utils.config_util import config_to_kwargs

from my_models.protos.rank_pb2 import CustomRankModelConfig  # NOQA


class MyRankModel(RankModel):
    \"\"\"A rank model defined outside of the TorchEasyRec source tree.\"\"\"

    def __init__(
        self,
        model_config: ModelConfig,
        features: List[BaseFeature],
        labels: List[str],
        sample_weights: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_config, features, labels, sample_weights, **kwargs)
        self.init_input()
        self.mlp = MLP(
            self.embedding_group.group_total_dim("deep"),
            **config_to_kwargs(self._model_config.mlp),
        )
        self.output_mlp = nn.Linear(self.mlp.output_dim(), self._num_class)

    def predict(self, batch: Batch) -> Dict[str, torch.Tensor]:
        \"\"\"Forward the model.\"\"\"
        grouped_features = self.build_input(batch)
        y = self.output_mlp(self.mlp(grouped_features["deep"]))
        return self._output_to_prediction(y)
"""

_CONFIG = """
train_input_path: ""
eval_input_path: ""
model_dir: "experiments/custom_rank_model"
train_config {
    sparse_optimizer {
        adagrad_optimizer {
            lr: 0.001
        }
        constant_learning_rate {
        }
    }
    dense_optimizer {
        adam_optimizer {
            lr: 0.001
        }
        constant_learning_rate {
        }
    }
    num_epochs: 1
}
eval_config {
}
data_config {
    batch_size: 8192
    dataset_type: ParquetDataset
    label_fields: "clk"
    num_workers: 2
}
feature_configs {
    id_feature {
        feature_name: "id_1"
        num_buckets: 100
        embedding_dim: 16
    }
}
feature_configs {
    id_feature {
        feature_name: "id_2"
        num_buckets: 1000
        embedding_dim: 8
    }
}
feature_configs {
    raw_feature {
        feature_name: "raw_1"
    }
}
model_config {
    feature_groups {
        group_name: "deep"
        feature_names: "id_1"
        feature_names: "id_2"
        feature_names: "raw_1"
        group_type: DEEP
    }
    custom_model {
        class_path: "my_models.models.rank.MyRankModel"
        config {
            [type.googleapis.com/my_models.protos.CustomRankModelConfig] {
                mlp {
                    hidden_units: [32, 16]
                }
            }
        }
    }
    metrics {
        auc {}
    }
    losses {
        binary_cross_entropy {}
    }
}
"""


class CustomModelIntegrationTest(unittest.TestCase):
    """A custom model defined and trained from outside the TorchEasyRec tree."""

    def setUp(self):
        self.success = False
        self.test_dir = make_test_dir()

    def tearDown(self):
        if self.success:
            if os.path.exists(self.test_dir):
                shutil.rmtree(self.test_dir)

    def _create_custom_package(self) -> str:
        """Write and compile a custom model package outside of tzrec."""
        project_dir = os.path.abspath(os.path.join(self.test_dir, "my-rec-project"))
        for pkg in ["my_models", "my_models/models", "my_models/protos"]:
            os.makedirs(os.path.join(project_dir, pkg))
            with open(os.path.join(project_dir, pkg, "__init__.py"), "w"):
                pass
        with open(os.path.join(project_dir, "my_models/protos/rank.proto"), "w") as f:
            f.write(_PROTO)
        with open(os.path.join(project_dir, "my_models/models/rank.py"), "w") as f:
            f.write(_MODEL)

        # the protoc command documented in docs/source/models/user_define.md
        tzrec_path = os.path.dirname(os.path.dirname(os.path.abspath(tzrec.__file__)))
        subprocess.run(
            [
                sys.executable,
                "-m",
                "grpc_tools.protoc",
                "-I.",
                f"-I{tzrec_path}",
                "my_models/protos/rank.proto",
                "--python_out=.",
                "--pyi_out=.",
            ],
            cwd=project_dir,
            check=True,
        )
        return project_dir

    def test_custom_model_train_eval_export(self):
        project_dir = self._create_custom_package()
        config_path = os.path.join(project_dir, "custom_rank_model.config")
        with open(config_path, "w") as f:
            f.write(_CONFIG)
        # the model module has to be importable to parse a config using it
        sys.path.insert(0, project_dir)
        pythonpath = f".:{project_dir}"

        self.success = utils.test_train_eval(
            config_path,
            self.test_dir,
            pythonpath=pythonpath,
            env_str="CHECKPOINT_TAG=ci-{data_ts}",
        )
        self.assertTrue(self.success)

        # eval and export reload the config saved beside the model in a fresh
        # process, class_path alone has to resolve the model there
        saved_config_path = os.path.join(self.test_dir, "train/pipeline.config")
        self.assertTrue(os.path.exists(saved_config_path))
        # CHECKPOINT_TAG names the checkpoints, and every one is marked complete
        # for external schedulers. The parquet source carries no event-time, so
        # {data_ts} renders 0.
        ckpt_dirs = glob.glob(os.path.join(self.test_dir, "train", "model.ckpt-ci-0-*"))
        self.assertTrue(ckpt_dirs)
        for ckpt_dir in ckpt_dirs:
            self.assertTrue(
                os.path.exists(
                    os.path.join(ckpt_dir, checkpoint_util.CKPT_SUCCESS_FILENAME)
                )
            )
        self.success = utils.test_eval(
            saved_config_path, self.test_dir, pythonpath=pythonpath
        )
        self.assertTrue(self.success)
        self.success = utils.test_export(
            saved_config_path, self.test_dir, pythonpath=pythonpath
        )
        self.assertTrue(self.success)
        self.assertTrue(
            os.path.exists(os.path.join(self.test_dir, "export/scripted_model.pt"))
        )


if __name__ == "__main__":
    unittest.main()
