# Copyright (c) 2024, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, List

import torch
from torch import nn

from tzrec.datasets.utils import Batch
from tzrec.features.feature import BaseFeature
from tzrec.models.multi_task_rank import MultiTaskRank
from tzrec.modules.mmoe import MMoE as MMoEModule
from tzrec.modules.task_tower import TaskTower
from tzrec.protos.model_pb2 import ModelConfig
from tzrec.utils.config_util import config_to_kwargs


class MMoE(MultiTaskRank):
    """Multi-gate Mixture-of-Experts model.

    Args:
        model_config (ModelConfig): an instance of ModelConfig.
        features (list): list of features.
        labels (list): list of label names.
        sample_weights (list): sample weight names.
    """

    def __init__(
        self, model_config: ModelConfig, features: List[BaseFeature], labels: List[str], sample_weights: List[str] = []
    ) -> None:
        super().__init__(model_config, features, labels, sample_weights)

        self.init_input()
        self.group_name = self.embedding_group.group_names()[0]
        mmoe_feature_in = self.embedding_group.group_total_dim(self.group_name)
        self.mmoe = MMoEModule(
            in_features=mmoe_feature_in,
            expert_mlp=config_to_kwargs(self._model_config.expert_mlp),
            num_expert=self._model_config.num_expert,
            num_task=len(self._task_tower_cfgs),
            gate_mlp=config_to_kwargs(self._model_config.gate_mlp)
            if self._model_config.HasField("gate_mlp")
            else None,
        )

        tower_feature_in = self.mmoe.output_dim()
        self._task_tower = nn.ModuleList()
        for task_tower_cfg in self._task_tower_cfgs:
            task_tower_cfg = config_to_kwargs(task_tower_cfg)
            mlp = task_tower_cfg["mlp"] if "mlp" in task_tower_cfg else None
            self._task_tower.append(
                TaskTower(tower_feature_in, task_tower_cfg["num_class"], mlp=mlp)
            )

    def predict(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """Forward the model.

        Args:
            batch (Batch): input batch data.

        Return:
            predictions (dict): a dict of predicted result.
        """
        grouped_features = self.build_input(batch)
        task_input_list = self.mmoe(grouped_features[self.group_name])

        tower_outputs = {}
        for i, task_tower_cfg in enumerate(self._task_tower_cfgs):
            tower_name = task_tower_cfg.tower_name
            tower_output = self._task_tower[i](task_input_list[i])
            tower_outputs[tower_name] = tower_output

        return self._multi_task_output_to_prediction(tower_outputs)
