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
from tzrec.modules.intervention import Intervention
from tzrec.modules.mlp import MLP
from tzrec.modules.mmoe import MMoE as MMoEModule
from tzrec.protos.model_pb2 import ModelConfig
from tzrec.protos.models import multi_task_rank_pb2
from tzrec.utils.config_util import config_to_kwargs


class DC2VR(MultiTaskRank):
    """DeCoudounding Conversion Rate.

    Args:
        model_config (ModelConfig): an instance of ModelConfig.
        features (list): list of features.
        labels (list): list of label names.
    """

    def __init__(
        self, model_config: ModelConfig, features: List[BaseFeature], labels: List[str]
    ) -> None:
        super().__init__(model_config, features, labels)
        assert model_config.WhichOneof("model") == "dc2vr", (
            "invalid model config: %s" % self._model_config.WhichOneof("model")
        )
        assert isinstance(self._model_config, multi_task_rank_pb2.DC2VR)

        self._task_tower_cfgs = self._model_config.task_towers
        self.init_input()
        self.group_name = self.embedding_group.group_names()[0]
        feature_in = self.embedding_group.group_total_dim(self.group_name)

        self.bottom_mlp = None
        if self._model_config.HasField("bottom_mlp"):
            self.bottom_mlp = MLP(
                feature_in, **config_to_kwargs(self._model_config.bottom_mlp)
            )
            feature_in = self.bottom_mlp.output_dim()

        self.mmoe = None
        if self._model_config.HasField("expert_mlp"):
            self.mmoe = MMoEModule(
                in_features=feature_in,
                expert_mlp=config_to_kwargs(self._model_config.expert_mlp),
                num_expert=self._model_config.num_expert,
                num_task=len(self._task_tower_cfgs),
                gate_mlp=config_to_kwargs(self._model_config.gate_mlp)
                if self._model_config.HasField("gate_mlp")
                else None,
            )
            feature_in = self.mmoe.output_dim()

        self.task_mlps = nn.ModuleDict()
        for task_tower_cfg in self._task_tower_cfgs:
            if task_tower_cfg.HasField("mlp"):
                tower_mlp = MLP(feature_in, **config_to_kwargs(task_tower_cfg.mlp))
                self.task_mlps[task_tower_cfg.tower_name] = tower_mlp

        self.intervention = nn.ModuleDict()
        for task_tower_cfg in self._task_tower_cfgs:
            tower_name = task_tower_cfg.tower_name
            if task_tower_cfg.HasField("low_rank_dim"):
                if tower_name in self.task_mlps:
                    base_intervention_dim = self.task_mlps[tower_name].output_dim()
                else:
                    base_intervention_dim = feature_in
                source_intervention_dim = 0
                for intervention_tower_name in task_tower_cfg.intervention_tower_names:
                    if intervention_tower_name in self.intervention:
                        source_intervention_dim += self.intervention[
                            intervention_tower_name
                        ].output_dim()
                    elif intervention_tower_name in self.task_mlps:
                        source_intervention_dim += self.task_mlps[
                            intervention_tower_name
                        ].output_dim()
                    else:
                        source_intervention_dim += feature_in
                intervention = Intervention(
                    base_intervention_dim,
                    source_intervention_dim,
                    task_tower_cfg.low_rank_dim,
                    task_tower_cfg.dropout_ratio,
                )
                self.intervention[tower_name] = intervention

        self.task_outputs = nn.ModuleList()
        for task_tower_cfg in self._task_tower_cfgs:
            tower_name = task_tower_cfg.tower_name
            if tower_name in self.intervention:
                input_dim = self.intervention[tower_name].output_dim()
            elif tower_name in self.task_mlps:
                input_dim = self.task_mlps[tower_name].output_dim()
            else:
                input_dim = feature_in
            self.task_outputs.append(nn.Linear(input_dim, task_tower_cfg.num_class))

    def predict(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """Forward the model.

        Args:
            batch (Batch): input batch data.

        Return:
            predictions (dict): a dict of predicted result.
        """
        grouped_features = self.build_input(batch)

        net = grouped_features[self.group_name]
        if self.bottom_mlp is not None:
            net = self.bottom_mlp(net)

        if self.mmoe is not None:
            task_input_list = self.mmoe(net)
        else:
            task_input_list = [net] * len(self._task_tower_cfgs)

        task_net = {}
        for i, task_tower_cfg in enumerate(self._task_tower_cfgs):
            tower_name = task_tower_cfg.tower_name
            if tower_name in self.task_mlps.keys():
                task_net[tower_name] = self.task_mlps[tower_name](task_input_list[i])
            else:
                task_net[tower_name] = task_input_list[i]

        intervention = {}
        for task_tower_cfg in self._task_tower_cfgs:
            tower_name = task_tower_cfg.tower_name
            if task_tower_cfg.HasField("low_rank_dim"):
                intervention_base = task_net[tower_name]
                intervention_source = []
                for intervention_tower_name in task_tower_cfg.intervention_tower_names:
                    intervention_source.append(intervention[intervention_tower_name])
                intervention_source = torch.cat(intervention_source, dim=-1)  # .mean(0)
                intervention[tower_name] = self.intervention[tower_name](
                    intervention_base, intervention_source
                )
            else:
                intervention[tower_name] = task_net[tower_name]

        tower_outputs = {}
        for i, task_tower_cfg in enumerate(self._task_tower_cfgs):
            tower_name = task_tower_cfg.tower_name
            tower_output = self.task_outputs[i](intervention[tower_name])
            tower_outputs[tower_name] = tower_output

        return self._multi_task_output_to_prediction(tower_outputs)
