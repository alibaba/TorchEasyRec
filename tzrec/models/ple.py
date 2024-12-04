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

from typing import Dict, List, Optional

import torch
from torch import nn

from tzrec.datasets.utils import Batch
from tzrec.features.feature import BaseFeature
from tzrec.models.multi_task_rank import MultiTaskRank
from tzrec.modules.extraction_net import ExtractionNet
from tzrec.modules.task_tower import TaskTower
from tzrec.protos.model_pb2 import ModelConfig
from tzrec.protos.models import multi_task_rank_pb2
from tzrec.utils.config_util import config_to_kwargs


class PLE(MultiTaskRank):
    """Progressive Layered Extraction model.

    Args:
        model_config (ModelConfig): an instance of ModelConfig.
        features (list): list of features.
        labels (list): list of label names.
        sample_weights (list): sample weight names.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        features: List[BaseFeature],
        labels: List[str],
        sample_weights: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        super().__init__(model_config, features, labels, sample_weights, **kwargs)
        assert model_config.WhichOneof("model") == "ple", (
            "invalid model config: %s" % self._model_config.WhichOneof("model")
        )
        assert isinstance(self._model_config, multi_task_rank_pb2.PLE)

        self._task_nums = len(self._model_config.task_towers)
        self._layer_nums = len(self._model_config.extraction_networks)

        self.init_input()
        self.group_name = self.embedding_group.group_names()[0]
        feature_in = self.embedding_group.group_total_dim(self.group_name)

        self._extraction_nets = nn.ModuleList()
        in_extraction_networks = [feature_in] * self._task_nums
        in_shared_expert = feature_in
        for i, extraction_network_cfg in enumerate(
            self._model_config.extraction_networks
        ):
            if i == self._layer_nums - 1:
                final_flag = True
            else:
                final_flag = False
            extraction_network_cfg = config_to_kwargs(extraction_network_cfg)
            extraction = ExtractionNet(
                in_extraction_networks,
                in_shared_expert,
                final_flag=final_flag,
                **extraction_network_cfg,
            )
            self._extraction_nets.append(extraction)
            output_dims = extraction.output_dim()
            in_extraction_networks = output_dims[:-1]
            in_shared_expert = output_dims[-1]
        self._task_tower = nn.ModuleList()
        for i, tower_cgf in enumerate(self._task_tower_cfgs):
            tower_cgf = config_to_kwargs(tower_cgf)
            mlp = tower_cgf["mlp"] if "mlp" in tower_cgf else None
            self._task_tower.append(
                TaskTower(in_extraction_networks[i], tower_cgf["num_class"], mlp=mlp)
            )

    def predict(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """Forward the model.

        Args:
            batch (Batch): input batch data.

        Return:
            predictions (dict): a dict of predicted result.
        """
        grouped_features = self.build_input(batch)
        net = grouped_features[self.group_name]
        extraction_network_fea = [net] * self._task_nums
        shared_expert_fea = net
        for extraction_net in self._extraction_nets:
            extraction_network_fea, shared_expert_fea = extraction_net(
                extraction_network_fea, shared_expert_fea
            )
        tower_outputs = {}
        for i, task_tower_cfg in enumerate(self._task_tower_cfgs):
            tower_name = task_tower_cfg.tower_name
            tower_output = self._task_tower[i](extraction_network_fea[i])
            tower_outputs[tower_name] = tower_output
        return self._multi_task_output_to_prediction(tower_outputs)
