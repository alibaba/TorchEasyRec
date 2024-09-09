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

# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Dict, List

import torch
from torch import nn
from torch.nn import functional as F

from tzrec.datasets.utils import Batch
from tzrec.features.feature import BaseFeature
from tzrec.models.multi_task_rank import MultiTaskRank
from tzrec.modules.embedding import EmbeddingGroup
from tzrec.modules.mmoe import MMoE as MMoEModule
from tzrec.modules.sequence import SimpleAttention
from tzrec.modules.task_tower import TaskTower
from tzrec.protos.model_pb2 import ModelConfig
from tzrec.utils.config_util import config_to_kwargs


class TMMoE(MultiTaskRank):
    """Multi-gate Mixture-of-Experts model.

    Args:
        model_config (ModelConfig): an instance of ModelConfig.
        features (list): list of features.
        labels (list): list of label names.
    """

    def __init__(
        self, model_config: ModelConfig, features: List[BaseFeature], labels: List[str]
    ) -> None:
        super().__init__(model_config, features, labels)
        self._task_tower_cfgs = self._model_config.task_towers

        self.embedding_group = EmbeddingGroup(
            features, list(model_config.feature_groups)
        )

        mmoe_feature_in = self.embedding_group.group_total_dim("dnn")
        self._use_skill = self.embedding_group.has_group("skill")
        self._use_lastnapplyjob = self.embedding_group.has_group("lastnapplyjob")
        self._use_label = self.embedding_group.has_group("label")
        self._use_pair = self.embedding_group.has_group("pair")
        if self._use_skill:
            self.skill_attn = SimpleAttention(
                sequence_dim=self.embedding_group.group_total_dim("skill.sequence"),
                query_dim=self.embedding_group.group_total_dim("skill.query"),
                input="skill",
            )
            mmoe_feature_in += self.skill_attn.output_dim()
        if self._use_lastnapplyjob:
            mmoe_feature_in += self.embedding_group.group_total_dim(
                "lastnapplyjob.sequence"
            )

        self._pair_features_dims = []
        if self._use_pair:
            self._pair_features_dims = self.embedding_group.group_dims("pair")

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
        self._task_output = nn.ModuleList()
        for task_tower_cfg in self._task_tower_cfgs:
            task_tower_cfg = config_to_kwargs(task_tower_cfg)
            mlp = task_tower_cfg["mlp"] if "mlp" in task_tower_cfg else None
            self._task_tower.append(
                TaskTower(tower_feature_in, task_tower_cfg["num_class"], mlp=mlp)
            )
            output_feature_in = task_tower_cfg["num_class"]
            if self._use_label:
                output_feature_in += self.embedding_group.group_total_dim("label")
            if self._use_pair:
                output_feature_in += len(self._pair_features_dims) // 2
            self._task_output.append(
                nn.Linear(output_feature_in, task_tower_cfg["num_class"], bias=False)
            )

    def predict(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """Forward the model.

        Args:
            batch (Batch): input batch data.

        Return:
            predictions (dict): a dict of predicted result.
        """
        grouped_features = self.embedding_group(batch)
        mmoe_features = [grouped_features["dnn"]]
        if self._use_skill:
            mmoe_features.append(self.skill_attn(grouped_features))
        if self._use_lastnapplyjob:
            sequence_length = grouped_features["lastnapplyjob.sequence_length"]
            sequence_length = torch.max(
                sequence_length, torch.ones_like(sequence_length)
            )
            mmoe_features.append(
                F.normalize(
                    torch.sum(grouped_features["lastnapplyjob.sequence"], dim=1)
                    / sequence_length.unsqueeze(1)
                )
            )

        task_input_list = self.mmoe(torch.cat(mmoe_features, dim=1))
        label_features = None
        sim_value = None
        if self._use_label:
            label_features = grouped_features["label"]
        if self._use_pair:
            pair_features = torch.reshape(
                grouped_features["pair"],
                (-1, len(self._pair_features_dims), self._pair_features_dims[0]),
            )
            pair_feature_list = pair_features.chunk(2, dim=1)
            sim_value = torch.cosine_similarity(
                pair_feature_list[0], pair_feature_list[1], 2
            )

        tower_outputs = {}
        for i, task_tower_cfg in enumerate(self._task_tower_cfgs):
            tower_name = task_tower_cfg.tower_name
            tower_output = self._task_tower[i](task_input_list[i])
            cat_outputs = [tower_output]
            if self._use_label:
                cat_outputs.append(label_features)
            if self._use_pair:
                cat_outputs.append(sim_value)
            tower_outputs[tower_name] = self._task_output[i](
                torch.cat(cat_outputs, dim=1)
            )

        return self._multi_task_output_to_prediction(tower_outputs)
