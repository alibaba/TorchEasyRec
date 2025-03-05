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

from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from tzrec.datasets.utils import Batch
from tzrec.features.feature import BaseFeature
from tzrec.models.rank_model import RankModel
from tzrec.modules.mlp import MLP
from tzrec.modules.utils import div_no_nan
from tzrec.protos.model_pb2 import ModelConfig
from tzrec.protos.simi_pb2 import Similarity
from tzrec.utils.config_util import config_to_kwargs


class RocketLaunching(RankModel):
    """RocketLaunching model.

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
        **kwargs: Any,
    ) -> None:
        super().__init__(model_config, features, labels, sample_weights, **kwargs)
        self.hidden_layer_feature_output = self._model_config.feature_based_distillation
        self.init_input()
        self.group_name = self.embedding_group.group_names()[0]
        feature_in = self.embedding_group.group_total_dim(self.group_name)
        self.share_mlp = MLP(
            feature_in, **config_to_kwargs(self._model_config.share_mlp)
        )

        self.booster_mlp = MLP(
            self.share_mlp.output_dim(),
            hidden_layer_feature_output=self.hidden_layer_feature_output,
            **config_to_kwargs(self._model_config.booster_mlp),
        )
        self.booster_liner = torch.nn.Linear(
            self.booster_mlp.output_dim(), self._num_class
        )

        self.light_mlp = MLP(
            self.share_mlp.output_dim(),
            hidden_layer_feature_output=self.hidden_layer_feature_output,
            **config_to_kwargs(self._model_config.light_mlp),
        )
        self.light_liner = torch.nn.Linear(self.light_mlp.output_dim(), self._num_class)
        self.hint_loss_name = "hint_l2_loss"

    def predict(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """Forward the model.

        Args:
            batch (Batch): input batch data.

        Return:
            predictions (dict): a dict of predicted result.
        """
        grouped_features = self.build_input(batch)
        net = grouped_features[self.group_name]
        share_net = self.share_mlp(net)
        self.light_net = self.light_mlp(share_net.detach())
        if self.hidden_layer_feature_output:
            light_out = self.light_liner(self.light_net["hidden_layer_end"])
        else:
            light_out = self.light_liner(self.light_net)
        prediction_dict = {}
        prediction_dict.update(self._output_to_prediction(light_out, suffix="_light"))

        if self.training:
            self.booster_net = self.booster_mlp(share_net)
            if self.hidden_layer_feature_output:
                booster_out = self.booster_liner(self.booster_net["hidden_layer_end"])
            else:
                booster_out = self.booster_liner(self.booster_net)
            prediction_dict.update(
                self._output_to_prediction(booster_out, suffix="_booster")
            )
        return prediction_dict

    def feature_based_sim(
        self, feature_distillation_function: Similarity, i: int, j: int
    ) -> torch.Tensor:
        """Compute similarity between booster_net and light_net."""
        booster_feature_no_gradient = self.booster_net["hidden_layer" + str(j)].detach()
        light_feature = self.light_net["hidden_layer" + str(i)]
        if feature_distillation_function == Similarity.COSINE:
            booster_feature_no_gradient_norm = F.normalize(
                booster_feature_no_gradient, p=2, dim=1
            )
            light_feature_norm = F.normalize(light_feature, p=2, dim=1)
            multi_middle_layer = torch.mul(
                booster_feature_no_gradient_norm, light_feature_norm
            )
            sim_middle_layer = torch.mean(
                torch.sum(multi_middle_layer, dim=1, keepdim=True)
            )
            return sim_middle_layer
        else:
            squared_diff = torch.square(booster_feature_no_gradient - light_feature)
            return torch.sqrt(torch.sum(squared_diff))

    def init_loss(self) -> None:
        """Initialize loss modules."""
        reduction = "none" if self._sample_weight_name else "mean"
        for loss_cfg in self._base_model_config.losses:
            self._init_loss_impl(
                loss_cfg, self._num_class, reduction=reduction, suffix="_booster"
            )
            self._init_loss_impl(
                loss_cfg, self._num_class, reduction=reduction, suffix="_light"
            )
        self._loss_modules[self.hint_loss_name] = nn.MSELoss(reduction=reduction)

    def init_metric(self) -> None:
        """Initialize metric modules."""
        for metric_cfg in self._base_model_config.metrics:
            self._init_metric_impl(metric_cfg, self._num_class, "_booster")
            self._init_metric_impl(metric_cfg, self._num_class, "_light")

        for loss_cfg in self._base_model_config.losses:
            self._init_loss_metric_impl(loss_cfg, "_booster")
            self._init_loss_metric_impl(loss_cfg, "_light")

    def loss(
        self, predictions: Dict[str, torch.Tensor], batch: Batch
    ) -> Dict[str, torch.Tensor]:
        """Compute loss of the model."""
        losses = {}
        if self._sample_weight_name:
            loss_weight = batch.sample_weights[self._sample_weight_name]
            loss_weight = div_no_nan(loss_weight, torch.mean(loss_weight))
        else:
            loss_weight = None
        # compute booster and light net classifier loss
        for loss_cfg in self._base_model_config.losses:
            if self.training:
                losses.update(
                    self._loss_impl(
                        predictions,
                        batch,
                        self._label_name,
                        loss_weight,
                        loss_cfg,
                        num_class=self._num_class,
                        suffix="_booster",
                    )
                )
            losses.update(
                self._loss_impl(
                    predictions,
                    batch,
                    self._label_name,
                    loss_weight,
                    loss_cfg,
                    num_class=self._num_class,
                    suffix="_light",
                )
            )
        losses.update(self._loss_collection)
        if self.training:
            # compute booster net and light net similarity loss
            if self._model_config.feature_based_distillation:
                booster_hidden_units = self._model_config.booster_mlp.hidden_units
                light_hidden_units = self._model_config.light_mlp.hidden_units
                self.feature_distillation_function = (
                    self._model_config.feature_distillation_function
                )
                count = 0
                for i, unit_i in enumerate(light_hidden_units):
                    for j, unit_j in enumerate(booster_hidden_units):
                        if unit_i == unit_j:
                            losses["similarity_" + str(count)] = (
                                -0.1
                                * self.feature_based_sim(
                                    self.feature_distillation_function, i, j
                                )
                            )
                            count += 1
                            break
            # computer booster logits and light logits mse loss
            logits_booster = predictions["logits_booster"]
            logits_light = predictions["logits_light"]
            losses[self.hint_loss_name] = self._loss_modules[self.hint_loss_name](
                logits_light, logits_booster.detach()
            )
        return losses

    def update_metric(
        self,
        predictions: Dict[str, torch.Tensor],
        batch: Batch,
        losses: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        """Update metric state.

        Args:
                predictions (dict): a dict of predicted result.
                batch (Batch): input batch data.
                losses (dict, optional): a dict of loss.
        """
        for metric_cfg in self._base_model_config.metrics:
            if self.training:
                self._update_metric_impl(
                    predictions,
                    batch,
                    self._label_name,
                    metric_cfg,
                    num_class=self._num_class,
                    suffix="_booster",
                )
            self._update_metric_impl(
                predictions,
                batch,
                self._label_name,
                metric_cfg,
                num_class=self._num_class,
                suffix="_light",
            )
        if losses is not None:
            for loss_cfg in self._base_model_config.losses:
                if self.training:
                    self._update_loss_metric_impl(
                        losses, batch, self._label_name, loss_cfg, suffix="_booster"
                    )
                self._update_loss_metric_impl(
                    losses, batch, self._label_name, loss_cfg, suffix="_light"
                )
