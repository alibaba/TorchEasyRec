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


from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.functional import F

from tzrec.utils.logging_util import logger


@torch.fx.wrap
def _feature_tile(
    feature_p: torch.Tensor,
    feature: torch.Tensor,
) -> Tensor:
    return feature_p.tile([feature.size(0), 1])


@torch.fx.wrap
def _update_dict_tensor(
    group_name: str, features: Dict[str, torch.Tensor], new_feature: torch.Tensor
) -> Dict[str, torch.Tensor]:
    features[group_name] = new_feature
    return features


class VariationalDropout(nn.Module):
    """Rank features by variational dropout.

    Args:
        features_dimension: features dimension.
        name: group name.
        regularization_lambda: regularization lambda
    """

    def __init__(
        self,
        features_dimension: Dict[str, int],
        name: str,
        regularization_lambda: Optional[float] = 0.01,
        **kwargs: Optional[Dict[str, Any]],
    ) -> None:
        super().__init__()
        self.group_name = name
        self.features_dimension = features_dimension
        self._regularization_lambda = regularization_lambda
        self.feature_p = nn.parameter.Parameter(
            torch.randn(len(features_dimension), requires_grad=True)
        )
        self.feature_dim_repeat = nn.parameter.Parameter(
            torch.tensor(list(features_dimension.values()), dtype=torch.int),
            requires_grad=False,
        )
        logger.info(
            f"group name: {name} has VariationalDropout ! "
            f"feature number:{len(features_dimension)}, "
            f"features:{features_dimension.keys()}"
        )

    def concrete_dropout_neuron(
        self, dropout_p: torch.Tensor, temp: float = 0.1
    ) -> Tensor:
        """Add disturbance to dropout probability."""
        EPSILON = torch.finfo(torch.float32).eps
        unif_noise = torch.rand_like(dropout_p)
        approx = (
            torch.log(dropout_p + EPSILON)
            - torch.log(1.0 - dropout_p + EPSILON)
            + torch.log(unif_noise + EPSILON)
            - torch.log(1.0 - unif_noise + EPSILON)
        )
        approx_output = F.sigmoid(approx / temp)
        return approx_output

    def sample_noisy_input(self, feature: Tensor) -> Tensor:
        """Add noisy for feature."""
        if self.training:
            dropout_p = self.feature_p.sigmoid()
            dropout_p = torch.unsqueeze(dropout_p, dim=0)
            dropout_p = _feature_tile(dropout_p, feature)
            bern_val = self.concrete_dropout_neuron(dropout_p)
            bern_val = torch.repeat_interleave(
                bern_val, self.feature_dim_repeat, dim=-1
            )
            noisy_input = feature * (1 - bern_val)
        else:
            dropout_p = self.feature_p.sigmoid()
            dropout_p = torch.unsqueeze(dropout_p, dim=0)
            dropout_p = _feature_tile(dropout_p, feature)
            dropout_p = torch.repeat_interleave(
                dropout_p, self.feature_dim_repeat, dim=-1
            )
            noisy_input = feature * (1 - dropout_p)
        return noisy_input

    def forward(self, feature: Tensor) -> Tuple[Tensor, Tensor]:
        """Add dropout to feature."""
        noisy_input = self.sample_noisy_input(feature)
        dropout_p = self.feature_p.sigmoid()
        variational_dropout_penalty = 1.0 - dropout_p
        sample_num = feature.size(0)
        # pyre-ignore [58]
        variational_dropout_penalty_lambda = self._regularization_lambda / sample_num
        variational_dropout_loss_sum = variational_dropout_penalty_lambda * torch.sum(
            variational_dropout_penalty
        )
        return noisy_input, variational_dropout_loss_sum
