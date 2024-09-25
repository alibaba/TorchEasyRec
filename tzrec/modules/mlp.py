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

from typing import List, Optional, Union

import torch
from torch import nn

from tzrec.utils.load_class import load_by_path


@torch.fx.wrap
def _check_perceptron_input_dim(input: torch.Tensor, use_bn: bool) -> None:
    if input.dim() != 2 and use_bn:
        raise ValueError(f"expected 2D input when use_bn (got {input.dim()}D input)")


class Perceptron(nn.Module):
    """Applies a linear transformation and activation.

    Args:
        in_features (int): number of elements in each input sample.
        out_features (int): number of elements in each output sample.
        activation (str, optional):
            the activation function to apply to the output of linear transformation.
            Default: torch.nn.Relu.
        use_bn (bool): use batch_norm or not.
        dropout_ratio (float): dropout ratio of the layer.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: Optional[str] = "nn.ReLU",
        use_bn: bool = False,
        dropout_ratio: float = 0.0,
    ) -> None:
        super().__init__()
        self.activation = activation
        self.use_bn = use_bn
        self.dropout_ratio = dropout_ratio

        self.perceptron = nn.Sequential(
            nn.Linear(in_features, out_features, bias=False if use_bn else True)
        )
        if use_bn:
            self.perceptron.append(nn.BatchNorm1d(out_features))
        if activation and len(activation) > 0:
            act_module = load_by_path(activation)
            if act_module:
                self.perceptron.append(act_module())
            else:
                raise ValueError(f"Unknown activation method: {activation}")
        if dropout_ratio > 0.0:
            self.perceptron.append(nn.Dropout(dropout_ratio))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward the module."""
        _check_perceptron_input_dim(input, self.use_bn)
        return self.perceptron(input)


class MLP(nn.Module):
    """Applies a stack of Perceptron modules sequentially.

    Args:
        in_features (int): in_size of the input.
        hidden_units (list): out_size of each Perceptron module.
        bias (bool): if set to False, the layer will not learn an additive bias.
            Default: True.
        activation (str, optional): the activation function to apply to the output of
            linear transformation. Default: torch.nn.ReLU.
        use_bn (bool): use batch_norm or not.
        dropout_ratio (float|list, optional): dropout ratio of each layer.
    """

    def __init__(
        self,
        in_features: int,
        hidden_units: List[int],
        activation: Optional[str] = "nn.ReLU",
        use_bn: bool = False,
        dropout_ratio: Optional[Union[List[float], float]] = None,
    ) -> None:
        super().__init__()
        self.hidden_units = hidden_units
        self.activation = activation
        self.use_bn = use_bn

        if dropout_ratio is None:
            dropout_ratio = [0.0] * len(hidden_units)
        elif isinstance(dropout_ratio, list):
            if len(dropout_ratio) == 0:
                dropout_ratio = [0.0] * len(hidden_units)
            elif len(dropout_ratio) == 1:
                dropout_ratio = dropout_ratio * len(hidden_units)
            else:
                assert len(dropout_ratio) == len(hidden_units), (
                    "length of dropout_ratio and hidden_units must be same, "
                    f"but got {len(dropout_ratio)} vs {len(hidden_units)}"
                )
        else:
            dropout_ratio = [dropout_ratio] * len(hidden_units)
        self.dropout_ratio = dropout_ratio

        self.mlp = nn.Sequential(
            *[
                Perceptron(
                    in_features=in_features if i == 0 else hidden_units[i - 1],
                    out_features=hidden_units[i],
                    activation=activation,
                    use_bn=use_bn,
                    dropout_ratio=dropout_ratio[i],
                )
                for i in range(len(hidden_units))
            ]
        )

    def output_dim(self) -> int:
        """Output dimension of the module."""
        return self.hidden_units[-1]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward the module."""
        return self.mlp(input)
