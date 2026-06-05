# Copyright (c) 2026, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data types for SID generation: enums and output tuples."""

from enum import Enum
from typing import NamedTuple

import torch


class QuantizeForwardMode(Enum):
    """Forward mode for vector quantization.

    Attributes:
        GUMBEL_SOFTMAX: use Gumbel-Softmax reparameterization.
        STE: use Straight-Through Estimator.
    """

    GUMBEL_SOFTMAX = 1
    STE = 2


class QuantizeOutput(NamedTuple):
    """Output of a single vector quantization layer.

    Attributes:
        embeddings (Tensor): quantized embeddings, shape (B, D).
        ids (Tensor): codebook indices, shape (B,).
    """

    embeddings: torch.Tensor
    ids: torch.Tensor


class ResidualQuantizerOutput(NamedTuple):
    """Output of the residual quantization module.

    Attributes:
        cluster_ids (Tensor): codebook indices per layer, shape (B, n_layers).
        quantized_embeddings (Tensor): sum of quantized embeddings, shape (B, D).
        quantization_loss (Tensor): total commitment loss scalar.
    """

    cluster_ids: torch.Tensor
    quantized_embeddings: torch.Tensor
    quantization_loss: torch.Tensor
