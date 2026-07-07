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

"""Data types for SID generation: enums and output tuples shared across quantizers."""

from enum import Enum
from typing import List, NamedTuple, Optional

import torch


class QuantizeForwardMode(Enum):
    """Forward mode for vector quantization (RQ-VAE backend).

    Attributes:
        GUMBEL_SOFTMAX: use Gumbel-Softmax reparameterization.
        STE: use Straight-Through Estimator.
    """

    GUMBEL_SOFTMAX = 1
    STE = 2


class QuantizeOutput(NamedTuple):
    """One quantize layer's output.

    Attributes:
        embeddings (Tensor): quantized embeddings, shape (B, D).
        ids (Tensor): codebook indices, shape (B,).
        scores (Tensor, optional): selected-code distances/scores, shape (B,).
        topk_ids (Tensor, optional): top-k nearest codebook indices, shape (B, K).
        topk_scores (Tensor, optional): top-k nearest distances/scores, shape (B, K).
    """

    embeddings: torch.Tensor
    ids: torch.Tensor
    scores: Optional[torch.Tensor] = None
    topk_ids: Optional[torch.Tensor] = None
    topk_scores: Optional[torch.Tensor] = None


class ResidualQuantizerOutput(NamedTuple):
    """Output of a residual quantization module.

    The per-layer cumulative quantized vectors are exposed as ``latents`` so the
    model-side commitment loss can consume them when needed.

    Attributes:
        cluster_ids (Tensor): codebook indices per layer, shape (B, n_layers).
        quantized_embeddings (Tensor): sum of quantized embeddings, shape (B, D).
        latents (Tensor): per-layer cumulative quantized vectors, shape
            (B, n_layers, D) (``latents[:, i]`` is the sum after layer ``i``).
        candidate_codes (Tensor, optional): candidate code tuples, shape
            (B, K, n_layers).
        candidate_scores (Tensor, optional): candidate scores, shape (B, K).
    """

    cluster_ids: torch.Tensor
    quantized_embeddings: torch.Tensor
    latents: torch.Tensor
    candidate_codes: Optional[torch.Tensor] = None
    candidate_scores: Optional[torch.Tensor] = None


class ResidualPassOutput(NamedTuple):
    """Internal result of :meth:`ResidualQuantizer._residual_pass` (shared walk).

    Distinct from :class:`ResidualQuantizerOutput`: it carries the raw running
    sums (``cumulative``) and the un-STE'd ``aggregated`` sum, which the public
    ``forward`` turns into ``latents`` / ``quantized_embeddings``.

    Attributes:
        cluster_ids (Tensor): stacked codes, shape (B, n_layers).
        aggregated (Tensor): sum of quantized vectors, shape (B, D).
        cumulative (List[Tensor]): running sum after each layer
            (``cumulative[-1]`` is ``aggregated``).
        candidate_codes (Tensor, optional): candidate code tuples, shape
            (B, K, n_layers).
        candidate_scores (Tensor, optional): candidate scores, shape (B, K).
    """

    cluster_ids: torch.Tensor
    aggregated: torch.Tensor
    cumulative: List[torch.Tensor]
    candidate_codes: Optional[torch.Tensor] = None
    candidate_scores: Optional[torch.Tensor] = None
