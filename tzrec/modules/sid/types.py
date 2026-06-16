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

"""Data types for SID generation: output tuples shared across quantizers."""

from typing import NamedTuple

import torch


class QuantizeOutput(NamedTuple):
    """One quantize layer's output.

    Attributes:
        embeddings (Tensor): quantized embeddings, shape (B, D).
        ids (Tensor): codebook indices, shape (B,).
    """

    embeddings: torch.Tensor
    ids: torch.Tensor
