# Copyright (c) 2025, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, Optional, Type

import torch
import torch.nn as nn
import torch.quantization as quant
from torchrec.modules.embedding_configs import QuantConfig
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
    EmbeddingCollection,
)
from torchrec.quant.embedding_modules import (
    EmbeddingBagCollection as QuantEmbeddingBagCollection,
)
from torchrec.quant.embedding_modules import (
    EmbeddingCollection as QuantEmbeddingCollection,
)

from tzrec.acc import utils as acc_utils


def quantize_embeddings(
    module: nn.Module,
    inplace: bool,
    output_dtype: torch.dtype = torch.float,
    per_table_weight_dtype: Optional[Dict[str, torch.dtype]] = None,
) -> nn.Module:
    """Quant Embedding Tables."""
    qconfig_spec: Dict[Type[nn.Module], QuantConfig] = {}
    mapping: Dict[Type[nn.Module], Type[nn.Module]] = {}
    if acc_utils.is_quant():
        qconfig = QuantConfig(
            activation=quant.PlaceholderObserver.with_args(dtype=output_dtype),
            weight=quant.PlaceholderObserver.with_args(dtype=acc_utils.quant_dtype()),
            per_table_weight_dtype=per_table_weight_dtype,
        )
        qconfig_spec[EmbeddingBagCollection] = qconfig
        mapping[EmbeddingBagCollection] = QuantEmbeddingBagCollection
    if acc_utils.is_ec_quant():
        qconfig = QuantConfig(
            activation=quant.PlaceholderObserver.with_args(dtype=output_dtype),
            weight=quant.PlaceholderObserver.with_args(
                dtype=acc_utils.ec_quant_dtype()
            ),
            per_table_weight_dtype=per_table_weight_dtype,
        )
        qconfig_spec[EmbeddingCollection] = qconfig
        mapping[EmbeddingCollection] = QuantEmbeddingCollection
    return quant.quantize_dynamic(
        module,
        qconfig_spec=qconfig_spec,
        mapping=mapping,
        inplace=inplace,
    )
