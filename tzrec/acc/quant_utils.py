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

import copy
from typing import Any, Dict, List, Optional, Tuple, Type, cast

import torch
import torch.nn as nn
import torch.quantization as quant
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    BoundsCheckMode,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_inference import (
    IntNBitTableBatchedEmbeddingBagsCodegen,
)
from torch import Tensor
from torchrec.modules.embedding_configs import (
    BaseEmbeddingConfig,
    EmbeddingBagConfig,
    QuantConfig,
)
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
)
from torchrec.modules.mc_embedding_modules import (
    ManagedCollisionEmbeddingBagCollection,
)
from torchrec.modules.mc_modules import ManagedCollisionCollection
from torchrec.quant.embedding_modules import (
    DEFAULT_ROW_ALIGNMENT,
    MODULE_ATTR_QUANT_STATE_DICT_SPLIT_SCALE_BIAS,
    MODULE_ATTR_REGISTER_TBES_BOOL,
    MODULE_ATTR_ROW_ALIGNMENT_INT,
    _get_device,
    _update_embedding_configs,
)
from torchrec.quant.embedding_modules import (
    EmbeddingBagCollection as QuantEmbeddingBagCollection,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, KeyedTensor


# we remove inputs_to_device temporally to make IntNBitTableBatchedEmbeddingBagsCodegen
# support torch.jit.script
def _forward_impl(
    self,
    indices: Tensor,
    offsets: Tensor,
    per_sample_weights: Optional[Tensor] = None,
) -> Tensor:
    assert self.weight_initialized, (
        "weight needs to be initialized before forward function"
    )

    # indices, offsets, per_sample_weights = inputs_to_device(
    #     indices, offsets, per_sample_weights, self.bounds_check_warning.device
    # )

    # First bound check: check if the indices/offsets are within the boundary
    # of the original embedding rows before pruning.
    # Note that this is only applied when we enable pruning (if the perf becomes
    # an issue, we can fuse it inside the remapping kernel).
    if (
        self.index_remapping_hash_table_cpu is not None
        or self.index_remapping_hash_table.numel() > 0
        or self.index_remappings_array.numel() > 0
    ):
        if self.bounds_check_mode_int != BoundsCheckMode.NONE.value:
            torch.ops.fbgemm.bounds_check_indices(
                self.original_rows_per_table,
                indices,
                offsets,
                self.bounds_check_mode_int,
                self.bounds_check_warning,
                per_sample_weights,
            )

    # Index remapping changes input indices, and some of them becomes -1 (prunned rows).
    # Hence, remapping should be done before prefetch and emb lookup
    # so that these operations are with the remapped indices.
    if self.index_remapping_hash_table_cpu is not None:
        indices = self.index_remapping_hash_table_cpu.lookup(indices, offsets)
    elif self.index_remapping_hash_table.numel() > 0:
        # Convert from raw indices to pruned indices
        indices = torch.ops.fbgemm.pruned_hashmap_lookup(
            indices,
            offsets,
            self.index_remapping_hash_table,
            self.index_remapping_hash_table_offsets,
        )
    elif self.index_remappings_array.numel() > 0:
        indices = torch.ops.fbgemm.pruned_array_lookup(
            indices,
            offsets,
            self.index_remappings_array,
            self.index_remappings_array_offsets,
        )
    # pyre-fixme[29]: `Union[(self: TensorBase) -> int, Module, Tensor]` is not
    #  a function.
    if self.lxu_cache_weights.numel() > 0:
        if self.timestep_prefetch_size.get() <= 0:
            self.prefetch(indices, offsets)
        self.timestep_prefetch_size.decrement()

    lxu_cache_locations = self.lxu_cache_locations_list.pop()

    # Second bound check: check if the indices/offsets are within the boundary
    # of the pruned embedding rows after pruning.
    # Note: we cast to int as a TorchScript workaround.
    if self.bounds_check_mode_int != BoundsCheckMode.NONE.value:
        torch.ops.fbgemm.bounds_check_indices(
            self.rows_per_table,
            indices,
            offsets,
            self.bounds_check_mode_int,
            self.bounds_check_warning,
            per_sample_weights,
        )
    # Note: CPU and CUDA ops use the same interface to facilitate JIT IR
    # generation for CUDA/CPU. For CPU op, we don't need weights_uvm and
    # weights_placements
    return torch.ops.fbgemm.int_nbit_split_embedding_codegen_lookup_function(
        dev_weights=self.weights_host if self.host_size > 0 else self.weights_dev,
        uvm_weights=self.weights_uvm,
        weights_placements=self.weights_placements,
        weights_offsets=self.weights_offsets,
        weights_tys=self.weights_tys,
        D_offsets=self.D_offsets,
        total_D=self.total_D,
        max_int2_D=self.max_int2_D,
        max_int4_D=self.max_int4_D,
        max_int8_D=self.max_int8_D,
        max_float16_D=self.max_float16_D,
        max_float32_D=self.max_float32_D,
        indices=indices,
        offsets=offsets,
        pooling_mode=int(self.pooling_mode),
        indice_weights=per_sample_weights,
        output_dtype=self.output_dtype,
        lxu_cache_weights=self.lxu_cache_weights,
        lxu_cache_locations=lxu_cache_locations,
        row_alignment=self.row_alignment,
        max_float8_D=self.max_float8_D,
        fp8_exponent_bits=self.fp8_exponent_bits,
        fp8_exponent_bias=self.fp8_exponent_bias,
    )


IntNBitTableBatchedEmbeddingBagsCodegen._forward_impl = _forward_impl


class QuantManagedCollisionEmbeddingBagCollection(QuantEmbeddingBagCollection):
    """QuantManagedCollisionEmbeddingBagCollection represents a quantized EBC module.

    The inputs into the MC-EC/EBC will first be modified by the managed collision module
    before being passed into the embedding collection.

    Args:
        tables (List[EmbeddingBagConfig]): A list of EmbeddingBagConfig
            objects representing the embedding tables in the collection.
        is_weighted (bool): whether input `KeyedJaggedTensor` is weighted.
        device (torch.device): The device on which the embedding bag collection will
            be allocated.
        output_dtype (torch.dtype, optional): The data type of the output embeddings.
            Defaults to torch.float.
        table_name_to_quantized_weights (Dict[str, Tuple[Tensor, Tensor]], optional):
            A dictionary mapping table names to their corresponding quantized weights.
            Defaults to None.
        register_tbes (bool, optional): Whether to register the TBEs in the model.
            Defaults to False.
        quant_state_dict_split_scale_bias (bool, optional): Whether to split the scale
            and bias parameters when saving the quantized state dict. Defaults to False.
        row_alignment (int, optional): The alignment of rows in the quantized weights.
            Defaults to DEFAULT_ROW_ALIGNMENT.
        managed_collision_collection (ManagedCollisionCollection, optional): The managed
            collision collection to use for managing collisions. Defaults to None.
        return_remapped_features (bool, optional): Whether to return the remapped input
            features in addition to the embeddings. Defaults to False.
    """

    def __init__(
        self,
        tables: List[EmbeddingBagConfig],
        is_weighted: bool,
        device: torch.device,
        output_dtype: torch.dtype = torch.float,
        table_name_to_quantized_weights: Optional[
            Dict[str, Tuple[Tensor, Tensor]]
        ] = None,
        register_tbes: bool = False,
        quant_state_dict_split_scale_bias: bool = False,
        row_alignment: int = DEFAULT_ROW_ALIGNMENT,
        managed_collision_collection: Optional[ManagedCollisionCollection] = None,
        return_remapped_features: bool = False,
    ) -> None:
        super().__init__(
            tables,
            is_weighted,
            device,
            output_dtype,
            table_name_to_quantized_weights,
            register_tbes,
            quant_state_dict_split_scale_bias,
            row_alignment,
        )
        assert managed_collision_collection, (
            "Managed collision collection cannot be None"
        )
        self._managed_collision_collection: ManagedCollisionCollection = (
            managed_collision_collection
        )
        self._return_remapped_features = return_remapped_features

        assert str(self.embedding_bag_configs()) == str(
            self._managed_collision_collection.embedding_configs()
        ), (
            "EmbeddingBagCollection and Managed Collision Collection must contain the "
            "same Embedding Configs"
        )

        # Assuming quantized MCEC is used in inference only
        for (
            managed_collision_module
        ) in self._managed_collision_collection._managed_collision_modules.values():
            managed_collision_module.reset_inference_mode()

    def to(
        self, *args: List[Any], **kwargs: Dict[str, Any]
    ) -> "QuantManagedCollisionEmbeddingBagCollection":
        """To device and dtype."""
        device, dtype, non_blocking, _ = torch._C._nn._parse_to(
            *args,  # pyre-ignore
            **kwargs,  # pyre-ignore
        )
        for param in self.parameters():
            if param.device.type != "meta":
                param.to(device)

        for buffer in self.buffers():
            if buffer.device.type != "meta":
                buffer.to(device)
        # Skip device movement and continue with other args
        super().to(
            dtype=dtype,
            non_blocking=non_blocking,
        )
        return self

    def forward(
        self,
        features: KeyedJaggedTensor,
    ) -> Tuple[KeyedTensor, Optional[KeyedJaggedTensor]]:
        """Forward the module."""
        features = self._managed_collision_collection(features)
        embedding_res = super().forward(features)

        if not self._return_remapped_features:
            return embedding_res, None
        return embedding_res, features

    def _get_name(self) -> str:
        return "QuantManagedCollisionEmbeddingBagCollection"

    @classmethod
    # pyre-ignore
    def from_float(
        cls,
        module: ManagedCollisionEmbeddingBagCollection,
        return_remapped_features: bool = False,
    ) -> "QuantManagedCollisionEmbeddingBagCollection":
        """Convert MC-EBC to Quant MC-EBC."""
        mc_ebc = module
        ebc = module._embedding_module

        # pyre-ignore[9]
        qconfig: torch.quantization.QConfig = module.qconfig
        assert hasattr(module, "qconfig"), (
            "QuantManagedCollisionEmbeddingBagCollection input float module must "
            "have qconfig defined"
        )

        # pyre-ignore[29]
        embedding_configs = copy.deepcopy(ebc.embedding_bag_configs())
        _update_embedding_configs(
            cast(List[BaseEmbeddingConfig], embedding_configs),
            qconfig,
        )
        _update_embedding_configs(
            mc_ebc._managed_collision_collection._embedding_configs,
            qconfig,
        )

        # pyre-ignore[9]
        table_name_to_quantized_weights: Dict[str, Tuple[Tensor, Tensor]] | None = (
            ebc._table_name_to_quantized_weights
            if hasattr(ebc, "_table_name_to_quantized_weights")
            else None
        )
        device = _get_device(ebc)
        return cls(
            embedding_configs,
            ebc.is_weighted(),
            device=device,
            output_dtype=qconfig.activation().dtype,
            table_name_to_quantized_weights=table_name_to_quantized_weights,
            register_tbes=getattr(module, MODULE_ATTR_REGISTER_TBES_BOOL, False),
            quant_state_dict_split_scale_bias=getattr(
                ebc, MODULE_ATTR_QUANT_STATE_DICT_SPLIT_SCALE_BIAS, False
            ),
            row_alignment=getattr(
                ebc, MODULE_ATTR_ROW_ALIGNMENT_INT, DEFAULT_ROW_ALIGNMENT
            ),
            managed_collision_collection=mc_ebc._managed_collision_collection,
            return_remapped_features=mc_ebc._return_remapped_features,
        )


def quantize_embeddings(
    module: nn.Module,
    dtype: torch.dtype,
    inplace: bool,
    additional_qconfig_spec_keys: Optional[List[Type[nn.Module]]] = None,
    additional_mapping: Optional[Dict[Type[nn.Module], Type[nn.Module]]] = None,
    output_dtype: torch.dtype = torch.float,
    per_table_weight_dtype: Optional[Dict[str, torch.dtype]] = None,
) -> nn.Module:
    """Converts a float embedding to quantized embedding."""
    qconfig = QuantConfig(
        activation=quant.PlaceholderObserver.with_args(dtype=output_dtype),
        weight=quant.PlaceholderObserver.with_args(dtype=dtype),
        per_table_weight_dtype=per_table_weight_dtype,
    )
    # TODO: add EmbeddingCollection and ManagedCollisionEmbeddingCollection into qconfig
    qconfig_spec: Dict[Type[nn.Module], QuantConfig] = {
        EmbeddingBagCollection: qconfig,
        ManagedCollisionEmbeddingBagCollection: qconfig,
    }
    mapping: Dict[Type[nn.Module], Type[nn.Module]] = {
        EmbeddingBagCollection: QuantEmbeddingBagCollection,
        ManagedCollisionEmbeddingBagCollection: QuantManagedCollisionEmbeddingBagCollection,  # NOQA
    }
    if additional_qconfig_spec_keys is not None:
        for t in additional_qconfig_spec_keys:
            qconfig_spec[t] = qconfig
    if additional_mapping is not None:
        mapping.update(additional_mapping)
    return quant.quantize_dynamic(
        module,
        qconfig_spec=qconfig_spec,
        mapping=mapping,
        inplace=inplace,
    )
