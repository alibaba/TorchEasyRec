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

from typing import Any, Callable, Dict, List, Optional, Union

import torch
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    BoundsCheckMode,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_inference import (
    IntNBitTableBatchedEmbeddingBagsCodegen,
)
from torch import Tensor
from torchrec.fx import symbolic_trace as _symbolic_trace


def symbolic_trace(
    # pyre-ignore[24]
    root: Union[torch.nn.Module, Callable],
    concrete_args: Optional[Dict[str, Any]] = None,
    leaf_modules: Optional[List[str]] = None,
) -> torch.fx.GraphModule:
    """Symbolic tracing API.

    Given an `nn.Module` or function instance `root`, this function will return a
    `GraphModule` constructed by recording operations seen while tracing through `root`.

    `concrete_args` allows you to partially specialize your function, whether it's to
    remove control flow or data structures.

    Args:
        root (Union[torch.nn.Module, Callable]): Module or function to be traced and
            converted into a Graph representation.
        concrete_args (Optional[Dict[str, any]]): Inputs to be partially specialized
        leaf_modules (Optional[List[str]]): modules do not trace

    Returns:
        GraphModule: a Module created from the recorded operations from ``root``.
    """
    # ComputeJTDictToKJT could not be traced
    _leaf_modules = ["ComputeJTDictToKJT"]
    if leaf_modules:
        _leaf_modules.extend(leaf_modules)
    return _symbolic_trace(root, concrete_args, _leaf_modules)


# We remove `inputs_to_device` to allow `IntNBitTableBatchedEmbeddingBagsCodegen`
# temporarily to run on both CPU and GPU after applying `symbolic_trace`. Additionally,
# we also can uncomment the following code to ensure it functions correctly, this may
# introduce unnecessary to_device operations.
# @torch.fx.wrap
# def inputs_to_device(
#     indices: torch.Tensor,
#     offsets: torch.Tensor,
#     per_sample_weights: Optional[torch.Tensor],
#     bounds_check_warning: torch.Tensor,
# ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
#     if bounds_check_warning.device.type == "meta":
#         return indices, offsets, per_sample_weights

#     non_blocking = bounds_check_warning.device.type != "cpu"
#     if indices.device != bounds_check_warning.device:
#         indices = indices.to(bounds_check_warning.device, non_blocking=non_blocking)
#     if offsets.device != bounds_check_warning.device:
#         offsets = offsets.to(bounds_check_warning.device, non_blocking=non_blocking)
#     if (
#         per_sample_weights is not None
#         and per_sample_weights.device != bounds_check_warning.device
#     ):
#         per_sample_weights = per_sample_weights.to(
#             bounds_check_warning.device, non_blocking=non_blocking
#         )
#     return indices, offsets, per_sample_weights


def _forward_impl(
    # pyre-ignore[2]
    self,
    indices: Tensor,
    offsets: Tensor,
    per_sample_weights: Optional[Tensor] = None,
) -> Tensor:
    assert self.weight_initialized, (
        "weight needs to be initialized before forward function"
    )

    # indices, offsets, per_sample_weights = inputs_to_device(
    #     indices, offsets, per_sample_weights, self.bounds_check_warning
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
