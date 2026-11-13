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

from typing import Dict, List, Optional

import torch
from torchrec.distributed.planner.constants import (
    BIGINT_DTYPE,
    KV_CACHING_RATIO,
    UVM_CACHING_RATIO,
)
from torchrec.distributed.planner.enumerators import (
    ParameterConstraints,
    ShardEstimator,
)
from torchrec.distributed.planner.shard_estimators import (
    calculate_shard_storages as _torchrec_calculate_shard_storages,
)
from torchrec.distributed.planner.shard_estimators import get_num_poolings
from torchrec.distributed.planner.types import (
    SharderData,
    SharderDataMap,
    ShardingOption,
    Storage,
    Topology,
)
from torchrec.distributed.types import (
    KeyValueParams,
    PipelineType,
)
from torchrec.modules.embedding_configs import DATA_TYPE_NUM_BITS

from tzrec.utils.dynamicemb_util import DynamicEmbTableStorage


class EmbeddingStorageEstimator(ShardEstimator):
    """Embedding Storage Usage Estimator.

    Args:
        topology (Topology): device topology.
        constraints (Optional[Dict[str, ParameterConstraints]]): parameter constraints.
        pipeline_type (PipelineType): The type of pipeline, if any. Will determine the
            input replication factor during memory estimation.
        run_embedding_at_peak_memory (bool): If the embedding fwd/bwd will be execute
            when HBM usage is at peak. When set to TRUE, any temporary memory allocation
            during embedding forward/backward, as long as output sizes before
            output_dist will be counted towards HBM storage cost. Otherwise they won't
            since they'll be "hidden" by the real memory peak.

            Only take effect if pipeline_type is set for backward compatibility (not
            affecting models using old pipeline-agnostic formula)

            Default to false because this is typically false for RecSys since memory
            peak happens at the end of dense forwrad / beginning of dense backward
            instead.
        is_inference (bool): If the model is inference model. Default to False.
    """

    def __init__(
        self,
        topology: Topology,
        constraints: Optional[Dict[str, ParameterConstraints]] = None,
        pipeline_type: PipelineType = PipelineType.NONE,
        run_embedding_at_peak_memory: bool = False,
        is_inference: bool = False,
    ) -> None:
        self._topology = topology
        self._constraints = constraints
        self._pipeline_type = pipeline_type
        self._run_embedding_at_peak_memory = run_embedding_at_peak_memory
        self._is_inference = is_inference

    def estimate(
        self,
        sharding_options: List[ShardingOption],
        sharder_data_map: SharderDataMap,
    ) -> None:
        """Estimate the storage cost of each sharding option.

        Args:
            sharding_options (List[ShardingOption]): list of sharding options.
            sharder_data_map (SharderDataMap): map from module type to sharder data.
        """
        for sharding_option in sharding_options:
            sharder_key = sharding_option.module_type_key
            sharder_data = sharder_data_map[sharder_key]

            caching_ratio = sharding_option.cache_load_factor
            # TODO: remove after deprecating fused_params in sharder
            if caching_ratio is None:
                caching_ratio = (
                    sharder_data.fused_params.get("cache_load_factor")
                    if hasattr(sharder_data, "fused_params")
                    and sharder_data.fused_params
                    else None
                )
            constraints: Optional[ParameterConstraints] = (
                self._constraints.get(sharding_option.name, None)
                if self._constraints
                else None
            )
            num_poolings = get_num_poolings(self._constraints, sharding_option)
            assert len(num_poolings) == sharding_option.num_inputs
            batch_sizes = (
                constraints.batch_sizes
                if constraints and constraints.batch_sizes
                else [sharding_option.batch_size] * sharding_option.num_inputs
            )

            key_value_params: Optional[KeyValueParams] = (
                constraints.key_value_params
                if constraints and constraints.key_value_params
                else None
            )
            kv_cache_load_factor: float = (
                sharder_data.fused_params.get("cache_load_factor", KV_CACHING_RATIO)
                if sharder_data.fused_params
                else KV_CACHING_RATIO
            )
            use_virtual_table: bool = (
                constraints.use_virtual_table if constraints else False
            )

            # hardcoded as 8 bytes
            # input indices can be of int32,
            # but in TBE they get converted to int64 anyway
            input_data_type_size = BIGINT_DTYPE

            output_data_type_size: float = (
                DATA_TYPE_NUM_BITS[sharding_option.output_dtype] / 8
                if sharding_option.output_dtype
                else sharding_option.tensor.element_size()
            )

            mpp_conf = (
                sharding_option.cache_params.multipass_prefetch_config
                if sharding_option.cache_params
                else None
            )
            # TODO: remove after deprecating fused_params in sharder
            if mpp_conf is None:
                mpp_conf = (
                    sharder_data.fused_params.get("multipass_prefetch_config", None)
                    if hasattr(sharder_data, "fused_params")
                    and sharder_data.fused_params
                    else None
                )

            use_dynamicemb = (
                hasattr(sharding_option, "use_dynamicemb")
                and sharding_option.use_dynamicemb
            )
            dynamicemb_options = None
            if use_dynamicemb:
                dynamicemb_options = sharding_option.dynamicemb_options
            elif not caching_ratio:
                caching_ratio = UVM_CACHING_RATIO

            shard_storages = calculate_shard_storages(
                sharder_data=sharder_data,
                sharding_type=sharding_option.sharding_type,
                tensor=sharding_option.tensor,
                compute_device=self._topology.compute_device,
                compute_kernel=sharding_option.compute_kernel,
                shard_sizes=[shard.size for shard in sharding_option.shards],
                batch_sizes=batch_sizes,
                world_size=self._topology.world_size,
                local_world_size=self._topology.local_world_size,
                input_lengths=sharding_option.input_lengths,
                num_poolings=num_poolings,
                caching_ratio=caching_ratio,
                is_pooled=sharding_option.is_pooled,
                input_data_type_size=input_data_type_size,
                output_data_type_size=output_data_type_size,
                pipeline_type=self._pipeline_type,
                count_ephemeral_storage_cost=self._run_embedding_at_peak_memory,
                is_inference=self._is_inference,
                multipass_prefetch_max_pass=mpp_conf.num_passes if mpp_conf else None,
                key_value_params=key_value_params,
                kv_cache_load_factor=kv_cache_load_factor,
                use_virtual_table=use_virtual_table,
                dynamicemb_options=dynamicemb_options,
                dynamicemb_storage=getattr(sharding_option, "dynamicemb_storage", None),
            )
            for shard, storage in zip(sharding_option.shards, shard_storages):
                shard.storage = storage


def calculate_shard_storages(
    sharder_data: SharderData,
    sharding_type: str,
    tensor: torch.Tensor,
    compute_device: str,
    compute_kernel: str,
    shard_sizes: List[List[int]],
    batch_sizes: List[int],
    world_size: int,
    local_world_size: int,
    input_lengths: List[float],
    num_poolings: List[float],
    caching_ratio: float,
    is_pooled: bool,
    input_data_type_size: float,
    output_data_type_size: float,
    pipeline_type: PipelineType = PipelineType.NONE,
    count_ephemeral_storage_cost: bool = False,
    is_inference: bool = False,
    multipass_prefetch_max_pass: Optional[int] = None,
    key_value_params: Optional[KeyValueParams] = None,
    kv_cache_load_factor: float = KV_CACHING_RATIO,
    use_virtual_table: bool = False,
    dynamicemb_options=None,
    dynamicemb_storage: Optional[DynamicEmbTableStorage] = None,
) -> List[Storage]:
    """Calculates estimated storage sizes.

    for each sharded tensor, comprised of input, output, tensor, gradient,
    and optimizer sizes.

    Args:
        sharder_data (SharderData): precomputed sharder data for the module.
        sharding_type (str): provided ShardingType value.
        tensor (torch.Tensor): tensor to be sharded.
        compute_device (str): compute device to be used.
        compute_kernel (str): compute kernel to be used.
        shard_sizes (List[List[int]]): list of dimensions of each sharded tensor.
        batch_sizes (List[int]): batch size for each input feature.
        world_size (int): total number of devices in topology.
        local_world_size (int): total number of devices in host group topology.
        input_lengths (List[float]): average input lengths synonymous with pooling
            factors.
        num_poolings (List[float]): average number of poolings per sample
            (typically 1.0).
        caching_ratio (float): ratio of HBM to DDR memory for UVM caching.
        is_pooled (bool): True if embedding output is pooled (ie. `EmbeddingBag`),
            False if unpooled/sequential (ie. `Embedding`).
        input_data_type_size (int): number of bytes of input data type.
        output_data_type_size (int): number of bytes of output data type.
        pipeline_type: PipelineType: pipeline type if for training.
        count_ephemeral_storage_cost (bool): count ephemeral storage cost
        is_inference: bool, whether the model is for inference.
        multipass_prefetch_max_pass (int): multipass prefetch max_pass
        key_value_params (Optional[KeyValueParams]): fused params for SSD/DRAM KV
            cache.
        kv_cache_load_factor (float): ratio of kv caching.
        use_virtual_table (bool): use virtual table or not.
        dynamicemb_options (DynamicEmbTableOptions): dynamic embedding options.
        dynamicemb_storage (DynamicEmbTableStorage): per-shard capacities and
            table storage costs of the DynamicEmb candidate.

    Returns:
        List[Storage]: storage object for each device in topology.
    """
    if dynamicemb_options is not None:
        from tzrec.utils.dynamicemb_util import dynamicemb_calculate_shard_storages

        assert dynamicemb_storage is not None
        return dynamicemb_calculate_shard_storages(
            sharder_data=sharder_data,
            sharding_type=sharding_type,
            tensor=tensor,
            compute_device=compute_device,
            compute_kernel=compute_kernel,
            shard_sizes=shard_sizes,
            batch_sizes=batch_sizes,
            world_size=world_size,
            local_world_size=local_world_size,
            input_lengths=input_lengths,
            num_poolings=num_poolings,
            is_pooled=is_pooled,
            input_data_type_size=input_data_type_size,
            output_data_type_size=output_data_type_size,
            pipeline_type=pipeline_type,
            count_ephemeral_storage_cost=count_ephemeral_storage_cost,
            is_inference=is_inference,
            multipass_prefetch_max_pass=multipass_prefetch_max_pass,
            dynamicemb_options=dynamicemb_options,
            dynamicemb_storage=dynamicemb_storage,
        )
    else:
        return _torchrec_calculate_shard_storages(
            sharder_data=sharder_data,
            sharding_type=sharding_type,
            tensor=tensor,
            compute_device=compute_device,
            compute_kernel=compute_kernel,
            shard_sizes=shard_sizes,
            batch_sizes=batch_sizes,
            world_size=world_size,
            local_world_size=local_world_size,
            input_lengths=input_lengths,
            num_poolings=num_poolings,
            caching_ratio=caching_ratio,
            is_pooled=is_pooled,
            input_data_type_size=input_data_type_size,
            output_data_type_size=output_data_type_size,
            pipeline_type=pipeline_type,
            count_ephemeral_storage_cost=count_ephemeral_storage_cost,
            is_inference=is_inference,
            multipass_prefetch_max_pass=multipass_prefetch_max_pass,
            key_value_params=key_value_params,
            kv_cache_load_factor=kv_cache_load_factor,
            use_virtual_table=use_virtual_table,
        )
