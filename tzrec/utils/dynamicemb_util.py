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

import math
from typing import List, Optional, Tuple, Type, cast

import torch
from torch import nn
from torchrec.distributed.embedding_types import (
    EmbeddingComputeKernel,
    ShardedEmbeddingTable,
)
from torchrec.distributed.planner import (
    constants,
    enumerators,
    planners,
    shard_estimators,
)
from torchrec.distributed.planner.types import (
    ParameterConstraints,
    ShardingOption,
    Storage,
    Topology,
)
from torchrec.distributed.sharding_plan import placement
from torchrec.distributed.types import (
    CacheParams,
    EmbeddingModuleShardingPlan,
    EnumerableShardingSpec,
    KeyValueParams,
    ModuleSharder,
    ParameterSharding,
    PipelineType,
    ShardingPlan,
    ShardingType,
    ShardMetadata,
)
from torchrec.modules.embedding_configs import BaseEmbeddingConfig, DataType
from torchrec.optim.apply_optimizer_in_backward import apply_optimizer_in_backward
from torchrec.optim.optimizers import SGD

from tzrec.optim import optimizer_builder
from tzrec.protos import feature_pb2
from tzrec.protos.train_pb2 import TrainConfig

has_dynamicemb = False
try:
    import dynamicemb
    from dynamicemb import (
        DynamicEmbEvictStrategy,
        DynamicEmbInitializerArgs,
        DynamicEmbInitializerMode,
        DynamicEmbScoreStrategy,
        batched_dynamicemb_compute_kernel,
    )
    from dynamicemb.dynamicemb_config import DynamicEmbKernel
    from dynamicemb.planner import (
        DynamicEmbParameterConstraints,
        DynamicEmbParameterSharding,
    )
    from dynamicemb.shard import (
        DynamicEmbeddingBagCollectionSharder,
        DynamicEmbeddingCollectionSharder,
    )

    has_dynamicemb = True
except Exception:
    pass


def _next_power_of_2(n: int) -> int:
    # Handle the case where n is 0
    if n == 0:
        return 1

    # If n is already a power of 2, return n
    if (n & (n - 1)) == 0:
        return n

    # Find the next power of 2
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32  # This line is necessary for 64-bit integers
    return n + 1


def _build_dynamicemb_initializer(
    init_cfg: Optional[feature_pb2.DynamicEmbInitializerArgs],
    num_embeddings: int,
    embedding_dim: int,
    is_eval: bool = False,
) -> "DynamicEmbInitializerArgs":
    """Build initializer for Dynamic Embedding."""
    if init_cfg is None:
        if is_eval:
            return DynamicEmbInitializerArgs(mode=DynamicEmbInitializerMode.CONSTANT)
        else:
            # compatible with torchrec init
            return DynamicEmbInitializerArgs(
                mode=DynamicEmbInitializerMode.UNIFORM,
                lower=-1 / math.sqrt(num_embeddings),
                upper=1 / math.sqrt(num_embeddings),
            )
    init_kwargs = {}
    if init_cfg.HasField("mode"):
        mode = init_cfg.mode
    elif is_eval:
        mode = "CONSTANT"
    else:
        mode = "UNIFORM"
    if mode in DynamicEmbInitializerMode.__members__.keys():
        init_kwargs["mode"] = DynamicEmbInitializerMode[mode]
    else:
        raise ValueError(
            "DynamicEmb init mode support: "
            f"{DynamicEmbInitializerMode.__members__.keys}, but got {mode}."
        )

    if mode in ["NORMAL", "TRUNCATED_NORMAL"]:
        init_kwargs["mean"] = init_cfg.mean
        init_kwargs["std_dev"] = (
            init_cfg.std_dev
            if init_cfg.HasField("std_dev")
            else 1 / math.sqrt(embedding_dim)
        )
    elif mode == "UNIFORM":
        # compatible with torchrec init
        init_kwargs["lower"] = (
            init_cfg.lower
            if init_cfg.HasField("lower")
            else -1 / math.sqrt(num_embeddings)
        )
        init_kwargs["upper"] = (
            init_cfg.upper
            if init_cfg.HasField("upper")
            else 1 / math.sqrt(num_embeddings)
        )
    elif mode == "CONSTANT":
        init_kwargs["value"] = init_cfg.value
    return DynamicEmbInitializerArgs(**init_kwargs)


def build_dynamicemb_constraints(
    dynamicemb_cfg: feature_pb2.DynamicEmbedding, emb_config: BaseEmbeddingConfig
) -> ParameterConstraints:
    """Build ParameterConstraints for DynamicEmbedding."""
    embedding_dim = emb_config.embedding_dim
    num_embeddings = emb_config.num_embeddings

    evict_strategy = None
    if dynamicemb_cfg.evict_strategy in DynamicEmbEvictStrategy.__members__.keys():
        evict_strategy = DynamicEmbEvictStrategy[dynamicemb_cfg.evict_strategy]
    else:
        raise ValueError(
            "DynamicEmbEvictStrategy support: "
            f"{DynamicEmbEvictStrategy.__members__.keys()}, "
            "but got {dynamicemb_cfg.evict_strategy}."
        )
    score_strategy = None
    if dynamicemb_cfg.score_strategy in DynamicEmbScoreStrategy.__members__.keys():
        score_strategy = DynamicEmbScoreStrategy[dynamicemb_cfg.score_strategy]
    else:
        raise ValueError(
            f"DynamicEmbScoreStrategy support: "
            f"{DynamicEmbScoreStrategy.__members__.keys()}, "
            f"but got {dynamicemb_cfg.score_strategy}."
        )

    init_capacity = None
    if dynamicemb_cfg.HasField("init_capacity_per_rank"):
        init_capacity = _next_power_of_2(dynamicemb_cfg.init_capacity_per_rank)

    dynamicemb_options = dynamicemb.DynamicEmbTableOptions(
        max_capacity=dynamicemb_cfg.max_capacity,
        init_capacity=init_capacity,
        # TODO: convert eb_config.init_fn to dynamicemb initializer
        initializer_args=_build_dynamicemb_initializer(
            dynamicemb_cfg.initializer_args, num_embeddings, embedding_dim
        ),
        eval_initializer_args=_build_dynamicemb_initializer(
            dynamicemb_cfg.eval_initializer_args,
            num_embeddings,
            embedding_dim,
            is_eval=True,
        ),
        evict_strategy=evict_strategy,
        score_strategy=score_strategy,
    )

    constraints_kwargs = {}
    if dynamicemb_cfg.HasField("cache_load_factor"):
        # for ShardingPlan estimator hdm storage
        constraints_kwargs["cache_params"] = (
            CacheParams(load_factor=dynamicemb_cfg.cache_load_factor),
        )

    constraints = DynamicEmbParameterConstraints(
        use_dynamicemb=True,
        sharding_types=[ShardingType.ROW_WISE.value],
        compute_kernels=[
            EmbeddingComputeKernel.CUSTOMIZED_KERNEL.value
        ],  # workaround for ShardingPlan estimator
        dynamicemb_options=dynamicemb_options,
        **constraints_kwargs,
    )
    return constraints


def _patch_dynamicemb_eval_model(model: nn.Module, train_config: TrainConfig) -> None:
    """Patch model with optimizer when eval.

    because DynamicEmbedding Eval need optimizer now.
    """
    if has_dynamicemb:
        with_dynamicemb_feature = False
        for feature in model.model._features:
            if hasattr(feature.config, "dynamicemb") and feature.config.HasField(
                "dynamicemb"
            ):
                with_dynamicemb_feature = True
                break
        if with_dynamicemb_feature:
            sparse_optim_cls, _ = optimizer_builder.create_sparse_optimizer(
                train_config.sparse_optimizer
            )
            trainable_params, frozen_params = model.model.sparse_parameters()
            apply_optimizer_in_backward(sparse_optim_cls, trainable_params, {"lr": 0.0})
            if len(frozen_params) > 0:
                apply_optimizer_in_backward(SGD, frozen_params, {"lr": 0.0})


if has_dynamicemb:
    enumerators.GUARDED_COMPUTE_KERNELS.add(EmbeddingComputeKernel.CUSTOMIZED_KERNEL)

    def _ebc_compute_kernels(
        self,  # pyre-ignore [2]
        sharding_type: str,
        compute_device_type: str,
    ) -> List[str]:
        # pyre-ignore [16]
        compute_kernels = super(
            DynamicEmbeddingBagCollectionSharder, self
        ).compute_kernels(sharding_type, compute_device_type)
        if compute_device_type == "cuda":
            compute_kernels += [EmbeddingComputeKernel.CUSTOMIZED_KERNEL.value]
        return compute_kernels

    def _ec_compute_kernels(
        self,  # pyre-ignore [2]
        sharding_type: str,
        compute_device_type: str,
    ) -> List[str]:
        # pyre-ignore [16]
        compute_kernels = super(
            DynamicEmbeddingCollectionSharder, self
        ).compute_kernels(sharding_type, compute_device_type)
        if compute_device_type == "cuda":
            compute_kernels += [EmbeddingComputeKernel.CUSTOMIZED_KERNEL.value]
        return compute_kernels

    DynamicEmbeddingBagCollectionSharder.compute_kernels = _ebc_compute_kernels
    DynamicEmbeddingCollectionSharder.compute_kernels = _ec_compute_kernels

    def _round_up(a: int, b: int) -> int:
        return int((a + b - 1) // b) * b

    def _calculate_dynamicemb_table_storage_specific_size(
        size: List[int],
        element_size: int,
        optimizer_multipler: float = 0.0,
        cache_ratio: Optional[float] = None,
        is_hbm: bool = True,
        only_values: bool = False,
    ) -> int:
        """Calculate dynamic embedding table storage.

        total_value_memory = max_capacity x aligned16(embedding+optimizer states)
        num_buckets = max_capacity/bucket_capacity
        hbm_budget = min(global_hbm_for_values//world_size, total_value_memory) +
            max_capacity x (key<8byte> + score<8byte> + digest<1byte>) +
            num_buckets x (bucket_size<4byte> + 4 x pointer<8byte>)
        ddr_budget = max(total_value_memory - global_hbm_for_values//world_size, 0)
        """
        if cache_ratio is None:
            cache_ratio = 1.0
        # TODO: get bucket_capacity from DynamicEmbTableOptions
        bucket_capacity = 128
        return math.ceil(
            _next_power_of_2(size[0])
            * (
                _round_up(
                    math.ceil(size[1] * (1 + optimizer_multipler) * element_size),
                    16,
                )
                * (cache_ratio if is_hbm else 1 - cache_ratio)
                + (8 + 8 + 1 + (4 + 4) / bucket_capacity) * (is_hbm and not only_values)
            )
        )

    def _to_sharding_plan(
        sharding_options: List[ShardingOption],
        topology: Topology,
    ) -> ShardingPlan:
        compute_device = topology.compute_device
        local_size = topology.local_world_size
        world_size = topology.world_size

        plan = {}
        for sharding_option in sharding_options:
            shards = sharding_option.shards
            sharding_type = sharding_option.sharding_type

            module_plan = plan.get(sharding_option.path, EmbeddingModuleShardingPlan())

            sharding_spec = (
                None
                if sharding_type == ShardingType.DATA_PARALLEL.value
                else EnumerableShardingSpec(
                    [
                        ShardMetadata(
                            shard_sizes=shard.size,
                            shard_offsets=shard.offset,
                            placement=placement(
                                compute_device, cast(int, shard.rank), local_size
                            ),
                        )
                        for shard in shards
                    ]
                )
            )

            if (
                hasattr(sharding_option, "use_dynamicemb")
                # pyre-ignore [16]
                and sharding_option.use_dynamicemb
            ):
                # only support row-wise now
                # pyre-ignore [16]
                dynamicemb_options = sharding_option.dynamicemb_options

                # calc local_hbm_for_values
                tensor = sharding_option.tensor
                optimizer_class = getattr(tensor, "_optimizer_classes", [None])[0]
                optimizer_multipler = shard_estimators._get_optimizer_multipler(
                    optimizer_class, tensor.shape
                )
                dynamicemb_options.training = optimizer_class is not None
                dynamicemb_options.local_hbm_for_values = (
                    _calculate_dynamicemb_table_storage_specific_size(
                        shards[0].size,
                        tensor.element_size(),
                        optimizer_multipler,
                        sharding_option.cache_load_factor,
                        is_hbm=True,
                        only_values=True,
                    )
                )

                # align to next_power_of_2
                num_aligned_embedding_per_rank = _next_power_of_2(shards[0].size[0])
                num_embeddings_per_shard = shards[0].size[0]
                if num_aligned_embedding_per_rank < dynamicemb_options.bucket_capacity:
                    num_aligned_embedding_per_rank = dynamicemb_options.bucket_capacity
                if num_embeddings_per_shard != num_aligned_embedding_per_rank:
                    dynamicemb_options.num_aligned_embedding_per_rank = (
                        num_aligned_embedding_per_rank
                    )

                module_plan[sharding_option.name] = DynamicEmbParameterSharding(
                    sharding_spec=sharding_spec,
                    sharding_type=ShardingType.ROW_WISE.value,
                    ranks=[i for i in range(world_size)],
                    compute_kernel=EmbeddingComputeKernel.CUSTOMIZED_KERNEL.value,
                    customized_compute_kernel=DynamicEmbKernel,
                    dist_type="roundrobin",
                    dynamicemb_options=dynamicemb_options,
                )
            else:
                module_plan[sharding_option.name] = ParameterSharding(
                    sharding_spec=sharding_spec,
                    sharding_type=sharding_type,
                    compute_kernel=sharding_option.compute_kernel,
                    ranks=[cast(int, shard.rank) for shard in shards],
                    cache_params=sharding_option.cache_params,
                    enforce_hbm=sharding_option.enforce_hbm,
                    stochastic_rounding=sharding_option.stochastic_rounding,
                    bounds_check_mode=sharding_option.bounds_check_mode,
                    output_dtype=sharding_option.output_dtype,
                    key_value_params=sharding_option.key_value_params,
                )
            plan[sharding_option.path] = module_plan
        return ShardingPlan(plan)

    # pyre-ignore [9]
    planners.to_sharding_plan = _to_sharding_plan

    def _kernel_bw_lookup(
        compute_device: str,
        compute_kernel: str,
        hbm_mem_bw: float,
        ddr_mem_bw: float,
        hbm_to_ddr_mem_bw: float,
        caching_ratio: Optional[float] = None,
        prefetch_pipeline: bool = False,
    ) -> Optional[float]:
        """Calculates the device bandwidth.

        Args:
            compute_kernel (str): compute kernel.
            compute_device (str): compute device.
            hbm_mem_bw (float): the bandwidth of the device HBM.
            ddr_mem_bw (float): the bandwidth of the system DDR memory.
            hbm_to_ddr_mem_bw (float): the bandwidth between device HBM and system DDR.
            caching_ratio (Optional[float]): caching ratio used to determine device
                bandwidth if UVM caching is enabled.
            prefetch_pipeline (bool): whether prefetch pipeline is enabled.

        Returns:
            Optional[float]: the device bandwidth.
        """
        if compute_kernel == EmbeddingComputeKernel.CUSTOMIZED_KERNEL.value:
            # for dynamic embedding table
            caching_ratio = caching_ratio if caching_ratio else 0.0
            return (
                caching_ratio * hbm_mem_bw + (1 - caching_ratio) * hbm_to_ddr_mem_bw
            ) / 10
        else:
            return constants.kernel_bw_lookup(
                compute_device=compute_device,
                compute_kernel=compute_kernel,
                hbm_mem_bw=hbm_mem_bw,
                ddr_mem_bw=ddr_mem_bw,
                hbm_to_ddr_mem_bw=hbm_to_ddr_mem_bw,
                caching_ratio=caching_ratio,
                prefetch_pipeline=prefetch_pipeline,
            )

    # pyre-ignore [9]
    shard_estimators.kernel_bw_lookup = _kernel_bw_lookup

    def _calculate_dynamicemb_storage_specific_sizes(
        tensor: torch.Tensor,
        shard_sizes: List[List[int]],
        optimizer_class: Optional[Type[torch.optim.Optimizer]] = None,
        cache_ratio: float = 1.0,
        is_inference: bool = False,
    ) -> Tuple[List[int], List[int]]:
        """Calculate storage for dynamicemb."""
        optimizer_multipler = 0.0
        optimizer_class = getattr(tensor, "_optimizer_classes", [None])[0]
        if not is_inference:
            optimizer_multipler = shard_estimators._get_optimizer_multipler(
                optimizer_class, tensor.shape
            )

        hdm_value_sizes = [
            _calculate_dynamicemb_table_storage_specific_size(
                size,
                tensor.element_size(),
                optimizer_multipler,
                cache_ratio,
                is_hbm=True,
            )
            for size in shard_sizes
        ]

        ddr_value_sizes = [
            _calculate_dynamicemb_table_storage_specific_size(
                size,
                tensor.element_size(),
                optimizer_multipler,
                cache_ratio,
                is_hbm=False,
            )
            for size in shard_sizes
        ]
        return hdm_value_sizes, ddr_value_sizes

    _tzrec_calculate_shard_storages = shard_estimators.calculate_shard_storages

    def _calculate_shard_storages(
        sharder: ModuleSharder[nn.Module],
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
        kv_cache_load_factor: float = constants.KV_CACHING_RATIO,
    ) -> List[Storage]:
        """Calculates estimated storage sizes.

        for each sharded tensor, comprised of input, output, tensor, gradient,
        and optimizer sizes.

        Args:
            sharder (ModuleSharder[nn.Module]): sharder for module that supports
                sharding.
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

        Returns:
            List[Storage]: storage object for each device in topology.
        """
        if compute_kernel == EmbeddingComputeKernel.CUSTOMIZED_KERNEL.value:
            # storage estimator for dynamicemb
            assert compute_device == "cuda"

            input_sizes, output_sizes = shard_estimators._calculate_shard_io_sizes(
                sharding_type=sharding_type,
                batch_sizes=batch_sizes,
                world_size=world_size,
                local_world_size=local_world_size,
                input_lengths=input_lengths,
                emb_dim=tensor.shape[1],
                shard_sizes=shard_sizes,
                input_data_type_size=input_data_type_size,
                output_data_type_size=output_data_type_size,
                num_poolings=num_poolings,
                is_pooled=is_pooled,
            )

            hbm_specific_sizes, ddr_specific_sizes = (
                _calculate_dynamicemb_storage_specific_sizes(
                    tensor=tensor,
                    shard_sizes=shard_sizes,
                    cache_ratio=caching_ratio if caching_ratio else 1.0,
                    is_inference=is_inference,
                )
            )

            hbm_sizes: List[int] = [
                (
                    hbm_specific_size
                    + shard_estimators.calculate_pipeline_io_cost(
                        input_size=input_size,
                        output_size=output_size,
                        prefetch_size=0,
                        pipeline_type=pipeline_type,
                        multipass_prefetch_max_pass=multipass_prefetch_max_pass,
                        count_ephemeral_storage_cost=count_ephemeral_storage_cost,
                        is_inference=is_inference,
                    )
                )
                for input_size, output_size, hbm_specific_size in zip(
                    input_sizes,
                    output_sizes,
                    hbm_specific_sizes,
                )
            ]
            ddr_sizes: List[int] = [
                (
                    input_size + output_size + ddr_specific_size
                    if compute_device == "cpu" and not is_inference
                    else ddr_specific_size
                )
                for input_size, output_size, ddr_specific_size in zip(
                    input_sizes,
                    output_sizes,
                    ddr_specific_sizes,
                )
            ]

            return [
                Storage(
                    hbm=hbm_size,
                    ddr=ddr_size,
                )
                for hbm_size, ddr_size in zip(hbm_sizes, ddr_sizes)
            ]

        else:
            return _tzrec_calculate_shard_storages(
                sharder=sharder,
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
            )

    # pyre-ignore [9]
    shard_estimators.calculate_shard_storages = _calculate_shard_storages

    _dynamicemb_get_dynamicemb_options_per_table = (
        batched_dynamicemb_compute_kernel._get_dynamicemb_options_per_table
    )

    def _get_dynamicemb_options_per_table(
        local_row: int,
        local_col: int,
        data_type: DataType,
        optimizer: dynamicemb.EmbOptimType,
        table: ShardedEmbeddingTable,
    ) -> dynamicemb.DynamicEmbTableOptions:
        # pyre-ignore [16]
        dynamicemb_options = table.fused_params["dynamicemb_options"]
        bak_local_hbm_for_values = None
        if dynamicemb_options.num_aligned_embedding_per_rank is not None:
            bak_local_hbm_for_values = dynamicemb_options.local_hbm_for_values

        dynamicemb_options = _dynamicemb_get_dynamicemb_options_per_table(
            local_row=local_row,
            local_col=local_col,
            data_type=data_type,
            optimizer=optimizer,
            table=table,
        )

        # do not improve the HBM budget, already aligned in planner.
        if bak_local_hbm_for_values is not None:
            dynamicemb_options.local_hbm_for_values = bak_local_hbm_for_values

        return dynamicemb_options

    # pyre-ignore [9]
    batched_dynamicemb_compute_kernel._get_dynamicemb_options_per_table = (
        _get_dynamicemb_options_per_table
    )
