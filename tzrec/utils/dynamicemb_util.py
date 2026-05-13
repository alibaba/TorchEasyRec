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

import dataclasses
import math
import os
from typing import Any, List, Optional, Tuple, Type, cast

import torch
from torch import nn
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.planner import (
    enumerators,
    planners,
    shard_estimators,
)
from torchrec.distributed.planner.estimator.types import (
    HardwarePerfConfig,
    ShardPerfContext,
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
    ModuleSharder,
    ParameterSharding,
    PipelineType,
    ShardingPlan,
    ShardingType,
    ShardMetadata,
)
from torchrec.modules.embedding_configs import BaseEmbeddingConfig

from tzrec.protos import feature_pb2

# Empirical x_eff constants fitted from an on-device dynamicemb sweep
# (4M-row table, dim=128, adam, pow-law alpha=1.05, A10 GPU; see
# experiments/sweep_20260513-161030/full_a10gpu1.json). Median fwd+bwd
# latency clustered into three regimes:
#   * HYBRID @ x=1.0:   0.80 ms (HBM-only fast path; runtime drops the
#                        host tier when total_value_memory <= local_hbm)
#   * CACHING @ x<1.0:  2.63 ms  (~3.3x slower than HBM-only)
#   * HYBRID  @ x<1.0:  5.44 ms  (~6.8x slower than HBM-only)
# Within each <1.0 block the ratio dependence is noise-dominated.
# Inverting the linear bw model bw = x_eff*HBM + (1-x_eff)*HBM_TO_DDR
# (torchrec defaults: HBM=897 GB/s, HBM_TO_DDR=32 GB/s) yields the
# constants below. The +0.01*x term is a tiebreaker so the DP can produce
# strictly ordered proposals within each block.
_DYNAMICEMB_CACHING_X_EFF_BASE = 0.28
_DYNAMICEMB_HYBRID_X_EFF_BASE = 0.11
_DYNAMICEMB_X_EFF_TIEBREAK = 0.01


def _dynamicemb_effective_cache_ratio(
    cache_load_factor: Optional[float],
    caching: bool,
    stats: Optional[Any] = None,
) -> float:
    """Effective HBM-hit ratio for the dynamicemb perf model.

    Returns the value passed to torchrec's perf bandwidth formula
    ``bw = x_eff*hbm + (1-x_eff)*hbm_to_ddr_bw``. Larger value = faster path.

    The ratio is derived from an on-device perf sweep, not a heuristic.
    Empirical pattern (alpha=1.05 pow-law access on A10):

      * ``x == 1.0``:  the runtime drops the host tier (HBM-only); both
        modes hit the fastest path. Return ``1.0``.
      * ``caching=True``, ``x < 1.0``: 3.3x slower than HBM-only -> base 0.28.
      * ``caching=False``, ``x < 1.0``: 6.8x slower than HBM-only -> base 0.11.

    Within each ``x < 1.0`` block the perf is roughly flat in ratio, but we
    add a tiny monotonic perturbation so the DP can break ties.

    If ``stats`` is provided, ``1 - stats.expected_miss_rate(x)`` overrides
    the heuristic verbatim (clamped to ``[0, 1]``); the caller opts in to
    their own measurement.
    """
    x = float(cache_load_factor) if cache_load_factor is not None else 0.0
    x = max(0.0, min(1.0, x))
    if stats is not None:
        miss_rate = float(stats.expected_miss_rate(x))
        return max(0.0, min(1.0, 1.0 - miss_rate))
    if x >= 1.0:
        return 1.0
    base = _DYNAMICEMB_CACHING_X_EFF_BASE if caching else _DYNAMICEMB_HYBRID_X_EFF_BASE
    return base + _DYNAMICEMB_X_EFF_TIEBREAK * x


has_dynamicemb = False
try:
    import dynamicemb
    from dynamicemb import (
        DynamicEmbInitializerArgs,
        DynamicEmbInitializerMode,
        DynamicEmbScoreStrategy,
        FrequencyAdmissionStrategy,
        KVCounter,
        align_to_table_size,
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
        init_capacity = align_to_table_size(dynamicemb_cfg.init_capacity_per_rank)

    admission_counter = None
    admit_strategy = None
    admission_strategy_type = dynamicemb_cfg.WhichOneof("admission_strategy")
    if admission_strategy_type is not None:
        if admission_strategy_type == "frequency_admission_strategy":
            admission_strategy_cfg = getattr(dynamicemb_cfg, admission_strategy_type)
            counter_capacity = (
                admission_strategy_cfg.counter_capacity
                if admission_strategy_cfg.HasField("counter_capacity")
                else num_embeddings
            )
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            admission_counter = KVCounter(
                capacity=align_to_table_size(int(counter_capacity / world_size)),
                bucket_capacity=admission_strategy_cfg.counter_bucket_capacity,
            )
            admit_strategy = FrequencyAdmissionStrategy(
                threshold=admission_strategy_cfg.threshold,
                initializer_args=_build_dynamicemb_initializer(
                    admission_strategy_cfg.initializer_args,
                    num_embeddings,
                    embedding_dim,
                    is_eval=True,
                ),
            )
        else:
            raise ValueError(f"Unknown AdmissionStrategy: {admission_strategy_type}")

    demb_opt_kwargs = {}
    if dynamicemb_cfg.HasField("bucket_capacity"):
        demb_opt_kwargs["bucket_capacity"] = dynamicemb_cfg.bucket_capacity

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
        score_strategy=score_strategy,
        admit_strategy=admit_strategy,
        admission_counter=admission_counter,
        **demb_opt_kwargs,
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
        compute_kernels=[EmbeddingComputeKernel.CUSTOMIZED_KERNEL.value],
        dynamicemb_options=dynamicemb_options,
        **constraints_kwargs,
    )
    return constraints


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
        bucket_capacity: int = 128,
        caching: bool = False,
    ) -> int:
        """Per-shard storage size for a dynamicemb table -- HBM or DDR (bytes).

        Byte budget (single shard, rows x dim):

            value_bytes_per_row = round_up16(dim * (1 + opt_mult) * element)
            total_value_memory  = align(rows) * value_bytes_per_row
            num_buckets         = align(rows) / bucket_capacity

            hbm_budget = cache_ratio * total_value_memory                 # values
                       + align(rows) * (key<8B> + score<8B> + digest<1B>) # per-row
                       + num_buckets * bucket_header<4B>                  # per-bucket

            ddr_budget = HYBRID  (caching=False): (1 - cache_ratio) * total_value_memory
                         CACHING (caching=True):  total_value_memory  # full backing

        HYBRID hash-partitions values across HBM and host; ``cache_ratio`` is
        HBM's value share. CACHING keeps the full backing store on host and
        uses HBM as a hot-row cache of size
        ``cache_ratio * total_value_memory``. Hash-table metadata
        (key + score + digest + bucket header) is accounted on HBM only --
        matches the existing tzrec convention.
        """
        if cache_ratio is None:
            cache_ratio = 1.0
        if is_hbm:
            value_ratio = cache_ratio
        else:
            value_ratio = 1.0 if caching else (1.0 - cache_ratio)
        return math.ceil(
            align_to_table_size(size[0])
            * (
                _round_up(
                    math.ceil(size[1] * (1 + optimizer_multipler) * element_size),
                    16,
                )
                * value_ratio
                + (8 + 8 + 1 + 4 / bucket_capacity) * (is_hbm and not only_values)
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
                        bucket_capacity=dynamicemb_options.bucket_capacity,
                    )
                )
                # Fill in per-shard fields that used to be populated by
                # dynamicemb's internal ``_get_dynamicemb_options_per_table``.
                # After the fused-storage refactor (NVIDIA recsys-examples
                # PR #343) that upstream function became a pass-through
                # validator, so the caller must set ``dim``, ``max_capacity``
                # (per-shard row count) and ``embedding_dtype`` directly.
                dynamicemb_options.dim = shards[0].size[1]
                dynamicemb_options.max_capacity = shards[0].size[0]
                if dynamicemb_options.embedding_dtype is None:
                    dynamicemb_options.embedding_dtype = tensor.dtype
                if dynamicemb_options.index_type is None:
                    dynamicemb_options.index_type = torch.int64

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

    _orig_hw_perf_config_get_device_bw = HardwarePerfConfig.get_device_bw

    def _customized_kernel_aware_get_device_bw(
        self,  # pyre-ignore [2]
        compute_device: str,
        compute_kernel: str,
        hbm_mem_bw: float,
        ddr_mem_bw: float,
        ssd_mem_bw: float,
        hbm_to_ddr_mem_bw: float,
        caching_ratio: Optional[float] = None,
        prefetch_pipeline: bool = False,
    ) -> Optional[float]:
        if compute_kernel == EmbeddingComputeKernel.CUSTOMIZED_KERNEL.value:
            cr = caching_ratio if caching_ratio is not None else 0.0
            return (cr * hbm_mem_bw + (1 - cr) * hbm_to_ddr_mem_bw) / 10
        return _orig_hw_perf_config_get_device_bw(
            self,
            compute_device,
            compute_kernel,
            hbm_mem_bw,
            ddr_mem_bw,
            ssd_mem_bw,
            hbm_to_ddr_mem_bw,
            caching_ratio,
            prefetch_pipeline,
        )

    # pyre-ignore [9]
    HardwarePerfConfig.get_device_bw = _customized_kernel_aware_get_device_bw

    _orig_build_shard_perf_contexts = (
        ShardPerfContext.build_shard_perf_contexts.__func__
    )

    def _dynamicemb_aware_build_shard_perf_contexts(
        cls,  # pyre-ignore [2]
        config,  # pyre-ignore [2]
        shard_sizes,  # pyre-ignore [2]
        sharding_option,  # pyre-ignore [2]
        topology,  # pyre-ignore [2]
        constraints,  # pyre-ignore [2]
        sharder,  # pyre-ignore [2]
        *args,  # pyre-ignore [2]
        **kwargs,  # pyre-ignore [2]
    ):
        """Inject the empirical x_eff into the perf estimator for both modes.

        Temporarily replace ``sharding_option.cache_params`` with a clone
        whose ``load_factor`` is the empirically-fitted x_eff for the
        (mode, cache_load_factor) combination. Restored before returning so
        the (separately invoked) storage estimator still sees the un-boosted
        ratio.
        """
        dynamicemb_options = getattr(sharding_option, "dynamicemb_options", None)
        original_cache_params = sharding_option.cache_params
        if dynamicemb_options is not None:
            caching = bool(getattr(dynamicemb_options, "caching", False))
            stats = original_cache_params.stats if original_cache_params else None
            x_eff = _dynamicemb_effective_cache_ratio(
                sharding_option.cache_load_factor, caching=caching, stats=stats
            )
            sharding_option.cache_params = (
                dataclasses.replace(original_cache_params, load_factor=x_eff)
                if original_cache_params is not None
                else CacheParams(load_factor=x_eff)
            )
        result = _orig_build_shard_perf_contexts(
            cls,
            config,
            shard_sizes,
            sharding_option,
            topology,
            constraints,
            sharder,
            *args,
            **kwargs,
        )
        sharding_option.cache_params = original_cache_params
        return result

    # pyre-ignore [9]
    ShardPerfContext.build_shard_perf_contexts = classmethod(
        _dynamicemb_aware_build_shard_perf_contexts
    )

    def _calculate_dynamicemb_storage_specific_sizes(
        tensor: torch.Tensor,
        shard_sizes: List[List[int]],
        optimizer_class: Optional[Type[torch.optim.Optimizer]] = None,
        cache_ratio: float = 1.0,
        is_inference: bool = False,
        bucket_capacity: int = 128,
        caching: bool = False,
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
                bucket_capacity=bucket_capacity,
                caching=caching,
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
                bucket_capacity=bucket_capacity,
                caching=caching,
            )
            for size in shard_sizes
        ]
        return hdm_value_sizes, ddr_value_sizes

    def dynamicemb_calculate_shard_storages(
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
        dynamicemb_options: Optional[dynamicemb.DynamicEmbTableOptions] = None,
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
            dynamicemb_options (DynamicEmbTableOptions): dynamice embedding options

        Returns:
            List[Storage]: storage object for each device in topology.
        """
        # storage estimator for dynamicemb
        assert compute_kernel == EmbeddingComputeKernel.CUSTOMIZED_KERNEL.value
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
                bucket_capacity=dynamicemb_options.bucket_capacity,
                caching=bool(getattr(dynamicemb_options, "caching", False)),
            )
        )
        counter_hbm_specific_size = 0
        if dynamicemb_options.admission_counter is not None:
            counter = dynamicemb_options.admission_counter
            if isinstance(counter, KVCounter):
                counter_hbm_specific_size = (
                    _calculate_dynamicemb_table_storage_specific_size(
                        [counter.capacity, 0],
                        element_size=0,  # counter does not contain embedding value
                        bucket_capacity=counter.bucket_capacity,
                    )
                )

        hbm_sizes: List[int] = [
            (
                hbm_specific_size
                + counter_hbm_specific_size
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

    from torchrec.sparse import jagged_tensor_validator as _jtv

    _orig_validate_feature_range = _jtv._validate_feature_range

    def _validate_feature_range_with_dynamicemb(kjt, configs):
        """Skip range check for dynamicemb features.

        DynamicEmb uses hash tables that accept arbitrary uint64 keys.
        max_capacity is a storage limit, not a valid key range.
        """
        filtered_configs = [
            c for c in configs if not getattr(c, "use_dynamicemb", False)
        ]
        if not filtered_configs:
            return True
        return _orig_validate_feature_range(kjt, filtered_configs)

    _jtv._validate_feature_range = _validate_feature_range_with_dynamicemb
