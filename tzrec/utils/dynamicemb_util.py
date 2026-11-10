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
from typing import Any, Dict, List, NamedTuple, Optional, Type, cast

import torch
from torchrec import optim as trec_optim
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
    SharderData,
    ShardingOption,
    Storage,
    Topology,
)
from torchrec.distributed.sharding_plan import placement
from torchrec.distributed.types import (
    CacheParams,
    EmbeddingModuleShardingPlan,
    EnumerableShardingSpec,
    ParameterSharding,
    PipelineType,
    ShardingPlan,
    ShardingType,
    ShardMetadata,
)
from torchrec.modules.embedding_configs import BaseEmbeddingConfig

from tzrec.optim.optimizer import (
    FTRL,
    has_fbgemm_ftrl,
    sparse_init_accumulator_value,
)
from tzrec.protos import feature_pb2
from tzrec.utils.logging_util import logger

_DYNAMICEMB_CACHING_X_EFF_BASE = 0.28
_DYNAMICEMB_HYBRID_X_EFF_BASE = 0.11
_DYNAMICEMB_X_EFF_TIEBREAK = 0.01

DYNAMICEMB_CACHE_BUCKET_CAPACITY = 1024
_DYNAMICEMB_CACHE_OVERFLOW_BUCKETS = 3


class DynamicEmbTableMemory(NamedTuple):
    """Physical per-shard capacities of one dynamicemb candidate.

    Fields:
        hbm_values: value+optimizer bytes in HBM (cache tier plus its overflow
            area for CACHING).
        host_values: value+optimizer bytes on the host (full backing table for
            CACHING, remainder partition for HYBRID).
        hbm_index: HBM hash-map metadata bytes: per-slot key/digest/score
            words and reference counters, plus per-bucket size counters.
            Host tiers keep their key maps on HBM too.
        hbm_rows: HBM value rows including the cache overflow area.
        host_rows: host value rows.
        resident_rows: rows credited toward normal cache residency -- the cache
            tier without its overflow area (all HBM value rows otherwise).
        local_hbm_for_values: requested value bytes, before physical tier rounding.
    """

    hbm_values: int
    host_values: int
    hbm_index: int
    hbm_rows: int
    host_rows: int
    resident_rows: int
    local_hbm_for_values: int


def _bucket_rounded_rows(rows: int, width: int) -> int:
    """Round a tier to whole native buckets."""
    return (rows + width - 1) // width * width


def _keymap_bytes(slots: int, width: int, score_words: int) -> int:
    """Count keys, digests, score words, reference counters and bucket sizes."""
    return slots * (8 + 1 + 8 * score_words + 4) + 4 * (slots // width)


def dynamicemb_table_memory(
    *,
    caching: bool,
    ratio: float,
    shard_rows: int,
    dim: int,
    element_size: int,
    optimizer_state_dim: int = 0,
    bucket_capacity: Optional[int] = None,
    score_words: int = 1,
    no_eviction: bool = False,
) -> DynamicEmbTableMemory:
    """Estimate physical storage and the value budget of a normalized row-wise shard.

    Constraints align shard_rows and the runtime max_capacity before enumeration.
    Zero and full budgets may share a HYBRID group: reserve its minimum tiers
    as well as the standalone endpoint. NO_EVICTION cannot form a HYBRID group.
    """
    if not 0.0 <= ratio <= 1.0 or (caching and ratio in (0.0, 1.0)):
        raise ValueError(f"invalid dynamicemb placement: caching={caching}, {ratio=}")
    width = bucket_capacity if bucket_capacity is not None else 128
    row_bytes = (dim + optimizer_state_dim) * element_size
    full_bytes = shard_rows * row_bytes
    budget = (
        full_bytes
        if ratio == 1
        else min(int(ratio * shard_rows * row_bytes), full_bytes - 1)
    )
    full_index = _keymap_bytes(
        shard_rows * (2 if no_eviction else 1), width, score_words
    )

    if no_eviction and ratio in (0, 1):
        resident = hbm = shard_rows if ratio == 1 else 0
        host = shard_rows - hbm
        index = full_index
    else:
        fraction = budget / full_bytes
        cache_width = DYNAMICEMB_CACHE_BUCKET_CAPACITY
        resident = _bucket_rounded_rows(max(1, int(shard_rows * fraction)), cache_width)
        if caching:
            overflow = _DYNAMICEMB_CACHE_OVERFLOW_BUCKETS * cache_width
            cache_words = 1 if no_eviction else score_words
            hbm, host = resident + overflow, shard_rows
            index = (
                full_index
                + _keymap_bytes(resident, cache_width, cache_words)
                + _keymap_bytes(overflow, overflow, cache_words)
            )
        else:
            hbm = resident
            host = _bucket_rounded_rows(max(1, int(shard_rows * (1 - fraction))), width)
            index = _keymap_bytes(hbm, cache_width, score_words) + _keymap_bytes(
                host, width, score_words
            )
            if ratio == 1:
                index = max(index, full_index)
    return DynamicEmbTableMemory(
        hbm * row_bytes, host * row_bytes, index, hbm, host, resident, budget
    )


# Armed by train() when delta dump publishes evicted-key tombstones: that
# dump's pop_evicted_keys drain is the only consumer of the evicted-key
# buffer, so retention is enabled exactly when a consumer exists (an
# unconsumed buffer would grow without bound in GPU memory).
_auto_retain_evicted_keys = False
_warned_missing_evicted_item_mode = False


def set_auto_retain_evicted_keys(enabled: bool) -> None:
    """Toggle RETAIN_KEY eviction recording for dynamicemb tables built later.

    Takes effect at plan/shard time (``build_dynamicemb_constraints``), which
    cannot see the train config, hence this module-level switch.

    Args:
        enabled: Whether new dynamicemb tables should retain evicted keys.
    """
    global _auto_retain_evicted_keys
    _auto_retain_evicted_keys = bool(enabled)


def _arm_evicted_key_retention(demb_opt_kwargs: Dict[str, Any]) -> None:
    """Add evicted_item_mode=RETAIN_KEY to table options when auto-retain is on.

    Older dynamicemb builds have no EvictedItemMode; warn once and keep the
    default DISCARD mode rather than raise, so the default-on tombstone dump
    cannot break existing jobs -- tombstones are then silently missing.

    Args:
        demb_opt_kwargs: Keyword arguments for DynamicEmbTableOptions.
    """
    global _warned_missing_evicted_item_mode
    try:
        from dynamicemb.dynamicemb_config import EvictedItemMode
    except ImportError:
        if not _warned_missing_evicted_item_mode:
            _warned_missing_evicted_item_mode = True
            logger.warning(
                "dynamicemb lacks EvictedItemMode; evicted-key tombstones "
                "will be missing from delta embedding dumps."
            )
        return
    demb_opt_kwargs["evicted_item_mode"] = EvictedItemMode.RETAIN_KEY


def _validate_eval_initializer_for_tombstones(
    dynamicemb_cfg: feature_pb2.DynamicEmbedding, table_name: str
) -> None:
    """Reject non-zero eval initializers when evicted-key tombstones are armed.

    Tombstone dumps publish constant-zero rows for evicted keys, so a
    non-zero eval initializer would make the same missing key resolve to
    different values on the dump consumer and in eval lookups.

    Args:
        dynamicemb_cfg: The feature's dynamic embedding config.
        table_name: Embedding table name, for the error message.

    Raises:
        ValueError: if eval_initializer_args is set and not CONSTANT 0.0.
    """
    if not _auto_retain_evicted_keys:
        return
    if not dynamicemb_cfg.HasField("eval_initializer_args"):
        return
    init_cfg = dynamicemb_cfg.eval_initializer_args
    mode = init_cfg.mode if init_cfg.HasField("mode") else "CONSTANT"
    if mode != "CONSTANT" or init_cfg.value != 0.0:
        raise ValueError(
            f"dynamic embedding table {table_name} sets eval_initializer_args "
            f"(mode={mode}, value={init_cfg.value}), but delta embedding dump "
            "with dump_evicted_tombstones publishes constant-zero tombstones "
            "for evicted keys; leave eval_initializer_args unset or set it "
            "to CONSTANT 0.0."
        )


def _dynamicemb_effective_cache_ratio(
    allocated_fraction: float,
    caching: bool,
    stats: Optional[Any] = None,
    *,
    full_endpoint: bool = False,
) -> float:
    """Estimate HBM hits from allocated residency or measured cache misses.

    Only the full non-caching endpoint takes the HBM-direct path. A cache
    covering its entire backing table still uses the empirical cache model.
    """
    x = max(0.0, min(1.0, float(allocated_fraction)))
    if full_endpoint:
        return 1.0
    if x <= 0.0 and not caching:
        return 0.0
    if caching and stats is not None:
        miss_rate = float(stats.expected_miss_rate(x))
        return max(0.0, min(1.0, 1.0 - miss_rate))
    base = _DYNAMICEMB_CACHING_X_EFF_BASE if caching else _DYNAMICEMB_HYBRID_X_EFF_BASE
    return base + _DYNAMICEMB_X_EFF_TIEBREAK * x


def _log_dynamicemb_table_plan(
    *,
    fqn: str,
    caching: bool,
    cache_load_factor: float,
    hbm_bytes: int,
    ddr_bytes: int,
) -> None:
    """Per-table candidate log on rank 0.

    The line records the planner's candidate -- the caching choice, the
    requested ratio, and the per-shard storage estimates. The layout of the
    fused module is the runtime's own decision (its ``StorageLayout``); do
    not read a requested endpoint here as that table's final placement.
    """
    if int(os.environ.get("RANK", 0)) != 0:
        return
    hbm_gib = hbm_bytes / (1 << 30)
    dram_gib = ddr_bytes / (1 << 30)
    logger.info(
        f"[dynamicemb plan] {fqn}: mode={'CACHING' if caching else 'NON_CACHING'} "
        f"cache_load_factor={cache_load_factor:.4g} "
        f"local_hbm={hbm_gib:.3f}GiB "
        f"local_dram={dram_gib:.3f}GiB"
    )


def _dynamicemb_optimizer_state_dim(
    optimizer_class: Optional[Type[torch.optim.Optimizer]],
    shape: torch.Size,
    dtype: torch.dtype,
) -> int:
    """Optimizer state elements per row in embedding-dtype units.

    The value layout stores embedding and optimizer state in one row, so the
    state width in table elements is the embedding width times the
    per-element multiplier, rounded the way the value sizing has always
    rounded it.
    """
    return round(shape[1] * _get_optimizer_multipler(optimizer_class, shape, dtype))


def _get_optimizer_multipler(
    optimizer_class: Optional[Type[torch.optim.Optimizer]],
    shape: torch.Size,
    dtype: torch.dtype,
) -> float:
    """Optimizer state size per embedding element in DynamicEmb's value layout.

    torchrec's table maps any class it does not know to 1 and counts row-wise
    Adagrad as a single element per row. DynamicEmb reserves a 16-byte state slot
    per row for row-wise Adagrad, and FTRL keeps a linear and an accumulator term
    per element, so its state is twice the embedding width.

    Args:
        optimizer_class (type, optional): in-backward optimizer class of the
            table, None when the table is not trained.
        shape (torch.Size): unsharded table shape, used by row-wise optimizers.
        dtype (torch.dtype): embedding dtype, which DynamicEmb also stores
            optimizer state in.

    Returns:
        the multiplier applied to the embedding width.
    """
    if optimizer_class is FTRL:
        return 2.0
    if optimizer_class is trec_optim.RowWiseAdagrad:
        return 16 / (shape[-1] * dtype.itemsize)
    return shard_estimators._get_optimizer_multipler(optimizer_class, shape)


has_dynamicemb = False
try:
    import dynamicemb
    from dynamicemb import (
        DynamicEmbInitializerArgs,
        DynamicEmbInitializerMode,
        DynamicEmbOptimType,
        DynamicEmbScoreStrategy,
        FrequencyAdmissionStrategy,
        KVCounter,
        ProbabilisticAdmissionStrategy,
        align_to_table_size,
    )
    from dynamicemb.dynamicemb_config import (
        DEFAULT_BUCKET_CAPACITY,
        _sharded_table_bucket_layout,
    )
    from dynamicemb.key_value_table import get_score_policy
    from dynamicemb.planner import (
        DynamicEmbParameterConstraints,
        DynamicEmbParameterSharding,
    )
    from dynamicemb.scored_hashtable import score_policy_num_scores
    from dynamicemb.shard import (
        DynamicEmbeddingBagCollectionSharder,
        DynamicEmbeddingCollectionSharder,
    )
    from dynamicemb.shard.embedding import DynamicEmbeddingCollectionContext
    from dynamicemb.types import BUCKET_ALIGNMENT

    # torchrec 1.7.0 no longer coerces EmbeddingCollectionContext's None list-fields
    # to []; dynamicemb still passes None (breaks sharded EC forward), so re-coerce.
    _orig_demb_ctx_init = DynamicEmbeddingCollectionContext.__init__

    def _demb_ctx_init_coerce_none(
        self,
        sharding_contexts=None,  # pyre-ignore [2]
        input_features=None,  # pyre-ignore [2]
        reverse_indices=None,  # pyre-ignore [2]
        seq_vbe_ctx=None,  # pyre-ignore [2]
        frequency_counters=None,  # pyre-ignore [2]
    ):  # pyre-ignore [3]
        _orig_demb_ctx_init(
            self,
            sharding_contexts if sharding_contexts is not None else [],
            input_features if input_features is not None else [],
            reverse_indices if reverse_indices is not None else [],
            seq_vbe_ctx if seq_vbe_ctx is not None else [],
            frequency_counters,
        )

    # pyre-ignore [9]
    DynamicEmbeddingCollectionContext.__init__ = _demb_ctx_init_coerce_none

    def _dynamicemb_score_words(dynamicemb_options) -> int:
        """Score words the table's policy occupies per key in its key map.

        Mirrors ``get_score_policy``/``score_policy_num_scores``: two words
        for the compound LRU+LFU policy, one for every other policy
        (NO_EVICTION alone keeps a slot-occupancy flag word).
        """
        return score_policy_num_scores(
            get_score_policy(dynamicemb_options.score_strategy).policy
        )

    def _dynamicemb_no_eviction(dynamicemb_options) -> bool:
        """Whether the table's strategy never evicts.

        The pinned runtime has no compound NO_EVICTION policies, so the
        single-strategy comparison is exact.
        """
        return dynamicemb_options.score_strategy == DynamicEmbScoreStrategy.NO_EVICTION

    @dataclasses.dataclass
    class TZRecDynamicEmbParameterSharding(DynamicEmbParameterSharding):
        """DynamicEmbParameterSharding carrying the Adagrad accumulator init value.

        The value is not a DynamicEmbTableOptions field, and FBGEMM TBE rejects it as
        a kwarg, so it can only reach BatchedDynamicEmbeddingTables as a per-table
        fused param of the customized kernel.
        """

        initial_accumulator_value: float = 0.0
        ftrl_fused_params: Dict[str, Any] = dataclasses.field(default_factory=dict)

        def get_additional_fused_params(self) -> Dict[str, Any]:
            """Extra fused params of the customized kernel."""
            params = super().get_additional_fused_params()
            # torchrec hashes fused params to group tables, so flatten the dict.
            params.pop("ftrl_fused_params", None)
            params["initial_accumulator_value"] = self.initial_accumulator_value
            params.update(self.ftrl_fused_params)
            return params

    def _dynamicemb_ftrl_fused_params(
        optimizer_class: Optional[Type[torch.optim.Optimizer]],
        optimizer_kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Fused params turning torchrec's fused FTRL into dynamicemb's.

        torchrec maps its FTRL class to FBGEMM's ``EmbOptimType.FTRL``, which
        dynamicemb tables reject, and passes the hyperparameters under the TBE
        kwarg names; these per-table params override both.

        Args:
            optimizer_class (type, optional): in-backward optimizer class of the
                table, None when the table is not trained.
            optimizer_kwargs (dict): in-backward optimizer kwargs of the table.

        Returns:
            the overriding fused params, empty unless the table uses FBGEMM FTRL.
        """
        # pyrefly: ignore[missing-attribute]  # only in the TorchEasyRec torchrec build
        if not has_fbgemm_ftrl() or optimizer_class is not trec_optim.FTRL:
            return {}
        return {
            "optimizer": DynamicEmbOptimType.FTRL,
            "learning_rate_power": optimizer_kwargs["ftrl_learning_rate_power"],
            "l1_reg": optimizer_kwargs["ftrl_l1_reg"],
            "l2_reg": optimizer_kwargs["ftrl_l2_reg"],
        }

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
    """Build row-wise constraints with matching tensor and table capacities.

    Normalize the shared table config before EBC/EC materializes its tensor.
    Initializer and admission defaults continue to use the configured row count.
    """
    if not has_dynamicemb:
        raise RuntimeError(
            "dynamicemb is not installed; required by features with "
            "`dynamicemb { }` set."
        )
    _validate_eval_initializer_for_tombstones(dynamicemb_cfg, emb_config.name)
    embedding_dim = emb_config.embedding_dim
    num_embeddings = emb_config.num_embeddings
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    bucket_capacity = (
        dynamicemb_cfg.bucket_capacity
        if dynamicemb_cfg.HasField("bucket_capacity")
        else DEFAULT_BUCKET_CAPACITY
    )
    num_buckets, bucket_capacity = _sharded_table_bucket_layout(
        emb_config, world_size, bucket_capacity
    )
    local_rows = num_buckets * bucket_capacity
    emb_config.num_embeddings = local_rows * world_size

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
        init_capacity = min(
            align_to_table_size(dynamicemb_cfg.init_capacity_per_rank, bucket_capacity),
            local_rows,
        )

    admit_strategy = None
    admission_strategy_type = dynamicemb_cfg.WhichOneof("admission_strategy")
    if admission_strategy_type is not None:
        admission_strategy_cfg = getattr(dynamicemb_cfg, admission_strategy_type)
        non_admitted_initializer_args = _build_dynamicemb_initializer(
            admission_strategy_cfg.initializer_args,
            num_embeddings,
            embedding_dim,
            is_eval=True,
        )
        if admission_strategy_type == "frequency_admission_strategy":
            counter_capacity = (
                admission_strategy_cfg.counter_capacity
                if admission_strategy_cfg.HasField("counter_capacity")
                else num_embeddings
            )
            admit_strategy = FrequencyAdmissionStrategy(
                threshold=admission_strategy_cfg.threshold,
                counter=KVCounter(
                    capacity=align_to_table_size(int(counter_capacity / world_size)),
                    bucket_capacity=admission_strategy_cfg.counter_bucket_capacity,
                ),
                initializer_args=non_admitted_initializer_args,
            )
        elif admission_strategy_type == "probabilistic_admission_strategy":
            admit_strategy = ProbabilisticAdmissionStrategy(
                probability=admission_strategy_cfg.probability,
                initializer_args=non_admitted_initializer_args,
            )
        else:
            raise ValueError(f"Unknown AdmissionStrategy: {admission_strategy_type}")

    demb_opt_kwargs = {"bucket_capacity": bucket_capacity}
    if _auto_retain_evicted_keys:
        _arm_evicted_key_retention(demb_opt_kwargs)

    dynamicemb_options = dynamicemb.DynamicEmbTableOptions(
        max_capacity=local_rows,
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
        **demb_opt_kwargs,
    )

    constraints_kwargs = {}
    if dynamicemb_cfg.HasField("cache_load_factor"):
        # for ShardingPlan estimator hdm storage
        load_factor = dynamicemb_cfg.cache_load_factor
        if not math.isfinite(load_factor) or not 0.0 <= load_factor <= 1.0:
            raise ValueError(
                f"dynamicemb table {emb_config.name} sets cache_load_factor="
                f"{load_factor}; it must be a finite ratio in [0, 1]."
            )
        constraints_kwargs["cache_params"] = CacheParams(load_factor=load_factor)

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

    # pyrefly: ignore[unbound-name]
    DynamicEmbeddingBagCollectionSharder.compute_kernels = _ebc_compute_kernels
    # pyrefly: ignore[unbound-name]
    DynamicEmbeddingCollectionSharder.compute_kernels = _ec_compute_kernels

    def _dynamicemb_shard_memory(
        dynamicemb_options,
        tensor: torch.Tensor,
        shard_rows: int,
        ratio: float,
        *,
        is_inference: bool = False,
        caching: Optional[bool] = None,
    ) -> DynamicEmbTableMemory:
        """Capacity mirror for one shard of a planned dynamicemb table.

        Derives every input from the table itself so the estimator, the
        candidate generator and plan conversion all see the same physical
        capacities from the same options.
        """
        optimizer_class = getattr(tensor, "_optimizer_classes", [None])[0]
        optimizer_state_dim = (
            0
            if is_inference
            else _dynamicemb_optimizer_state_dim(
                optimizer_class, tensor.shape, tensor.dtype
            )
        )
        # the candidate's caching choice is the planner's own field on the
        # option; the runtime's fused-group layout is derived later
        return dynamicemb_table_memory(
            caching=bool(dynamicemb_options.caching) if caching is None else caching,
            ratio=ratio,
            shard_rows=shard_rows,
            dim=tensor.shape[1],
            element_size=tensor.element_size(),
            optimizer_state_dim=optimizer_state_dim,
            bucket_capacity=dynamicemb_options.bucket_capacity,
            score_words=_dynamicemb_score_words(dynamicemb_options),
            no_eviction=_dynamicemb_no_eviction(dynamicemb_options),
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

                tensor = sharding_option.tensor
                optimizer_class = getattr(tensor, "_optimizer_classes", [None])[0]
                optimizer_kwargs = getattr(tensor, "_optimizer_kwargs", [{}])[0]
                table_memory = _dynamicemb_shard_memory(
                    dynamicemb_options,
                    tensor,
                    shards[0].size[0],
                    float(sharding_option.cache_load_factor),
                    is_inference=False,
                )
                dynamicemb_options.training = optimizer_class is not None
                dynamicemb_options.local_hbm_for_values = (
                    table_memory.local_hbm_for_values
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

                module_plan[sharding_option.name] = TZRecDynamicEmbParameterSharding(
                    sharding_spec=sharding_spec,
                    sharding_type=ShardingType.ROW_WISE.value,
                    ranks=[i for i in range(world_size)],
                    compute_kernel=EmbeddingComputeKernel.CUSTOMIZED_KERNEL.value,
                    dynamicemb_options=dynamicemb_options,
                    initial_accumulator_value=sparse_init_accumulator_value(),
                    ftrl_fused_params=_dynamicemb_ftrl_fused_params(
                        optimizer_class, optimizer_kwargs
                    ),
                )
                _log_dynamicemb_table_plan(
                    fqn=f"{sharding_option.path}.{sharding_option.name}",
                    caching=bool(dynamicemb_options.caching),
                    cache_load_factor=float(sharding_option.cache_load_factor),
                    hbm_bytes=int(shards[0].storage.hbm),
                    ddr_bytes=int(shards[0].storage.ddr),
                )
            else:
                # A data_parallel table is replicated and updated by the dense
                # optimizer, so the sparse optimizer never reaches it.
                if (
                    getattr(sharding_option.tensor, "_optimizer_classes", [None])[0]
                    is FTRL
                    and sharding_type != ShardingType.DATA_PARALLEL.value
                ):
                    raise ValueError(
                        "sparse ftrl_optimizer only supports dynamicemb embedding "
                        "tables, but table["
                        f"{sharding_option.path}.{sharding_option.name}] is planned "
                        f"with sharding_type[{sharding_type}] and "
                        f"compute_kernel[{sharding_option.compute_kernel}]. "
                        "The installed fbgemm-gpu has no FTRL embedding kernel: "
                        "install the TorchEasyRec fbgemm-gpu and torchrec wheels, "
                        "set `dynamicemb { }` on every sparse feature, or use "
                        "another sparse optimizer."
                    )
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
        *args,  # pyre-ignore [2]
        **kwargs,  # pyre-ignore [2]
    ):
        """Estimate performance from allocated residency without changing the budget."""
        dynamicemb_options = getattr(sharding_option, "dynamicemb_options", None)
        original_cache_params = sharding_option.cache_params
        if dynamicemb_options is not None and shard_sizes:
            requested = sharding_option.cache_load_factor
            requested = 1.0 if requested is None else requested
            caching = bool(dynamicemb_options.caching)
            fraction = requested
            if 0 < requested < 1:
                memory = _dynamicemb_shard_memory(
                    dynamicemb_options,
                    sharding_option.tensor,
                    shard_sizes[0][0],
                    requested,
                )
                fraction = (
                    memory.resident_rows / memory.host_rows
                    if caching
                    else memory.hbm_rows / (memory.hbm_rows + memory.host_rows)
                )
            x_eff = _dynamicemb_effective_cache_ratio(
                fraction,
                caching=caching,
                stats=original_cache_params.stats if original_cache_params else None,
                full_endpoint=requested == 1 and not caching,
            )
            sharding_option.cache_params = (
                dataclasses.replace(original_cache_params, load_factor=x_eff)
                if original_cache_params is not None
                else CacheParams(load_factor=x_eff)
            )
        # try/finally so an estimator exception cannot leak the boosted
        # cache_params clone into the storage estimator's view of the
        # same ShardingOption.
        try:
            result = _orig_build_shard_perf_contexts(
                cls,
                config,
                shard_sizes,
                sharding_option,
                topology,
                constraints,
                *args,
                **kwargs,
            )
        finally:
            sharding_option.cache_params = original_cache_params
        return result

    # pyre-ignore [9]
    ShardPerfContext.build_shard_perf_contexts = classmethod(
        _dynamicemb_aware_build_shard_perf_contexts
    )

    def dynamicemb_calculate_shard_storages(
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
        dynamicemb_options: Optional[dynamicemb.DynamicEmbTableOptions] = None,
    ) -> List[Storage]:
        """Calculates estimated storage sizes.

        for each sharded tensor, comprised of input, output, tensor, gradient,
        and optimizer sizes.

        Args:
            sharder_data (SharderData): precomputed sharder data (unused; kept for
                signature parity with the upstream storage estimator).
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
            caching_ratio (float): requested HBM share of the dynamicemb
                table's values, paired with the candidate's caching choice
                on ``dynamicemb_options``: a partial non-caching candidate
                splits values by this ratio, a caching candidate caches this
                fraction above a full host backing store, and the zero and
                full endpoints carry a zero or full value budget whose
                mixed-group costs the estimate envelopes.
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

        # A zero ratio is a HOST_ONLY placement, not a request for the
        # UVM default; only an unset ratio falls back.
        shard_caching_ratio = caching_ratio if caching_ratio is not None else 1.0
        # Per-shard (HBM, DDR) bytes from the capacity mirror: the HBM side
        # holds this placement's value tier plus every tier's key-map
        # metadata (host tiers keep their key maps on HBM); the DDR side
        # holds the host value tier.
        shard_memories = [
            _dynamicemb_shard_memory(
                dynamicemb_options,
                tensor,
                size[0],
                shard_caching_ratio,
                is_inference=is_inference,
            )
            for size in shard_sizes
        ]
        hbm_specific_sizes = [m.hbm_values + m.hbm_index for m in shard_memories]
        ddr_specific_sizes = [m.host_values for m in shard_memories]
        # The counter belongs to the strategy that counts with it; a
        # probabilistic admitter keeps no state, so it costs nothing.
        counter_hbm_specific_size = 0
        if isinstance(dynamicemb_options.admit_strategy, FrequencyAdmissionStrategy):
            counter = dynamicemb_options.admit_strategy.counter
            width = _bucket_rounded_rows(counter.bucket_capacity, BUCKET_ALIGNMENT)
            slots = _bucket_rounded_rows(counter.capacity, width)
            counter_hbm_specific_size = _keymap_bytes(slots, width, score_words=1)

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
