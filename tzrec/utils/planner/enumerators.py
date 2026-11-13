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

import copy
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from torch import nn
from torchrec.distributed.planner.constants import POOLING_FACTOR
from torchrec.distributed.planner.enumerators import (
    GUARDED_COMPUTE_KERNELS,
    GUARDED_SHARDING_TYPES_FOR_FP_MODULES,
    EmbeddingComputeKernel,
    EmbeddingTower,
    EmbeddingTowerCollection,
    ParameterConstraints,
    Shard,
    ShardEstimator,
    _get_tower_index,
    calculate_shard_sizes_and_offsets,
    get_partition_by_type,
    sharder_name,
)
from torchrec.distributed.planner.enumerators import (
    EmbeddingEnumerator as _EmbeddingEnumerator,
)
from torchrec.distributed.planner.shard_estimators import EmbeddingPerfEstimator
from torchrec.distributed.planner.types import (
    ShardingOption,
    Topology,
)
from torchrec.distributed.planner.utils import build_sharder_data_map
from torchrec.distributed.types import (
    BoundsCheckMode,
    CacheParams,
    KeyValueParams,
    ModuleSharder,
    ShardingType,
)
from torchrec.modules.embedding_configs import DataType
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
    EmbeddingCollection,
)

from tzrec.utils.dynamicemb_util import DynamicEmbTableStorage
from tzrec.utils.logging_util import logger
from tzrec.utils.planner.shard_estimators import EmbeddingStorageEstimator


def _extract_constraints_for_param(
    constraints: Optional[Dict[str, ParameterConstraints]], name: str
) -> Tuple[
    List[float],
    Optional[int],
    Optional[CacheParams],
    Optional[bool],
    Optional[bool],
    Optional[BoundsCheckMode],
    Optional[List[str]],
    Optional[DataType],
    Optional[str],
    Optional[KeyValueParams],
    bool,
    Any,
]:
    """Extract TorchRec and DynamicEmb constraints for one table."""
    input_lengths = [POOLING_FACTOR]
    col_wise_shard_dim = None
    cache_params = None
    enforce_hbm = None
    stochastic_rounding = None
    bounds_check_mode = None
    feature_names = None
    output_dtype = None
    device_group = None
    key_value_params = None
    use_dynamicemb = False
    dynamicemb_options = None

    if constraints and constraints.get(name):
        input_lengths = constraints[name].pooling_factors
        col_wise_shard_dim = constraints[name].min_partition
        cache_params = constraints[name].cache_params
        enforce_hbm = constraints[name].enforce_hbm
        stochastic_rounding = constraints[name].stochastic_rounding
        bounds_check_mode = constraints[name].bounds_check_mode
        feature_names = constraints[name].feature_names
        output_dtype = constraints[name].output_dtype
        device_group = constraints[name].device_group
        key_value_params = constraints[name].key_value_params
        if hasattr(constraints[name], "use_dynamicemb"):
            # pyre-ignore [16]
            use_dynamicemb = constraints[name].use_dynamicemb
        if hasattr(constraints[name], "dynamicemb_options"):
            # pyre-ignore [16]
            dynamicemb_options = constraints[name].dynamicemb_options
    return (
        input_lengths,
        col_wise_shard_dim,
        cache_params,
        enforce_hbm,
        stochastic_rounding,
        bounds_check_mode,
        feature_names,
        output_dtype,
        device_group,
        key_value_params,
        use_dynamicemb,
        dynamicemb_options,
    )


_DYNAMICEMB_WORKING_SET_MULTIPLIER = 5


def _dynamicemb_partial_ratios(low: float) -> List[float]:
    """Sample five geometric points up to 0.5, then fixed tenths up to 0.9."""
    if low >= 1.0:
        return []
    ratios = {low, *(r for r in (0.5, 0.6, 0.7, 0.8, 0.9) if r >= low)}
    if low < 0.5:
        ratios.update(float(r) for r in np.geomspace(low, 0.5, 5)[1:-1])
    return sorted(ratios)


def _emit_dynamicemb_variants(
    base_option: ShardingOption,
    unique_id_factor: Optional[float],
    allow_rocksdb: bool = False,
) -> List[ShardingOption]:
    """Expand a DynamicEmb sharding option into caching and non-caching candidates.

    Without an explicit load factor, search the zero and full value budgets
    and both partial-budget modes. A configured unique-id factor sets the
    working-set floor; a zero factor uses a one-row floor.
    HYBRID starts at min(floor, 0.1), CACHING at max(floor, 0.1).
    Without that factor, built-in grids start at 0.1 and RocksDB at 0.01.
    The built-in partial grids use five geometric points through 0.5, then fixed
    tenths up to 0.9. An explicit partial ratio requests both modes at that
    budget. A cache must cover the working set; a cache covering the entire
    shard is represented by the full non-caching endpoint.

    In automatic search, keep the first built-in budget for each set of
    rounded tier capacities within a mode. Zero, partial and full budgets
    remain distinct: equal row estimates do not imply the same layout after
    native fusion.
    Within a mode the HBM value rows follow from the resident rows -- equal
    for HYBRID, plus the fixed overflow area for CACHING -- so the key tracks
    only the resident and backing rows. The planner uses the candidates'
    resource and performance costs to select a plan.

    With RocksDB planning, eligible tables additionally get CACHING_PS
    candidates on one geometric grid of ten ratios from the working-set floor
    up to ``0.5``, both endpoints included. A floor above ``0.5`` leaves the
    table without RocksDB candidates. DynamicEmb preserves each table's value budget
    when fusing tables with different ratios.

    The caller supplies base_option.dynamicemb_options. Each emitted option
    owns a deep copy of its table options and reports the allocated share as
    its load factor; its storage estimate retains the byte budget for runtime.
    """
    tensor = base_option.tensor
    template = copy.copy(base_option)
    template.dynamicemb_options = copy.copy(base_option.dynamicemb_options)
    options = template.dynamicemb_options
    rows = base_option.shards[0].size[0]
    if any(shard.size[0] != options.max_capacity for shard in base_option.shards):
        raise ValueError("DynamicEmb shard rows must match the per-rank max_capacity")
    if unique_id_factor is None:
        required = math.ceil(base_option.batch_size * sum(base_option.input_lengths))
        builtin_floor, rocksdb_floor = 0.1, 0.01
    else:
        if not math.isfinite(unique_id_factor) or unique_id_factor < 0:
            raise ValueError(
                f"{base_option.name}: unique_id_factor must be finite and non-negative"
            )
        required = math.ceil(
            _DYNAMICEMB_WORKING_SET_MULTIPLIER
            * unique_id_factor
            * base_option.batch_size
        )
        required = min(rows, required)
        builtin_floor = rocksdb_floor = max(required, 1) / rows

    params = base_option.cache_params
    requests: List[Tuple[bool, float, Any]]
    if params is None:
        requests = [(False, 0.0, None)]
        requests.extend(
            (False, r, None)
            for r in _dynamicemb_partial_ratios(min(builtin_floor, 0.1))
        )
        requests.extend(
            (True, r, None) for r in _dynamicemb_partial_ratios(max(builtin_floor, 0.1))
        )
        requests.append((False, 1.0, None))
    else:
        ratio = float(params.load_factor)
        requests = [(False, ratio, None)]
        if 0 < ratio < 1:
            requests.append((True, ratio, None))

    if allow_rocksdb and params is None and options.score_strategy.name != "CUSTOMIZED":
        if rocksdb_floor <= 0.5:
            from tzrec_ext.rocksdb.storage import RocksDBStorageFactory

            factory = RocksDBStorageFactory(None, base_option.fqn)
            requests.extend(
                (True, ratio, factory)
                for ratio in sorted(
                    {float(r) for r in np.geomspace(rocksdb_floor, 0.5, num=10)}
                )
            )

    candidates = []
    seen = set()
    for caching, requested, external_storage in requests:
        options.caching = caching
        options.external_storage = external_storage
        ratio = requested
        dynamicemb_storage = DynamicEmbTableStorage.from_options(
            tensor, options, ratio=ratio
        )
        if ratio > 0 and dynamicemb_storage.local_hbm_for_values == 0:
            if options.caching:
                continue
            ratio = 0.0
        elif (
            options.caching
            and dynamicemb_storage.resident_rows >= dynamicemb_storage.backing_rows
        ):
            if params is None:
                continue  # Automatic search already includes the full endpoint.
            options.caching = False
            ratio = 1.0
        if ratio != requested:
            dynamicemb_storage = DynamicEmbTableStorage.from_options(
                tensor, options, ratio=ratio
            )
        if options.caching and dynamicemb_storage.resident_rows < required:
            continue
        if params is None and external_storage is None:
            endpoint = ratio if ratio in (0, 1) else None
            key = (
                options.caching,
                endpoint,
                dynamicemb_storage.backing_rows,
                dynamicemb_storage.resident_rows,
            )
            if key in seen:
                continue
            seen.add(key)

        if options.caching:
            allocated = (
                dynamicemb_storage.resident_rows / dynamicemb_storage.backing_rows
            )
        elif 0 < ratio < 1:
            allocated = dynamicemb_storage.resident_rows / (
                dynamicemb_storage.resident_rows + dynamicemb_storage.backing_rows
            )
        else:
            allocated = ratio
        option = copy.deepcopy(template)
        option.cache_params = CacheParams(
            load_factor=allocated, stats=params.stats if params else None
        )
        option.dynamicemb_storage = dynamicemb_storage
        candidates.append(option)

    return candidates


class EmbeddingEnumerator(_EmbeddingEnumerator):
    """Generates embedding sharding options for given `nn.Module` with constraints.

    Args:
        topology (Topology): device topology.
        batch_size (int): batch size.
        constraints (Optional[Dict[str, ParameterConstraints]]): dict of parameter names
            to provided ParameterConstraints.
        estimator (Optional[Union[ShardEstimator, List[ShardEstimator]]]): shard
            performance estimators.
        use_exact_enumerate_order (bool): whether to enumerate shardable parameters in
            the exact name_children enumeration order
        fqn_constraints (Optional[Dict[str, ParameterConstraints]]): dict of parameter
            fqns to provided ParameterConstraints.
        global_constraints (Optional[ParameterConstraints]): all parameters provided
            ParameterConstraints.
        allow_rocksdb: Whether to enumerate eligible RocksDB candidates.
    """

    def __init__(
        self,
        topology: Topology,
        batch_size: int,
        constraints: Optional[Dict[str, ParameterConstraints]] = None,
        estimator: Optional[Union[ShardEstimator, List[ShardEstimator]]] = None,
        use_exact_enumerate_order: Optional[bool] = False,
        fqn_constraints: Optional[Dict[str, ParameterConstraints]] = None,
        global_constraints: Optional[ParameterConstraints] = None,
        allow_rocksdb: bool = False,
    ) -> None:
        if not estimator:
            estimator = [
                EmbeddingPerfEstimator(topology=topology, constraints=constraints),
                EmbeddingStorageEstimator(topology=topology, constraints=constraints),
            ]
        super().__init__(
            topology, batch_size, constraints, estimator, use_exact_enumerate_order
        )
        self._fqn_constraints = fqn_constraints
        self._global_constraints = global_constraints
        self.allow_rocksdb = allow_rocksdb

    def _get_constraints(
        self, child_path: str, name: str
    ) -> Tuple[Optional[Dict[str, ParameterConstraints]], str]:
        """Resolve table constraints with global sharding and kernel defaults."""
        if self._fqn_constraints is not None:
            constraint_key = child_path + "." + name
            # pyre-ignore [58]
            if constraint_key in self._fqn_constraints:
                constraint = self._fqn_constraints[constraint_key]
                if self._global_constraints is not None:
                    # an entry that does not restrict a field falls back to
                    # the global constraint for that field, so statistics-only
                    # entries keep the global lists applied
                    merged = copy.copy(constraint)
                    if merged.sharding_types is None:
                        merged.sharding_types = self._global_constraints.sharding_types
                    if merged.compute_kernels is None:
                        merged.compute_kernels = (
                            self._global_constraints.compute_kernels
                        )
                    constraint = merged
                return {constraint_key: constraint}, constraint_key
        if self._global_constraints is not None:
            return {"__GLOBAL__": self._global_constraints}, "__GLOBAL__"
        return self._constraints, name

    def enumerate(
        self,
        module: nn.Module,
        sharders: List[ModuleSharder[nn.Module]],
    ) -> List[ShardingOption]:
        """Generates relevant sharding options given module and sharders.

        Args:
            module (nn.Module): module to be sharded.
            sharders (List[ModuleSharder[nn.Module]]): provided sharders for module.

        Returns:
            List[ShardingOption]: valid sharding options with values populated.
        """
        self._sharder_map = {
            sharder_name(sharder.module_type): sharder for sharder in sharders
        }
        self._sharder_data_map = build_sharder_data_map(self._sharder_map)
        sharding_options: List[ShardingOption] = []

        named_modules_queue = [("", module)]
        while named_modules_queue:
            if not self._use_exact_enumerate_order:
                child_path, child_module = named_modules_queue.pop()
            else:
                child_path, child_module = named_modules_queue.pop(0)
            sharder_key = sharder_name(type(child_module))
            sharder = self._sharder_map.get(sharder_key, None)
            if not sharder:
                for n, m in child_module.named_children():
                    if child_path != "":
                        named_modules_queue.append((child_path + "." + n, m))
                    else:
                        named_modules_queue.append((n, m))
                continue

            # Determine the pooling state for all sharding_options using this
            # (child_module, child_path). With this optimization, we change enumerate()
            # from being O(N^2) with respect to the number of tables to O(N). The
            # previous quadratic behavior is because in populate_estimates() invoked
            # below, each sharding_option needs to determine its pooling state, which
            # is does via an expensive O(N) walk through the list of embedding tables.
            # With this change sharding_option.is_pooled becomes O(1).
            is_pooled = ShardingOption.module_pooled(child_module, child_path)

            for name, param in sharder.shardable_parameters(child_module).items():
                _constraints, key = self._get_constraints(child_path, name)
                (
                    input_lengths,
                    col_wise_shard_dim,
                    cache_params,
                    enforce_hbm,
                    stochastic_rounding,
                    bounds_check_mode,
                    feature_names,
                    output_dtype,
                    device_group,
                    key_value_params,
                    use_dynamicemb,
                    dynamicemb_options,
                ) = _extract_constraints_for_param(_constraints, key)
                # the union measurement rides on the resolved constraint;
                # absence stays None, distinct from a measured zero
                unique_id_factor = (
                    getattr(_constraints[key], "unique_id_factor", None)
                    if _constraints and _constraints.get(key)
                    else None
                )

                # skip for other device groups
                if device_group and device_group != self._compute_device:
                    continue

                num_buckets = self._get_num_buckets(name, child_module)
                sharding_options_per_table: List[ShardingOption] = []

                for sharding_type in self._filter_sharding_types(
                    name,
                    sharder.sharding_types(self._compute_device),
                    child_path,
                    sharder_key=sharder_key,
                ):
                    for compute_kernel in self._filter_compute_kernels(
                        name,
                        sharder.compute_kernels(sharding_type, self._compute_device),
                        sharding_type,
                        child_path,
                    ):
                        (
                            shard_sizes,
                            shard_offsets,
                        ) = calculate_shard_sizes_and_offsets(
                            tensor=param,
                            world_size=self._world_size,
                            local_world_size=self._local_world_size,
                            sharding_type=sharding_type,
                            col_wise_shard_dim=col_wise_shard_dim,
                            device_memory_sizes=self._device_memory_sizes,
                            num_buckets=num_buckets,
                        )
                        dependency = None
                        if isinstance(child_module, EmbeddingTower):
                            dependency = child_path
                        elif isinstance(child_module, EmbeddingTowerCollection):
                            tower_index = _get_tower_index(name, child_module)
                            dependency = child_path + ".tower_" + str(tower_index)
                        sharding_option = ShardingOption(
                            name=name,
                            tensor=param,
                            module=(child_path, child_module),
                            input_lengths=input_lengths,
                            batch_size=self._batch_size,
                            compute_kernel=compute_kernel,
                            sharding_type=sharding_type,
                            partition_by=get_partition_by_type(sharding_type),
                            shards=[
                                Shard(size=size, offset=offset)
                                for size, offset in zip(shard_sizes, shard_offsets)
                            ],
                            cache_params=cache_params,
                            enforce_hbm=enforce_hbm,
                            stochastic_rounding=stochastic_rounding,
                            bounds_check_mode=bounds_check_mode,
                            dependency=dependency,
                            is_pooled=is_pooled,
                            feature_names=feature_names,
                            output_dtype=output_dtype,
                            key_value_params=key_value_params,
                        )
                        # hack sharding option for dynamicemb
                        if use_dynamicemb:
                            # pyre-ignore [16]
                            sharding_option.use_dynamicemb = use_dynamicemb
                            # pyre-ignore [16]
                            sharding_option.dynamicemb_options = dynamicemb_options
                            sharding_options_per_table.extend(
                                _emit_dynamicemb_variants(
                                    sharding_option,
                                    unique_id_factor,
                                    self.allow_rocksdb,
                                )
                            )
                        else:
                            sharding_options_per_table.append(sharding_option)

                if not sharding_options_per_table:
                    raise RuntimeError(
                        "No available sharding type and compute kernel combination "
                        f"after applying user provided constraints for {name}. "
                        f"Module: {sharder_key}, sharder: {sharder.__class__.__name__},"
                        f" compute device: {self._compute_device}. "
                        "To debug, search above for warning logs about no available "
                        f"sharding types/compute kernels for table: {name}"
                    )

                sharding_options.extend(sharding_options_per_table)

        self.populate_estimates(sharding_options)

        return sharding_options

    def _get_num_buckets(self, parameter: str, module: nn.Module) -> Optional[int]:
        """Get total_num_buckets for virtual-table sharding, or None."""
        if isinstance(module, EmbeddingBagCollection):
            embedding_configs = module.embedding_bag_configs()
        elif isinstance(module, EmbeddingCollection):
            embedding_configs = module.embedding_configs()
        else:
            return None
        for config in embedding_configs:
            if config.name == parameter and getattr(config, "use_virtual_table", False):
                return getattr(config, "total_num_buckets", None)
        return None

    # pyre-ignore [14]
    def _filter_sharding_types(
        self,
        name: str,
        allowed_sharding_types: List[str],
        child_path: str,
        sharder_key: str = "",
    ) -> List[str]:
        """Filter sharding types using explicit constraints and guarded defaults."""
        _constraints, key = self._get_constraints(child_path, name)
        # GRID_SHARD and row-wise on FP modules require explicit opt-in.
        is_fp_module = "FeatureProcessedEmbeddingBagCollection" in sharder_key
        if not _constraints or not _constraints.get(key):
            filtered = [
                t for t in allowed_sharding_types if t != ShardingType.GRID_SHARD.value
            ]
            if is_fp_module:
                filtered = [
                    t
                    for t in filtered
                    if t not in GUARDED_SHARDING_TYPES_FOR_FP_MODULES
                ]
            return filtered
        constraints: ParameterConstraints = _constraints[key]
        if not constraints.sharding_types:
            filtered = [
                t for t in allowed_sharding_types if t != ShardingType.GRID_SHARD.value
            ]
            if is_fp_module:
                filtered = [
                    t
                    for t in filtered
                    if t not in GUARDED_SHARDING_TYPES_FOR_FP_MODULES
                ]
            return filtered
        constrained_sharding_types: List[str] = constraints.sharding_types

        filtered_sharding_types = list(
            set(constrained_sharding_types) & set(allowed_sharding_types)
        )
        if not filtered_sharding_types:
            logger.warn(
                "No available sharding types after applying user provided "
                f"constraints for {key}. Constrained sharding types: "
                f"{constrained_sharding_types}, allowed sharding types: "
                f"{allowed_sharding_types}, filtered sharding types: "
                f"{filtered_sharding_types}. Please check if the constrained "
                "sharding types are too restrictive, if the sharder allows the "
                "sharding types, or if non-strings are passed in."
            )
        return filtered_sharding_types

    # pyre-ignore [14]
    def _filter_compute_kernels(
        self,
        name: str,
        allowed_compute_kernels: List[str],
        sharding_type: str,
        child_path: str,
    ) -> List[str]:
        """Filter kernels using table constraints and dense-kernel rules."""
        _constraints, key = self._get_constraints(child_path, name)
        # setup constrained_compute_kernels
        if _constraints and _constraints.get(key) and _constraints[key].compute_kernels:
            # pyre-ignore
            constrained_compute_kernels: List[str] = _constraints[key].compute_kernels
        else:
            constrained_compute_kernels: List[str] = [
                compute_kernel.value
                for compute_kernel in EmbeddingComputeKernel
                if compute_kernel not in GUARDED_COMPUTE_KERNELS
            ]

        # setup filtered_compute_kernels
        filtered_compute_kernels = list(
            set(constrained_compute_kernels) & set(allowed_compute_kernels)
        )

        # special rules
        if EmbeddingComputeKernel.DENSE.value in filtered_compute_kernels:
            if (
                EmbeddingComputeKernel.FUSED.value in filtered_compute_kernels
            ):  # always false for data_parallel
                filtered_compute_kernels.remove(EmbeddingComputeKernel.DENSE.value)

        if not filtered_compute_kernels:
            logger.warn(
                "No available compute kernels after applying user provided "
                f"constraints for {key}. Constrained compute kernels: "
                f"{constrained_compute_kernels}, allowed compute kernels: "
                f"{allowed_compute_kernels}, filtered compute kernels: "
                f"{filtered_compute_kernels}, sharding type: {sharding_type}. "
                "Please check if the constrained "
                "compute kernels are too restrictive, if the sharder allows the "
                "compute kernels, or if non-strings are passed in."
            )
        return filtered_compute_kernels
