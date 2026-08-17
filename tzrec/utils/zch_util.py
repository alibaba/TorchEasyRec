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

"""ZCH (managed collision) helpers shared by delta dump, export and conversion."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, Tuple

import torch
from torch import nn
from torchrec.distributed.mc_embedding_modules import (
    BaseShardedManagedCollisionEmbeddingCollection,
)
from torchrec.modules.mc_embedding_modules import ManagedCollisionEmbeddingBagCollection
from torchrec.modules.mc_modules import MCHManagedCollisionModule

from tzrec.utils.logging_util import logger

# MCH marks the unoccupied slots of _mch_sorted_raw_ids with this delimiter.
ZCH_EMPTY_SLOT: int = torch.iinfo(torch.int64).max

ZchEventTrackerFn = Callable[[nn.Module, torch.Tensor, torch.Tensor], None]


@dataclass(frozen=True)
class ZchTable:
    """One managed collision table and the embedding table it remaps.

    Attributes:
        wrapper_fqn: ``named_modules`` path of the managed collision wrapper.
        wrapper: Managed collision wrapper, sharded or not.
        inner: Embedding collection the wrapper remaps into.
        inner_kind: ``embedding_bags`` for pooled tables, ``embeddings`` for
            sequence ones.
        table_name: Raw table name within the collection.
        mc_module: Managed collision module owning this table's id mapping.
        table_fqn: Module path of the embedding table, unstripped.
        mc_module_fqn: Module path of ``mc_module``, unstripped.
    """

    wrapper_fqn: str
    wrapper: nn.Module
    inner: nn.Module
    inner_kind: str
    table_name: str
    mc_module: nn.Module
    table_fqn: str
    mc_module_fqn: str


def iter_zch_tables(model: nn.Module) -> Iterator[ZchTable]:
    """Yield every managed collision table of a model.

    A managed collision wrapper holds its ZCH modules at
    ``<P>._managed_collision_collection._managed_collision_modules.<table>`` and
    the matching weight at
    ``<P>._embedding_module.{embedding_bags,embeddings}.<table>``, for pooled
    (``mc_ebc``) and sequence (``mc_ec_dict.<dim>``) collections alike, in both
    the sharded and the unsharded module tree.

    Callers keep their own FQN normalization and their own policy for managed
    collision modules that are not :class:`MCHManagedCollisionModule`, so
    neither is applied here.

    Args:
        model: model to walk, sharded or not.

    Yields:
        One :class:`ZchTable` per remapped table.
    """
    for wrapper_fqn, wrapper in model.named_modules():
        mc_collection = getattr(wrapper, "_managed_collision_collection", None)
        if mc_collection is None or not hasattr(wrapper, "_embedding_module"):
            continue
        if isinstance(wrapper, BaseShardedManagedCollisionEmbeddingCollection):
            bagged = wrapper.bagged
        else:
            bagged = isinstance(wrapper, ManagedCollisionEmbeddingBagCollection)
        inner_kind = "embedding_bags" if bagged else "embeddings"
        inner = wrapper._embedding_module
        for table_name, mc_module in mc_collection._managed_collision_modules.items():
            yield ZchTable(
                wrapper_fqn=wrapper_fqn,
                wrapper=wrapper,
                inner=inner,
                inner_kind=inner_kind,
                table_name=table_name,
                mc_module=mc_module,
                table_fqn=f"{wrapper_fqn}._embedding_module.{inner_kind}.{table_name}",
                mc_module_fqn=(
                    f"{wrapper_fqn}._managed_collision_collection."
                    f"_managed_collision_modules.{table_name}"
                ),
            )


def _mc_compute(self: nn.Module, ctx: Any, dist_input: Any) -> Any:
    output = _ORIG_MC_COMPUTE(self, ctx, dist_input)
    if self.post_lookup_tracker_fn is not None:
        # dist_input carries this shard's raw ids keyed by feature name, while
        # the inner embedding module only ever sees the remapped rows.
        for features in dist_input:
            self.post_lookup_tracker_fn(features, torch.empty(0), self, None)
    return output


def _mc_output_dist(self: nn.Module, ctx: Any, output: Any) -> Any:
    awaitable = _ORIG_MC_OUTPUT_DIST(self, ctx, output)
    if self.post_odist_tracker_fn is not None:
        self.post_odist_tracker_fn()
    return awaitable


# TorchRec declares the tracker callbacks on ShardedEmbeddingModule but never
# invokes them on the managed collision wrapper, and its own collection-level
# hook only fires for HashZch modules, so a delta tracker registered on a ZCH
# module would otherwise silently record nothing.
_ORIG_MC_COMPUTE = BaseShardedManagedCollisionEmbeddingCollection.compute
_ORIG_MC_OUTPUT_DIST = BaseShardedManagedCollisionEmbeddingCollection.output_dist
# pyre-ignore [8]
BaseShardedManagedCollisionEmbeddingCollection.compute = _mc_compute
# pyre-ignore [8]
BaseShardedManagedCollisionEmbeddingCollection.output_dist = _mc_output_dist


def register_post_zch_event_tracker_fn(
    mc_module: MCHManagedCollisionModule, record_fn: ZchEventTrackerFn
) -> None:
    """Register a function to be called when a ZCH table admits and evicts ids.

    ``MCHManagedCollisionModule._update_and_evict`` writes the admitted raw ids
    over the evicted ones in a single store, so the outgoing ids only exist
    between the eviction policy returning its slot indices and that store, and
    the incoming ones only between the store and the trailing buffer re-sort.
    Both boundaries are wrapped to capture the exact tensors, rather than
    reconstructing the two sets by comparing snapshots of the whole table.

    Args:
        mc_module: ZCH module to track.
        record_fn: Called with the module, the evicted raw ids and the admitted
            raw ids, once per eviction round.
    """
    already_wrapped = "_sort_mch_buffers" in mc_module.__dict__
    if already_wrapped:
        logger.warning(
            "[ModelDeltaTracker] zch event function already defined, "
            "overriding with new callable"
        )
    # pyre-ignore [16]
    mc_module.post_zch_event_tracker_fn = record_fn
    if already_wrapped:
        return

    policy = mc_module._eviction_policy
    orig_score: Callable[..., Tuple[torch.Tensor, torch.Tensor]] = (
        policy.update_metadata_and_generate_eviction_scores
    )
    orig_sort: Callable[[], None] = mc_module._sort_mch_buffers
    pending: Dict[str, torch.Tensor] = {}

    def scored(*args: Any, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        evicted_indices, selected_new_indices = orig_score(*args, **kwargs)
        # Advanced indexing copies, so this survives the in-place store below.
        pending["indices"] = evicted_indices
        pending["evicted"] = mc_module._mch_sorted_raw_ids[evicted_indices]
        return evicted_indices, selected_new_indices

    def sorted_buffers() -> None:
        indices = pending.pop("indices", None)
        if indices is not None:
            admitted = mc_module._mch_sorted_raw_ids[indices]
            mc_module.post_zch_event_tracker_fn(
                mc_module, pending.pop("evicted"), admitted
            )
        orig_sort()

    # pyre-ignore [8]
    policy.update_metadata_and_generate_eviction_scores = scored
    # pyre-ignore [8]
    mc_module._sort_mch_buffers = sorted_buffers
