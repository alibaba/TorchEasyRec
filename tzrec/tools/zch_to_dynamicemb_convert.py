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

"""Convert a ZCH-based tzrec checkpoint to a dynamicemb-based checkpoint.

Reads a checkpoint produced by training a tzrec model that uses Zero
Collision Hashing (ZCH) and emits a checkpoint in the format expected by
``DynamicEmbLoad``, so the next training run -- whose pipeline.config has
swapped ``zch{}`` for ``dynamicemb{}`` on the relevant features -- can warm
start from the trained embeddings (and scores, when the target score
strategy is recurrence-compatible).

Output layout (under ``<save_dir>/model.ckpt-0/``)::

    model/                       # byte-copied from source; PartialLoadPlanner
                                 # silently skips MCH/EBC keys absent in the
                                 # new dynamicemb-flavored model
    optimizer/                   # byte-copied from source; non-MCH opt state
                                 # warm-starts as-is, MCH-EBC opt entries are
                                 # silently ignored on load
    dynamicemb/<full_mod_path>/<tbl>_emb_keys.rank_R.world_size_W
    dynamicemb/<full_mod_path>/<tbl>_emb_values.rank_R.world_size_W
    dynamicemb/<full_mod_path>/<tbl>_emb_scores.rank_R.world_size_W   # only when
                                 # target score_strategy in {LFU, STEP}
    dynamicemb/<full_mod_path>/<tbl>_opt_args.json
    plan                         # copied if present (advisory)
    meta                         # {load_model, load_optim,
                                 #  dynamicemb_load_table_names,
                                 #  dynamicemb_load_optim=false}

Memory: one ZCH table is processed at a time, but the source DCP load
materializes the *unsharded* weight plus per-table MCH state in host RAM.
Peak host memory ≈ ``zch_size * (embedding_dim * 4 + 24)`` bytes per
table (e.g. a 100M x 128 fp32 table is ~50 GB). Scale up the conversion
host accordingly; streaming-load is not implemented in v1.
"""

import argparse
import json
import os
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from queue import Queue
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
from torch.distributed.checkpoint import (
    FileSystemReader,
    TensorStorageMetadata,
    load,
)
from torchrec.modules.mc_embedding_modules import (
    ManagedCollisionEmbeddingBagCollection,
    ManagedCollisionEmbeddingCollection,
)
from torchrec.modules.mc_modules import MCHManagedCollisionModule

from tzrec.main import _create_features, _create_model
from tzrec.models.model import TrainWrapper
from tzrec.utils import checkpoint_util, config_util
from tzrec.utils.logging_util import logger

# dynamicemb is an optional dependency. Import lazily so the module is
# loadable on CPU-only / non-dynamicemb environments (e.g. CI test discovery
# imports every *_test.py, which in turn imports this module). All call
# paths that actually need dynamicemb re-import these names locally and
# will raise a clear ImportError there.

# Score / value / key dtypes -- match dynamicemb.types.
_KEY_DTYPE = torch.int64
_VALUE_DTYPE = torch.float32
_SCORE_DTYPE = torch.int64
_IINFO_MAX = torch.iinfo(torch.int64).max

# Target score strategies that the converter migrates from MCH metadata.
# Other strategies (TIMESTAMP, CUSTOMIZED, NO_EVICTION) have score units or
# semantics we cannot recover from MCH state, so we skip score migration and
# let dynamicemb cold-start scores at load time.
_SCORES_MIGRATABLE = frozenset({"LFU", "STEP"})

# Trailing collection-wrapper suffix of an EBC/EC module path. Pooled
# collections add one segment (``mc_ebc`` / ``ebc`` / ``..._user``); sequence
# (EC dict) collections add two segments because ``ec_dict`` is an
# ``nn.ModuleDict`` keyed by embedding-dim (so the path ends in
# ``mc_ec_dict.<dim>`` / ``ec_dict.<dim>``). The regex below strips both
# shapes in one pass.
_COLLECTION_SUFFIX_RE = re.compile(
    r"\.(mc_ebc|mc_ebc_user|ebc|ebc_user"
    r"|mc_ec_dict\.\d+|mc_ec_dict_user\.\d+"
    r"|ec_dict\.\d+|ec_dict_user\.\d+)$"
)


def _strip_collection_suffix(mod_path: str) -> str:
    """Strip the trailing EBC/EC collection-wrapper segment(s).

    ``model.<...>.<group_impl>.mc_ebc`` -> ``model.<...>.<group_impl>``
    ``model.<...>.<group_impl>.mc_ec_dict.16`` -> ``model.<...>.<group_impl>``

    Both shapes reduce to the path of the containing
    EmbeddingGroupImpl / SequenceEmbeddingGroupImpl.
    """
    m = _COLLECTION_SUFFIX_RE.search(mod_path)
    if m is None:
        raise ValueError(
            f"path {mod_path!r} does not end with a known EBC/EC collection wrapper"
        )
    return mod_path[: m.start()]


def _normalize_group_key(path: str) -> str:
    """Strip leading ``model.`` prefixes so source and target paths line up.

    Source-side paths have one ``model.`` (from ``named_modules()``);
    target-side paths have two (from the init-script-style BFS +
    ``parameter_constraints`` walk). Both forms reduce to the same
    ``embedding_group.<group_impl>`` key.
    """
    while path.startswith("model."):
        path = path[len("model.") :]
    return path


@dataclass
class _SourceZchTable:
    """One ZCH-wrapped logical table discovered in the source model.

    ``group_key`` discriminates between tables with the same ``emb_name``
    appearing in different EmbeddingGroupImpl / SequenceEmbeddingGroupImpl
    instances. It is the wrapper-stripped, leading-``model.``-normalized
    path of the containing group impl.
    """

    group_key: str
    emb_name: str
    embedding_dim: int
    zch_size: int
    mch_prefix: str
    weight_fqn: str
    metadata_buffer_names: List[str]
    eviction_policy: str  # "lfu" | "lru" | "distance_lfu" | "none"


@dataclass
class _TargetDynamicEmbTable:
    """One dynamicemb-bound logical table in the target model.

    ``group_key`` mirrors ``_SourceZchTable.group_key``.

    ``full_mod_path`` keeps the wrapper segments and the doubled-``model.``
    prefix because that's the on-disk directory ``DynamicEmbLoad`` looks
    for when restoring shards (verified in
    ``recsys-examples/corelib/dynamicemb/dynamicemb/dump_load.py``:
    ``find_sharded_modules`` walks down to the ShardedEmbeddingCollection
    living *inside* ``ec_dict[dim]``, and ``DynamicEmbLoad`` does
    ``os.path.join(path, collection_path)``).
    """

    group_key: str
    emb_name: str
    full_mod_path: str
    embedding_dim: int
    score_strategy: str
    dist_type: str


def _classify_eviction_policy(metadata_buffer_names: List[str]) -> str:
    """Map MCH metadata buffer presence to the proto eviction-policy name."""
    has_counts = "_mch_counts" in metadata_buffer_names
    has_last = "_mch_last_access_iter" in metadata_buffer_names
    if has_counts and has_last:
        return "distance_lfu"
    if has_counts:
        return "lfu"
    if has_last:
        return "lru"
    return "none"


def _find_zch_tables(
    model: torch.nn.Module,
) -> Dict[Tuple[str, str], _SourceZchTable]:
    """Walk source model and discover MCH-wrapped tables.

    Returns a dict keyed by ``(group_key, emb_name)`` so that two tables
    with the same logical name living under different EmbeddingGroupImpl
    instances do not collide.
    """
    out: Dict[Tuple[str, str], _SourceZchTable] = {}
    for raw_mod_name, m in model.named_modules():
        if not isinstance(
            m,
            (
                ManagedCollisionEmbeddingBagCollection,
                ManagedCollisionEmbeddingCollection,
            ),
        ):
            continue
        mod_name = checkpoint_util._strip_dmp_prefix(raw_mod_name)
        group_path = _strip_collection_suffix(mod_name)
        group_key = _normalize_group_key(group_path)

        mc_coll = m._managed_collision_collection
        emb_coll = m._embedding_module

        if isinstance(m, ManagedCollisionEmbeddingBagCollection):
            emb_configs = emb_coll.embedding_bag_configs()
            inner_kind = "embedding_bags"
        else:
            emb_configs = emb_coll.embedding_configs()
            inner_kind = "embeddings"
        name_to_dim = {c.name: c.embedding_dim for c in emb_configs}

        for tbl_name, mch in mc_coll._managed_collision_modules.items():
            if not isinstance(mch, MCHManagedCollisionModule):
                logger.warning(
                    f"Skipping table '{tbl_name}' under {mod_name}: "
                    f"unsupported MCH type {type(mch).__name__}. "
                    "Only MCHManagedCollisionModule is supported for now."
                )
                continue
            metadata_buffers: List[str] = []
            zch_size = int(mch._zch_size)
            non_persistent = set(mch._non_persistent_buffers_set)
            for buf_name, buf in mch._buffers.items():
                if not buf_name.startswith("_mch_"):
                    continue
                if buf_name in ("_mch_sorted_raw_ids", "_mch_remapped_ids_mapping"):
                    continue
                if buf_name in non_persistent:
                    continue
                if buf is None or buf.dim() != 1 or buf.shape[0] != zch_size:
                    continue
                metadata_buffers.append(buf_name)

            policy = _classify_eviction_policy(metadata_buffers)
            weight_fqn = f"{mod_name}._embedding_module.{inner_kind}.{tbl_name}.weight"
            mch_prefix = (
                f"{mod_name}._managed_collision_collection."
                f"_managed_collision_modules.{tbl_name}"
            )
            key = (group_key, tbl_name)
            if key in out:
                raise RuntimeError(
                    f"Duplicate ZCH-wrapped table '{tbl_name}' found under the "
                    f"same group '{group_key}' (existing: {out[key].mch_prefix}, "
                    f"new: {mch_prefix}). This indicates a misconfigured model."
                )
            if tbl_name not in name_to_dim:
                raise RuntimeError(
                    f"Table '{tbl_name}' has an MCH module but no matching "
                    f"embedding config under {mod_name}._embedding_module."
                )
            out[key] = _SourceZchTable(
                group_key=group_key,
                emb_name=tbl_name,
                embedding_dim=int(name_to_dim[tbl_name]),
                zch_size=zch_size,
                mch_prefix=mch_prefix,
                weight_fqn=weight_fqn,
                metadata_buffer_names=metadata_buffers,
                eviction_policy=policy,
            )
    return out


def _find_dynamicemb_tables(
    target_model: torch.nn.Module,
) -> Dict[Tuple[str, str], _TargetDynamicEmbTable]:
    """Mirror of create_dynamicemb_init_ckpt.py's walk to discover dynamicemb tables.

    Returns dict keyed by ``(group_key, emb_name)``. ``full_mod_path`` (the
    on-disk directory under ``dynamicemb/``) keeps the wrapper segments and
    the doubled-``model.`` prefix so ``DynamicEmbLoad`` finds the shards.
    """
    from dynamicemb.planner import DynamicEmbParameterConstraints

    out: Dict[Tuple[str, str], _TargetDynamicEmbTable] = {}
    q: Queue = Queue()
    q.put(("", target_model))
    while not q.empty():
        path, m = q.get()
        if hasattr(m, "parameter_constraints"):
            # ``path`` here is exactly the group-impl path (with trailing dot)
            # passed in by the BFS as the prefix to ``parameter_constraints``.
            group_key = _normalize_group_key(path.rstrip("."))

            # Read embedding_dim from the impl's actual EBC/EC configs --
            # ``opts.dim`` is unset before the planner runs, so the v3 fallback
            # of ``int(opts.dim) if opts.dim else -1`` silently allowed dim
            # mismatches through.
            name_to_dim: Dict[str, int] = {}
            for attr in ("ebc", "ebc_user"):
                coll = getattr(m, attr, None)
                if coll is not None:
                    for c in coll.embedding_bag_configs():
                        name_to_dim[c.name] = int(c.embedding_dim)
            for attr in ("ec_dict", "ec_dict_user"):
                d = getattr(m, attr, None)
                if d is not None:
                    for ec in d.values():
                        for c in ec.embedding_configs():
                            name_to_dim[c.name] = int(c.embedding_dim)

            for fqn, const in m.parameter_constraints(path).items():
                if not isinstance(const, DynamicEmbParameterConstraints):
                    continue
                mod_path, emb_name = fqn.rsplit(".", 1)
                full_mod_path = "model." + mod_path
                opts = const.dynamicemb_options
                score_strategy = (
                    opts.score_strategy.name
                    if opts.score_strategy is not None
                    else "TIMESTAMP"
                )
                dist_type = getattr(opts, "dist_type", None) or "roundrobin"
                if emb_name not in name_to_dim:
                    raise RuntimeError(
                        f"Target dynamicemb table '{emb_name}' under group "
                        f"'{group_key}' has no matching embedding config; "
                        "check the target pipeline.config."
                    )
                key = (group_key, emb_name)
                if key in out:
                    raise RuntimeError(
                        f"Duplicate dynamicemb table '{emb_name}' under group "
                        f"'{group_key}' (existing: {out[key].full_mod_path}, "
                        f"new: {full_mod_path})."
                    )
                out[key] = _TargetDynamicEmbTable(
                    group_key=group_key,
                    emb_name=emb_name,
                    full_mod_path=full_mod_path,
                    embedding_dim=name_to_dim[emb_name],
                    score_strategy=score_strategy,
                    dist_type=dist_type,
                )
        else:
            for name, child in m.named_children():
                q.put((f"{path}{name}.", child))
    return out


def _select_meta_fqn_for_suffix(
    sd_meta: Mapping[str, object], suffix: str
) -> Optional[str]:
    """Find a single saved FQN whose stripped form matches the given suffix.

    Strict match: ``_strip_dmp_prefix(fqn) == suffix``. Returns None if no
    match; raises if there are multiple (ambiguity).
    """
    hits = [
        fqn
        for fqn, md in sd_meta.items()
        if isinstance(md, TensorStorageMetadata)
        and checkpoint_util._strip_dmp_prefix(fqn) == suffix
    ]
    if not hits:
        return None
    if len(hits) > 1:
        raise RuntimeError(
            f"Multiple saved keys match suffix '{suffix}': {hits}. Cannot disambiguate."
        )
    return hits[0]


def _load_dcp_subset(
    ckpt_path: str, requested_suffixes: List[str]
) -> Dict[str, Optional[torch.Tensor]]:
    """Single-process partial load of selected tensors from a DCP directory.

    Args:
        ckpt_path: directory containing ``.metadata`` + ``__*.distcp`` files.
        requested_suffixes: canonical (post-DMP-strip) FQNs of tensors to load.

    Returns:
        Dict {suffix -> tensor or None if the key wasn't found in the
        checkpoint}. Tensors are CPU-resident at the global (unsharded) shape.
    """
    reader = FileSystemReader(path=ckpt_path)
    meta = reader.read_metadata()
    sd_meta = meta.state_dict_metadata

    target_sd: Dict[str, torch.Tensor] = {}
    suffix_to_meta_fqn: Dict[str, str] = {}
    for suffix in requested_suffixes:
        meta_fqn = _select_meta_fqn_for_suffix(sd_meta, suffix)
        if meta_fqn is None:
            continue
        md = sd_meta[meta_fqn]
        dtype = md.properties.dtype
        target_sd[meta_fqn] = torch.empty(tuple(md.size), dtype=dtype)
        suffix_to_meta_fqn[suffix] = meta_fqn

    if target_sd:
        load(target_sd, storage_reader=reader, no_dist=True)

    return {
        suffix: (target_sd[meta_fqn] if suffix in suffix_to_meta_fqn else None)
        for suffix, meta_fqn in (
            (s, suffix_to_meta_fqn.get(s)) for s in requested_suffixes
        )
    }


def _derive_scores(
    metadata_tensors: Dict[str, torch.Tensor],
    eviction_policy: str,
    target_score_strategy: str,
    valid_mask: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Project MCH eviction metadata onto dynamicemb's int64 score.

    Returns ``None`` for target score strategies whose units / semantics
    don't admit a faithful mapping from MCH counters (TIMESTAMP scores are
    ``device_timestamp()`` ns ~ 10^18; CUSTOMIZED is user-defined;
    NO_EVICTION ignores scores). The caller then omits score files for the
    affected table and dynamicemb cold-starts scores at load.

    Cross-policy combinations also return ``None`` with a warning: there is
    no faithful unit mapping from LFU counts to STEP iteration numbers (or
    vice versa). Only same-semantic source/target pairs migrate:

      - target=LFU   ← source has ``_mch_counts`` (LFU or DistanceLFU)
      - target=STEP  ← source has ``_mch_last_access_iter`` (LRU or DistanceLFU)
    """
    target = (target_score_strategy or "").upper()
    if target not in _SCORES_MIGRATABLE:
        return None

    if target == "LFU":
        src = metadata_tensors.get("_mch_counts")
        if src is None:
            logger.warning(
                f"Target score_strategy=LFU but source has no _mch_counts "
                f"(eviction_policy={eviction_policy!r}); omitting score files."
            )
            return None
        return src[valid_mask].to(_SCORE_DTYPE)

    # target == "STEP"
    src = metadata_tensors.get("_mch_last_access_iter")
    if src is None:
        logger.warning(
            f"Target score_strategy=STEP but source has no _mch_last_access_iter "
            f"(eviction_policy={eviction_policy!r}); omitting score files."
        )
        return None
    return src[valid_mask].to(_SCORE_DTYPE)


def _gather_and_shard_writes(
    raw_ids: torch.Tensor,
    values: torch.Tensor,
    scores: Optional[torch.Tensor],
    table_name: str,
    save_path: str,
    world_size: int,
) -> None:
    """Shard converted entries by raw_id % world_size and write per-rank files.

    When ``scores is None`` no ``_emb_scores.*`` file is written for any
    rank -- dynamicemb's load path tolerates absent score files and falls
    back to runtime score generation.

    Empty shards still get zero-byte ``keys``/``values`` files so
    ``dynamicemb.batched_dynamicemb_tables.find_files`` counts a full set
    of W shards per item.
    """
    from dynamicemb.batched_dynamicemb_tables import encode_checkpoint_file_path

    assert raw_ids.dim() == 1
    assert values.dim() == 2 and values.shape[0] == raw_ids.shape[0]
    if scores is not None:
        assert scores.dim() == 1 and scores.shape[0] == raw_ids.shape[0]

    raw_ids_np = raw_ids.to(_KEY_DTYPE).cpu().numpy()
    values_np = values.to(_VALUE_DTYPE).cpu().numpy()
    scores_np = scores.to(_SCORE_DTYPE).cpu().numpy() if scores is not None else None

    # Bucket-sort by destination rank in O(N) instead of W mask scans. The
    # stable argsort groups same-rank entries together so per-rank slices
    # (views, not copies) are contiguous in the rank order. Empty shards
    # still produce zero-byte files because the slice is empty.
    keys_mod = raw_ids_np % world_size if world_size > 1 else np.zeros_like(raw_ids_np)
    order = np.argsort(keys_mod, kind="stable")
    sizes = np.bincount(keys_mod, minlength=world_size)
    ends = np.cumsum(sizes)
    starts = ends - sizes
    raw_ids_sorted = raw_ids_np[order]
    values_sorted = values_np[order]
    scores_sorted = scores_np[order] if scores_np is not None else None

    for rank in range(world_size):
        s, e = int(starts[rank]), int(ends[rank])
        with open(
            encode_checkpoint_file_path(
                save_path, table_name, rank, world_size, "keys"
            ),
            "wb",
        ) as f:
            if e > s:
                f.write(raw_ids_sorted[s:e].astype(np.int64).tobytes())
        with open(
            encode_checkpoint_file_path(
                save_path, table_name, rank, world_size, "values"
            ),
            "wb",
        ) as f:
            if e > s:
                f.write(values_sorted[s:e].astype(np.float32).tobytes())
        if scores_sorted is not None:
            with open(
                encode_checkpoint_file_path(
                    save_path, table_name, rank, world_size, "scores"
                ),
                "wb",
            ) as f:
                if e > s:
                    f.write(scores_sorted[s:e].astype(np.int64).tobytes())


def _copy_dcp_dir(src: str, dst: str) -> None:
    """Byte-copy a DCP directory (model/ or optimizer/)."""
    if not os.path.isdir(src):
        return
    os.makedirs(dst, exist_ok=True)
    for fn in os.listdir(src):
        sp = os.path.join(src, fn)
        dp = os.path.join(dst, fn)
        if os.path.isfile(sp):
            shutil.copyfile(sp, dp)


def convert(
    source_checkpoint_path: str,
    source_pipeline_config_path: str,
    target_pipeline_config_path: str,
    save_dir: str,
    world_size: Optional[int],
) -> None:
    """Top-level conversion routine. See module docstring."""
    if not os.path.isdir(source_checkpoint_path):
        raise RuntimeError(
            f"source_checkpoint_path '{source_checkpoint_path}' is not a directory."
        )
    source_model_dir = os.path.join(source_checkpoint_path, "model")
    source_opt_dir = os.path.join(source_checkpoint_path, "optimizer")
    if not os.path.isdir(source_model_dir):
        raise RuntimeError(f"Missing model/ subdir under {source_checkpoint_path}.")

    # Import dynamicemb here so the module stays importable in environments
    # without it (CI test discovery). Raises a clear ImportError if missing.
    from dynamicemb.batched_dynamicemb_tables import encode_meta_json_file_path

    out_dir = os.path.join(save_dir, "model.ckpt-0")
    # Fail loudly rather than silently mixing fresh shards with stale ones --
    # per-rank shard filenames embed ``world_size_W``, so a re-run with a
    # different ``--world_size`` would leave both sets co-resident.
    if os.path.exists(out_dir):
        raise RuntimeError(
            f"Output directory {out_dir} already exists. Remove it before re-running."
        )
    os.makedirs(out_dir, exist_ok=False)

    # 1. Auto-detect world_size from the source DCP if not provided.
    if world_size is None:
        world_size = checkpoint_util._ckpt_world_size(source_model_dir)
        logger.info(f"Auto-detected source world_size={world_size}.")
    if world_size < 1:
        raise ValueError(f"--world_size must be >= 1, got {world_size}.")

    # 2. Build source + target models on CPU.
    src_pipeline = config_util.load_pipeline_config(source_pipeline_config_path)
    tgt_pipeline = config_util.load_pipeline_config(target_pipeline_config_path)

    src_features = _create_features(
        list(src_pipeline.feature_configs), src_pipeline.data_config
    )
    src_model = _create_model(
        src_pipeline.model_config,
        src_features,
        list(src_pipeline.data_config.label_fields),
        sample_weights=list(src_pipeline.data_config.sample_weight_fields),
    )
    src_model = TrainWrapper(src_model)

    tgt_features = _create_features(
        list(tgt_pipeline.feature_configs), tgt_pipeline.data_config
    )
    tgt_model = _create_model(
        tgt_pipeline.model_config,
        tgt_features,
        list(tgt_pipeline.data_config.label_fields),
        sample_weights=list(tgt_pipeline.data_config.sample_weight_fields),
    )
    tgt_model = TrainWrapper(tgt_model)

    # 3. Discover ZCH tables in source and dynamicemb tables in target.
    src_zch = _find_zch_tables(src_model)
    tgt_demb = _find_dynamicemb_tables(tgt_model)

    if not src_zch:
        raise RuntimeError(
            "No ZCH-wrapped tables found in the source model. Check that "
            "the source pipeline.config has features with zch{} set."
        )
    if not tgt_demb:
        raise RuntimeError(
            "No dynamicemb-bound tables found in the target model. Check "
            "that the target pipeline.config has features with dynamicemb{} set."
        )

    # 4. Match by (group_key, emb_name), validate dim equality.
    matched: Dict[Tuple[str, str], Tuple[_SourceZchTable, _TargetDynamicEmbTable]] = {}
    for key, demb in tgt_demb.items():
        if key not in src_zch:
            raise RuntimeError(
                f"Target dynamicemb table {key} has no matching ZCH "
                f"table in source. Source ZCH tables: {sorted(src_zch)}."
            )
        zch = src_zch[key]
        if demb.embedding_dim != zch.embedding_dim:
            raise RuntimeError(
                f"Table {key}: source dim={zch.embedding_dim} vs target "
                f"dim={demb.embedding_dim}."
            )
        if demb.dist_type != "roundrobin":
            raise RuntimeError(
                f"Table {key}: target dist_type={demb.dist_type!r} not "
                "supported. Only 'roundrobin' is implemented in v1."
            )
        matched[key] = (zch, demb)

    # 5. Per-table conversion.
    dynamicemb_load_table_names: Dict[str, List[str]] = defaultdict(list)

    for key, (zch, demb) in matched.items():
        group_key, name = key
        logger.info(
            f"Converting table '{name}' under group '{group_key}' "
            f"(zch_size={zch.zch_size})..."
        )

        # 5a. Load MCH state + EBC weight in one shot.
        canonical_sorted = zch.mch_prefix + "._mch_sorted_raw_ids"
        canonical_remapped = zch.mch_prefix + "._mch_remapped_ids_mapping"
        canonical_metadata = [
            zch.mch_prefix + "." + b for b in zch.metadata_buffer_names
        ]
        canonical_weight = zch.weight_fqn

        requested = [canonical_sorted, canonical_remapped, canonical_weight] + list(
            canonical_metadata
        )
        loaded = _load_dcp_subset(source_model_dir, requested)
        sorted_raw = loaded.get(canonical_sorted)
        remapped_global = loaded.get(canonical_remapped)
        weight = loaded.get(canonical_weight)
        if sorted_raw is None or remapped_global is None or weight is None:
            raise RuntimeError(
                f"Table {key}: missing required tensors in source checkpoint. "
                f"Found: sorted_raw={'ok' if sorted_raw is not None else 'MISSING'}, "
                f"remapped={'ok' if remapped_global is not None else 'MISSING'}, "
                f"weight={'ok' if weight is not None else 'MISSING'}."
            )
        metadata_tensors = {
            b: loaded[zch.mch_prefix + "." + b]
            for b in zch.metadata_buffer_names
            if loaded.get(zch.mch_prefix + "." + b) is not None
        }

        valid_mask = sorted_raw != _IINFO_MAX
        raw_ids = sorted_raw[valid_mask].to(_KEY_DTYPE)
        remapped = remapped_global[valid_mask].to(torch.long)
        if int(remapped.max().item() if remapped.numel() else -1) >= weight.shape[0]:
            raise RuntimeError(
                f"Table {key}: remapped id {int(remapped.max().item())} "
                f"out of range for weight shape {tuple(weight.shape)}."
            )
        values = weight.index_select(0, remapped).to(_VALUE_DTYPE)

        # 5b. Scores (may be None for TIMESTAMP/CUSTOMIZED/NO_EVICTION).
        scores = _derive_scores(
            metadata_tensors,
            zch.eviction_policy,
            demb.score_strategy,
            valid_mask,
        )

        # 5c. Write the dynamicemb files for this table.
        save_path = os.path.join(out_dir, "dynamicemb", demb.full_mod_path)
        os.makedirs(save_path, exist_ok=True)
        with open(encode_meta_json_file_path(save_path, name), "w") as f:
            f.write(json.dumps({}))

        _gather_and_shard_writes(
            raw_ids=raw_ids,
            values=values,
            scores=scores,
            table_name=name,
            save_path=save_path,
            world_size=world_size,
        )

        dynamicemb_load_table_names[demb.full_mod_path].append(name)
        has_scores = "yes" if scores is not None else "no"
        logger.info(
            f"  table '{name}': wrote {raw_ids.shape[0]} entries across "
            f"{world_size} ranks (scores={has_scores})."
        )

    # 6. Byte-copy model/ and optimizer/ verbatim. No filtering.
    _copy_dcp_dir(source_model_dir, os.path.join(out_dir, "model"))
    has_optim = os.path.isdir(source_opt_dir)
    if has_optim:
        _copy_dcp_dir(source_opt_dir, os.path.join(out_dir, "optimizer"))

    # Copy plan if present (advisory).
    plan_src = os.path.join(source_checkpoint_path, "plan")
    if os.path.isfile(plan_src):
        shutil.copyfile(plan_src, os.path.join(out_dir, "plan"))

    # 7. Top-level meta. dynamicemb_load_optim is hard-coded false in v1
    # since we don't migrate MCH-EBC optimizer state into dynamicemb opt_values.
    meta = {
        "load_model": True,
        "load_optim": has_optim,
        "dynamicemb_load_table_names": dict(dynamicemb_load_table_names),
        "dynamicemb_load_optim": False,
    }
    with open(os.path.join(out_dir, "meta"), "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Conversion complete. Output: {out_dir}.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a ZCH-based tzrec checkpoint to a dynamicemb-based "
            "checkpoint. The source pipeline.config has features with "
            "zch{}; the target pipeline.config replaces those with "
            "dynamicemb{}."
        )
    )
    parser.add_argument(
        "--source_checkpoint_path",
        type=str,
        required=True,
        help="Path to source checkpoint directory (e.g. .../model.ckpt-N).",
    )
    parser.add_argument(
        "--source_pipeline_config_path",
        type=str,
        required=True,
        help="Path to source pipeline.config (ZCH version).",
    )
    parser.add_argument(
        "--target_pipeline_config_path",
        type=str,
        required=True,
        help="Path to target pipeline.config (dynamicemb version).",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Output directory; writes <save_dir>/model.ckpt-0/.",
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=None,
        help=(
            "World size for the rank-sharded output files. Defaults to the "
            "source checkpoint's world size (auto-detected from .distcp files)."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    convert(
        source_checkpoint_path=args.source_checkpoint_path,
        source_pipeline_config_path=args.source_pipeline_config_path,
        target_pipeline_config_path=args.target_pipeline_config_path,
        save_dir=args.save_dir,
        world_size=args.world_size,
    )
