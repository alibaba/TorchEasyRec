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
start from the trained embeddings, scores, and (when compatible) optimizer
states.

Output layout (under ``<save_dir>/model.ckpt-0/``)::

    model/                       # copied verbatim from source (loader skips
                                 # extra ZCH keys absent in the target model)
    optimizer/                   # copied verbatim from source
    dynamicemb/<mod_path>/<tbl>_emb_keys.rank_R.world_size_W
    dynamicemb/<mod_path>/<tbl>_emb_values.rank_R.world_size_W
    dynamicemb/<mod_path>/<tbl>_emb_scores.rank_R.world_size_W
    dynamicemb/<mod_path>/<tbl>_emb_opt_values.rank_R.world_size_W   # optional
    dynamicemb/<mod_path>/<tbl>_opt_args.json
    plan                         # copied if present (advisory)
    meta                         # {load_model, load_optim,
                                 #  dynamicemb_load_table_names,
                                 #  dynamicemb_load_optim}
"""

import argparse
import json
import os
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

# Score / value / key dtypes — match dynamicemb.types.
_KEY_DTYPE = torch.int64
_VALUE_DTYPE = torch.float32
_SCORE_DTYPE = torch.int64
_OPT_DTYPE = torch.float32
_IINFO_MAX = torch.iinfo(torch.int64).max


@dataclass
class _SourceZchTable:
    """One ZCH-wrapped logical table discovered in the source model.

    Fields capture everything we need to drive the conversion: where its
    state lives in the source DCP, its global zch_size, and which MCH
    metadata buffers exist (so we know the eviction policy).
    """

    emb_name: str
    embedding_dim: int
    zch_size: int
    mch_prefix: str
    weight_fqn: str
    metadata_buffer_names: List[str]
    eviction_policy: str  # "lfu" | "lru" | "distance_lfu" | "none"


@dataclass
class _TargetDynamicEmbTable:
    """One dynamicemb-bound logical table in the target model."""

    emb_name: str
    mod_path: str
    embedding_dim: int
    score_strategy: str
    dist_type: str
    opt_label: Optional[str] = None  # filled by inspecting source opt state


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


def _find_zch_tables(model: torch.nn.Module) -> Dict[str, _SourceZchTable]:
    """Walk source model and discover MCH-wrapped tables.

    Returns a dict keyed by logical embedding name (the EBC/EC table name,
    which is also the embedding-config ``name`` field tzrec uses to match
    source and target).
    """
    out: Dict[str, _SourceZchTable] = {}
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
            if tbl_name in out:
                # Two MC collections cannot share an embedding name; if they
                # do, tzrec is misconfigured -- fail loud.
                raise RuntimeError(
                    f"Duplicate ZCH-wrapped table name '{tbl_name}' found at "
                    f"both {out[tbl_name].mch_prefix} and {mch_prefix}."
                )
            if tbl_name not in name_to_dim:
                raise RuntimeError(
                    f"Table '{tbl_name}' has an MCH module but no matching "
                    f"embedding config under {mod_name}._embedding_module."
                )
            out[tbl_name] = _SourceZchTable(
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
) -> Dict[str, _TargetDynamicEmbTable]:
    """Mirror of create_dynamicemb_init_ckpt.py's walk to discover dynamicemb tables.

    Returns dict keyed by emb_name. Asserts no table appears under more
    than one module path (each emb_name -> one mod_path).
    """
    from dynamicemb.planner import DynamicEmbParameterConstraints

    out: Dict[str, _TargetDynamicEmbTable] = {}
    q: Queue = Queue()
    q.put(("", target_model))
    while not q.empty():
        path, m = q.get()
        if hasattr(m, "parameter_constraints"):
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
                if emb_name in out:
                    raise RuntimeError(
                        f"Dynamicemb table '{emb_name}' is bound under two "
                        f"distinct module paths "
                        f"({out[emb_name].mod_path} and {full_mod_path}); "
                        "this is not supported."
                    )
                out[emb_name] = _TargetDynamicEmbTable(
                    emb_name=emb_name,
                    mod_path=full_mod_path,
                    embedding_dim=int(opts.dim) if opts.dim else -1,
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


def _classify_source_optimizer_for_weight(
    opt_meta: Mapping[str, object], weight_fqn: str, zch_size: int, dim: int
) -> Tuple[str, Dict[str, str]]:
    """Infer the source optimizer type from per-weight state-dict keys.

    Returns ``(label, {state_name -> full_meta_fqn})``. ``label`` is one of
    ``{"sgd", "adagrad", "rowwise_adagrad", "adam", "unknown"}``.
    """
    # The optimizer state_dict, when DCP-saved, flattens to keys that contain
    # the param FQN (or its stripped form) plus the opt-state name. We match
    # on substring of the weight FQN.
    state_name_to_fqn: Dict[str, str] = {}
    for full_fqn, md in opt_meta.items():
        if not isinstance(md, TensorStorageMetadata):
            continue
        stripped = checkpoint_util._strip_dmp_prefix(full_fqn)
        if weight_fqn not in stripped:
            continue
        # The trailing component is the opt-state name. For example
        # ``state.<param_fqn>.exp_avg`` -> ``exp_avg``.
        tail = stripped.rsplit(".", 1)[-1]
        # Filter to known opt-state names so we don't misread the .weight key.
        if tail in ("exp_avg", "exp_avg_sq", "sum", "momentum_buffer"):
            state_name_to_fqn[tail] = full_fqn

    names = set(state_name_to_fqn)
    if names == set():
        return "sgd", state_name_to_fqn
    if names == {"exp_avg", "exp_avg_sq"}:
        return "adam", state_name_to_fqn
    if names == {"sum"}:
        sum_fqn = state_name_to_fqn["sum"]
        sum_shape = tuple(opt_meta[sum_fqn].size)
        if sum_shape == (zch_size, dim):
            return "adagrad", state_name_to_fqn
        if sum_shape == (zch_size,):
            return "rowwise_adagrad", state_name_to_fqn
        logger.warning(
            f"Unexpected shape {sum_shape} for AdaGrad-style state at "
            f"{sum_fqn}; expected ({zch_size}, {dim}) or ({zch_size},). "
            "Treating as unknown."
        )
        return "unknown", state_name_to_fqn
    return "unknown", state_name_to_fqn


_SPARSE_OPT_LABELS = {
    "sgd_optimizer": "sgd",
    "adagrad_optimizer": "adagrad",
    "adam_optimizer": "adam",
    "rowwise_adagrad_optimizer": "rowwise_adagrad",
}


def _infer_target_opt_label_from_pipeline_config(pipeline_config) -> str:
    """Read pipeline_config.train_config.sparse_optimizer to label the target opt.

    tzrec wires the sparse-optimizer choice globally; that's the optimizer
    every embedding (incl. dynamicemb-bound tables) ends up using. Returns
    ``"unknown"`` for optimizers we don't know how to migrate (LARS, LAMB,
    RMSprop, Adadelta, partial-rowwise variants).
    """
    if not pipeline_config.HasField("train_config"):
        return "unknown"
    train_cfg = pipeline_config.train_config
    if not train_cfg.HasField("sparse_optimizer"):
        return "unknown"
    kind = train_cfg.sparse_optimizer.WhichOneof("optimizer")
    return _SPARSE_OPT_LABELS.get(kind, "unknown")


def _derive_scores(
    metadata_tensors: Dict[str, torch.Tensor],
    eviction_policy: str,
    target_score_strategy: str,
    valid_mask: torch.Tensor,
    init_score_offset: int,
) -> torch.Tensor:
    """Project MCH eviction metadata onto dynamicemb's int64 score.

    Direction is consistent in both systems (higher = keep longer). For
    LFU-target we prefer counts; for STEP/TIMESTAMP/CUSTOMIZED we prefer
    last_access. NO_EVICTION ignores the score, so we emit a constant.

    See plan §5 for the full mapping table. ``init_score_offset`` is added
    to every score after the base mapping (useful to shift LRU iter values
    into the wall-clock-ns range when target is TIMESTAMP).
    """
    target = (target_score_strategy or "").upper()
    n_valid = int(valid_mask.sum().item())

    if target == "NO_EVICTION":
        base = torch.zeros(n_valid, dtype=_SCORE_DTYPE)
        return base + int(init_score_offset)

    counts = metadata_tensors.get("_mch_counts")
    last = metadata_tensors.get("_mch_last_access_iter")

    if target == "LFU":
        preferred = counts if counts is not None else last
    else:
        # STEP / TIMESTAMP / CUSTOMIZED / unknown -> recency wins
        preferred = last if last is not None else counts

    if preferred is None:
        logger.warning(
            f"ZCH module has no MCH eviction metadata "
            f"(eviction_policy={eviction_policy!r}); falling back to score=0."
        )
        base = torch.zeros(n_valid, dtype=_SCORE_DTYPE)
    else:
        base = preferred[valid_mask].to(_SCORE_DTYPE)

    if target == "TIMESTAMP" and preferred is not None and base.numel() > 0:
        if int(base.max().item()) < 10**15 and init_score_offset == 0:
            logger.warning(
                "Target score_strategy=TIMESTAMP but mapped scores are far "
                "below typical device_timestamp() values (~1e18 ns). "
                "Converted entries will be evicted first when new entries "
                "are inserted. Consider passing --init_score_offset close "
                "to the current device_timestamp() to protect them until "
                "first touch."
            )
    return base + int(init_score_offset)


def _gather_and_shard_writes(
    raw_ids: torch.Tensor,
    values: torch.Tensor,
    scores: torch.Tensor,
    opt_values: Optional[torch.Tensor],
    table_name: str,
    save_path: str,
    world_size: int,
) -> None:
    """Shard converted entries by raw_id % world_size and write per-rank files.

    Empty shards still get zero-byte files so ``find_files`` sees the full
    expected count of W shards per item.
    """
    from dynamicemb.batched_dynamicemb_tables import encode_checkpoint_file_path

    assert raw_ids.dim() == 1
    assert values.dim() == 2 and values.shape[0] == raw_ids.shape[0]
    assert scores.dim() == 1 and scores.shape[0] == raw_ids.shape[0]
    if opt_values is not None:
        assert opt_values.dim() == 2 and opt_values.shape[0] == raw_ids.shape[0]

    raw_ids_np = raw_ids.to(_KEY_DTYPE).cpu().numpy()
    values_np = values.to(_VALUE_DTYPE).cpu().numpy()
    scores_np = scores.to(_SCORE_DTYPE).cpu().numpy()
    opt_np = opt_values.to(_OPT_DTYPE).cpu().numpy() if opt_values is not None else None

    keys_mod = raw_ids_np % world_size if world_size > 1 else np.zeros_like(raw_ids_np)
    for rank in range(world_size):
        mask = keys_mod == rank
        with open(
            encode_checkpoint_file_path(
                save_path, table_name, rank, world_size, "keys"
            ),
            "wb",
        ) as f:
            if mask.any():
                f.write(raw_ids_np[mask].astype(np.int64).tobytes())
        with open(
            encode_checkpoint_file_path(
                save_path, table_name, rank, world_size, "values"
            ),
            "wb",
        ) as f:
            if mask.any():
                f.write(values_np[mask].astype(np.float32).tobytes())
        with open(
            encode_checkpoint_file_path(
                save_path, table_name, rank, world_size, "scores"
            ),
            "wb",
        ) as f:
            if mask.any():
                f.write(scores_np[mask].astype(np.int64).tobytes())
        if opt_np is not None:
            with open(
                encode_checkpoint_file_path(
                    save_path, table_name, rank, world_size, "opt_values"
                ),
                "wb",
            ) as f:
                if mask.any():
                    f.write(opt_np[mask].astype(np.float32).tobytes())


def _gather_opt_rows(
    opt_state_tensors: Dict[str, torch.Tensor],
    label: str,
    remapped: torch.Tensor,
    dim: int,
) -> torch.Tensor:
    """Pack source optimizer state rows into dynamicemb's flat opt_values layout.

    Output shape: ``(N, opt_state_dim)`` where ``opt_state_dim`` is the
    checkpoint-on-disk width expected by dynamicemb (see
    ``dynamicemb.optimizer.get_optimizer_ckpt_state_dim``).
    """
    if label == "sgd":
        # opt_state_dim == 0 -> no opt file written; caller should not call us.
        return torch.empty((remapped.shape[0], 0), dtype=_OPT_DTYPE)
    if label == "adam":
        m = opt_state_tensors["exp_avg"][remapped].to(_OPT_DTYPE)
        v = opt_state_tensors["exp_avg_sq"][remapped].to(_OPT_DTYPE)
        return torch.cat([m, v], dim=1)
    if label == "adagrad":
        return opt_state_tensors["sum"][remapped].to(_OPT_DTYPE)
    if label == "rowwise_adagrad":
        # Per-row scalar -> reshape to (N, 1) so the on-disk layout matches
        # get_optimizer_ckpt_state_dim() == 1.
        return opt_state_tensors["sum"][remapped].to(_OPT_DTYPE).reshape(-1, 1)
    raise ValueError(f"Unsupported optimizer label '{label}' for opt-state packing")


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
    init_score_offset: int,
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
    os.makedirs(out_dir, exist_ok=True)

    # 1. Auto-detect world_size from the source DCP if not provided.
    if world_size is None:
        world_size = checkpoint_util._ckpt_world_size(source_model_dir)
        logger.info(f"Auto-detected source world_size={world_size}.")
    assert world_size >= 1

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

    # 4. Match by emb_name, validate dim equality.
    matched: Dict[str, Tuple[_SourceZchTable, _TargetDynamicEmbTable]] = {}
    for name, demb in tgt_demb.items():
        if name not in src_zch:
            raise RuntimeError(
                f"Target dynamicemb table '{name}' has no matching ZCH "
                f"table in source. Source ZCH tables: {sorted(src_zch)}."
            )
        zch = src_zch[name]
        if demb.embedding_dim > 0 and demb.embedding_dim != zch.embedding_dim:
            raise RuntimeError(
                f"Table '{name}': source dim={zch.embedding_dim} vs target "
                f"dim={demb.embedding_dim}."
            )
        if demb.dist_type != "roundrobin":
            raise RuntimeError(
                f"Table '{name}': target dist_type={demb.dist_type!r} not "
                "supported. Only 'roundrobin' is implemented in v1."
            )
        matched[name] = (zch, demb)

    # Per-table opt-label inference uses target pipeline_config's sparse
    # optimizer choice -- tzrec applies this uniformly to every embedding.
    target_opt_label = _infer_target_opt_label_from_pipeline_config(tgt_pipeline)
    logger.info(f"Target dynamicemb optimizer label: {target_opt_label}")

    # 5. Per-table conversion.
    opt_reader_meta = None
    if os.path.isdir(source_opt_dir):
        opt_reader_meta = FileSystemReader(path=source_opt_dir).read_metadata()

    # Whether *every* table got opt_values written -- needed to set the
    # top-level dynamicemb_load_optim flag honestly.
    all_tables_have_opt_values = True
    dynamicemb_load_table_names: Dict[str, List[str]] = defaultdict(list)

    for name, (zch, demb) in matched.items():
        logger.info(f"Converting table '{name}' (zch_size={zch.zch_size})...")

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
                f"Table '{name}': missing required tensors in source checkpoint. "
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
                f"Table '{name}': remapped id {int(remapped.max().item())} "
                f"out of range for weight shape {tuple(weight.shape)}."
            )
        values = weight.index_select(0, remapped).to(_VALUE_DTYPE)

        # 5b. Scores.
        scores = _derive_scores(
            metadata_tensors,
            zch.eviction_policy,
            demb.score_strategy,
            valid_mask,
            init_score_offset,
        )

        # 5c. Optimizer state -> opt_values (best effort).
        opt_values: Optional[torch.Tensor] = None
        if opt_reader_meta is not None and target_opt_label not in ("unknown", "sgd"):
            src_opt_label, name_to_fqn = _classify_source_optimizer_for_weight(
                opt_reader_meta.state_dict_metadata,
                zch.weight_fqn,
                zch.zch_size,
                zch.embedding_dim,
            )
            if src_opt_label != target_opt_label:
                logger.warning(
                    f"Table '{name}': source optimizer label '{src_opt_label}' "
                    f"!= target '{target_opt_label}'. Skipping opt-state migration "
                    "for this table -- dynamicemb will initialize fresh state on load."
                )
                all_tables_have_opt_values = False
            elif not name_to_fqn:
                logger.warning(
                    f"Table '{name}': no optimizer state tensors found in source "
                    "checkpoint for this weight; skipping opt-state migration."
                )
                all_tables_have_opt_values = False
            else:
                # Load each opt-state tensor (canonical form == stripped saved key).
                canonical_opt_suffixes = [
                    checkpoint_util._strip_dmp_prefix(fqn)
                    for fqn in name_to_fqn.values()
                ]
                opt_loaded = _load_dcp_subset(source_opt_dir, canonical_opt_suffixes)
                opt_state_tensors: Dict[str, torch.Tensor] = {}
                for state_name, full_fqn in name_to_fqn.items():
                    stripped = checkpoint_util._strip_dmp_prefix(full_fqn)
                    t = opt_loaded.get(stripped)
                    if t is None:
                        opt_state_tensors = {}
                        break
                    opt_state_tensors[state_name] = t
                if not opt_state_tensors:
                    logger.warning(
                        f"Table '{name}': opt-state tensors could not be loaded "
                        "from source DCP; skipping opt-state migration."
                    )
                    all_tables_have_opt_values = False
                else:
                    opt_values = _gather_opt_rows(
                        opt_state_tensors,
                        src_opt_label,
                        remapped,
                        zch.embedding_dim,
                    )
        else:
            # No source optimizer/, OR target is sgd/unknown -> no opt_values.
            if target_opt_label not in ("sgd", "unknown"):
                all_tables_have_opt_values = False

        # 5d. Write the dynamicemb files for this table.
        save_path = os.path.join(out_dir, "dynamicemb", demb.mod_path)
        os.makedirs(save_path, exist_ok=True)
        with open(encode_meta_json_file_path(save_path, name), "w") as f:
            f.write(json.dumps({}))

        _gather_and_shard_writes(
            raw_ids=raw_ids,
            values=values,
            scores=scores,
            opt_values=opt_values,
            table_name=name,
            save_path=save_path,
            world_size=world_size,
        )

        dynamicemb_load_table_names[demb.mod_path].append(name)
        has_opt = "yes" if opt_values is not None else "no"
        logger.info(
            f"  table '{name}': wrote {raw_ids.shape[0]} entries across "
            f"{world_size} ranks (opt_state={has_opt})."
        )

    # 6. Copy model/ and optimizer/ verbatim.
    _copy_dcp_dir(source_model_dir, os.path.join(out_dir, "model"))
    if os.path.isdir(source_opt_dir):
        _copy_dcp_dir(source_opt_dir, os.path.join(out_dir, "optimizer"))

    # Copy plan if present (advisory).
    plan_src = os.path.join(source_checkpoint_path, "plan")
    if os.path.isfile(plan_src):
        shutil.copyfile(plan_src, os.path.join(out_dir, "plan"))

    # 7. Top-level meta.
    meta = {
        "load_model": True,
        "load_optim": os.path.isdir(source_opt_dir),
        "dynamicemb_load_table_names": dict(dynamicemb_load_table_names),
        "dynamicemb_load_optim": (
            all_tables_have_opt_values and target_opt_label not in ("sgd", "unknown")
        ),
    }
    with open(os.path.join(out_dir, "meta"), "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(
        f"Conversion complete. Output: {out_dir}. "
        f"dynamicemb_load_optim={meta['dynamicemb_load_optim']}."
    )


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
    parser.add_argument(
        "--init_score_offset",
        type=int,
        default=0,
        help=(
            "Constant int64 added to every derived score. Useful for "
            "TIMESTAMP score_strategy targets (set close to current "
            "device_timestamp() to protect converted entries from being "
            "evicted by fresh inserts)."
        ),
    )
    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    args = _parse_args()
    convert(
        source_checkpoint_path=args.source_checkpoint_path,
        source_pipeline_config_path=args.source_pipeline_config_path,
        target_pipeline_config_path=args.target_pipeline_config_path,
        save_dir=args.save_dir,
        world_size=args.world_size,
        init_score_offset=args.init_score_offset,
    )
