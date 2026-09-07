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

"""Writes what a serving runtime needs beside the HuggingFace weights.

The HuggingFace config is rewritten into composite shape, with the backbone
nested under ``text_config``. That is not cosmetic: it names the backbone to a
runtime that composes an arbitrary causal LM behind one registered
architecture, and it is what makes an engine treat the model as carrying a
second modality, which is exactly what a projected slot is.

The front-end is exported whether or not any slot is projected, as a standard
tzrec model directory so the online processor loads it unchanged: the scripted
module, ``fg.json``, ``pipeline.config`` and ``model_acc.json``. Lookup is a
stage of its own. By default the artifact carries the slot tables and takes
raw ids; under ``USE_DISTRIBUTED_EMBEDDING=1`` the processor's embedding stage
owns them, the module becomes the dense stage that reads its looked-up rows,
and the tables ship beside it as the sparse npz files that stage already
loads.
"""

import copy
import glob
import json
import os
from typing import Any, Dict, List, Optional, Tuple, cast

import torch
from torch import nn
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollectionInterface,
    EmbeddingCollectionInterface,
)
from torchrec.modules.mc_embedding_modules import (
    ManagedCollisionEmbeddingBagCollection,
    ManagedCollisionEmbeddingCollection,
)

from tzrec.acc import utils as acc_utils
from tzrec.features.feature import BaseFeature, create_feature_configs, create_fg_json
from tzrec.prompt.frontend import (
    OUT_CU_SEQLENS,
    OUT_HOLE_KEYS,
    OUT_HOLE_POSITIONS,
    OUT_HOLE_SLOT_COUNTS,
    OUT_INPUT_IDS,
    OUT_RESPONSE_LENGTHS,
    OUT_SLOT_EMBEDS,
    PromptAssembler,
    PromptFrontEnd,
    SlotTable,
    host_lengths_key,
)
from tzrec.prompt.types import CompiledPrompt, FillMode, SlotSeg
from tzrec.protos.model_pb2 import FeatureGroupType
from tzrec.protos.pipeline_pb2 import EasyRecConfig
from tzrec.utils import config_util
from tzrec.utils.logging_util import logger

SERVING_ARCH = "PromptGenRecForCausalLM"
SERVING_MODEL_TYPE = "prompt_genrec"
FRONTEND_DIR = "frontend"
SCRIPTED_MODEL_FILENAME = "scripted_model.pt"
LOOKUP_ARTIFACT = "artifact"
LOOKUP_HOST = "host"
_CONFIG = "config.json"
_SPARSE_DIR = "sparse"


def write_composite_config(export_dir: str) -> None:
    """Rewrite ``config.json`` so the backbone sits under ``text_config``.

    Args:
        export_dir: the HuggingFace export directory.
    """
    path = os.path.join(export_dir, _CONFIG)
    with open(path, "r") as f:
        backbone: Dict[str, Any] = json.load(f)
    if backbone.get("model_type") == SERVING_MODEL_TYPE:
        return
    composite = {
        "architectures": [SERVING_ARCH],
        "model_type": SERVING_MODEL_TYPE,
        "text_config": backbone,
    }
    # a runtime that reads only the outer config still needs to size its cache
    for key in ("vocab_size", "hidden_size", "num_hidden_layers", "torch_dtype"):
        if key in backbone:
            composite[key] = backbone[key]
    with open(path, "w") as f:
        json.dump(composite, f, indent=2)
    logger.info(
        f"wrote a composite config naming backbone "
        f"{backbone.get('architectures', ['?'])[0]} under {SERVING_ARCH}."
    )


def _feature_tables(embedding_group: nn.Module) -> Dict[str, Tuple[torch.Tensor, str]]:
    """Map every feature with a plain table to its weight and pooling mode.

    Managed-collision tables are skipped: their ids are remapped before the
    lookup, which a plain ``EmbeddingBag`` cannot reproduce.
    """
    collision_prefixes = [
        name + "."
        for name, module in embedding_group.named_modules()
        if isinstance(
            module,
            (
                ManagedCollisionEmbeddingBagCollection,
                ManagedCollisionEmbeddingCollection,
            ),
        )
    ]
    tables: Dict[str, Tuple[torch.Tensor, str]] = {}
    for name, module in embedding_group.named_modules():
        if any(name.startswith(prefix) for prefix in collision_prefixes):
            continue
        if isinstance(module, EmbeddingBagCollectionInterface):
            weights = module.state_dict()
            for cfg in module.embedding_bag_configs():
                weight = weights[f"embedding_bags.{cfg.name}.weight"]
                for feature_name in cfg.feature_names:
                    tables.setdefault(
                        feature_name, (weight, str(cfg.pooling.value).lower())
                    )
        elif isinstance(module, EmbeddingCollectionInterface):
            # a multi-value item pools its codes with the feature's own
            # pooling; a single-value item is that pooling over one row
            weights = module.state_dict()
            for cfg in module.embedding_configs():
                weight = weights[f"embeddings.{cfg.name}.weight"]
                for feature_name in cfg.feature_names:
                    tables.setdefault(feature_name, (weight, ""))
    return tables


def _is_multi_valued(feature: BaseFeature) -> bool:
    """Whether the parser emits ``key_lengths`` for this feature."""
    return feature.is_sequence and feature.value_dim != 1


def build_slot_tables(
    model: nn.Module, compiled_prompt: CompiledPrompt, features: List[BaseFeature]
) -> List[SlotTable]:
    """One ``SlotTable`` per projected slot, filled from the restored model.

    Args:
        model: the restored genrec model, whose ``embedding_group`` holds the
            trained tables.
        compiled_prompt: the compiled prompt.
        features: every created feature.

    Returns:
        The tables in ``projected_slots`` order.

    Raises:
        ValueError: a member has no plain table the artifact could carry.
    """
    tables = _feature_tables(model.embedding_group)
    by_name = {feature.name: feature for feature in features}
    result: List[SlotTable] = []
    for seg in compiled_prompt.prompt_plan.projected_slots:
        rows: List[int] = []
        dims: List[int] = []
        modes: List[str] = []
        key_length_keys: List[str] = []
        for name in seg.feature_names:
            feature = by_name[name]
            if not feature.is_sparse or name not in tables:
                raise ValueError(
                    f"prompt slot [{seg.name}] member [{name}] has no plain "
                    f"embedding table the front-end could carry (dense, "
                    f"managed-collision and dynamic tables cannot be). Export "
                    f"with USE_DISTRIBUTED_EMBEDDING=1 so the serving host "
                    f"owns the lookup."
                )
            weight, mode = tables[name]
            rows.append(int(weight.shape[0]))
            dims.append(int(weight.shape[1]))
            modes.append(mode or str(feature.pooling_type.value).lower())
            key_length_keys.append(
                f"{name}.key_lengths" if _is_multi_valued(feature) else ""
            )
        table = SlotTable(
            [f"{name}.values" for name in seg.feature_names],
            [f"{name}.lengths" for name in seg.feature_names],
            key_length_keys,
            rows,
            dims,
            seg.group_type == FeatureGroupType.JAGGED_SEQUENCE,
            modes,
        )
        with torch.no_grad():
            for module, name in zip(table.tables, seg.feature_names):
                bag = cast(nn.EmbeddingBag, module)
                bag.weight.copy_(tables[name][0].detach().to(bag.weight.dtype))
        result.append(table)
    return result


def host_embed_keys(
    compiled_prompt: CompiledPrompt,
) -> Tuple[List[List[str]], Dict[str, List[str]]]:
    """Batch keys a host lookup stage fills, and the ``dense_meta`` naming them.

    The names follow the dense graph a distributed-embedding export produces:
    a sequence member arrives as ``{feature}`` rows with ``{feature}__lengths``,
    and a pooled slot as one ``{slot}__ebc`` tensor concatenating its members.

    Returns:
        Per projected slot, the keys to concatenate on the feature axis, and
        the ``dense_meta.json`` the processor reads to produce them.
    """
    embed_keys: List[List[str]] = []
    dense_meta: Dict[str, List[str]] = {"sequence__ec": []}
    for seg in compiled_prompt.prompt_plan.projected_slots:
        if seg.group_type == FeatureGroupType.JAGGED_SEQUENCE:
            embed_keys.append(list(seg.feature_names))
            for name in seg.feature_names:
                dense_meta["sequence__ec"].extend(
                    [f"{name}__ec", host_lengths_key(name)]
                )
        else:
            key = f"{seg.name}__ebc"
            embed_keys.append([key])
            dense_meta[key] = [f"{name}__ebc" for name in seg.feature_names]
    return embed_keys, dense_meta


def _frontend_inputs(
    compiled_prompt: CompiledPrompt,
    features: List[BaseFeature],
    lookup: str,
    embed_keys: List[List[str]],
) -> List[str]:
    """Every batch key the exported module reads, for the serving contract."""
    by_name = {feature.name: feature for feature in features}
    inputs: List[str] = []

    def raw(name: str) -> None:
        inputs.append(f"{name}.values")
        inputs.append(f"{name}.lengths")
        if _is_multi_valued(by_name[name]):
            inputs.append(f"{name}.key_lengths")

    plan = compiled_prompt.prompt_plan
    for seg in plan.segments:
        if isinstance(seg, SlotSeg) and seg.fill is FillMode.INLINE:
            raw(seg.feature_names[0])
    for seg, keys in zip(plan.projected_slots, embed_keys):
        for name in seg.feature_names:
            raw(name)
        if lookup == LOOKUP_HOST:
            inputs.extend(keys)
            if seg.group_type == FeatureGroupType.JAGGED_SEQUENCE:
                inputs.extend(host_lengths_key(name) for name in seg.feature_names)
    return list(dict.fromkeys(inputs))


def build_front_end(
    model: nn.Module,
    compiled_prompt: CompiledPrompt,
    features: List[BaseFeature],
    carry_tables: bool,
) -> Tuple[PromptFrontEnd, Dict[str, Any], Optional[Dict[str, List[str]]]]:
    """Assemble the serving module from a restored model's parts.

    The projections are the trained modules themselves, by reference, so the
    artifact carries the weights the model learned rather than a copy that can
    drift from them.

    Args:
        model: the restored genrec model.
        compiled_prompt: the compiled prompt.
        features: every created feature.
        carry_tables: whether the artifact holds the slot tables, or expects a
            host lookup stage to hand it rows.

    Returns:
        The front-end, the ``frontend`` block of the serving contract, and the
        ``dense_meta`` a host lookup stage needs (None when tables are carried).
    """
    plan = compiled_prompt.prompt_plan
    sid_space = compiled_prompt.sid_space
    assembler = PromptAssembler(
        plan,
        sid_space,
        features_are_dense={f.name: not f.is_sparse for f in features},
        features_are_multi_valued={f.name: _is_multi_valued(f) for f in features},
        plan_hash=compiled_prompt.plan_hash,
        # serving has no answer to assemble; the LM generates it
        include_response=False,
    )
    projections = list(model._slot_projections)
    if carry_tables:
        lookup = LOOKUP_ARTIFACT
        tables: Optional[List[SlotTable]] = build_slot_tables(
            model, compiled_prompt, features
        )
        embed_keys: List[List[str]] = [[] for _ in plan.projected_slots]
        dense_meta: Optional[Dict[str, List[str]]] = None
    else:
        lookup = LOOKUP_HOST
        tables = None
        embed_keys, dense_meta = host_embed_keys(compiled_prompt)
    front_end = PromptFrontEnd(
        assembler,
        projections,
        embed_keys,
        tables=tables,
        vocab_hash=compiled_prompt.vocab_hash,
        plan_hash=compiled_prompt.plan_hash,
        bundle_uuid=sid_space.bundle_uuid if sid_space is not None else "",
    )
    meta = {
        "dir": FRONTEND_DIR,
        "model": SCRIPTED_MODEL_FILENAME,
        "lookup": lookup,
        "inputs": _frontend_inputs(compiled_prompt, features, lookup, embed_keys),
        "outputs": [
            OUT_INPUT_IDS,
            OUT_CU_SEQLENS,
            OUT_HOLE_POSITIONS,
            OUT_HOLE_KEYS,
            OUT_HOLE_SLOT_COUNTS,
            OUT_SLOT_EMBEDS,
            OUT_RESPONSE_LENGTHS,
        ],
    }
    return front_end, meta, dense_meta


def write_front_end_dir(
    front_end: PromptFrontEnd,
    pipeline_config: EasyRecConfig,
    features: List[BaseFeature],
    export_dir: str,
    dense_meta: Optional[Dict[str, List[str]]] = None,
) -> str:
    """Script the front-end into ``frontend/``, laid out as a tzrec model dir.

    Args:
        front_end: the module to export.
        pipeline_config: the pipeline config, whose feature configs are
            rewritten beside the copied assets.
        features: every created feature.
        export_dir: the HuggingFace export directory.
        dense_meta: the processor's dense-stage input map, written when a host
            lookup stage feeds the module.

    Returns:
        The ``frontend/`` directory written.
    """
    frontend_dir = os.path.join(export_dir, FRONTEND_DIR)
    os.makedirs(frontend_dir, exist_ok=True)
    torch.jit.script(front_end.eval()).save(
        os.path.join(frontend_dir, SCRIPTED_MODEL_FILENAME)
    )

    feature_configs = create_feature_configs(features, asset_dir=frontend_dir)
    served_config = copy.deepcopy(pipeline_config)
    served_config.ClearField("feature_configs")
    served_config.feature_configs.extend(feature_configs)
    config_util.save_message(
        served_config, os.path.join(frontend_dir, "pipeline.config")
    )
    with open(os.path.join(frontend_dir, "fg.json"), "w") as f:
        json.dump(create_fg_json(features, asset_dir=frontend_dir), f, indent=4)
    with open(os.path.join(frontend_dir, "model_acc.json"), "w") as f:
        json.dump(acc_utils.export_acc_config(), f, indent=4)
    if dense_meta is not None:
        with open(os.path.join(frontend_dir, "dense_meta.json"), "w") as f:
            json.dump(dense_meta, f, indent=4)
    logger.info(f"wrote the scripted prompt front-end to {frontend_dir}.")
    return frontend_dir


def export_sparse_tables(
    wrapped_model: nn.Module, checkpoint_path: str, frontend_dir: str
) -> None:
    """Write the slot tables as the sparse npz files a host lookup stage loads.

    Single rank, in the layout ``export_distributed_embedding`` produces.

    Args:
        wrapped_model: the restored model under its inference wrapper, so the
            table names match the checkpoint's.
        checkpoint_path: the checkpoint, read for dynamic tables.
        frontend_dir: the ``frontend/`` directory to write ``sparse/`` into.
    """
    from tzrec.utils import export_util, npz_util

    bag_info, emb_info = export_util._get_sparse_table_to_embedding_info(wrapped_model)
    local, dynamic, emb_meta, feat_meta = export_util._get_sparse_embedding_tensor(
        wrapped_model, checkpoint_path, emb_info, bag_info
    )
    sparse_dir = os.path.join(frontend_dir, _SPARSE_DIR)
    os.makedirs(sparse_dir, exist_ok=True)
    shard = "sparse_embeddings-00-of-01"
    npz_util.savez_streaming(os.path.join(sparse_dir, f"{shard}.npz"), local)
    if dynamic:
        npz_util.savez_streaming(
            os.path.join(sparse_dir, "sparse_dynamic_embedding-00-of-01.npz"), dynamic
        )
    with open(os.path.join(sparse_dir, f"{shard}.json"), "w") as f:
        json.dump(emb_meta, f, indent=4)
    with open(os.path.join(sparse_dir, "sparse_features.json"), "w") as f:
        json.dump(feat_meta, f, indent=4)
    shards = []
    for path in glob.glob(os.path.join(sparse_dir, "sparse_embeddings*.json")):
        with open(path, "r") as f:
            shards.append(json.load(f))
    with open(os.path.join(sparse_dir, "sparse_embedding.json"), "w") as f:
        json.dump(export_util._merge_sharded_embedding_json(shards), f, indent=4)
    logger.info(f"wrote {len(local)} slot table(s) to {sparse_dir}.")
