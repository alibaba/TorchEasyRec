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


import os
from typing import Any, Dict, Union

import torch
from torch import nn
from torchrec.distributed.train_pipeline.utils import Tracer

from tzrec.models.model import CombinedModelWrapper, UnifiedAOTIModelWrapper
from tzrec.utils.fx_util import (
    fx_mark_keyed_tensor,
    fx_mark_seq_len,
    fx_mark_seq_tensor,
    fx_mark_tensor,
    symbolic_trace,
)
from tzrec.utils.logging_util import logger


def load_model_aot(
    model_path: str, device: torch.device
) -> Union[CombinedModelWrapper, UnifiedAOTIModelWrapper]:
    """Load AOTInductor model.

    Supports both unified (single AOTI) and legacy (sparse JIT + dense AOTI) models.

    Args:
        model_path (str): model directory.
        device (torch.device): model placement.

    Return:
        AOTInductor model wrapper.
    """
    sparse_model_path = os.path.join(model_path, "scripted_sparse_model.pt")
    aoti_model_path = os.path.join(model_path, "aoti_model.pt2")

    if os.path.exists(sparse_model_path):
        # Legacy two-stage path: sparse JIT + dense AOTI
        sparse_model: torch.jit.ScriptModule = torch.jit.load(
            sparse_model_path, map_location=device
        )
        dense_model: torch.export.pt2_archive._package.AOTICompiledModel = (
            torch._inductor.aoti_load_package(
                aoti_model_path,
                device_index=device.index,
            )
        )
        return CombinedModelWrapper(sparse_model, dense_model)
    else:
        # Unified single-model path
        model = torch._inductor.aoti_load_package(
            aoti_model_path,
            device_index=device.index,
        )
        return UnifiedAOTIModelWrapper(model)


def export_model_aot(
    sparse_model: nn.Module,
    dense_model: nn.Module,
    data: Dict[str, torch.Tensor],
    meta_info: Dict[str, Any],
    save_dir: str,
) -> str:
    """Export AOTInductor model.

    Args:
        sparse_model (nn.Module): the sparse model
        dense_model (nn.Module): the dense model
        data (Dict[str, torch.Tensor]): the test data
        meta_info (Dict[str, Any]): split meta info
        save_dir (str): model save dir
    """
    sparse_output, _ = sparse_model(data, "cuda:0")
    sparse_model_traced = symbolic_trace(sparse_model)

    with open(os.path.join(save_dir, "gm_sparse.code"), "w") as f:
        f.write(sparse_model_traced.code)
    sparse_model_scripted = torch.jit.script(sparse_model_traced)
    sparse_model_scripted.save(os.path.join(save_dir, "scripted_sparse_model.pt"))

    batch = torch.export.Dim("batch", min=1, max=499999999)
    dynamic_shapes = {}
    seq_tensor_names = meta_info.get("seq_tensor_names", [])
    jagged_seq_tensor_names = meta_info.get("jagged_seq_tensor_names", [])
    for key in sparse_output.keys():
        if key in seq_tensor_names:
            dynamic_shapes[key] = {
                0: batch,
                1: torch.export.Dim(f"{key}__seq_len", min=1, max=999999993),
            }
        elif key in jagged_seq_tensor_names:
            dynamic_shapes[key] = {
                0: torch.export.Dim(f"{key}__batch", min=1, max=999999993)
            }
        else:
            dynamic_shapes[key] = {0: batch}

    logger.info("dynamic shapes=%s" % dynamic_shapes)

    # pre_hook requires running arbitrary code at runtime
    with torch._inductor.config.patch(
        {"unsafe_ignore_unsupported_triton_autotune_args": True}
    ):
        exported_pg = torch.export.export(
            dense_model, args=(sparse_output,), dynamic_shapes=(dynamic_shapes,)
        )
    # AsserScalar codegen is not correct.
    with torch._inductor.config.patch(
        {
            "scalar_asserts": False,
            "unsafe_ignore_unsupported_triton_autotune_args": True,
        }
    ):
        torch._inductor.aoti_compile_and_package(
            exported_pg,
            package_path=os.path.join(save_dir, "aoti_model.pt2"),
        )
    return save_dir


def _build_dynamic_shapes(
    data: Dict[str, torch.Tensor],
    features: Any,
    model_config: Any,
) -> Dict[str, Dict[int, torch.export.Dim]]:
    """Build dynamic shapes for the full model input.

    Uses structural knowledge from feature configs and model config:
    - .lengths → batch dim (always)
    - .values for non-sequence single-value sparse features → batch dim
    - .values for sequence features → data-dependent Dim, shared by features
      in the same FeatureGroupConfig (JAGGED_SEQUENCE/SEQUENCE) or SeqGroupConfig
    - .values without a .lengths sibling → batch dim (dense feature)
    - .weights → shares Dim with corresponding .values (same prefix)
    - .key_lengths → data-dependent (own Dim)
    - scalars → no dynamic dims
    - everything else (labels, sample_weights) → batch dim

    Args:
        data: input tensor dict from Batch.to_dict().
        features: list of BaseFeature from model._features.
        model_config: ModelConfig proto with feature_groups.

    Returns:
        dynamic_shapes dict for torch.export.export().
    """
    from tzrec.protos.model_pb2 import FeatureGroupType

    # Step 1: Group grouped sequence features by their sequence_name.
    # Features in the same SequenceFeature config share per-sample lengths,
    # so their .values always have the same nnz — they must share a Dim.
    # This takes precedence over FeatureGroupConfig because it reflects the
    # authoritative data structure (shared sequence), not model organization.
    feat_to_seq_dim_group: Dict[str, str] = {}
    seq_feat_names: set = set()
    for feat in features:
        if feat.is_sequence:
            seq_feat_names.add(feat.name)
            if getattr(feat, "_is_grouped_seq", False):
                seq_name = getattr(feat, "sequence_name", None)
                if seq_name:
                    feat_to_seq_dim_group[feat.name] = f"seq_{seq_name}"

    # Step 2: For standalone sequence features not yet grouped, fall back to
    # model_config.feature_groups structure.
    for fg in model_config.feature_groups:
        if fg.group_type == FeatureGroupType.JAGGED_SEQUENCE:
            # In JAGGED_SEQUENCE groups, all sequence features share a single
            # jagged structure → same nnz → share Dim.
            for name in fg.feature_names:
                if name in seq_feat_names and name not in feat_to_seq_dim_group:
                    feat_to_seq_dim_group[name] = f"fg_{fg.group_name}"
        # SEQUENCE (DIN-style) groups: standalone sequence features have
        # independent lengths, so don't auto-share. Only grouped sequence
        # features (from SequenceFeature config via sequence_groups) share nnz.
        for sg in fg.sequence_groups:
            for name in sg.feature_names:
                # Only sequence features in seq_groups share nnz — non-sequence
                # features mixed into seq_groups are candidates, not sequences.
                if name in seq_feat_names and name not in feat_to_seq_dim_group:
                    feat_to_seq_dim_group[name] = f"sg_{fg.group_name}_{sg.group_name}"

    # Step 3: Collect prefixes with .lengths siblings (sparse/sequence features)
    lengths_prefixes = set()
    for key in data:
        if key.endswith(".lengths"):
            lengths_prefixes.add(key[: -len(".lengths")])

    if not lengths_prefixes:
        raise ValueError(
            "Cannot infer batch size: no '.lengths' tensor found in input data. "
            "Unified AOTI export requires at least one sparse/sequence feature."
        )

    # Step 4: Build dynamic shapes
    batch = torch.export.Dim("batch", min=1, max=499999999)
    dynamic_shapes = {}
    group_to_dim: Dict[str, torch.export.Dim] = {}
    prefix_to_dim: Dict[str, torch.export.Dim] = {}
    dim_counter = 0

    for key, tensor in data.items():
        if tensor.dim() == 0:
            dynamic_shapes[key] = {}
            continue

        prefix = key
        for suffix in (".values", ".lengths", ".weights", ".key_lengths"):
            if key.endswith(suffix):
                prefix = key[: -len(suffix)]
                break

        is_sparse_values = key.endswith(".values") and prefix in lengths_prefixes
        is_sparse_weights = key.endswith(".weights") and prefix in lengths_prefixes
        is_key_lengths = key.endswith(".key_lengths")

        if is_sparse_values:
            if prefix in feat_to_seq_dim_group:
                # Sequence feature: share Dim with same-group features
                group = feat_to_seq_dim_group[prefix]
                if group not in group_to_dim:
                    group_to_dim[group] = torch.export.Dim(
                        f"g_{dim_counter}", min=1, max=999999993
                    )
                    dim_counter += 1
                dim = group_to_dim[group]
                dynamic_shapes[key] = {0: dim}
                prefix_to_dim[prefix] = dim
            else:
                # Multi-value non-sequence or ungrouped: own Dim
                dim = torch.export.Dim(f"g_{dim_counter}", min=1, max=999999993)
                dim_counter += 1
                dynamic_shapes[key] = {0: dim}
                prefix_to_dim[prefix] = dim
        elif is_sparse_weights:
            dim = prefix_to_dim.get(prefix)
            if dim is None:
                dim = torch.export.Dim(f"g_{dim_counter}", min=1, max=999999993)
                dim_counter += 1
                prefix_to_dim[prefix] = dim
            dynamic_shapes[key] = {0: dim}
        elif is_key_lengths:
            dim = torch.export.Dim(f"g_{dim_counter}", min=1, max=999999993)
            dim_counter += 1
            dynamic_shapes[key] = {0: dim}
        else:
            dynamic_shapes[key] = {0: batch}

    return dynamic_shapes


def export_unified_model_aot(
    model: nn.Module,
    data: Dict[str, torch.Tensor],
    save_dir: str,
) -> str:
    """Export a unified AOTInductor model (sparse+dense fused).

    Args:
        model (nn.Module): the full model (ScriptWrapper).
        data (Dict[str, torch.Tensor]): sample input data.
        save_dir (str): model save dir.
    """
    os.makedirs(save_dir, exist_ok=True)
    graph_dir = os.path.join(save_dir, "graph")
    os.makedirs(graph_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.set_is_inference(True)
    model.eval()

    # Trace the full model
    logger.info("tracing full model for unified AOTI export...")
    tracer = Tracer()
    full_graph = tracer.trace(model)

    with open(os.path.join(graph_dir, "gm_full.graph"), "w") as f:
        f.write(str(full_graph))

    # Remove fx_mark_* no-op nodes (they are split markers, not needed for unified)
    fx_mark_targets = {
        fx_mark_keyed_tensor,
        fx_mark_tensor,
        fx_mark_seq_tensor,
        fx_mark_seq_len,
    }
    for node in list(full_graph.nodes):
        if node.op == "call_function" and node.target in fx_mark_targets:
            if node.users:
                node.replace_all_uses_with(None)
            full_graph.erase_node(node)

    # Bake device into the graph as a string constant, removing it as an input.
    # torch.export doesn't support torch.device as input, but the graph's .to()
    # calls work fine with string device specs like "cuda:0".
    device_str = str(device)
    device_node = None
    for node in full_graph.nodes:
        if node.op == "placeholder" and node.target == "device":
            device_node = node
    if device_node is not None:
        for user in list(device_node.users):
            user.args = tuple(device_str if a is device_node else a for a in user.args)
            user.kwargs = {
                k: device_str if v is device_node else v for k, v in user.kwargs.items()
            }
        full_graph.erase_node(device_node)

    full_gm = torch.fx.GraphModule(model, full_graph)
    full_gm.graph.eliminate_dead_code()

    from tzrec.utils.export_util import _prune_unused_param_and_buffer

    full_gm = _prune_unused_param_and_buffer(full_gm)

    with open(os.path.join(graph_dir, "gm_unified.graph"), "w") as f:
        f.write(str(full_gm.graph))
    with open(os.path.join(save_dir, "gm_unified.code"), "w") as f:
        f.write(full_gm.code)

    # Move data to the target device (AOTI models always run on CUDA)
    data_on_device = {k: v.to(device) for k, v in data.items()}

    # Verify the unified model produces correct output
    result = full_gm(data_on_device)
    result_info = {k: (v.size(), v.dtype) for k, v in result.items()}
    logger.info(f"Unified Model Outputs: {result_info}")

    # Build dynamic shapes using feature metadata for correct Dim grouping
    dynamic_shapes = _build_dynamic_shapes(
        data,
        features=model._features,
        model_config=model.model._base_model_config,
    )
    logger.info("dynamic shapes=%s" % dynamic_shapes)

    # Export with torch.export
    logger.info("exporting unified model with torch.export...")
    with torch._inductor.config.patch(
        {"unsafe_ignore_unsupported_triton_autotune_args": True}
    ):
        exported_pg = torch.export.export(
            full_gm,
            args=(data_on_device,),
            dynamic_shapes=(dynamic_shapes,),
        )

    # Compile with AOTI
    logger.info("compiling unified model with AOTI...")
    with torch._inductor.config.patch(
        {
            "scalar_asserts": False,
            "unsafe_ignore_unsupported_triton_autotune_args": True,
        }
    ):
        torch._inductor.aoti_compile_and_package(
            exported_pg,
            package_path=os.path.join(save_dir, "aoti_model.pt2"),
        )

    logger.info("unified AOTI model exported to %s", save_dir)
    return save_dir
