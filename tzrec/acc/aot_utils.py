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
) -> tuple:
    """Build dynamic shapes for the full model input.

    Uses key-suffix structural knowledge to classify tensors:
    - .lengths → batch dim (always)
    - .values with a matching .lengths sibling → data-dependent (own Dim)
    - .values without a .lengths sibling → batch dim (dense feature)
    - .weights → data-dependent (own Dim)
    - .key_lengths → data-dependent (own Dim)
    - scalars → no dynamic dims
    - everything else (labels, sample_weights) → batch dim

    Args:
        data: input tensor dict from Batch.to_dict().

    Returns:
        Tuple of (dynamic_shapes dict, dim_name_map dict) where dim_name_map
        maps Dim name strings to Dim objects for constraint resolution.
    """
    # Collect prefixes that have a .lengths sibling — these are sparse/sequence
    lengths_prefixes = set()
    for key in data:
        if key.endswith(".lengths"):
            lengths_prefixes.add(key[: -len(".lengths")])

    if not lengths_prefixes:
        raise ValueError(
            "Cannot infer batch size: no '.lengths' tensor found in input data. "
            "Unified AOTI export requires at least one sparse/sequence feature."
        )

    batch = torch.export.Dim("batch", min=1, max=499999999)
    dynamic_shapes = {}
    dim_name_map: Dict[str, torch.export.Dim] = {"batch": batch}
    dim_counter = 0

    for key, tensor in data.items():
        if tensor.dim() == 0:
            dynamic_shapes[key] = {}
            continue

        # Determine prefix for matching .values/.lengths/.weights
        prefix = key
        for suffix in (".values", ".lengths", ".weights", ".key_lengths"):
            if key.endswith(suffix):
                prefix = key[: -len(suffix)]
                break

        is_sparse_values = key.endswith(".values") and prefix in lengths_prefixes
        is_sparse_weights = key.endswith(".weights") and prefix in lengths_prefixes
        is_key_lengths = key.endswith(".key_lengths")

        if is_sparse_values or is_sparse_weights or is_key_lengths:
            # Data-dependent dim — each gets its own independent Dim
            dim_name = f"s_{dim_counter}"
            dim = torch.export.Dim(dim_name, min=1, max=999999993)
            dynamic_shapes[key] = {0: dim}
            dim_name_map[dim_name] = dim
            dim_counter += 1
        else:
            # Batch dim: .lengths, dense .values, labels, sample_weights, etc.
            dynamic_shapes[key] = {0: batch}

    return dynamic_shapes, dim_name_map


def _apply_suggested_dim_fixes(
    dynamic_shapes: Dict[str, Dict[int, torch.export.Dim]],
    dim_name_map: Dict[str, torch.export.Dim],
    error_msg: str,
) -> bool:
    """Parse torch.export constraint violation and merge Dims as suggested.

    When torch.export detects that two independently-named Dims must always be
    equal (e.g., features in the same data group share nnz), it emits
    "Suggested fixes:" with lines like "s_0 = s_1". This function parses
    those lines and merges the Dims in dynamic_shapes.

    Args:
        dynamic_shapes: the dynamic shapes dict to modify in-place.
        dim_name_map: mapping from Dim name string to Dim object.
        error_msg: the UserError message from torch.export.

    Returns True if any fixes were applied, False otherwise.
    """
    import re

    # Extract suggested fixes section
    fixes_match = re.search(r"Suggested fixes:\s*\n((?:\s+\S+ = \S+\n?)+)", error_msg)
    if not fixes_match:
        return False

    fixes_text = fixes_match.group(1)
    # Parse lines like "  s_0 = s_1"
    fix_pairs = re.findall(r"(\S+)\s*=\s*(\S+)", fixes_text)
    if not fix_pairs:
        return False

    # For each fix pair, replace all occurrences of dim_b with dim_a
    applied = False
    for dim_name_a, dim_name_b in fix_pairs:
        dim_a = dim_name_map.get(dim_name_a)
        dim_b = dim_name_map.get(dim_name_b)
        if dim_a is None or dim_b is None:
            continue
        if dim_a is dim_b:
            continue
        # Replace dim_b with dim_a everywhere
        for key in dynamic_shapes:
            for axis, dim in list(dynamic_shapes[key].items()):
                if dim is dim_b:
                    dynamic_shapes[key][axis] = dim_a
                    applied = True
        # Update mapping so chained fixes work (s_0 = s_1, s_1 = s_2)
        dim_name_map[dim_name_b] = dim_a

    return applied


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

    # Build dynamic shapes for the full input
    dynamic_shapes, dim_name_map = _build_dynamic_shapes(data)
    logger.info("dynamic shapes=%s" % dynamic_shapes)

    # Export with torch.export, retrying if constraint violations suggest
    # merging Dims (features in the same data group share nnz).
    logger.info("exporting unified model with torch.export...")
    max_retries = 5
    exported_pg = None
    for attempt in range(max_retries):
        try:
            with torch._inductor.config.patch(
                {"unsafe_ignore_unsupported_triton_autotune_args": True}
            ):
                exported_pg = torch.export.export(
                    full_gm,
                    args=(data_on_device,),
                    dynamic_shapes=(dynamic_shapes,),
                )
            break
        except torch._dynamo.exc.UserError as e:
            error_msg = str(e)
            if _apply_suggested_dim_fixes(dynamic_shapes, dim_name_map, error_msg):
                logger.info(
                    "Applied suggested Dim fixes (attempt %d), retrying export...",
                    attempt + 1,
                )
            else:
                raise
    if exported_pg is None:
        raise RuntimeError(
            f"torch.export failed after {max_retries} constraint resolution attempts"
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
