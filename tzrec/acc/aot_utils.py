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


def _prune_unused_param_and_buffer(
    gm: torch.fx.GraphModule,
) -> torch.fx.GraphModule:
    """Prune unused parameters and buffers in GraphModule."""
    # Import here to avoid circular dependency
    from tzrec.utils.export_util import _prune_unused_param_and_buffer

    return _prune_unused_param_and_buffer(gm)


def _build_dynamic_shapes(
    data: Dict[str, torch.Tensor],
) -> Dict[str, Dict[int, torch.export.Dim]]:
    """Build dynamic shapes for the full model input.

    Args:
        data: input tensor dict from Batch.to_dict().

    Returns:
        dynamic_shapes dict for torch.export.export().
    """
    batch = torch.export.Dim("batch", min=1, max=499999999)
    dynamic_shapes = {}

    for key, tensor in data.items():
        if tensor.dim() == 0:
            # Scalar tensors (e.g., batch_size for INPUT_TILE) - no dynamic dims
            continue
        elif key.endswith(".lengths") or key.endswith(".key_lengths"):
            # Lengths tensors: (batch,) or (batch * num_features,)
            dynamic_shapes[key] = {0: batch}
        elif key.endswith(".values"):
            # Could be dense (batch, dim) or sparse (nnz,)
            if tensor.dim() == 1:
                # Sparse values: data-dependent size
                dim_name = key.replace(".", "_").replace("-", "_")
                dynamic_shapes[key] = {
                    0: torch.export.Dim(f"nnz_{dim_name}", min=1, max=999999993)
                }
            else:
                # Dense values: (batch, dim) or (batch, seq_len, dim)
                dynamic_shapes[key] = {0: batch}
        elif key.endswith(".weights"):
            # Sparse weights: same size as corresponding .values
            dim_name = key.replace(".", "_").replace("-", "_")
            dynamic_shapes[key] = {
                0: torch.export.Dim(f"nnz_{dim_name}", min=1, max=999999993)
            }
        else:
            # Labels, sample_weights, etc.: (batch,) or (batch, dim)
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
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    graph_dir = os.path.join(save_dir, "graph")
    if not os.path.exists(graph_dir):
        os.makedirs(graph_dir)

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
            full_graph.erase_node(node)

    full_gm = torch.fx.GraphModule(model, full_graph)
    full_gm.graph.eliminate_dead_code()
    full_gm = _prune_unused_param_and_buffer(full_gm)

    with open(os.path.join(graph_dir, "gm_unified.graph"), "w") as f:
        f.write(str(full_gm.graph))
    with open(os.path.join(save_dir, "gm_unified.code"), "w") as f:
        f.write(full_gm.code)

    # Verify the unified model produces correct output
    result = full_gm(data, device=device)
    result_info = {k: (v.size(), v.dtype) for k, v in result.items()}
    logger.info(f"Unified Model Outputs: {result_info}")

    # Build dynamic shapes for the full input
    dynamic_shapes = _build_dynamic_shapes(data)
    logger.info("dynamic shapes=%s" % dynamic_shapes)

    # Export with torch.export
    logger.info("exporting unified model with torch.export...")
    with torch._inductor.config.patch(
        {"unsafe_ignore_unsupported_triton_autotune_args": True}
    ):
        exported_pg = torch.export.export(
            full_gm,
            args=(data,),
            kwargs={"device": device},
            dynamic_shapes={"data": dynamic_shapes},
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
