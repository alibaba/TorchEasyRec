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
from typing import Any, Dict, Optional, Set, Union

import torch
from torch import nn

from tzrec.acc.utils import is_unified_aot_predict
from tzrec.models.model import (
    CombinedModelWrapper,
    CudaAutocastWrapper,
    UnifiedAOTIModelWrapper,
)
from tzrec.utils.fx_util import symbolic_trace
from tzrec.utils.logging_util import logger

# Eagerly register custom ops referenced by AOT-packaged models so that
# torch._inductor.aoti_load_package() can resolve them by name. AOT packages
# reference ops via their qualified name (e.g. ``tzrec::cutlass_hstu_mha_fwd``)
# and PyTorch only knows about an op once its registering module has been
# imported. Wrap in try/except so this stays optional for environments
# without the corresponding native dependencies installed.
try:
    from tzrec.ops._cuda import cutlass_hstu_attention  # noqa: F401
except ImportError:
    logger.debug("cutlass_hstu_attention not available; skipping op registration")


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
    aoti_model_path = os.path.join(model_path, "aoti", "aoti_model.pt2")

    if is_unified_aot_predict(model_path):
        # Unified single-model path
        model = torch._inductor.aoti_load_package(
            aoti_model_path,
            device_index=device.index,
        )
        return UnifiedAOTIModelWrapper(model)
    else:
        # Legacy two-stage path: sparse JIT + dense AOTI.
        # Disable TensorExpr fuser: its lazy first-call compile is not
        # thread-safe under the predict worker pool.
        torch._C._jit_set_texpr_fuser_enabled(False)
        sparse_model: torch.jit.ScriptModule = torch.jit.load(
            os.path.join(model_path, "scripted_sparse_model.pt"),
            map_location=device,
        )
        dense_model: torch.export.pt2_archive._package.AOTICompiledModel = (
            torch._inductor.aoti_load_package(
                aoti_model_path,
                device_index=device.index,
            )
        )
        return CombinedModelWrapper(sparse_model, dense_model)


def export_model_aot(
    sparse_model: nn.Module,
    dense_model: nn.Module,
    data: Dict[str, torch.Tensor],
    meta_info: Dict[str, Any],
    save_dir: str,
    mixed_precision: Optional[str] = None,
) -> str:
    """Export AOTInductor model.

    Args:
        sparse_model (nn.Module): the sparse model
        dense_model (nn.Module): the dense model
        data (Dict[str, torch.Tensor]): the test data
        meta_info (Dict[str, Any]): split meta info
        save_dir (str): model save dir
        mixed_precision (Optional[str]): "BF16", "FP16", or None. When set,
            the dense sub-graph is wrapped in a CudaAutocastWrapper so that
            torch.export captures the autocast region as a wrap_with_autocast
            Higher Order Op. The sparse sub-graph is left untouched because
            it is only embedding lookups, which don't benefit from AMP and
            which would complicate torch.jit.script compilation.
    """
    sparse_output, _ = sparse_model(data, "cuda:0")
    sparse_model_traced = symbolic_trace(sparse_model)

    with open(os.path.join(save_dir, "gm_sparse.code"), "w") as f:
        f.write(sparse_model_traced.code)
    sparse_model_scripted = torch.jit.script(sparse_model_traced)
    sparse_model_scripted.save(os.path.join(save_dir, "scripted_sparse_model.pt"))

    batch = torch.export.Dim("batch", min=1, max=499999999)
    dynamic_shapes: Dict[str, Dict[int, torch.export.Dim]] = {}
    seq_tensor_names = meta_info.get("seq_tensor_names", [])
    jagged_seq_tensor_names = meta_info.get("jagged_seq_tensor_names", [])
    seq_share_groups = meta_info.get("seq_share_groups", {})

    # Separate caches per axis: a SEQUENCE group and a JAGGED_SEQUENCE
    # group from the same parent SequenceFeature share a lengths source,
    # but the Dims they need are different symbolic quantities — max
    # seq_len (axis 1, padded) vs total nnz (axis 0, jagged).
    seq_len_dims: Dict[str, torch.export.Dim] = {}
    jagged_batch_dims: Dict[str, torch.export.Dim] = {}

    for key in sparse_output.keys():
        if key in seq_tensor_names:
            share_key = seq_share_groups.get(key, key)
            if share_key not in seq_len_dims:
                seq_len_dims[share_key] = torch.export.Dim(
                    f"{share_key}__seq_len", min=1, max=999999993
                )
            dynamic_shapes[key] = {0: batch, 1: seq_len_dims[share_key]}
        elif key in jagged_seq_tensor_names:
            share_key = seq_share_groups.get(key, key)
            if share_key not in jagged_batch_dims:
                jagged_batch_dims[share_key] = torch.export.Dim(
                    f"{share_key}__batch", min=1, max=999999993
                )
            dynamic_shapes[key] = {0: jagged_batch_dims[share_key]}
        else:
            dynamic_shapes[key] = {0: batch}

    logger.info("dynamic shapes=%s" % dynamic_shapes)

    # Wrap the dense module so torch.export captures the autocast region
    # as a `wrap_with_autocast` HOP that AOT Inductor lowers correctly.
    dense_to_export: nn.Module = dense_model
    if mixed_precision:
        dense_to_export = CudaAutocastWrapper(dense_model, mixed_precision)

    # pre_hook requires running arbitrary code at runtime
    with torch._inductor.config.patch(
        {"unsafe_ignore_unsupported_triton_autotune_args": True}
    ):
        exported_pg = torch.export.export(
            dense_to_export,
            args=(sparse_output,),
            dynamic_shapes=(dynamic_shapes,),
        )
    # AsserScalar codegen is not correct.
    with torch._inductor.config.patch(
        {
            "scalar_asserts": False,
            "unsafe_ignore_unsupported_triton_autotune_args": True,
        }
    ):
        aoti_dir = os.path.join(save_dir, "aoti")
        os.makedirs(aoti_dir, exist_ok=True)

        torch._inductor.aoti_compile_and_package(
            exported_pg,
            package_path=os.path.join(aoti_dir, "aoti_model.pt2"),
        )
    return save_dir


def _pad_empty_sparse_values(
    data: Dict[str, torch.Tensor],
    seq_feat_names: Set[str],
) -> Dict[str, torch.Tensor]:
    """Pad 0-size non-sequence sparse .values tensors to have at least 1 element.

    When a non-sequence sparse feature has all-zero lengths in the example
    batch, its .values tensor has size 0. torch.export traces the code with
    this concrete size and specializes on 0, making the dimension incompatible
    with a dynamic Dim spec. To avoid this, we inject a dummy value and set
    one length entry to 1 so the total nnz becomes >= 1.

    Only non-sequence sparse features are padded; sequence features are left
    as-is.

    This must be called AFTER model verification and BEFORE torch.export.
    """
    lengths_prefixes = set()
    for key in data:
        if key.endswith(".lengths"):
            lengths_prefixes.add(key[: -len(".lengths")])

    for prefix in lengths_prefixes:
        if prefix in seq_feat_names:
            continue
        values_key = f"{prefix}.values"
        lengths_key = f"{prefix}.lengths"
        if values_key not in data or lengths_key not in data:
            continue
        values = data[values_key]
        lengths = data[lengths_key]
        if values.numel() < 2 and lengths.numel() > 0:
            # Pad to at least 2 elements — torch.export specializes on
            # sizes 0 and 1 as special cases but treats >= 2 as dynamic.
            pad_n = 2 - values.numel()
            data[values_key] = torch.zeros(2, dtype=values.dtype, device=values.device)
            new_lengths = lengths.clone()
            new_lengths[0] = new_lengths[0] + pad_n
            data[lengths_key] = new_lengths
            # Also pad .weights if present.
            weights_key = f"{prefix}.weights"
            if weights_key in data:
                weights = data[weights_key]
                data[weights_key] = torch.ones(
                    2, dtype=weights.dtype, device=weights.device
                )

    return data


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
    feat_by_name: Dict[str, Any] = {}
    for feat in features:
        feat_by_name[feat.name] = feat
        if feat.is_sequence:
            seq_feat_names.add(feat.name)
            if getattr(feat, "_is_grouped_seq", False):
                seq_name = getattr(feat, "sequence_name", None)
                if seq_name:
                    vdim = getattr(feat, "value_dim", 1)
                    if vdim == 1:
                        # Single-valued features share nnz within a sequence.
                        feat_to_seq_dim_group[feat.name] = f"seq_{seq_name}"
                    else:
                        # Multi-valued (value_dim=0 variable, or >1 fixed multi)
                        # have their own nnz, so they must NOT share a Dim.
                        pass

    def _is_single_valued(name: str) -> bool:
        feat = feat_by_name.get(name)
        if feat is None:
            return True
        return getattr(feat, "value_dim", 1) == 1

    # Step 2: For standalone sequence features not yet grouped, fall back to
    # model_config.feature_groups structure.
    for fg in model_config.feature_groups:
        if fg.group_type == FeatureGroupType.JAGGED_SEQUENCE:
            # In JAGGED_SEQUENCE groups, single-valued features share nnz.
            # Multi-valued features have independent nnz and are NOT grouped.
            for name in fg.feature_names:
                if name in seq_feat_names and name not in feat_to_seq_dim_group:
                    if _is_single_valued(name):
                        feat_to_seq_dim_group[name] = f"fg_{fg.group_name}"
        # SEQUENCE (DIN-style) groups: standalone sequence features have
        # independent lengths, so don't auto-share. Only grouped sequence
        # features (from SequenceFeature config via sequence_groups) share nnz.
        for sg in fg.sequence_groups:
            for name in sg.feature_names:
                # Only sequence features in seq_groups share nnz — non-sequence
                # features mixed into seq_groups are candidates, not sequences.
                if name in seq_feat_names and name not in feat_to_seq_dim_group:
                    if _is_single_valued(name):
                        feat_to_seq_dim_group[name] = (
                            f"sg_{fg.group_name}_{sg.group_name}"
                        )

    # Step 3: Collect prefixes with .lengths siblings (sparse/sequence features)
    lengths_prefixes = set()
    for key in data:
        if key.endswith(".lengths"):
            lengths_prefixes.add(key[: -len(".lengths")])

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
            elif prefix in seq_feat_names:
                # Ungrouped sequence feature: own Dim, min=1
                dim = torch.export.Dim(f"g_{dim_counter}", min=1, max=999999993)
                dim_counter += 1
                dynamic_shapes[key] = {0: dim}
                prefix_to_dim[prefix] = dim
            else:
                # Non-sequence sparse feature: own Dim, min=0
                dim = torch.export.Dim(f"g_{dim_counter}", min=0, max=999999993)
                dim_counter += 1
                dynamic_shapes[key] = {0: dim}
                prefix_to_dim[prefix] = dim
        elif is_sparse_weights:
            dim = prefix_to_dim.get(prefix)
            if dim is None:
                dim = torch.export.Dim(f"g_{dim_counter}", min=0, max=999999993)
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
    mixed_precision: Optional[str] = None,
) -> str:
    """Export a unified AOTInductor model (sparse+dense fused).

    Args:
        model (nn.Module): the full model (ScriptWrapper).
        data (Dict[str, torch.Tensor]): sample input data.
        save_dir (str): model save dir.
        mixed_precision (Optional[str]): "BF16", "FP16", or None.
    """
    os.makedirs(save_dir, exist_ok=True)

    # AOTInductor export requires CUDA.
    device = torch.device("cuda:0")

    model.set_is_inference(True)
    model.eval()

    # Bind device and optional autocast into a single wrapper so the
    # traced graph sees only `data` as input.
    trace_root = CudaAutocastWrapper(model, mixed_precision, device=str(device))

    logger.info("tracing full model for unified AOTI export...")
    full_gm = symbolic_trace(trace_root)

    with open(os.path.join(save_dir, "gm.code"), "w") as f:
        f.write(full_gm.code)

    # Pad any 0-size non-sequence sparse .values tensors so torch.export
    # doesn't specialize on the empty size (which conflicts with dynamic Dims).
    seq_feat_names = {f.name for f in model._features if f.is_sequence}
    data = _pad_empty_sparse_values(data, seq_feat_names)

    # Build dynamic shapes using feature metadata for correct Dim grouping
    dynamic_shapes = _build_dynamic_shapes(
        data,
        features=model._features,
        model_config=model.model._base_model_config,
    )
    logger.info("dynamic shapes=%s" % dynamic_shapes)

    # Export with torch.export (CPU inputs; graph handles its own H2D).
    logger.info("exporting unified model with torch.export...")
    with torch._inductor.config.patch(
        {"unsafe_ignore_unsupported_triton_autotune_args": True}
    ):
        exported_pg = torch.export.export(
            full_gm,
            args=(data,),
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
        aoti_dir = os.path.join(save_dir, "aoti")
        os.makedirs(aoti_dir, exist_ok=True)

        torch._inductor.aoti_compile_and_package(
            exported_pg,
            package_path=os.path.join(aoti_dir, "aoti_model.pt2"),
        )

    logger.info("unified AOTI model exported to %s", save_dir)
    return save_dir
