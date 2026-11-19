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


import json
import os
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast

import torch
import torch._inductor.codecache  # noqa: F401 -- populate torch._inductor.codecache so torch.export.pt2_archive._package._load_aoti can read it on PPU torch (host torch auto-imports it; PPU's tzrec-test:1.1-ppu image does not).
from torch import nn

from tzrec.acc.utils import (
    is_autotune_with_sample_inputs,
    is_input_tile,
    is_unified_aot_predict,
)
from tzrec.features.feature import BaseFeature, get_hard_search_input_tile_features
from tzrec.models.model import (
    CombinedModelWrapper,
    CudaAutocastWrapper,
    UnifiedAOTIModelWrapper,
)
from tzrec.ops._cuda import cutlass_hstu_attention  # noqa: F401
from tzrec.protos import model_pb2
from tzrec.utils.fx_util import symbolic_trace
from tzrec.utils.logging_util import logger


def _aoti_compile_cfg() -> Dict[str, Any]:
    """torch._inductor.config overrides applied to AOTI compile."""
    cfg: Dict[str, Any] = {
        # AssertScalar codegen is incorrect; keep asserts off.
        "scalar_asserts": False,
        # Tolerate Triton Autotuner pre_hooks (HSTU TensorDescriptor pre_hook).
        "unsafe_ignore_unsupported_triton_autotune_args": True,
        # fp64 alpha forces Newton-Raphson tl.sigmoid on Ada/Ampere where
        # there is no native ex2.f64; fp32 specialization is the right default.
        "_use_fp64_for_unbacked_floats": False,
    }
    if is_autotune_with_sample_inputs():
        cfg["triton.autotune_with_sample_inputs"] = True
    return cfg


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
        tile_keys: Optional[Tuple[str, str]] = None
        hsk_checks: Optional[List[Tuple[str, str]]] = None
        meta_path = os.path.join(model_path, "aoti", "aoti_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                aoti_meta = json.load(f)
            raw_tile_keys = aoti_meta.get("tile_keys")
            if raw_tile_keys is not None:
                tile_keys = (raw_tile_keys[0], raw_tile_keys[1])
            raw_hsk_checks = aoti_meta.get("hsk_checks")
            if raw_hsk_checks is not None:
                hsk_checks = [(pair[0], pair[1]) for pair in raw_hsk_checks]
        return UnifiedAOTIModelWrapper(
            model, tile_keys=tile_keys, hsk_checks=hsk_checks
        )
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
    # no_grad: outputs become leaf tensors so AOTI's
    # extract_autotune_inputs() can deepcopy them when sample-input
    # autotune is on (upstream pytorch issue: only leaves support deepcopy).
    with torch.no_grad():
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
    with torch._inductor.config.patch(_aoti_compile_cfg()):
        aoti_dir = os.path.join(save_dir, "aoti")
        os.makedirs(aoti_dir, exist_ok=True)

        torch._inductor.aoti_compile_and_package(
            exported_pg,
            package_path=os.path.join(aoti_dir, "aoti_model.pt2"),
        )
    return save_dir


def _pad_empty_sparse_values(
    data: Dict[str, torch.Tensor],
    features: List[BaseFeature],
) -> Dict[str, torch.Tensor]:
    """Pad <2-element sparse and sequence tensors before torch.export.

    When a non-sequence sparse feature or sequence feature has total nnz or
    total sequence length < 2 in the example batch, torch.export specializes the
    axis to 0 or 1, making it incompatible with a dynamic Dim spec. Padding to
    >= 2 elements keeps the traced graph dynamic.

    This must be called AFTER model verification and BEFORE torch.export.
    """
    data = dict(data)
    seq_feat_names = {f.name for f in features if f.is_sequence}

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
            pad_n = 2 - values.numel()
            data[values_key] = torch.cat(
                [values, torch.zeros(pad_n, dtype=values.dtype, device=values.device)]
            )
            new_lengths = lengths.clone()
            new_lengths[0] = new_lengths[0] + pad_n
            data[lengths_key] = new_lengths
            weights_key = f"{prefix}.weights"
            if weights_key in data:
                weights = data[weights_key]
                data[weights_key] = torch.cat(
                    [
                        weights,
                        torch.ones(pad_n, dtype=weights.dtype, device=weights.device),
                    ]
                )

    # Group sequence features so shared sequences keep matching lengths after padding.
    seq_groups: Dict[str, List[BaseFeature]] = {}
    for feat in features:
        if not feat.is_sequence:
            continue
        seq_name = getattr(feat, "sequence_name", None)
        if getattr(feat, "_is_grouped_seq", False) and seq_name:
            group_key = f"seq_{seq_name}"
        else:
            group_key = f"feat_{feat.name}"
        seq_groups.setdefault(group_key, []).append(feat)

    for group_feats in seq_groups.values():
        features_in_batch = [
            feat
            for feat in group_feats
            if f"{feat.name}.lengths" in data and f"{feat.name}.values" in data
        ]
        if not features_in_batch:
            continue
        ref_lengths = data[f"{features_in_batch[0].name}.lengths"]
        if ref_lengths.numel() == 0:
            continue
        total_sequence_length = int(ref_lengths.sum().item())
        if total_sequence_length < 2:
            sequence_length_padding = 2 - total_sequence_length
            for feat in features_in_batch:
                lengths_key = f"{feat.name}.lengths"
                values_key = f"{feat.name}.values"
                kl_key = f"{feat.name}.key_lengths"
                weights_key = f"{feat.name}.weights"

                new_lengths = data[lengths_key].clone()
                new_lengths[0] = new_lengths[0] + sequence_length_padding
                data[lengths_key] = new_lengths

                values = data[values_key]
                if kl_key in data:
                    kl = data[kl_key]
                    data[kl_key] = torch.cat(
                        [
                            kl,
                            torch.ones(
                                sequence_length_padding,
                                dtype=kl.dtype,
                                device=kl.device,
                            ),
                        ]
                    )
                    data[values_key] = torch.cat(
                        [
                            values,
                            torch.zeros(
                                sequence_length_padding,
                                dtype=values.dtype,
                                device=values.device,
                            ),
                        ]
                    )
                elif feat.is_sparse:
                    data[values_key] = torch.cat(
                        [
                            values,
                            torch.zeros(
                                sequence_length_padding,
                                dtype=values.dtype,
                                device=values.device,
                            ),
                        ]
                    )
                else:
                    data[values_key] = torch.cat(
                        [
                            values,
                            values.new_zeros(
                                (sequence_length_padding, *values.shape[1:])
                            ),
                        ],
                        dim=0,
                    )
                if weights_key in data:
                    weights = data[weights_key]
                    data[weights_key] = torch.cat(
                        [
                            weights,
                            torch.ones(
                                sequence_length_padding,
                                dtype=weights.dtype,
                                device=weights.device,
                            ),
                        ]
                    )

        for feat in features_in_batch:
            kl_key = f"{feat.name}.key_lengths"
            values_key = f"{feat.name}.values"
            if kl_key not in data:
                continue
            values = data[values_key]
            kl = data[kl_key]
            if values.numel() < 2 and kl.numel() > 0:
                pad_vals = 2 - values.numel()
                new_kl = kl.clone()
                new_kl[0] = new_kl[0] + pad_vals
                data[kl_key] = new_kl
                data[values_key] = torch.cat(
                    [
                        values,
                        torch.zeros(pad_vals, dtype=values.dtype, device=values.device),
                    ]
                )
                weights_key = f"{feat.name}.weights"
                if weights_key in data:
                    weights = data[weights_key]
                    data[weights_key] = torch.cat(
                        [
                            weights,
                            torch.ones(
                                pad_vals, dtype=weights.dtype, device=weights.device
                            ),
                        ]
                    )

    return data


def _build_dynamic_shapes(
    data: Dict[str, torch.Tensor],
    features: Any,
    feature_groups: List[model_pb2.FeatureGroupConfig],
) -> Dict[str, Dict[int, torch.export.Dim]]:
    """Build dynamic shapes for the full model input.

    Uses structural knowledge from feature configs and feature groups:
    - .lengths -> batch dim (or user_batch / hard-search compact batch dim)
    - .values for non-sequence single-value sparse features -> own Dim (min=0)
    - sequence value tensors (.values for single-value sparse or dense sequence
      features, .key_lengths for multi-value sparse sequence features) -> shared
      total sequence length Dim per SequenceFeature / JAGGED_SEQUENCE /
      SeqGroupConfig (min=1 for JAGGED_SEQUENCE, min=0 otherwise)
    - .values without a .lengths sibling -> batch dim (dense feature)
    - .weights -> shares Dim with corresponding .values (same prefix)
    - .values for multi-value sparse sequence features (with .key_lengths) ->
      own sub-token Dim (min=0)
    - scalars and ``batch_size`` -> no dynamic dims
    - everything else (labels, sample_weights, inverse_indices) -> batch dim

    Args:
        data: input tensor dict from Batch.to_dict().
        features: list of BaseFeature from model.features.
        feature_groups: list of FeatureGroupConfig from model.feature_groups.

    Returns:
        dynamic_shapes dict for torch.export.export().
    """
    from tzrec.protos.model_pb2 import FeatureGroupType

    # Step 1: Group grouped sequence features by their sequence_name.
    # Features in the same SequenceFeature config share per-sample lengths,
    # so their .values or .key_lengths tensors share a total sequence length Dim.
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
    # feature_groups structure, and track JAGGED_SEQUENCE (min=1) members.
    jagged_seq_feat_names: Set[str] = set()
    jagged_seq_groups: Set[str] = set()
    for fg in feature_groups:
        if fg.group_type == FeatureGroupType.JAGGED_SEQUENCE:
            for name in fg.feature_names:
                if name in seq_feat_names:
                    jagged_seq_feat_names.add(name)
                    if name not in feat_to_seq_dim_group:
                        feat_to_seq_dim_group[name] = f"fg_{fg.group_name}"
                    jagged_seq_groups.add(feat_to_seq_dim_group[name])
        # SEQUENCE (DIN-style) groups: standalone sequence features have
        # independent lengths, so don't auto-share. Only grouped sequence
        # features (from SequenceFeature config via sequence_groups) share nnz.
        for sg in fg.sequence_groups:
            for name in sg.feature_names:
                if name in seq_feat_names and name not in feat_to_seq_dim_group:
                    feat_to_seq_dim_group[name] = f"sg_{fg.group_name}_{sg.group_name}"

    # Step 3: Collect prefixes with .lengths and .key_lengths siblings.
    lengths_prefixes = set()
    key_lengths_prefixes = set()
    for key in data:
        if key.endswith(".lengths"):
            lengths_prefixes.add(key[: -len(".lengths")])
        elif key.endswith(".key_lengths"):
            key_lengths_prefixes.add(key[: -len(".key_lengths")])

    # Step 4: Build dynamic shapes
    input_tile = is_input_tile()
    user_feat_names: Set[str] = (
        {f.name for f in features if f.is_user_feat} if input_tile else set()
    )
    hard_search_features: Dict[str, str] = (
        get_hard_search_input_tile_features(cast(List[BaseFeature], features))
        if input_tile
        else {}
    )

    batch = torch.export.Dim("batch", min=1, max=499999999)
    user_batch: Optional[torch.export.Dim] = None
    hsk_batch_dims: Dict[str, torch.export.Dim] = {}
    static_tile_keys: List[str] = []
    dynamic_shapes = {}
    group_to_dim: Dict[str, torch.export.Dim] = {}
    prefix_to_dim: Dict[str, torch.export.Dim] = {}
    dim_counter = 0

    def _get_total_sequence_length_dim(feat_prefix: str) -> torch.export.Dim:
        nonlocal dim_counter
        if feat_prefix in feat_to_seq_dim_group:
            group = feat_to_seq_dim_group[feat_prefix]
            if group not in group_to_dim:
                min_len = 1 if group in jagged_seq_groups else 0
                group_to_dim[group] = torch.export.Dim(
                    f"g_{dim_counter}", min=min_len, max=999999993
                )
                dim_counter += 1
            return group_to_dim[group]
        min_len = 1 if feat_prefix in jagged_seq_feat_names else 0
        total_sequence_length_dim = torch.export.Dim(
            f"g_{dim_counter}", min=min_len, max=999999993
        )
        dim_counter += 1
        return total_sequence_length_dim

    for key, tensor in data.items():
        if tensor.dim() == 0 or key == "batch_size":
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
            if prefix in seq_feat_names and prefix not in key_lengths_prefixes:
                dim = _get_total_sequence_length_dim(prefix)
                dynamic_shapes[key] = {0: dim}
                prefix_to_dim[prefix] = dim
            else:
                dim = prefix_to_dim.get(prefix)
                if dim is None:
                    dim = torch.export.Dim(f"g_{dim_counter}", min=0, max=999999993)
                    dim_counter += 1
                    prefix_to_dim[prefix] = dim
                dynamic_shapes[key] = {0: dim}
        elif is_sparse_weights:
            dim = prefix_to_dim.get(prefix)
            if dim is None:
                dim = torch.export.Dim(f"g_{dim_counter}", min=0, max=999999993)
                dim_counter += 1
                prefix_to_dim[prefix] = dim
            dynamic_shapes[key] = {0: dim}
        elif is_key_lengths:
            dynamic_shapes[key] = {0: _get_total_sequence_length_dim(prefix)}
        elif input_tile and prefix in hard_search_features:
            if tensor.size(0) < 2:
                # torch.export specializes size < 2 and rejects a dynamic Dim.
                dynamic_shapes[key] = {}
                static_tile_keys.append(key)
            else:
                seq_src = hard_search_features[prefix]
                if seq_src not in hsk_batch_dims:
                    hsk_batch_dims[seq_src] = torch.export.Dim(
                        f"hsk_{seq_src}_batch", min=1, max=499999999
                    )
                dynamic_shapes[key] = {0: hsk_batch_dims[seq_src]}
        elif input_tile and prefix in user_feat_names:
            if tensor.size(0) < 2:
                # torch.export specializes size < 2 and rejects a dynamic Dim.
                dynamic_shapes[key] = {}
                static_tile_keys.append(key)
            else:
                if user_batch is None:
                    user_batch = torch.export.Dim("user_batch", min=1, max=499999999)
                dynamic_shapes[key] = {0: user_batch}
        else:
            dynamic_shapes[key] = {0: batch}

    if static_tile_keys:
        logger.warning(
            "Input-tile tensor(s) %s have <2 rows in the export sample; "
            "freezing dim 0 as static in unified AOTI export.",
            static_tile_keys,
        )
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

    features = cast(List[BaseFeature], model.features)
    # Normalize KeyedTensor column slices so AOTI compiles for contiguous inputs.
    data = {k: v.contiguous() for k, v in sorted(data.items())}
    # Pad <2-element sparse/sequence tensors so torch.export does not specialize
    # them as constants and reject dynamic Dim constraints.
    data = _pad_empty_sparse_values(data, features)

    # Build dynamic shapes using feature metadata for correct Dim grouping
    dynamic_shapes = _build_dynamic_shapes(
        data,
        features=features,
        feature_groups=model.feature_groups,
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
    with torch._inductor.config.patch(_aoti_compile_cfg()):
        aoti_dir = os.path.join(save_dir, "aoti")
        os.makedirs(aoti_dir, exist_ok=True)

        torch._inductor.aoti_compile_and_package(
            exported_pg,
            package_path=os.path.join(aoti_dir, "aoti_model.pt2"),
        )
        data_parser = getattr(model, "_data_parser", None)
        aoti_meta: Dict[str, Any] = {}
        if is_input_tile() and data_parser is not None:
            if data_parser._tile_keys is not None:
                aoti_meta["tile_keys"] = list(data_parser._tile_keys)
            hsk_map = data_parser._hard_search_input_tile_features
            hsk_checks = [
                [f"{f.name}.lengths", f"{hsk_map[f.name]}__inverse_indices"]
                for f in features
                if not f.stub_type and f.is_sparse and f.name in hsk_map
            ]
            if hsk_checks:
                aoti_meta["hsk_checks"] = hsk_checks
        if aoti_meta:
            with open(os.path.join(aoti_dir, "aoti_meta.json"), "w") as f:
                json.dump(aoti_meta, f, indent=4)

    logger.info("unified AOTI model exported to %s", save_dir)
    return save_dir
