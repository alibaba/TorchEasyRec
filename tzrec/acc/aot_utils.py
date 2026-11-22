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
import operator
import os
import platform
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast

import torch

# Populate torch._inductor.codecache so torch.export.pt2_archive._package._load_aoti
# can read it in runtime images where torch does not auto-import it.
import torch._inductor.codecache  # noqa: F401
from torch import nn

from tzrec.acc.utils import (
    canonical_contiguous,
    get_aot_cpu_march,
    is_autotune_with_sample_inputs,
    is_input_tile,
    is_input_tile_3_online,
    is_input_tile_user_keyed_tensor,
    is_portable_cpu_aot,
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
    march = get_aot_cpu_march()
    simdlen_env = os.environ.get("AOTI_CPU_SIMDLEN")
    if march is not None:
        cfg["cpp.march"] = march
        if simdlen_env is not None:
            cfg["cpp.simdlen"] = int(simdlen_env)
        elif march in ("x86-64", "x86-64-v2"):
            cfg["cpp.simdlen"] = 0
        elif march in ("core-avx2", "haswell", "skylake", "x86-64-v3"):
            cfg["cpp.simdlen"] = 256
    elif is_portable_cpu_aot():
        if platform.machine() in ("x86_64", "AMD64"):
            cfg["cpp.march"] = "core-avx2"
            cfg["cpp.simdlen"] = 256
        else:
            cfg["cpp.march"] = ""
            cfg["cpp.simdlen"] = 0
        if simdlen_env is not None:
            cfg["cpp.simdlen"] = int(simdlen_env)
    return cfg


def _compile_and_package_aoti(
    exported_pg: torch.export.ExportedProgram, save_dir: str
) -> str:
    """Compile and package an ExportedProgram into an AOTInductor .pt2 archive."""
    with torch._inductor.config.patch(_aoti_compile_cfg()):
        aoti_dir = os.path.join(save_dir, "aoti")
        os.makedirs(aoti_dir, exist_ok=True)
        package_path = os.path.join(aoti_dir, "aoti_model.pt2")
        torch._inductor.aoti_compile_and_package(
            exported_pg,
            package_path=package_path,
        )
    return package_path


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
    if device.type != "cuda":
        raise ValueError(f"load_model_aot requires a CUDA device, got {device!r}.")
    device_index = -1 if device.index is None else device.index
    aoti_model_path = os.path.join(model_path, "aoti", "aoti_model.pt2")

    if is_unified_aot_predict(model_path):
        # Unified single-model path
        model = torch._inductor.aoti_load_package(
            aoti_model_path,
            device_index=device_index,
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
                device_index=device_index,
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
    if not torch.cuda.is_available():
        raise RuntimeError("AOTI export requires a CUDA device.")
    os.makedirs(save_dir, exist_ok=True)
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
    # but the Dims they need are different symbolic quantities -- max
    # seq_len (axis 1, padded) vs total nnz (axis 0, jagged).
    seq_len_dims: Dict[str, torch.export.Dim] = {}
    jagged_batch_dims: Dict[str, torch.export.Dim] = {}

    for key in sparse_output.keys():
        if key in seq_tensor_names:
            share_key = seq_share_groups.get(key, key)
            if share_key not in seq_len_dims:
                seq_len_dims[share_key] = torch.export.Dim(
                    f"{share_key}__seq_len", min=0, max=999999993
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

    sparse_output = {k: canonical_contiguous(v) for k, v in sparse_output.items()}
    # pre_hook requires running arbitrary code at runtime
    with torch._inductor.config.patch(
        {"unsafe_ignore_unsupported_triton_autotune_args": True}
    ):
        exported_pg = torch.export.export(
            dense_to_export,
            args=(sparse_output,),
            dynamic_shapes=(dynamic_shapes,),
        )
    _compile_and_package_aoti(exported_pg, save_dir)
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


def _collect_sequence_dim_groups(
    features: List[BaseFeature],
    feature_groups: List[model_pb2.FeatureGroupConfig],
) -> Tuple[Dict[str, str], Set[str], Set[str], Set[str]]:
    """Collect shared sequence Dim group mappings and JAGGED_SEQUENCE flags.

    Args:
        features: model feature definitions.
        feature_groups: model feature group configs.

    Returns:
        Tuple of ``(feat_to_seq_dim_group, seq_feat_names,
        jagged_seq_feat_names, jagged_seq_groups)``.
    """
    from tzrec.protos.model_pb2 import FeatureGroupType

    # Features in the same SequenceFeature config share per-sample lengths,
    # so their step tensors (.values or .key_lengths) share a Dim.
    feat_to_seq_dim_group: Dict[str, str] = {}
    seq_feat_names: Set[str] = set()
    for feat in features:
        if feat.is_sequence:
            seq_feat_names.add(feat.name)
            if getattr(feat, "_is_grouped_seq", False):
                seq_name = getattr(feat, "sequence_name", None)
                if seq_name:
                    feat_to_seq_dim_group[feat.name] = f"seq_{seq_name}"

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
        for sg in fg.sequence_groups:
            for name in sg.feature_names:
                if name in seq_feat_names and name not in feat_to_seq_dim_group:
                    feat_to_seq_dim_group[name] = f"sg_{fg.group_name}_{sg.group_name}"

    return (
        feat_to_seq_dim_group,
        seq_feat_names,
        jagged_seq_feat_names,
        jagged_seq_groups,
    )


def _can_prune_unused_input_keys(
    gm: torch.fx.GraphModule, input_node: torch.fx.Node
) -> bool:
    """Return whether every user of ``input_node`` reads known named keys."""
    del gm
    for user in input_node.users:
        if user.op == "call_function" and user.target == operator.getitem:
            if len(user.args) == 2 and isinstance(user.args[1], str):
                continue
            return False
        return False
    return True


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
    (
        feat_to_seq_dim_group,
        seq_feat_names,
        jagged_seq_feat_names,
        jagged_seq_groups,
    ) = _collect_sequence_dim_groups(cast(List[BaseFeature], features), feature_groups)

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
    hsk_lengths_keys: Set[str] = (
        {f"{name}.lengths" for name in hard_search_features} if input_tile else set()
    )

    batch = torch.export.Dim("batch", min=1, max=499999999)
    user_batch: Optional[torch.export.Dim] = None
    hsk_batch_dims: Dict[str, torch.export.Dim] = {}
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
        elif (input_tile and prefix in hard_search_features) or key in hsk_lengths_keys:
            if tensor.size(0) < 2:
                logger.warning(
                    "Dynamic shape for %r degraded to static {} (size(0)=%d < 2).",
                    key,
                    tensor.size(0),
                )
                dynamic_shapes[key] = {}
            else:
                seq_src = hard_search_features[prefix]
                if seq_src not in hsk_batch_dims:
                    hsk_batch_dims[seq_src] = torch.export.Dim(
                        f"hsk_{seq_src}_batch", min=1, max=499999999
                    )
                dynamic_shapes[key] = {0: hsk_batch_dims[seq_src]}
        elif input_tile and prefix in user_feat_names:
            if tensor.size(0) < 2:
                logger.warning(
                    "Dynamic shape for %r degraded to static {} (size(0)=%d < 2).",
                    key,
                    tensor.size(0),
                )
                dynamic_shapes[key] = {}
            else:
                if user_batch is None:
                    user_batch = torch.export.Dim("user_batch", min=1, max=499999999)
                dynamic_shapes[key] = {0: user_batch}
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
    if not torch.cuda.is_available():
        raise RuntimeError("AOTI export requires a CUDA device.")
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
    # Prune unconsumed keys (e.g. unused "batch_size") so in_spec stays minimal.
    input_node = next(n for n in full_gm.graph.nodes if n.op == "placeholder")
    if _can_prune_unused_input_keys(full_gm, input_node):
        consumed_keys = _extract_distributed_dense_consumed_keys(full_gm)
        if consumed_keys:
            data = {k: v for k, v in data.items() if k in consumed_keys}
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
    data = {k: canonical_contiguous(v) for k, v in data.items()}
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
    _compile_and_package_aoti(exported_pg, save_dir)
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
        with open(os.path.join(save_dir, "aoti", "aoti_meta.json"), "w") as f:
            json.dump(aoti_meta, f, indent=4)

    logger.info("unified AOTI model exported to %s", save_dir)
    return save_dir


def _extract_distributed_dense_consumed_keys(
    gm: torch.fx.GraphModule,
) -> Set[str]:
    """Extract input dict keys consumed by the rewritten dense graph."""
    input_node = next(n for n in gm.graph.nodes if n.op == "placeholder")
    consumed_keys: Set[str] = set()
    for node in gm.graph.nodes:
        if (
            node.op == "call_function"
            and node.target == operator.getitem
            and len(node.args) == 2
            and node.args[0] is input_node
            and isinstance(node.args[1], str)
            and len(node.users) > 0
        ):
            consumed_keys.add(node.args[1])
    return consumed_keys


def _increment_length_at(
    sample_data: Dict[str, torch.Tensor], key: str, delta: int, index: int = 0
) -> None:
    """Increment one entry of a length tensor without mutating in place.

    Args:
        sample_data: sample tensor dictionary to update.
        key: length tensor key in ``sample_data``.
        delta: positive length increment to add.
        index: element index in ``sample_data[key]`` to increment.
    """
    if key not in sample_data:
        return
    lengths = sample_data[key]
    if index < 0 or index >= lengths.numel():
        return
    updated = lengths.clone()
    updated[index] = updated[index] + delta
    sample_data[key] = updated


def _pad_rows_zeros(sample_data: Dict[str, torch.Tensor], key: str, pad_n: int) -> None:
    """Append ``pad_n`` zero rows along dim 0 for ``sample_data[key]``."""
    tensor = sample_data[key]
    sample_data[key] = torch.cat(
        [tensor, tensor.new_zeros((pad_n, *tensor.shape[1:]))], dim=0
    )


def _pad_distributed_dense_sample_inputs(
    sample_data: Dict[str, torch.Tensor],
    seq_jts: Dict[str, Any],
    features: List[BaseFeature],
    feature_groups: List[model_pb2.FeatureGroupConfig],
) -> Dict[str, torch.Tensor]:
    """Pad <2-row sequence and sparse tensors to prevent 0/1 specialization."""
    sample_data = dict(sample_data)
    feat_to_seq_dim_group, seq_feat_names, _, _ = _collect_sequence_dim_groups(
        features, feature_groups
    )
    feat_by_name = {f.name: f for f in features}
    raw_name_to_ec_keys: Dict[str, List[str]] = defaultdict(list)
    for name in seq_jts:
        raw_name_to_ec_keys[name.split("@", 1)[0]].append(name)

    # 1. Sequence groups sharing a total_sequence_length Dim
    # (EC, SequenceDense, mulval key_lengths).
    seq_groups: Dict[str, List[BaseFeature]] = defaultdict(list)
    for feat in features:
        if not feat.is_sequence:
            continue
        group_key = feat_to_seq_dim_group.get(feat.name, f"feat_{feat.name}")
        seq_groups[group_key].append(feat)

    for group_feats in seq_groups.values():
        total_sequence_length: Optional[int] = None
        for feat in group_feats:
            kl_key = f"{feat.name}.key_lengths"
            if kl_key in sample_data:
                total_sequence_length = int(sample_data[kl_key].size(0))
                break
            if not feat.is_sparse and f"{feat.name}.values" in sample_data:
                total_sequence_length = int(sample_data[f"{feat.name}.values"].size(0))
                break
            for ec_key in raw_name_to_ec_keys.get(feat.name, []):
                if ec_key in sample_data:
                    total_sequence_length = int(sample_data[ec_key].size(0))
                    break
            if total_sequence_length is not None:
                break
            if feat.is_sparse and f"{feat.name}.values" in sample_data:
                total_sequence_length = int(sample_data[f"{feat.name}.values"].size(0))
                break

        if total_sequence_length is not None and total_sequence_length < 2:
            sequence_length_padding = 2 - total_sequence_length
            for feat in group_feats:
                kl_key = f"{feat.name}.key_lengths"
                lengths_key = f"{feat.name}.lengths"
                values_key = f"{feat.name}.values"
                if kl_key in sample_data:
                    kl = sample_data[kl_key]
                    sample_data[kl_key] = torch.cat(
                        [
                            kl,
                            torch.ones(
                                sequence_length_padding,
                                dtype=kl.dtype,
                                device=kl.device,
                            ),
                        ],
                        dim=0,
                    )
                    _increment_length_at(
                        sample_data, lengths_key, sequence_length_padding
                    )
                    for ec_key in raw_name_to_ec_keys.get(feat.name, []):
                        if ec_key in sample_data:
                            _pad_rows_zeros(
                                sample_data, ec_key, sequence_length_padding
                            )
                        _increment_length_at(
                            sample_data, f"{ec_key}__lengths", sequence_length_padding
                        )
                    if values_key in sample_data:
                        _pad_rows_zeros(
                            sample_data, values_key, sequence_length_padding
                        )
                elif not feat.is_sparse:
                    if values_key in sample_data:
                        _pad_rows_zeros(
                            sample_data, values_key, sequence_length_padding
                        )
                        _increment_length_at(
                            sample_data, lengths_key, sequence_length_padding
                        )
                else:
                    for ec_key in raw_name_to_ec_keys.get(feat.name, []):
                        if ec_key in sample_data:
                            _pad_rows_zeros(
                                sample_data, ec_key, sequence_length_padding
                            )
                        _increment_length_at(
                            sample_data, f"{ec_key}__lengths", sequence_length_padding
                        )
                    if values_key in sample_data:
                        _pad_rows_zeros(
                            sample_data, values_key, sequence_length_padding
                        )
                        _increment_length_at(
                            sample_data, lengths_key, sequence_length_padding
                        )

        # 2. Multi-value sequence sub-token values may still be <2 even when
        # total_sequence_length >= 2 (e.g. key_lengths=[0, 0]).
        for feat in group_feats:
            kl_key = f"{feat.name}.key_lengths"
            if kl_key not in sample_data or sample_data[kl_key].numel() == 0:
                continue
            seq_lengths_key = f"{feat.name}.lengths"
            owner_sample = 0
            if seq_lengths_key in sample_data:
                nonzero = torch.nonzero(
                    sample_data[seq_lengths_key] > 0, as_tuple=False
                )
                if nonzero.numel() > 0:
                    owner_sample = int(nonzero[0].item())
            pad_delta = 0
            for ec_key in raw_name_to_ec_keys.get(feat.name, []):
                if ec_key in sample_data and sample_data[ec_key].size(0) < 2:
                    pad_vals = 2 - int(sample_data[ec_key].size(0))
                    _pad_rows_zeros(sample_data, ec_key, pad_vals)
                    _increment_length_at(
                        sample_data, f"{ec_key}__lengths", pad_vals, index=owner_sample
                    )
                    pad_delta = max(pad_delta, pad_vals)
            raw_vals_key = f"{feat.name}.values"
            if raw_vals_key in sample_data and sample_data[raw_vals_key].size(0) < 2:
                pad_vals = 2 - int(sample_data[raw_vals_key].size(0))
                _pad_rows_zeros(sample_data, raw_vals_key, pad_vals)
                pad_delta = max(pad_delta, pad_vals)
            if pad_delta > 0:
                _increment_length_at(sample_data, kl_key, pad_delta)

    # 3. Non-sequence Query EC entries in sequence__ec.
    for ec_key in seq_jts:
        raw_name = ec_key.split("@", 1)[0]
        feat = feat_by_name.get(raw_name)
        if raw_name in seq_feat_names or (feat is not None and feat.is_sequence):
            continue
        if ec_key in sample_data and sample_data[ec_key].size(0) < 2:
            pad_n = 2 - int(sample_data[ec_key].size(0))
            _pad_rows_zeros(sample_data, ec_key, pad_n)
            _increment_length_at(sample_data, f"{ec_key}__lengths", pad_n)

    return sample_data


def _resolve_user_batch_size(
    data: Dict[str, torch.Tensor],
    seq_jts: Dict[str, Any],
    features: List[BaseFeature],
    hsk_feats: Set[str],
) -> Optional[int]:
    """Resolve the pre-tile user batch size from warm-up inputs."""
    user_feats = {
        f.name
        for f in features
        if f.is_user_feat and not f.stub_type and f.name not in hsk_feats
    }
    for feat in features:
        if feat.name not in user_feats:
            continue
        lengths_key = f"{feat.name}.lengths"
        if lengths_key in data:
            return int(data[lengths_key].numel())
        if not feat.is_sparse and not feat.is_sequence:
            values_key = f"{feat.name}.values"
            if values_key in data:
                return int(data[values_key].size(0))
    for ec_key, jt in seq_jts.items():
        raw_name = ec_key.split("@", 1)[0]
        if raw_name in user_feats:
            return int(jt.lengths().numel())
    return None


def _prepare_distributed_dense_sample_inputs(
    gm: torch.fx.GraphModule,
    data: Dict[str, torch.Tensor],
    sparse_kts: Dict[str, Any],
    seq_jts: Dict[str, Any],
    features: List[BaseFeature],
    feature_groups: List[model_pb2.FeatureGroupConfig],
    device: torch.device,
    input_tile: Optional[bool] = None,
    input_tile_3_online: Optional[bool] = None,
    hard_search_features: Optional[Dict[str, str]] = None,
) -> Dict[str, torch.Tensor]:
    """Assemble and pad serving-style CUDA sample inputs for dense AOTI export."""
    consumed_keys = _extract_distributed_dense_consumed_keys(gm)
    online = (
        is_input_tile_3_online() if input_tile_3_online is None else input_tile_3_online
    )
    resolved_input_tile = is_input_tile() if input_tile is None else input_tile
    resolved_hsk: Dict[str, str] = (
        (
            get_hard_search_input_tile_features(features)
            if hard_search_features is None
            else hard_search_features
        )
        if resolved_input_tile
        else {}
    )
    user_bs = _resolve_user_batch_size(data, seq_jts, features, set(resolved_hsk))
    sample_data: Dict[str, torch.Tensor] = {}

    for name, kt in sparse_kts.items():
        if name not in consumed_keys:
            continue
        val = kt.values()
        if is_input_tile_user_keyed_tensor(name):
            # EmbeddingGroupImpl marks __ebc_user / __mc_ebc_user after tiling;
            # slice back to pre-tile user_bs so sample inputs match the dense graph.
            target_rows = user_bs
            if target_rows is None:
                if online:
                    target_rows = 1
                elif "batch_size" in data:
                    tile_size = int(data["batch_size"].item())
                    if tile_size > 0 and val.size(0) % tile_size == 0:
                        target_rows = val.size(0) // tile_size
            if target_rows is not None and 0 < target_rows < val.size(0):
                val = val[:target_rows]
        sample_data[name] = val

    for name, jt in seq_jts.items():
        if name in consumed_keys:
            sample_data[name] = jt.values()
        lengths_key = f"{name}__lengths"
        if lengths_key in consumed_keys:
            sample_data[lengths_key] = jt.lengths().to(torch.int32)

    for key in sorted(consumed_keys):
        if key not in sample_data and key in data:
            sample_data[key] = data[key]

    sample_data = {
        k: v.to(device=device).contiguous() for k, v in sorted(sample_data.items())
    }
    sample_data = _pad_distributed_dense_sample_inputs(
        sample_data, seq_jts, features, feature_groups
    )
    return dict(sorted(sample_data.items()))


def _build_distributed_dense_dynamic_shapes(
    sample_data: Dict[str, torch.Tensor],
    dense_graph_config: Dict[str, Any],
    features: List[BaseFeature],
    feature_groups: List[model_pb2.FeatureGroupConfig],
    input_tile: Optional[bool] = None,
    hard_search_features: Optional[Dict[str, str]] = None,
) -> Dict[str, Dict[int, torch.export.Dim]]:
    """Build torch.export dynamic_shapes for a distributed dense serving graph."""
    (
        feat_to_seq_dim_group,
        seq_feat_names,
        jagged_seq_feat_names,
        jagged_seq_groups,
    ) = _collect_sequence_dim_groups(features, feature_groups)

    feat_by_name = {f.name: f for f in features}
    mulval_seq_names: Set[str] = {
        f.name for f in features if f.is_sparse and f.is_sequence and f.value_dim != 1
    }
    resolved_input_tile = is_input_tile() if input_tile is None else input_tile
    user_feat_names: Set[str] = (
        {f.name for f in features if f.is_user_feat} if resolved_input_tile else set()
    )
    resolved_hsk: Dict[str, str] = (
        (
            get_hard_search_input_tile_features(features)
            if hard_search_features is None
            else hard_search_features
        )
        if resolved_input_tile
        else {}
    )

    seq_ec_entries = set(dense_graph_config.get("sequence__ec", []))
    seq_ec_names = {
        entry[: -len("__ec")] for entry in seq_ec_entries if entry.endswith("__ec")
    }
    seq_ec_lengths_names = {
        entry for entry in seq_ec_entries if entry.endswith("__lengths")
    }
    ebc_item_names = {
        k
        for k in dense_graph_config
        if k != "sequence__ec" and not is_input_tile_user_keyed_tensor(k)
    }
    ebc_user_names = {
        k
        for k in dense_graph_config
        if k != "sequence__ec" and is_input_tile_user_keyed_tensor(k)
    }

    batch = torch.export.Dim("batch", min=1, max=499999999)
    user_batch: Optional[torch.export.Dim] = None
    hsk_batch_dims: Dict[str, torch.export.Dim] = {}
    group_to_dim: Dict[str, torch.export.Dim] = {}
    prefix_to_dim: Dict[str, torch.export.Dim] = {}
    dynamic_shapes: Dict[str, Dict[int, torch.export.Dim]] = {}
    dim_counter = 0

    def _next_dim(min_len: int) -> torch.export.Dim:
        nonlocal dim_counter
        dim = torch.export.Dim(f"g_{dim_counter}", min=min_len, max=999999993)
        dim_counter += 1
        return dim

    def _user_batch_shape(
        key: str, tensor: torch.Tensor
    ) -> Dict[int, torch.export.Dim]:
        nonlocal user_batch
        if tensor.size(0) < 2:
            logger.warning(
                "Dynamic shape for %r degraded to static (size(0)=%d < 2).",
                key,
                tensor.size(0),
            )
            return {}
        if user_batch is None:
            user_batch = torch.export.Dim("user_batch", min=1, max=499999999)
        return {0: user_batch}

    def _hsk_batch_shape(
        key: str, feat_name: str, tensor: torch.Tensor
    ) -> Dict[int, torch.export.Dim]:
        if tensor.size(0) < 2:
            logger.warning(
                "Dynamic shape for %r degraded to static (size(0)=%d < 2).",
                key,
                tensor.size(0),
            )
            return {}
        seq_src = resolved_hsk[feat_name]
        if seq_src not in hsk_batch_dims:
            hsk_batch_dims[seq_src] = torch.export.Dim(
                f"hsk_{seq_src}_batch", min=1, max=499999999
            )
        return {0: hsk_batch_dims[seq_src]}

    def _get_total_sequence_length_dim(feat_prefix: str) -> torch.export.Dim:
        if feat_prefix in feat_to_seq_dim_group:
            group = feat_to_seq_dim_group[feat_prefix]
            if group not in group_to_dim:
                min_len = 1 if group in jagged_seq_groups else 0
                group_to_dim[group] = _next_dim(min_len)
            return group_to_dim[group]
        min_len = 1 if feat_prefix in jagged_seq_feat_names else 0
        return _next_dim(min_len)

    def _get_or_create_prefix_dim(feat_prefix: str) -> torch.export.Dim:
        if feat_prefix not in prefix_to_dim:
            prefix_to_dim[feat_prefix] = _next_dim(0)
        return prefix_to_dim[feat_prefix]

    for key, tensor in sample_data.items():
        if tensor.dim() == 0 or key == "batch_size":
            dynamic_shapes[key] = {}
        elif key in ebc_item_names or key.endswith("__ebc") or key.endswith("__mc_ebc"):
            dynamic_shapes[key] = {0: batch}
        elif key in ebc_user_names or is_input_tile_user_keyed_tensor(key):
            dynamic_shapes[key] = _user_batch_shape(key, tensor)
        elif key in seq_ec_lengths_names or key.endswith("__lengths"):
            raw_name = key[: -len("__lengths")].split("@", 1)[0]
            if resolved_input_tile and raw_name in resolved_hsk:
                dynamic_shapes[key] = _hsk_batch_shape(key, raw_name, tensor)
            elif resolved_input_tile and raw_name in user_feat_names:
                dynamic_shapes[key] = _user_batch_shape(key, tensor)
            else:
                dynamic_shapes[key] = {0: batch}
        elif key in seq_ec_names:
            raw_name = key.split("@", 1)[0]
            if raw_name in mulval_seq_names:
                # Sub-token rows vary independently of sequence step count.
                dynamic_shapes[key] = {0: _next_dim(0)}
            elif raw_name in seq_feat_names:
                dynamic_shapes[key] = {0: _get_total_sequence_length_dim(raw_name)}
            else:
                # Non-sequence Query EC entry inside a sequence group.
                dynamic_shapes[key] = {0: _next_dim(0)}
        elif key.endswith("__inverse_indices"):
            dynamic_shapes[key] = {0: batch}
        else:
            prefix = key
            for suffix in (".values", ".lengths", ".weights", ".key_lengths"):
                if key.endswith(suffix):
                    prefix = key[: -len(suffix)]
                    break
            feat = feat_by_name.get(prefix)
            if key.endswith(".values"):
                if prefix in seq_feat_names and feat is not None and not feat.is_sparse:
                    dynamic_shapes[key] = {0: _get_total_sequence_length_dim(prefix)}
                elif prefix in seq_feat_names and feat is not None and feat.is_sparse:
                    if prefix in mulval_seq_names:
                        dynamic_shapes[key] = {0: _get_or_create_prefix_dim(prefix)}
                    else:
                        dynamic_shapes[key] = {
                            0: _get_total_sequence_length_dim(prefix)
                        }
                elif feat is not None and feat.is_sparse:
                    dynamic_shapes[key] = {0: _get_or_create_prefix_dim(prefix)}
                elif resolved_input_tile and prefix in resolved_hsk:
                    dynamic_shapes[key] = _hsk_batch_shape(key, prefix, tensor)
                elif resolved_input_tile and prefix in user_feat_names:
                    dynamic_shapes[key] = _user_batch_shape(key, tensor)
                else:
                    dynamic_shapes[key] = {0: batch}
            elif key.endswith(".lengths"):
                if resolved_input_tile and prefix in resolved_hsk:
                    dynamic_shapes[key] = _hsk_batch_shape(key, prefix, tensor)
                elif resolved_input_tile and prefix in user_feat_names:
                    dynamic_shapes[key] = _user_batch_shape(key, tensor)
                else:
                    dynamic_shapes[key] = {0: batch}
            elif key.endswith(".key_lengths"):
                dynamic_shapes[key] = {0: _get_total_sequence_length_dim(prefix)}
            elif key.endswith(".weights"):
                dynamic_shapes[key] = {0: _get_or_create_prefix_dim(prefix)}
            else:
                dynamic_shapes[key] = {0: batch}

    return dynamic_shapes


def export_distributed_dense_model_aot(
    model: nn.Module,
    dense_model_traced: torch.fx.GraphModule,
    data: Dict[str, torch.Tensor],
    sparse_kts: Dict[str, Any],
    seq_jts: Dict[str, Any],
    dense_graph_config: Dict[str, Any],
    save_dir: str,
    device: torch.device,
    mixed_precision: Optional[str] = None,
) -> str:
    """Export a distributed-embedding dense serving graph to CUDA AOTInductor.

    Args:
        model: original wrapped model exposing ``features`` / ``feature_groups``.
        dense_model_traced: traced and dead-code-cleaned pure-dense serving
            ``GraphModule`` carrying restored weights on ``device``.
        data: warm-up input dict from ``create_dense_export_warmup_data``.
        sparse_kts: captured ``fx_mark_keyed_tensor`` outputs from sanity run.
        seq_jts: captured ``fx_mark_seq_ec_jt`` outputs from sanity run.
        dense_graph_config: ``dense_meta.json`` mapping.
        save_dir: export output directory receiving ``gm_dense.code`` and
            ``aoti/aoti_model.pt2``.
        device: CUDA device ``dense_model_traced`` lives on.
        mixed_precision: optional ``"BF16"``, ``"FP16"``, or None/empty string.

    Returns:
        ``save_dir``.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("AOTI export requires a CUDA device.")
    if device.type != "cuda":
        raise ValueError(
            "export_distributed_dense_model_aot requires a CUDA device, "
            f"got {device!r}."
        )

    os.makedirs(save_dir, exist_ok=True)
    features = cast(List[BaseFeature], getattr(model, "features", []) or [])
    feature_groups = list(getattr(model, "feature_groups", []) or [])
    input_tile = is_input_tile()
    input_tile_3_online = is_input_tile_3_online()
    hard_search_features = (
        get_hard_search_input_tile_features(features) if input_tile else {}
    )

    dense_model_traced.eval()
    with open(os.path.join(save_dir, "gm_dense.code"), "w") as f:
        f.write(dense_model_traced.code)

    dense_sample_data = _prepare_distributed_dense_sample_inputs(
        gm=dense_model_traced,
        data=data,
        sparse_kts=sparse_kts,
        seq_jts=seq_jts,
        features=features,
        feature_groups=feature_groups,
        device=device,
        input_tile=input_tile,
        input_tile_3_online=input_tile_3_online,
        hard_search_features=hard_search_features,
    )
    dynamic_shapes = _build_distributed_dense_dynamic_shapes(
        sample_data=dense_sample_data,
        dense_graph_config=dense_graph_config,
        features=features,
        feature_groups=feature_groups,
        input_tile=input_tile,
        hard_search_features=hard_search_features,
    )
    logger.info("distributed dense dynamic shapes=%s", dynamic_shapes)

    placeholders = [n for n in dense_model_traced.graph.nodes if n.op == "placeholder"]
    # Bind device="cuda" (not "cuda:0") so the compiled .pt2 works on any GPU.
    device_arg = "cuda" if len(placeholders) >= 2 else ""
    dense_to_export = CudaAutocastWrapper(
        dense_model_traced, mixed_precision, device=device_arg
    ).eval()

    dense_sample_data = {
        k: canonical_contiguous(v) for k, v in dense_sample_data.items()
    }
    logger.info("exporting distributed dense model with torch.export...")
    with torch._inductor.config.patch(
        {"unsafe_ignore_unsupported_triton_autotune_args": True}
    ):
        exported_pg = torch.export.export(
            dense_to_export,
            args=(dense_sample_data,),
            dynamic_shapes=(dynamic_shapes,),
        )

    logger.info("compiling distributed dense model with AOTI...")
    _compile_and_package_aoti(exported_pg, save_dir)

    logger.info("distributed dense AOTI model exported to %s", save_dir)
    return save_dir
