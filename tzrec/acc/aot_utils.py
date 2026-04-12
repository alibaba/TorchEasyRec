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
from typing import Any, Dict, Optional, Union

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
        # Legacy two-stage path: sparse JIT + dense AOTI
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

    # Wrap the dense module so torch.export captures the autocast region
    # as a `wrap_with_autocast` HOP that AOT Inductor lowers correctly.
    dense_to_export: nn.Module = dense_model
    if mixed_precision:
        dense_to_export = CudaAutocastWrapper(dense_model, mixed_precision)

    # Dry-run the wrapped module to capture output field names. Must run
    # through dense_to_export (not dense_model) so the autocast context is
    # active — kernels like CUTLASS HSTU attention reject fp32 inputs.
    with torch.no_grad():
        _out = dense_to_export(sparse_output)
        aoti_output_keys = list(_out.keys())
        del _out

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

        # Save original model output field names to aoti directory
        if aoti_output_keys:
            output_names_path = os.path.join(aoti_dir, "output_field_names.json")
            with open(output_names_path, "w") as f:
                json.dump(aoti_output_keys, f, indent=4)
            logger.info(
                f"Saved output field names to {output_names_path}: {aoti_output_keys}"
            )

        torch._inductor.aoti_compile_and_package(
            exported_pg,
            package_path=os.path.join(aoti_dir, "aoti_model.pt2"),
        )
    return save_dir


_fbgemm_patched = False


def _patch_fbgemm_fake_impls() -> None:
    """Patch fbgemm fake/meta implementations that create unbacked SymInts.

    ``keyed_jagged_index_select_dim1``'s abstract function calls
    ``.sum().item()`` to compute ``selected_lengths_sum``. During
    ``torch.export``, ``.item()`` on a FakeTensor scalar creates an
    unbacked SymInt that downstream ops can't guard on.

    The fix: drop ``.item()`` so the value stays as a 0-dim tensor
    (backed SymInt) that ``torch.export`` can trace symbolically.
    """
    global _fbgemm_patched
    if _fbgemm_patched:
        return
    _fbgemm_patched = True

    try:
        from fbgemm_gpu.sparse_ops import (
            keyed_jagged_index_select_dim1_abstract,
        )
    except ImportError:
        logger.debug("fbgemm_gpu.sparse_ops not available; skipping patch")
        return

    import inspect

    src = inspect.getsource(keyed_jagged_index_select_dim1_abstract)
    if ".sum().item()" not in src:
        logger.debug("fbgemm fake impl already patched or changed; skipping")
        return

    # Monkey-patch: replace .sum().item() with .sum() in the function

    _orig_fn = keyed_jagged_index_select_dim1_abstract

    def _patched_keyed_jagged_index_select_dim1_abstract(
        values,
        lengths,
        offsets,
        indices,
        batch_size,
        weights=None,
        selected_lengths_sum=None,
    ):
        num_batches = len(lengths) // batch_size
        torch._check(len(lengths) + 1 == len(offsets))
        torch._check(len(lengths) % batch_size == 0)
        if weights is not None:
            torch._check(values.shape == weights.shape)

        if selected_lengths_sum is None:
            length_indices = torch.cat(
                [indices + i * batch_size for i in range(num_batches)]
            )
            # Use .sum() without .item() to keep value as backed SymInt
            selected_lengths_sum = torch.index_select(lengths, 0, length_indices).sum()

        ret = [
            values.new_empty([selected_lengths_sum]),
            lengths.new_empty([indices.shape[0] * num_batches]),
        ]
        if weights is not None:
            ret.append(weights.new_empty([selected_lengths_sum]))
        return ret

    # Can't re-register Meta (torch.library.impl rejects double-reg).
    # Instead, replace the function's __code__ so the registered wrapper
    # calls our patched body.
    _orig_fn.__code__ = _patched_keyed_jagged_index_select_dim1_abstract.__code__
    logger.info("patched fbgemm keyed_jagged_index_select_dim1 fake impl")


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

    # Trace the full model. The torchrec-aware symbolic_trace decomposes
    # quantized EBC ops (e.g. fbgemm.bounds_check_indices) into primitives,
    # which is required for torch.export.export() to functionalize them.
    logger.info("tracing full model for unified AOTI export...")
    full_gm = symbolic_trace(trace_root)
    with open(os.path.join(save_dir, "gm.code"), "w") as f:
        f.write(full_gm.code)

    # Verify the unified model produces correct output.
    result = full_gm(data)
    result_info = {k: (v.size(), v.dtype) for k, v in result.items()}
    logger.info(f"Unified Model Outputs: {result_info}")
    aoti_output_keys = list(result.keys())
    del result

    # Build dynamic shapes using feature metadata for correct Dim grouping
    dynamic_shapes = _build_dynamic_shapes(
        data,
        features=model._features,
        model_config=model.model._base_model_config,
    )
    logger.info("dynamic shapes=%s" % dynamic_shapes)

    from tzrec.acc.torchrec_export_patches import export_patches

    logger.info("exporting unified model with torch.export...")
    with export_patches():
        with torch._inductor.config.patch(
            {"unsafe_ignore_unsupported_triton_autotune_args": True}
        ):
            exported_pg = torch.export.export(
                full_gm,
                args=(data,),
                dynamic_shapes=(dynamic_shapes,),
                strict=False,
            )

        # Compile with AOTI (inside export_patches for fbgemm schema)
        logger.info("compiling unified model with AOTI...")
        with torch._inductor.config.patch(
            {
                "scalar_asserts": False,
                "unsafe_ignore_unsupported_triton_autotune_args": True,
            }
        ):
            aoti_dir = os.path.join(save_dir, "aoti")
            os.makedirs(aoti_dir, exist_ok=True)

            if aoti_output_keys:
                output_names_path = os.path.join(aoti_dir, "output_field_names.json")
                with open(output_names_path, "w") as f:
                    json.dump(aoti_output_keys, f, indent=4)
                logger.info(
                    "Saved output field names to %s: %s",
                    output_names_path,
                    aoti_output_keys,
                )

            torch._inductor.aoti_compile_and_package(
                exported_pg,
                package_path=os.path.join(aoti_dir, "aoti_model.pt2"),
            )

    logger.info("unified AOTI model exported to %s", save_dir)
    return save_dir
