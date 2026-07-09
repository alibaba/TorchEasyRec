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

# Copyright (c) Alibaba, Inc. and its affiliates.
import json
import os
from typing import Dict, Optional, Union

import torch

from tzrec.protos.export_pb2 import ExportConfig
from tzrec.protos.pipeline_pb2 import EasyRecConfig
from tzrec.protos.train_pb2 import TrainConfig
from tzrec.utils.logging_util import logger


def is_input_tile() -> bool:
    """Judge is input file or not."""
    input_tile = os.environ.get("INPUT_TILE")
    if input_tile and (input_tile[0] == "2" or input_tile[0] == "3"):
        return True
    return False


def is_input_tile_predict(model_path: str) -> bool:
    """Judge is input tile or not in predict."""
    with open(model_path + "/model_acc.json", "r", encoding="utf-8") as file:
        data = json.load(file)
    input_tile = data.get("INPUT_TILE")
    if input_tile and (input_tile[0] == "2" or input_tile[0] == "3"):
        return True
    return False


def is_input_tile_emb() -> bool:
    """Judge is input file or not.

    Embedding Split user/item
    """
    input_tile = os.environ.get("INPUT_TILE")
    if input_tile and input_tile[0] == "3":
        return True
    return False


def is_input_tile_3_online() -> bool:
    """Judge is the sequential tensor use jt.values() directly or not.

    If INPUT_TILE_3_ONLINE=1, sequential tensor use jt.values() directly,
    offline prediction for the model will not be supported.
    """
    input_tile_3_online = os.environ.get("INPUT_TILE_3_ONLINE")
    if input_tile_3_online and input_tile_3_online[0] == "1":
        return True
    return False


def is_aot() -> bool:
    """Judge is inductor or not.

    ENABLE_AOT=1: legacy two-stage export (sparse JIT + dense AOTI).
    ENABLE_AOT=2: unified AOTI export (fused sparse+dense single .pt2).
    Legacy ``UNIFIED_AOT=1`` is honored as an alias of ``ENABLE_AOT=2``.
    """
    val = os.environ.get("ENABLE_AOT")
    if val and val[0] in ("1", "2"):
        return True
    legacy = os.environ.get("UNIFIED_AOT")
    return bool(legacy) and legacy[0] == "1"


def is_unified_aot() -> bool:
    """Judge whether to use the unified AOTI export (ENABLE_AOT=2).

    Legacy ``UNIFIED_AOT=1`` is honored as an alias of ``ENABLE_AOT=2``.
    """
    val = os.environ.get("ENABLE_AOT")
    if val and val[0] == "2":
        return True
    legacy = os.environ.get("UNIFIED_AOT")
    return bool(legacy) and legacy[0] == "1"


def is_unified_aot_predict(model_path: str) -> bool:
    """Judge whether the exported model uses unified AOTI in predict.

    Reads ``ENABLE_AOT`` from model_acc.json (=2 → unified). Falls back
    to the legacy ``UNIFIED_AOT`` key for pre-fold exports, then to
    file presence (legacy two-stage exports produce
    scripted_sparse_model.pt; unified exports do not) when neither
    JSON key is present — keeps unit tests that bypass the full export
    pipeline working.
    """
    acc_json_path = os.path.join(model_path, "model_acc.json")
    if os.path.exists(acc_json_path):
        with open(acc_json_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        enable_aot = data.get("ENABLE_AOT")
        if enable_aot is not None:
            return str(enable_aot)[:1] == "2"
        # Pre-fold compat: model_acc.json may carry UNIFIED_AOT.
        legacy = data.get("UNIFIED_AOT")
        if legacy is not None:
            return str(legacy)[:1] == "1"
    sparse_model_path = os.path.join(model_path, "scripted_sparse_model.pt")
    if not os.path.exists(sparse_model_path):
        logger.warning(
            "model_acc.json missing ENABLE_AOT; falling back to file-presence "
            "heuristic (no scripted_sparse_model.pt → unified)"
        )
        return True
    return False


def is_aot_predict(model_path: str) -> bool:
    """Judge is aot or not in predict (ENABLE_AOT={1,2})."""
    with open(model_path + "/model_acc.json", "r", encoding="utf-8") as file:
        data = json.load(file)
    enable_aot = data.get("ENABLE_AOT")
    if enable_aot and str(enable_aot)[:1] in ("1", "2"):
        return True
    legacy = data.get("UNIFIED_AOT")
    return bool(legacy) and str(legacy)[:1] == "1"


def is_trt() -> bool:
    """Judge is trt or not."""
    is_trt = os.environ.get("ENABLE_TRT")
    if is_trt and is_trt[0] == "1":
        return True
    return False


def is_trt_predict(model_path: str) -> bool:
    """Judge is trt or not in predict."""
    with open(model_path + "/model_acc.json", "r", encoding="utf-8") as file:
        data = json.load(file)
    is_trt = data.get("ENABLE_TRT")
    if is_trt and is_trt[0] == "1":
        return True
    return False


def is_cuda_export() -> bool:
    """Judge is trt/aot or not."""
    return is_trt() or is_aot()


def is_autotune_with_sample_inputs() -> bool:
    """Judge whether AOTI should autotune with realized sample inputs.

    AOTI_AUTOTUNE_WITH_SAMPLE_INPUTS=1: enable inductor's
    ``triton.autotune_with_sample_inputs``. Sample-input autotune walks
    the FX graph via ``torch.fx.Interpreter``, dispatching
    ``_assert_scalar`` / ``sym_constrain_*`` nodes that
    ``_AddRuntimeAssertionsForInlineConstraintsPass`` inserts upstream.
    If a node's predicate evaluates False on the materialized hint --
    typical for HSTU's data-dependent unbacked SymInts when the hint
    exceeds the derived bound -- AOTI compile aborts. Mitigate via
    ``stu.scaling_seqlen`` plus a bumped ``model_config.max_seq_len``
    (see FAQ Q16). AOTI_AUTOTUNE_WITH_SAMPLE_INPUTS=0 (default):
    inductor's default -- autotune benches with random tensors
    (``rand_strided``).
    """
    val = os.environ.get("AOTI_AUTOTUNE_WITH_SAMPLE_INPUTS")
    return bool(val) and val[0] == "1"


def is_debug_trt() -> bool:
    """Judge is debug trt or not.

    Embedding Split user/item
    """
    is_trt = os.environ.get("DEBUG_TRT")
    if is_trt and is_trt[0] == "1":
        return True
    return False


def is_quant() -> bool:
    """Judge is quant or not."""
    is_quant = os.environ.get("QUANT_EMB")
    if is_quant and is_quant[0] == "0":
        return False
    return True


def is_ec_quant() -> bool:
    """Judge EmbeddingCollection is quant or not."""
    is_ec_quant = os.environ.get("QUANT_EC_EMB", "0")
    if is_ec_quant[0] == "0":
        return False
    return True


_quant_str_to_dtype = {
    "FP32": torch.float,
    "FP16": torch.half,
    "INT8": torch.qint8,
    "INT4": torch.quint4x2,
    "INT2": torch.quint2x4,
}


def quant_dtype() -> torch.dtype:
    """Get EmbeddingBagCollection quant dtype."""
    quant_dtype_str = os.environ.get("QUANT_EMB", "INT8")
    if quant_dtype_str == "1":
        # for compatible
        quant_dtype_str = "INT8"
    if quant_dtype_str not in _quant_str_to_dtype:
        raise ValueError(
            f"Unknown QUANT_EMB: {quant_dtype_str},"
            f"available types: {list(_quant_str_to_dtype.keys())}"
        )
    else:
        return _quant_str_to_dtype[quant_dtype_str]


def ec_quant_dtype() -> torch.dtype:
    """Get EmbeddingCollection quant dtype."""
    quant_dtype_str = os.environ.get("QUANT_EC_EMB", "INT8")
    if quant_dtype_str not in _quant_str_to_dtype:
        raise ValueError(
            f"Unknown QUANT_EC_EMB: {quant_dtype_str},"
            f"available types: {list(_quant_str_to_dtype.keys())}"
        )
    else:
        return _quant_str_to_dtype[quant_dtype_str]


DISTRIBUTED_SPARSE_QUANT_FORMAT = "QUint8RowwiseF16"


def _normalized_distributed_sparse_quant() -> str:
    return os.environ.get("DIST_QUANT", "").strip().upper()


def is_distributed_sparse_quant() -> bool:
    """Whether distributed sparse artifacts should be rowwise quantized."""
    quant = _normalized_distributed_sparse_quant()
    if quant in ("", "0", "NONE"):
        return False
    if quant == "INT8":
        return True
    raise ValueError("Unsupported DIST_QUANT: %s, only INT8 is supported." % quant)


def distributed_sparse_quant_format() -> str:
    """Get distributed sparse artifact quantization format."""
    if not is_distributed_sparse_quant():
        return ""
    return DISTRIBUTED_SPARSE_QUANT_FORMAT


_MIXED_PRECISION_TO_DTYPE: Dict[str, torch.dtype] = {
    "BF16": torch.bfloat16,
    "FP16": torch.float16,
}


def mixed_precision_to_dtype(mixed_precision: Optional[str]) -> Optional[torch.dtype]:
    """Convert a TrainConfig.mixed_precision string to a torch dtype.

    Returns ``None`` when ``mixed_precision`` is ``None`` or empty.
    Raises ``ValueError`` on unknown values so typos fail loudly.
    """
    if not mixed_precision:
        return None
    if mixed_precision not in _MIXED_PRECISION_TO_DTYPE:
        raise ValueError(
            f"Unknown mixed_precision: {mixed_precision}, "
            f"available types: {list(_MIXED_PRECISION_TO_DTYPE.keys())}"
        )
    return _MIXED_PRECISION_TO_DTYPE[mixed_precision]


def mixed_precision_for_export(pipeline_config: EasyRecConfig) -> str:
    """Resolve mixed_precision for export.

    Compares ``train_config.mixed_precision`` with
    ``export_config.mixed_precision``; logs a warning when they differ.
    Returns ``export_config.mixed_precision`` — does not fall back to
    train_config. The export-time AMP intent must be expressed
    explicitly on ``export_config``.
    """
    train_mp = pipeline_config.train_config.mixed_precision
    export_mp = (
        pipeline_config.export_config.mixed_precision
        if pipeline_config.HasField("export_config")
        else ""
    )
    if train_mp != export_mp:
        logger.warning(
            "export_config.mixed_precision=%r differs from "
            "train_config.mixed_precision=%r; export uses %r",
            export_mp,
            train_mp,
            export_mp,
        )
    return export_mp


def write_mapping_file_for_input_tile(
    state_dict: Dict[str, torch.Tensor], remap_file_path: str
) -> None:
    r"""Mapping ebc params to ebc_user and ebc_item Updates the model's state.

    dictionary with adapted parameters for the input tile.

    Args:
        state_dict (Dict[str, torch.Tensor]): model state_dict
        remap_file_path (str) : store new_params_name\told_params_name\n
    """
    input_tile_mapping = {
        ".ebc_user.embedding_bags.": ".ebc.embedding_bags.",
        ".mc_ebc_user._embedding_module.": ".mc_ebc._embedding_module.",
        ".mc_ebc_user._managed_collision_collection.": ".mc_ebc._managed_collision_collection.",  # NOQA
        ".ec_list_user.": ".ec_list.",
        ".mc_ec_list_user.": ".mc_ec_list.",
        ".ec_dict_user.": ".ec_dict.",
        ".mc_ec_dict_user.": ".mc_ec_dict.",
    }

    remap_str = ""
    for key, _ in state_dict.items():
        for input_tile_key in input_tile_mapping:
            if input_tile_key in key:
                src_key = key.replace(
                    input_tile_key, input_tile_mapping[input_tile_key]
                )
                remap_str += key + "\t" + src_key + "\n"

    with open(remap_file_path, "w") as f:
        f.write(remap_str)


def export_acc_config(
    additional_export_config: Optional[Dict[str, Union[bool, str]]] = None,
) -> Dict[str, Union[bool, str]]:
    """Export acc config for model online inference.

    Args:
        additional_export_config (dict, optional): extra key/value pairs merged
            into the acc config.
    """
    # use int64 sparse id as input
    acc_config: Dict[str, Union[bool, str]] = {"SPARSE_INT64": "1"}
    if "INPUT_TILE" in os.environ:
        acc_config["INPUT_TILE"] = os.environ["INPUT_TILE"]
    if "QUANT_EMB" in os.environ:
        acc_config["QUANT_EMB"] = os.environ["QUANT_EMB"]
    if "QUANT_EC_EMB" in os.environ:
        acc_config["QUANT_EC_EMB"] = os.environ["QUANT_EC_EMB"]
    if "DIST_QUANT" in os.environ and is_distributed_sparse_quant():
        acc_config["DIST_QUANT"] = "INT8"
    if "ENABLE_TRT" in os.environ:
        acc_config["ENABLE_TRT"] = os.environ["ENABLE_TRT"]
    if "AOTI_AUTOTUNE_WITH_SAMPLE_INPUTS" in os.environ:
        acc_config["AOTI_AUTOTUNE_WITH_SAMPLE_INPUTS"] = os.environ[
            "AOTI_AUTOTUNE_WITH_SAMPLE_INPUTS"
        ]
    if "MAX_EXPORT_BATCH_SIZE" in os.environ:
        acc_config["MAX_EXPORT_BATCH_SIZE"] = os.environ["MAX_EXPORT_BATCH_SIZE"]
    # Normalize ENABLE_AOT — write "2" whenever the resolved mode is unified
    # (whether via the new env or the legacy UNIFIED_AOT=1).
    if is_aot():
        acc_config["ENABLE_AOT"] = "2" if is_unified_aot() else "1"
    if additional_export_config:
        acc_config.update(additional_export_config)
    if os.environ.get("USE_DISTRIBUTED_EMBEDDING", "0") == "1":
        acc_config["DISTRIBUTED_EMBEDDING"] = True
    return acc_config


def get_max_export_batch_size() -> int:
    """Get max export batch size.

    Returns:
        int: max_batch_size
    """
    batch_size = int(os.environ.get("MAX_EXPORT_BATCH_SIZE", 512))
    # compact with old trt batch size config
    if "TRT_MAX_BATCH_SIZE" in os.environ:
        # pyre-ignore [6]
        batch_size = int(os.environ.get("TRT_MAX_BATCH_SIZE"))
    return batch_size


def allow_tf32(config: Union[TrainConfig, ExportConfig]) -> None:
    """Apply TF32 flags from ``config`` to ``torch.backends``.

    The flags are hardware-level matmul precision settings, not
    distributed-backend-specific. Only fields that are explicitly set
    (``HasField``) are applied; unset fields leave torch.backends as-is.
    """
    if config.HasField("cudnn_allow_tf32"):
        torch.backends.cudnn.allow_tf32 = config.cudnn_allow_tf32
    if config.HasField("cuda_matmul_allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = config.cuda_matmul_allow_tf32


def allow_tf32_for_export(pipeline_config: EasyRecConfig) -> None:
    """Apply TF32 flags for AOTI export.

    Warns when ``train_config`` and ``export_config`` are both
    explicitly set on a field and disagree, then applies train_config
    first and export_config second so train applies for fields export
    leaves untouched and export wins where set. Must run BEFORE
    ``torch.export.export()`` so AOTI captures the resolved
    ``torch.backends`` globals into Triton ``ALLOW_TF32`` constexprs.
    """
    train_cfg = pipeline_config.train_config
    export_cfg = pipeline_config.export_config
    for field in ("cudnn_allow_tf32", "cuda_matmul_allow_tf32"):
        if (
            train_cfg.HasField(field)
            and export_cfg.HasField(field)
            and getattr(train_cfg, field) != getattr(export_cfg, field)
        ):
            logger.warning(
                "export_config.%s=%s overrides train_config.%s=%s",
                field,
                getattr(export_cfg, field),
                field,
                getattr(train_cfg, field),
            )
    allow_tf32(train_cfg)
    allow_tf32(export_cfg)
