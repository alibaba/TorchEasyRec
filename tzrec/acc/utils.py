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

import contextlib
import contextvars
import json
import os
from typing import Any, Dict, Iterator, Optional, TypedDict, Union, cast

import torch

from tzrec.protos.export_pb2 import ExportConfig
from tzrec.protos.pipeline_pb2 import EasyRecConfig
from tzrec.protos.train_pb2 import TrainConfig
from tzrec.utils import env_util, quant_util
from tzrec.utils.logging_util import logger


class ExportEnvConfig(TypedDict, total=False):
    """Scoped export environment overrides stored in ``_EXPORT_ENV_CTX``."""

    use_distributed_embedding: bool
    input_tile: str
    input_tile_3_online: bool
    world_size: int
    rank: int


_EXPORT_ENV_CTX: contextvars.ContextVar[Optional[ExportEnvConfig]] = (
    contextvars.ContextVar("tzrec_export_env", default=None)
)


@contextlib.contextmanager
def export_env_scope(
    use_distributed_embedding: Optional[bool] = None,
    input_tile: Optional[str] = None,
    input_tile_3_online: Optional[bool] = None,
    world_size: Optional[int] = None,
    rank: Optional[int] = None,
) -> Iterator[None]:
    """Scope export flags in a ContextVar without mutating ``os.environ``.

    Each argument only overrides when not ``None``. Passing ``None`` leaves the
    currently active value unchanged (enclosing ``export_env_scope``, then
    ``os.environ``) and does not clear or unset an outer scope value.

    Args:
        use_distributed_embedding (bool, optional): override
            ``USE_DISTRIBUTED_EMBEDDING`` within the scope; ``None`` leaves the
            active outer/env value intact.
        input_tile (str, optional): override ``INPUT_TILE`` within the scope;
            ``None`` leaves the active outer/env value intact.
        input_tile_3_online (bool, optional): override ``INPUT_TILE_3_ONLINE``
            within the scope; ``None`` leaves the active outer/env value intact.
        world_size (int, optional): override dataset ``WORLD_SIZE`` within the
            scope; ``None`` leaves the active outer/env value intact.
        rank (int, optional): override dataset ``RANK`` within the scope;
            ``None`` leaves the active outer/env value intact.
    """
    next_ctx: Dict[str, Any] = dict(_EXPORT_ENV_CTX.get() or {})
    if use_distributed_embedding is not None:
        next_ctx["use_distributed_embedding"] = use_distributed_embedding
    if input_tile is not None:
        next_ctx["input_tile"] = input_tile
    if input_tile_3_online is not None:
        next_ctx["input_tile_3_online"] = input_tile_3_online
    if world_size is not None:
        next_ctx["world_size"] = world_size
    if rank is not None:
        next_ctx["rank"] = rank
    token = _EXPORT_ENV_CTX.set(cast(ExportEnvConfig, next_ctx))
    try:
        yield
    finally:
        _EXPORT_ENV_CTX.reset(token)


def _is_distributed_embedding_enabled() -> bool:
    ctx = _EXPORT_ENV_CTX.get() or {}
    if "use_distributed_embedding" in ctx:
        return ctx["use_distributed_embedding"]
    return os.environ.get("USE_DISTRIBUTED_EMBEDDING", "0") == "1"


def use_distributed_embedding() -> bool:
    """Export model for distributed embedding mode of EAS processor."""
    return _is_distributed_embedding_enabled()


def get_input_tile() -> Optional[str]:
    """Return active ``INPUT_TILE`` value from export scope or ``os.environ``."""
    ctx = _EXPORT_ENV_CTX.get() or {}
    if "input_tile" in ctx:
        return ctx["input_tile"]
    return os.environ.get("INPUT_TILE")


def get_world_size() -> int:
    """Return active ``WORLD_SIZE`` from export scope or ``os.environ``."""
    ctx = _EXPORT_ENV_CTX.get() or {}
    if "world_size" in ctx:
        return ctx["world_size"]
    return int(os.environ.get("WORLD_SIZE", 1))


def get_rank() -> int:
    """Return active ``RANK`` from export scope or ``os.environ``."""
    ctx = _EXPORT_ENV_CTX.get() or {}
    if "rank" in ctx:
        return ctx["rank"]
    return int(os.environ.get("RANK", 0))


def is_distributed_embedding_predict(model_path: str) -> bool:
    """Judge is the model exported in distributed embedding mode in predict."""
    with open(model_path + "/model_acc.json", "r", encoding="utf-8") as file:
        data = json.load(file)
    return data.get("DISTRIBUTED_EMBEDDING") == "1"


def is_input_tile() -> bool:
    """Judge is input file or not."""
    input_tile = get_input_tile()
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
    input_tile = get_input_tile()
    if input_tile and input_tile[0] == "3":
        return True
    return False


def is_input_tile_user_keyed_tensor(name: str) -> bool:
    """Return whether a split KeyedTensor name is the INPUT_TILE=3 user side.

    Args:
        name: placeholder or KeyedTensor group name.

    Returns:
        True for ``__ebc_user`` or ``__mc_ebc_user`` suffixes.
    """
    return name.endswith("__ebc_user") or name.endswith("__mc_ebc_user")


def is_input_tile_3_online() -> bool:
    """Judge is the sequential tensor use jt.values() directly or not.

    If INPUT_TILE_3_ONLINE=1, sequential tensor uses jt.values() directly, and
    offline prediction is only supported with batch_size=1 (``tzrec.predict``
    rejects larger batch sizes).
    """
    ctx = _EXPORT_ENV_CTX.get() or {}
    if "input_tile_3_online" in ctx:
        return ctx["input_tile_3_online"]
    input_tile_3_online = os.environ.get("INPUT_TILE_3_ONLINE")
    if input_tile_3_online and input_tile_3_online[0] == "1":
        return True
    return False


def is_input_tile_3_online_predict(model_path: str) -> bool:
    """Return whether exported model enabled ``INPUT_TILE_3_ONLINE``."""
    acc_json_path = os.path.join(model_path, "model_acc.json")
    if not os.path.exists(acc_json_path):
        return False
    with open(acc_json_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    val = data.get("INPUT_TILE_3_ONLINE")
    return bool(val) and str(val)[:1] == "1"


def is_hard_search_input_tile() -> bool:
    """Judge is hard-search input tile or not."""
    if os.environ.get("HARD_SEARCH_INPUT_TILE", "0") != "1":
        return False
    if not is_input_tile():
        raise ValueError("HARD_SEARCH_INPUT_TILE=1 requires INPUT_TILE=2 or 3.")
    if is_trt() or (is_aot() and not is_unified_aot()) or env_util.use_rtp():
        raise ValueError(
            "HARD_SEARCH_INPUT_TILE only supports FX/TorchScript or unified AOT "
            "(ENABLE_AOT=2) export."
        )
    return True


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

    Reads ``ENABLE_AOT`` from model_acc.json (=2 -> unified). Falls back
    to the legacy ``UNIFIED_AOT`` key for pre-fold exports, then to
    file presence (legacy two-stage exports produce
    scripted_sparse_model.pt; unified exports do not) when neither
    JSON key is present -- keeps unit tests that bypass the full export
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
            "heuristic (no scripted_sparse_model.pt -> unified)"
        )
        return True
    return False


def is_aot_predict(model_path: str) -> bool:
    """Judge whether the exported model uses AOTI in predict (ENABLE_AOT={1,2}).

    Reads ``ENABLE_AOT`` (or legacy ``UNIFIED_AOT``) from model_acc.json,
    falling back to file presence (``aoti/aoti_model.pt2`` without
    ``scripted_model.pt``) when model_acc.json is missing or omits both keys.

    Args:
        model_path: exported model directory.

    Returns:
        True when prediction should load an AOTInductor package.
    """
    acc_json_path = os.path.join(model_path, "model_acc.json")
    if os.path.exists(acc_json_path):
        with open(acc_json_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        enable_aot = data.get("ENABLE_AOT")
        if enable_aot is not None:
            return str(enable_aot)[:1] in ("1", "2")
        legacy = data.get("UNIFIED_AOT")
        if legacy is not None:
            return str(legacy)[:1] == "1"
    return os.path.exists(
        os.path.join(model_path, "aoti", "aoti_model.pt2")
    ) and not os.path.exists(os.path.join(model_path, "scripted_model.pt"))


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
    return quant_util.DISTRIBUTED_SPARSE_SUPPORTED_QUANT_FORMATS[0]


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
    Returns ``export_config.mixed_precision`` -- does not fall back to
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


def export_acc_config(
    additional_export_config: Optional[Dict[str, Union[bool, str]]] = None,
    hard_search_input_tile_features: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Export acc config for model online inference.

    Args:
        additional_export_config (dict, optional): extra key/value pairs merged
            into the acc config.
        hard_search_input_tile_features (dict, optional): hard-search feature to
            sequence mapping for HARD_SEARCH_INPUT_TILE.
    """
    ctx = _EXPORT_ENV_CTX.get() or {}
    dist_emb = _is_distributed_embedding_enabled()
    resolved_input_tile = get_input_tile()
    # use int64 sparse id as input
    acc_config: Dict[str, Any] = {"SPARSE_INT64": "1"}
    if resolved_input_tile is not None:
        acc_config["INPUT_TILE"] = resolved_input_tile
    if "input_tile_3_online" in ctx:
        acc_config["INPUT_TILE_3_ONLINE"] = "1" if ctx["input_tile_3_online"] else "0"
    elif "INPUT_TILE_3_ONLINE" in os.environ:
        acc_config["INPUT_TILE_3_ONLINE"] = os.environ["INPUT_TILE_3_ONLINE"]
    if "QUANT_EMB" in os.environ:
        acc_config["QUANT_EMB"] = os.environ["QUANT_EMB"]
    if "QUANT_EC_EMB" in os.environ:
        acc_config["QUANT_EC_EMB"] = os.environ["QUANT_EC_EMB"]
    if dist_emb and is_distributed_sparse_quant():
        acc_config["DIST_QUANT"] = "INT8"
    if "ENABLE_TRT" in os.environ:
        acc_config["ENABLE_TRT"] = os.environ["ENABLE_TRT"]
    if "AOTI_AUTOTUNE_WITH_SAMPLE_INPUTS" in os.environ:
        acc_config["AOTI_AUTOTUNE_WITH_SAMPLE_INPUTS"] = os.environ[
            "AOTI_AUTOTUNE_WITH_SAMPLE_INPUTS"
        ]
    if "MAX_EXPORT_BATCH_SIZE" in os.environ:
        acc_config["MAX_EXPORT_BATCH_SIZE"] = os.environ["MAX_EXPORT_BATCH_SIZE"]
    # Normalize ENABLE_AOT -- write "2" whenever the resolved mode is unified
    # (whether via the new env or the legacy UNIFIED_AOT=1).
    if is_aot():
        acc_config["ENABLE_AOT"] = "2" if is_unified_aot() else "1"
    if additional_export_config:
        acc_config.update(additional_export_config)
    if dist_emb:
        acc_config["DISTRIBUTED_EMBEDDING"] = "1"
    if hard_search_input_tile_features:
        acc_config["hard_search_input_tile"] = {
            "version": 1,
            "features": hard_search_input_tile_features,
        }
    return acc_config


def is_portable_cpu_aot() -> bool:
    """Return whether opt-in portable CPU ISA flags are enabled for AOTI."""
    val = os.environ.get("AOTI_PORTABLE_CPU")
    return bool(val) and val[0] == "1"


def get_aot_cpu_march() -> Optional[str]:
    """Return explicit ``-march`` target for AOTI host compilation when set."""
    val = os.environ.get("AOTI_CPU_MARCH")
    return val if val is not None else None


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


def canonical_contiguous(tensor: torch.Tensor) -> torch.Tensor:
    """Return a tensor with exact row-major C-order strides.

    PyTorch treats size-0/size-1 dimensions as contiguous regardless of their
    stride, so ``.contiguous()`` keeps outer strides (e.g. ``(4, 1)`` on shape
    ``(1, 1)``) from stacked DataParser slices. AOTI runtime input checks
    verify exact C-order strides, so clone into contiguous format when needed.
    """
    tensor = tensor.contiguous()
    if tensor.dim() == 0:
        return tensor
    expected = 1
    strides = tensor.stride()
    shape = tensor.shape
    for i in range(tensor.dim() - 1, -1, -1):
        if strides[i] != expected:
            return tensor.clone(memory_format=torch.contiguous_format)
        expected *= shape[i]
    return tensor
