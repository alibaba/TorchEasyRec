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

"""HuggingFace export for HF-backed models."""

import json
import os
import shutil
import tempfile
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from safetensors.torch import save_file
from torch import nn
from transformers import GenerationConfig, PretrainedConfig

from tzrec.features.feature import BaseFeature
from tzrec.prompt.compile import compile_prompt
from tzrec.protos.pipeline_pb2 import EasyRecConfig
from tzrec.utils import checkpoint_util
from tzrec.utils.filesystem_util import url_to_fs
from tzrec.utils.logging_util import logger

SERVING_ARCH = "GenRecForCausalLM"
SERVING_MODEL_TYPE = "genrec"


def capture_hf_backbone(
    wrapped_model: nn.Module,
) -> Tuple[PretrainedConfig, Optional[GenerationConfig], str]:
    """Read what HF conversion needs off a live model, before export drops it.

    Nothing returned here references the backbone module: the caller deletes it
    right after this call and must be free to.

    Args:
        wrapped_model: the wrapper whose ``state_dict()`` names match the
            checkpoint's.

    Returns:
        The backbone config, its generation config when it has one, and the
        prefix its parameters carry in the checkpoint.

    Raises:
        ValueError: no module in the wrapper chain declares ``hf_backbone``.
    """
    inner = checkpoint_util.unwrap_to(wrapped_model, "hf_backbone")
    if inner is None:
        raise ValueError(
            f"capture_hf_backbone: {type(wrapped_model).__name__} wraps no "
            "module declaring hf_backbone()."
        )
    backbone = inner.hf_backbone()
    # named_modules() FQNs carry the DMP prefix that state_dict() strips.
    raw_prefix = next(
        (n for n, m in wrapped_model.named_modules() if m is backbone), ""
    )
    prefix = checkpoint_util._strip_dmp_prefix(raw_prefix)
    return (
        backbone.config,
        getattr(backbone, "generation_config", None),
        prefix + ("." if prefix else ""),
    )


def dcp_to_hf(
    ckpt_dir: str,
    out_dir: str,
    config: PretrainedConfig,
    state_dict_prefix: str,
) -> None:
    """Write the backbone of a checkpoint as a ``from_pretrained`` safetensors file.

    Keys that do not map 1:1 onto ``config`` raise rather than write a partial
    model. No config is written: the caller composes the one ``config.json``
    the export carries.

    Args:
        ckpt_dir: the checkpoint the weights come from.
        out_dir: where ``model.safetensors`` is written.
        config: the backbone config, read off the live model.
        state_dict_prefix: the prefix the backbone's keys carry in ``ckpt_dir``.
    """
    from torch.distributed.checkpoint.state_dict_loader import (
        _load_state_dict_from_keys,
        _storage_setup,
    )
    from transformers import AutoModelForCausalLM

    model_ckpt_path = os.path.join(ckpt_dir, "model")
    if not os.path.exists(model_ckpt_path):
        raise RuntimeError(f"dcp_to_hf: model DCP dir [{model_ckpt_path}] not exists.")

    with torch.device("meta"):
        empty = AutoModelForCausalLM.from_config(config)
    target_keys: Set[str] = set(empty.state_dict().keys())
    tied_keys: Set[str] = set(getattr(empty, "_tied_weights_keys", None) or [])
    del empty

    # the mapping is decided on names alone, so the load below reads the
    # backbone and not the sparse tables beside it
    reader = _storage_setup(None, model_ckpt_path, reader=True)
    ckpt_keys: Set[str] = set(reader.read_metadata().state_dict_metadata)

    def _strip_model_prefix() -> Optional[Dict[str, str]]:
        """Strip the export-time prefix; None unless it yields an EXACT match."""
        out = {
            k[len(state_dict_prefix) :]: k
            for k in ckpt_keys
            if k.startswith(state_dict_prefix)
        }
        return out if set(out) == target_keys else None

    def _derive_by_suffix() -> Optional[Dict[str, str]]:
        """Each target key is a unique suffix of exactly one DCP key; None if not."""
        out: Dict[str, str] = {}
        for tk in target_keys:
            matches = [k for k in ckpt_keys if k == tk or k.endswith("." + tk)]
            if len(matches) != 1:
                return None
            out[tk] = matches[0]
        return out

    key_map = _strip_model_prefix()
    if key_map is None:
        logger.warning(
            f"dcp_to_hf: export-time prefix [{state_dict_prefix}] did not map "
            "exactly onto the architecture; deriving the backbone prefix by "
            "suffix-matching."
        )
        key_map = _derive_by_suffix()

    if key_map is None:
        raise RuntimeError(
            "dcp_to_hf: cannot map the DCP state dict onto the backbone "
            f"architecture (state_dict_prefix={state_dict_prefix!r}). Wanted "
            f"{len(target_keys)} keys like {sorted(target_keys)[:3]}; the "
            f"checkpoint holds {len(ckpt_keys)} like {sorted(ckpt_keys)[:3]}. "
            "Refusing to write a partially-loaded HF model."
        )

    # non-distributed => full tensors locally
    raw_state: Dict[str, torch.Tensor] = _load_state_dict_from_keys(
        set(key_map.values()), checkpoint_id=model_ckpt_path
    )
    mapped = {tk: raw_state[ck] for tk, ck in key_map.items()}

    # from_pretrained re-ties them.
    if getattr(config, "tie_word_embeddings", False):
        mapped = {k: v for k, v in mapped.items() if k not in tied_keys}
    mapped = {k: v.contiguous() for k, v in mapped.items()}  # save_file rejects views
    del raw_state
    os.makedirs(out_dir, exist_ok=True)
    save_file(mapped, os.path.join(out_dir, "model.safetensors"))


def export_hf_assets(
    pipeline_config: EasyRecConfig,
    features: List[BaseFeature],
    checkpoint_path: str,
    export_dir: str,
    backbone_config: PretrainedConfig,
    generation_config: Optional[GenerationConfig],
    state_dict_prefix: str,
) -> None:
    """Write what an LLM engine reads beside the scripted front-end.

    The HuggingFace weights, the extended tokenizer and one composite
    ``config.json``: the backbone sits under ``text_config``, which names it to
    a runtime that composes an arbitrary causal LM behind one registered
    architecture and treats a projected prompt slot as a second modality. A
    remote ``export_dir`` is written locally and uploaded, as ``export_model``
    does for its own files.

    Args:
        pipeline_config: the pipeline being exported.
        features: the created features the prompt compiles against.
        checkpoint_path: the checkpoint the weights come from.
        export_dir: the export directory.
        backbone_config: the backbone config, read off the live model.
        generation_config: the backbone's generation config, when it has one.
        state_dict_prefix: the prefix the backbone's keys carry in the
            checkpoint.
    """
    fs, local_dir = url_to_fs(export_dir)
    if fs is not None:
        local_dir = tempfile.mkdtemp()
    try:
        dcp_to_hf(checkpoint_path, local_dir, backbone_config, state_dict_prefix)
        if generation_config is not None:
            generation_config.save_pretrained(local_dir)
        backbone = json.loads(backbone_config.to_json_string())
        compiled = compile_prompt(
            pipeline_config.prompt_config,
            features,
            list(pipeline_config.data_config.label_fields),
            tokenizer_dir=local_dir,
        )
        composite: Dict[str, Any] = {
            "architectures": [SERVING_ARCH],
            "model_type": SERVING_MODEL_TYPE,
            "text_config": backbone,
            "eos_token_id": compiled.sid_space.eos_token_id,
            "pad_token_id": compiled.sid_space.pad_token_id,
        }
        # a runtime that reads only the outer config still needs to size its cache
        for key in ("vocab_size", "hidden_size", "num_hidden_layers", "torch_dtype"):
            if key in backbone:
                composite[key] = backbone[key]
        with open(os.path.join(local_dir, "config.json"), "w") as f:
            json.dump(composite, f, indent=2)
        if fs is not None:
            fs.upload(
                local_dir, export_dir, recursive=True, file_thread_num=os.cpu_count()
            )
    finally:
        # the staging dir holds the full LM weights; drop it however this ends
        if fs is not None:
            shutil.rmtree(local_dir, ignore_errors=True)
