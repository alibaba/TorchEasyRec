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
from typing import Any, Dict, Optional, Set

import torch
from safetensors.torch import save_file
from transformers import GenerationConfig, PretrainedConfig

from tzrec.prompt.compile import save_tokenizer_dir
from tzrec.prompt.types import CompiledPrompt
from tzrec.utils.filesystem_util import url_to_fs

SERVING_ARCH = "GenRecForCausalLM"
SERVING_MODEL_TYPE = "genrec"


def dcp_to_hf(ckpt_dir: str, out_dir: str, config: PretrainedConfig) -> None:
    """Write the backbone of a checkpoint as a ``from_pretrained`` safetensors file.

    The backbone's keys are found by suffix, under whatever prefix the training
    wrapper gave them. Keys that do not map 1:1 onto ``config`` raise rather
    than write a partial model. No config is written: the caller composes the
    one ``config.json`` the export carries.

    Args:
        ckpt_dir: the checkpoint the weights come from.
        out_dir: where ``model.safetensors`` is written.
        config: the backbone config, read off the live model.
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
    target_shapes = {k: tuple(v.shape) for k, v in empty.state_dict().items()}
    target_keys: Set[str] = set(target_shapes)
    tied_keys: Set[str] = set(getattr(empty, "_tied_weights_keys", None) or [])
    del empty

    # the mapping is decided on names alone, so the load below reads the
    # backbone and not the sparse tables beside it
    reader = _storage_setup(None, model_ckpt_path, reader=True)
    metadata = reader.read_metadata().state_dict_metadata
    ckpt_keys: Set[str] = set(metadata)

    def _derive_by_suffix() -> Optional[Dict[str, str]]:
        """Each target key is a unique suffix of exactly one DCP key; None if not."""
        out: Dict[str, str] = {}
        for tk in target_keys:
            matches = [k for k in ckpt_keys if k == tk or k.endswith("." + tk)]
            if len(matches) != 1:
                return None
            out[tk] = matches[0]
        return out

    key_map = _derive_by_suffix()
    # one backbone under one prefix: a look-alike key elsewhere cannot stand in
    if key_map is not None and len({ck[: -len(tk)] for tk, ck in key_map.items()}) != 1:
        key_map = None
    if key_map is None:
        raise RuntimeError(
            "dcp_to_hf: cannot map the DCP state dict onto the backbone "
            f"architecture. Wanted {len(target_keys)} keys like "
            f"{sorted(target_keys)[:3]}; the checkpoint holds {len(ckpt_keys)} "
            f"like {sorted(ckpt_keys)[:3]}. Refusing to write a partially-loaded "
            "HF model."
        )
    # names alone would let a resized vocabulary or a swapped backbone through
    drifted = [
        (tk, target_shapes[tk], tuple(metadata[ck].size))
        for tk, ck in key_map.items()
        if tuple(metadata[ck].size) != target_shapes[tk]
    ]
    if drifted:
        raise RuntimeError(
            "dcp_to_hf: the checkpoint's tensors do not fit the exported config, "
            f"e.g. {drifted[:3]} as (key, config shape, checkpoint shape); was it "
            "trained under a different sid_space or backbone? Refusing to write "
            "a partially-loaded HF model."
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
    checkpoint_path: str,
    export_dir: str,
    compiled_prompt: CompiledPrompt,
    backbone_config: PretrainedConfig,
    generation_config: Optional[GenerationConfig],
) -> None:
    """Write what an LLM engine reads beside the scripted front-end.

    The HuggingFace weights, the extended tokenizer and one composite
    ``config.json``: the backbone sits under ``text_config``, which names it to
    a runtime that composes an arbitrary causal LM behind one registered
    architecture and treats a projected prompt slot as a second modality. A
    remote ``export_dir`` is written locally and uploaded, as ``export_model``
    does for its own files.

    Args:
        checkpoint_path: the checkpoint the weights come from.
        export_dir: the export directory.
        compiled_prompt: the prompt the model was built on; its tokenizer and
            SID space are what the engine decodes with.
        backbone_config: the backbone config, read off the live model.
        generation_config: the backbone's generation config, when it has one.
    """
    fs, local_dir = url_to_fs(export_dir)
    if fs is not None:
        local_dir = tempfile.mkdtemp()
    try:
        dcp_to_hf(checkpoint_path, local_dir, backbone_config)
        if generation_config is not None:
            generation_config.save_pretrained(local_dir)
        backbone = json.loads(backbone_config.to_json_string())
        save_tokenizer_dir(compiled_prompt, local_dir)
        composite: Dict[str, Any] = {
            "architectures": [SERVING_ARCH],
            "model_type": SERVING_MODEL_TYPE,
            "text_config": backbone,
            "eos_token_id": compiled_prompt.sid_space.eos_token_id,
            "pad_token_id": compiled_prompt.sid_space.pad_token_id,
        }
        # a runtime that reads only the outer config still needs to size its cache
        for key in ("vocab_size", "hidden_size", "num_hidden_layers", "dtype"):
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
