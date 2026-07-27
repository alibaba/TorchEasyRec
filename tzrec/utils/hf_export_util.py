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

"""HuggingFace export for HF-backed models (``GenerativeRecLM`` family).

Kept out of ``export_util`` (TorchScript/TRT/AOTI) so ``checkpoint_util`` can
call ``write_hf_assets`` without a circular import.
"""

import json
import os
import shutil
from typing import Dict, Optional, Set

import torch
from safetensors.torch import save_file
from torch import nn

from tzrec.utils import checkpoint_util
from tzrec.utils.logging_util import logger

# Missing files are skipped -- tokenizers emit different subsets.
_HF_ASSET_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "special_tokens_map.json",
)

_HF_EXPORT_META_FILENAME = "hf_export_meta.json"


def _unwrap_hf_model(wrapped_model: nn.Module) -> Optional[nn.Module]:
    """Walk DMP/TrainWrapper layers down to the model exposing ``hf_backbone``.

    Returns ``None`` if no backbone is found in the chain -- not an HF-backed
    model, so callers no-op.
    """
    m = wrapped_model
    while not hasattr(m, "hf_backbone"):
        if hasattr(m, "module"):  # DMP / DDP-style wrapper
            m = m.module
        elif hasattr(m, "model"):  # Train/Predict/Script wrapper
            m = m.model
        else:
            return None
    return m


def write_hf_assets(wrapped_model: nn.Module, save_dir: str) -> None:
    """Co-locate the HF config + tokenizer (NO weights) in a checkpoint dir.

    Records the backbone's FQN prefix, read off ``wrapped_model``'s live module
    graph, into ``hf_export_meta.json`` so ``dcp_to_hf`` can strip it without
    hard-coding a wrapper convention. Rank 0 only -- the dense backbone is
    data-parallel-replicated.
    """
    if int(os.environ.get("RANK", 0)) != 0:
        return
    inner = _unwrap_hf_model(wrapped_model)
    if inner is None:
        return
    os.makedirs(save_dir, exist_ok=True)

    backbone = inner.hf_backbone()
    backbone.config.save_pretrained(save_dir)
    gen_cfg = getattr(backbone, "generation_config", None)
    if gen_cfg is not None:
        gen_cfg.save_pretrained(save_dir)
    inner.hf_tokenizer().save_pretrained(save_dir)

    # named_modules() FQNs carry the DMP prefix that state_dict() strips.
    raw_prefix = next(
        (n for n, m in wrapped_model.named_modules() if m is backbone), ""
    )
    prefix = checkpoint_util._strip_dmp_prefix(raw_prefix)
    meta = {"backbone_state_dict_prefix": prefix + ("." if prefix else "")}
    with open(os.path.join(save_dir, _HF_EXPORT_META_FILENAME), "w") as f:
        json.dump(meta, f, indent=2)


def dcp_to_hf(ckpt_dir: str, out_dir: str) -> None:
    """Convert a self-contained checkpoint dir to a ``from_pretrained`` HF dir.

    Reads everything from ``ckpt_dir`` -- no live model is built and no weights
    are downloaded. Keys that do not map 1:1 onto the architecture in the
    co-located ``config.json`` raise rather than write a partial model.
    """
    from torch.distributed.checkpoint.state_dict_loader import (
        _load_state_dict_from_keys,
    )
    from transformers import AutoConfig, AutoModelForCausalLM

    model_ckpt_path = os.path.join(ckpt_dir, "model")
    if not os.path.exists(model_ckpt_path):
        raise RuntimeError(f"dcp_to_hf: model DCP dir [{model_ckpt_path}] not exists.")

    # No keys => load every key; non-distributed => full tensors locally.
    raw_state: Dict[str, torch.Tensor] = _load_state_dict_from_keys(
        checkpoint_id=model_ckpt_path
    )

    meta_path = os.path.join(ckpt_dir, _HF_EXPORT_META_FILENAME)
    prefix: Optional[str] = None
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            prefix = json.load(f).get("backbone_state_dict_prefix")

    cfg = AutoConfig.from_pretrained(ckpt_dir)
    with torch.device("meta"):
        empty = AutoModelForCausalLM.from_config(cfg)
    target_keys: Set[str] = set(empty.state_dict().keys())
    tied_keys: Set[str] = set(getattr(empty, "_tied_weights_keys", None) or [])
    del empty

    def _strip_recorded_prefix(
        state: Dict[str, torch.Tensor],
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Strip the recorded prefix; None unless it yields an EXACT match."""
        if not prefix:
            return None
        out = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
        return out if set(out) == target_keys else None

    def _derive_by_suffix(
        state: Dict[str, torch.Tensor],
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Each target key is a unique suffix of exactly one DCP key; None if not."""
        out: Dict[str, torch.Tensor] = {}
        for tk in target_keys:
            matches = [k for k in state if k == tk or k.endswith("." + tk)]
            if len(matches) != 1:
                return None
            out[tk] = state[matches[0]]
        return out

    # A stale recorded prefix falls back to prefix-free suffix matching.
    mapped = _strip_recorded_prefix(raw_state)
    if mapped is None:
        if prefix:
            logger.warning(
                f"dcp_to_hf: recorded prefix [{prefix}] did not map exactly onto "
                "the architecture; deriving the backbone prefix by suffix-matching."
            )
        mapped = _derive_by_suffix(raw_state)

    if mapped is None or set(mapped.keys()) != target_keys:
        got = set(mapped.keys()) if mapped is not None else set()
        missing = sorted(target_keys - got)
        extra = sorted(got - target_keys)
        raise RuntimeError(
            "dcp_to_hf: cannot map the DCP state dict onto the backbone "
            f"architecture (recorded prefix={prefix!r}). missing={missing[:10]} "
            f"extra={extra[:10]}. Refusing to write a partially-loaded HF model."
        )

    # Tied heads are dropped after validation; from_pretrained re-ties them.
    if getattr(cfg, "tie_word_embeddings", False):
        mapped = {k: v for k, v in mapped.items() if k not in tied_keys}
    mapped = {k: v.contiguous() for k, v in mapped.items()}  # save_file rejects views
    del raw_state
    os.makedirs(out_dir, exist_ok=True)
    save_file(mapped, os.path.join(out_dir, "model.safetensors"))

    for fname in _HF_ASSET_FILES:
        src = os.path.join(ckpt_dir, fname)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(out_dir, fname))
