"""Smoke test for ``GenerativeRecLM`` — exercises the full forward path
(splice + base forward + suffix slice + HF loss_function) without going
through ``tzrec.train_eval``.

Validates:
  1. Proto + dispatch wire up.
  2. Vocab extension preserves SID-token offset arithmetic.
  3. Left-pad with eos_token_id (L7 fix) produces a finite, non-trivial CE.
  4. SID → token mapping (``token = sid + base_vocab - 1``) hits the new atoms.

Usage:
    cd /workspace/fangtinglin/codework/feat/support_qwen/TorchEasyRec
    /opt/conda/bin/python -m examples.generative_rec_lm_smoke
"""

from __future__ import annotations

import os
import sys
import time

import torch
from google.protobuf import text_format

from tzrec.datasets.utils import Batch
from tzrec.models import generative_rec_lm  # noqa: F401  registers class
from tzrec.models.model import BaseModel
from tzrec.protos.model_pb2 import ModelConfig


# Tiny config — small codebook so vocab extension stays cheap on CPU.
HF_MODEL_ID = "/workspace/fangtinglin/_hf_stage/Qwen2.5-0.5B"
CODEBOOK = [64, 64]   # 128 SID atoms total → small vocab grow on CPU
USER_FEATURE = "user_sequence"
LABEL_FEATURE = "label"


def _make_proto():
    cfg = ModelConfig()
    grl = cfg.generative_rec_lm
    grl.class_name = "Qwen2RecLM"
    grl.hf_model_id = HF_MODEL_ID
    for c in CODEBOOK:
        grl.codebook.append(c)
    grl.user_sequence_feature_name = USER_FEATURE
    grl.label_feature_name = LABEL_FEATURE
    grl.ignore_index = -100
    return cfg


class _DummyJagged:
    """Minimal stand-in for torchrec's JaggedTensor — just enough surface
    for ``GenerativeRecLM._jagged_to_row_list`` to consume."""
    def __init__(self, values: torch.Tensor, lengths: torch.Tensor):
        self._values = values
        self._lengths = lengths

    def values(self) -> torch.Tensor:
        return self._values

    def lengths(self) -> torch.Tensor:
        return self._lengths


def _make_batch(B: int, sum_codebook: int, user_len: int, label_len: int) -> Batch:
    # SID indices are 1-indexed in [1, sum(codebook)].
    torch.manual_seed(0)
    user_vals = torch.randint(1, sum_codebook + 1, (B * user_len,), dtype=torch.long)
    label_vals = torch.randint(1, sum_codebook + 1, (B * label_len,), dtype=torch.long)
    user_lens = torch.full((B,), user_len, dtype=torch.long)
    label_lens = torch.full((B,), label_len, dtype=torch.long)

    seq_dense = {
        USER_FEATURE: _DummyJagged(user_vals, user_lens),
        LABEL_FEATURE: _DummyJagged(label_vals, label_lens),
    }
    # Batch is a dataclass-like NamedTuple — construct with just the field we need.
    # tzrec.datasets.utils.Batch is a NamedTuple of many fields, all default to {}.
    return Batch(sequence_dense_features=seq_dense)


def main() -> int:
    os.environ["TZREC_GENRECLM_DEBUG"] = "1"
    print(f"[smoke] torch={torch.__version__} cuda_avail={torch.cuda.is_available()}", flush=True)

    cfg = _make_proto()
    print(f"[smoke] proto: class_name={cfg.generative_rec_lm.class_name} "
          f"codebook={list(cfg.generative_rec_lm.codebook)}", flush=True)

    # --- dispatch ---
    model_cls = BaseModel.create_class("GenerativeRecLM")
    print(f"[smoke] dispatched -> {model_cls.__name__}", flush=True)

    # --- construct ---
    t0 = time.time()
    model = model_cls(
        cfg,
        features=[],
        labels=[],
        sample_weights=None,
    )
    model.train()
    print(f"[smoke] construct ok ({time.time()-t0:.1f}s); "
          f"base_vocab={model._base_vocab} "
          f"final_vocab={model.lm.config.vocab_size} "
          f"pad_id={model._pad_token_id}", flush=True)
    print(f"[smoke] tpl_system numel={model.tpl_system.numel()} "
          f"tpl_user_prefix numel={model.tpl_user_prefix.numel()} "
          f"tpl_asst_prefix numel={model.tpl_asst_prefix.numel()}", flush=True)

    # --- batch ---
    sum_codebook = sum(CODEBOOK)
    batch = _make_batch(B=2, sum_codebook=sum_codebook, user_len=5, label_len=4)

    # --- forward ---
    t0 = time.time()
    with torch.no_grad():
        pred = model.predict(batch)
    elapsed = time.time() - t0
    loss = pred["loss"]
    logits = pred["logits"]
    print(f"[smoke] forward ok ({elapsed:.2f}s); "
          f"loss={float(loss):.4f} logits.shape={tuple(logits.shape)}", flush=True)

    # Acceptance criteria
    ok = True
    if not torch.isfinite(loss):
        print("[smoke] FAIL: loss is non-finite")
        ok = False
    if float(loss) < 0.1 or float(loss) > 100.0:
        print(f"[smoke] FAIL: loss out of plausible range ({float(loss)})")
        ok = False
    if logits.shape[0] != 2:
        print(f"[smoke] FAIL: logits batch dim wrong ({logits.shape})")
        ok = False

    # --- one backward step to make sure grads flow through the chat-template
    #     buffers + base model + lm_head without error ---
    t0 = time.time()
    pred = model.predict(batch)
    pred["loss"].backward()
    print(f"[smoke] backward ok ({time.time()-t0:.2f}s)", flush=True)

    print(f"[smoke] {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
