# Copyright (c) 2026, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0

"""Generic generative-recommendation language-model base for TorchEasyRec.

* Per-family subclasses: ``GenerativeRecLM`` is the abstract base; each LLM
family is a concrete subclass implementing the ``_build_prompt_tokens`` and
``predict`` hooks (e.g. ``Qwen2RecLM``). The pipeline config selects the
family by its own oneof entry, whose message-type name resolves directly to
the same-named class via the BaseModel registry. Shared config lives in
``GenerativeRecLMConfig`` (the family message's ``common`` field).
* Streaming sample format: each row carries two raw-int64 sequence features,
``user_sequence`` and ``label``, both holding raw SID indices in
``[1, sum(codebook)]``.
* The chat template is tokenised ONCE at ``__init__`` and cached as
non-persistent buffers, so per-batch encoding is integer arithmetic only
(no HF tokenizer in the hot path).
* SID -> token id by integer offset: ``token = sid + base_vocab - 1`` (the SID
atoms ``C0..C{sum-1}`` are added right after the original vocabulary).
* Left padding with ``eos_token_id``: real content sits at the END of every
row so the suffix slice captures only ``[response + end_markers]``.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
import torchmetrics
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from tzrec.datasets.utils import Batch
from tzrec.features.feature import BaseFeature
from tzrec.models.model import BaseModel
from tzrec.modules.embedding import EmbeddingGroup
from tzrec.protos.model_pb2 import ModelConfig


class GenerativeRecLM(BaseModel):
    """Abstract base for HF-backed generative-recommendation LMs.

    The base owns the architecture-agnostic plumbing: model construction, SID
    vocab extension, the shared sample data-prep (``_sid_token_rows`` /
    ``_tokenize_sids``), loss, and metrics. Each family subclass implements two
    architecture-specific hooks:

        _build_prompt_tokens(tokenizer, cfg)  — cache the prompt template
        predict(batch)                        — build inputs + HF forward

    Family proto contract: every family message embeds ``GenerativeRecLMConfig
    common = 1`` and supplies a backbone (by default an ``hf_model_id`` field,
    overridable via ``_backbone_id``).

    ``Qwen2RecLM`` provides the decoder-only chat implementation, reusable by
    Llama/Mistral/Gemma/Phi-style families; GPT-NeoX/RWKV/Mamba/T5 each need
    their own. Each family registers directly (oneof message-type name == class
    name).
    """

    # Backbone PARAM dtype options (the fp32 MASTER weights). fp32 avoids
    # bf16-ULP underflow of Adam's small (lr=1e-5) updates; bf16 *compute* comes
    # from mixed_precision:"BF16" autocast, NOT the param dtype. Selected by
    # ``common.param_dtype`` (default "float32") -> ``self._param_dtype``.
    _DTYPE_BY_NAME = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }

    def __init__(
        self,
        model_config: ModelConfig,
        features: List[BaseFeature],
        labels: List[str],
        sample_weights: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_config, features, labels, sample_weights, **kwargs)
        cfg = self._model_config  # family message (e.g. Qwen2RecLM)
        common = cfg.common  # GenerativeRecLMConfig — shared by all families

        sid_atoms = self._read_common_config(common)

        self.lm = self._build_backbone()
        tokenizer, base = self._build_extended_tokenizer(sid_atoms)
        self._hf_tokenizer = tokenizer
        self._base_vocab = base

        self._pad_token_id = self._resolve_pad_token_id(tokenizer)

        self._build_prompt_tokens(tokenizer, cfg)

        # Build the (param-free, raw-passthrough) EmbeddingGroup for the SID
        # JAGGED_SEQUENCE groups — the HSTU retrieval idiom (see init_input).
        self.init_input()

        # one-shot debug dump of the first spliced batch
        self._smoke_log_once = os.environ.get("TZREC_GENRECLM_DEBUG", "0") == "1"
        self._first_predict = True

    @staticmethod
    def _resolve_pad_token_id(tokenizer: Any) -> int:
        """Pad id for the left-padded splice, falling back to eos.

        Fail loudly if a tokenizer has neither (vs an opaque ``int(None)``).
        """
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        if pad_id is None:
            raise ValueError(
                "GenerativeRecLM: tokenizer has neither pad_token_id nor "
                "eos_token_id; cannot choose a pad id for the left-padded splice."
            )
        return int(pad_id)

    def _read_common_config(self, common: Any) -> int:
        """Parse shared proto knobs into attributes; return the SID atom count."""
        self._input_name: str = common.user_sequence_feature_name
        self._label_name: str = common.label_feature_name
        # Which JAGGED_SEQUENCE feature_group carries each SID stream; build_input
        # keys the EmbeddingGroup output by these GROUP names (HSTU idiom),
        # decoupled from the feature names above.
        self._history_group: str = common.history_group_name
        self._label_group: str = common.label_group_name
        self._ignore_index: int = int(common.ignore_index)
        # Inference output key + backbone param dtype (configurable; default
        # "generated_sids" / "float32" = the fp32-master weights).
        self._generated_sids_key: str = common.generated_sids_key
        param_dtype = self._DTYPE_BY_NAME.get(common.param_dtype)
        if param_dtype is None:
            raise ValueError(
                f"{type(self).__name__}: param_dtype must be one of "
                f"{list(self._DTYPE_BY_NAME)}, got {common.param_dtype!r}."
            )
        self._param_dtype: torch.dtype = param_dtype
        # max history (SID codes) for activation pre-sizing = the user-sequence
        # feature's sequence_length. FG_NONE doesn't truncate, so _sid_token_rows
        # enforces this cap (item-aligned) model-side. 0 = off.
        self._max_seq_length: int = self._input_sequence_length()
        codebook = list(common.codebook)
        if len(codebook) == 0:
            raise ValueError("GenerativeRecLM: codebook must be non-empty.")
        # len(codebook) = SID codes per item (answer width); sum = vocab atoms.
        self._num_levels = len(codebook)
        # Per-level SID validity bands (buffers) for the inference gate
        # (_validate_sid_candidates). Non-persistent: derived config.
        lo, hi = self._sid_level_bands(codebook)
        self.register_buffer("_sid_lvl_lo", lo, persistent=False)
        self.register_buffer("_sid_lvl_hi", hi, persistent=False)
        self._vocab_pad_mult = int(common.vocab_pad_to_multiple_of) or 128
        return sum(int(c) for c in codebook)

    def _build_backbone(self) -> Any:
        """Build the EMPTY extended architecture in fp32 (master) — no download.

        Config reads only define the module shapes the DCP checkpoint expects;
        the GB-scale pretrained weights load once at cold start via
        ``init_from_pretrained``, then DCP fills them on every restore/eval.
        """
        hf_model_id = self._backbone_id()
        if not hf_model_id:
            raise ValueError(
                f"{type(self).__name__}: empty backbone id (see _backbone_id)."
            )
        hf_cfg = AutoConfig.from_pretrained(hf_model_id)
        # fp32 MASTER weights: bf16 params underflow Adam's small (lr=1e-5) updates
        # and freeze. bf16 compute comes from mixed_precision:"BF16"; ckpt stays fp32.
        lm = AutoModelForCausalLM.from_config(hf_cfg, torch_dtype=self._param_dtype)
        if next(lm.parameters()).dtype != self._param_dtype:
            lm = lm.to(self._param_dtype)
        return lm

    def _build_extended_tokenizer(self, sid_atoms: int) -> tuple[Any, int]:
        """Add the SID atoms ``C0..C{sid_atoms-1}`` and resize ``self.lm``.

        Returns ``(tokenizer, base)`` where ``base`` is the tokenizer's next free
        id BEFORE adding the atoms — use ``len(tokenizer)``, NOT
        ``config.vocab_size`` (which counts reserved slots). The atoms append
        directly after the existing vocab, so the splice offset is
        ``token = base + (sid - 1)``.
        """
        tokenizer = AutoTokenizer.from_pretrained(self._backbone_id(), use_fast=True)
        base = len(tokenizer)
        added = tokenizer.add_tokens([f"C{i}" for i in range(sid_atoms)])
        if added != sid_atoms:
            # pre-existing Cxxx tokens would silently break the offset arithmetic.
            raise RuntimeError(
                f"GenerativeRecLM: tokenizer was expected to grow by "
                f"{sid_atoms} new atoms, only added {added}. "
                f"Aborting to avoid silent SID-token mismatch."
            )
        # stash the resize target so init_from_pretrained re-extends identically.
        self._target_vocab = base + sid_atoms
        self.lm.resize_token_embeddings(
            self._target_vocab, pad_to_multiple_of=self._vocab_pad_mult
        )
        c0_id = tokenizer.convert_tokens_to_ids("C0")
        if c0_id != base:
            raise RuntimeError(
                f"GenerativeRecLM: SID atom layout mismatch — expected "
                f"C0 at token id {base}, got {c0_id}. "
                f"Splice arithmetic would produce wrong token ids."
            )
        return tokenizer, base

    def _backbone_id(self) -> str:
        """Family hook: the HF model id to load for ``self.lm``.

        Defaults to the family message's ``hf_model_id``; override if a family
        sources its backbone differently.
        """
        return self._model_config.hf_model_id

    def init_from_pretrained(self) -> None:
        """Load the pretrained HF backbone weights into ``self.lm``.

        The single ``from_pretrained`` call, run once at COLD START (no checkpoint
        to resume); on resume/eval/export the empty ``__init__`` arch is filled by
        DCP instead, skipping the download. Re-extends the vocab to ``__init__``'s
        target so the shapes match.
        """
        # fp32 master (not "auto", which keeps the stored bf16); must match
        # _build_backbone so cold-start and restore arches agree.
        lm = AutoModelForCausalLM.from_pretrained(
            self._backbone_id(), torch_dtype=self._param_dtype
        )
        lm.resize_token_embeddings(
            self._target_vocab, pad_to_multiple_of=self._vocab_pad_mult
        )
        self.lm = lm

    def _input_sequence_length(self) -> int:
        """The user-sequence feature's ``sequence_length`` (SID codes), or 0.

        FG_NONE does NOT truncate, so this is the cap ``_sid_token_rows`` enforces
        model-side (item-aligned), and the upper bound the activation pool is
        pre-sized to (see ``Qwen2RecLM._predict_train``). 0 if the feature has no
        length cap, which disables pre-allocation.
        """
        for feature in self._features:
            if feature.config.feature_name == self._input_name:
                return int(getattr(feature, "sequence_length", 0) or 0)
        return 0

    def hf_backbone(self):
        """The HF backbone module (``export_util.write_hf_assets``/``dcp_to_hf``)."""
        return self.lm

    def hf_tokenizer(self):
        """The extended tokenizer (base vocab + C0..C{sum-1}) to serialize."""
        return self._hf_tokenizer

    def _build_prompt_tokens(self, tokenizer, cfg) -> None:
        """Family hook: cache the tokenised prompt template as buffers.

        Called from ``__init__`` after vocab extension; consumed by the family's
        ``predict``. Subclasses MUST implement this.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _build_prompt_tokens "
            f"(GenerativeRecLM is abstract)."
        )

    @property
    def device(self) -> torch.device:
        """Device the HF backbone runs on — the single source for model I/O."""
        return self.lm.device

    def _tokenize_sids(self, sids: torch.Tensor) -> torch.Tensor:
        """Map raw 1-indexed SID values to extended-vocab token ids.

        Atom ``C{k}`` sits at ``base_vocab + k`` (atoms appended right after the
        original vocab), so ``token_id = sid + base_vocab - 1``. The integer
        counterpart of the HF tokenizer used for text; shape-agnostic.
        """
        return sids + (self._base_vocab - 1)

    def _detokenize_sids(self, tokens: torch.Tensor) -> torch.Tensor:
        """Inverse of ``_tokenize_sids``: token id -> raw 1-indexed SID."""
        return tokens - (self._base_vocab - 1)

    @staticmethod
    def _sid_level_bands(codebook: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-level closed SID bands ``(lo, hi)`` as ``(num_levels,)`` long tensors.

        Level ``j`` occupies a DISJOINT band ``[offset_j + 1, offset_j +
        codebook[j]]`` where ``offset_j = sum(codebook[:j])``. Single source of
        truth: both ``_read_common_config`` and the tests build bands from this.
        """
        lo, hi, acc = [], [], 0
        for c in codebook:
            lo.append(acc + 1)
            hi.append(acc + int(c))
            acc += int(c)
        return (
            torch.tensor(lo, dtype=torch.long),
            torch.tensor(hi, dtype=torch.long),
        )

    def _validate_sid_candidates(
        self, new_tokens: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        """Map a generated token tail back to SIDs and reject malformed candidates.

        ``new_tokens`` is the per-beam tail ``(B*num_return, w)`` (``w`` may be <
        ``num_levels`` when beams stop early). Returns ``(batch_size, num_return,
        num_levels)`` SIDs with every malformed candidate (early EOS / non-SID /
        wrong-level atom) set to ``-1`` — which can never match a real item.
        """
        sids = self._detokenize_sids(new_tokens)
        # pad each candidate to exactly num_levels with the -1 sentinel (an
        # early-EOS beam returns fewer tokens) so the reshape stays rectangular.
        sids = F.pad(sids, (0, self._num_levels - sids.shape[1]), value=-1)
        # valid only if every position-j atom is in level j's band; any violation
        # invalidates the WHOLE candidate -> -1.
        invalid = ((sids < self._sid_lvl_lo) | (sids > self._sid_lvl_hi)).any(dim=1)
        sids = sids.masked_fill(invalid.unsqueeze(1), -1)
        # generate() returns rows batch-major ([b0_beam0, b0_beam1, ...]) so this
        # groups beams per user; the row-wise mask above preserved that order.
        return sids.view(batch_size, -1, self._num_levels)

    def init_input(self) -> None:
        """Build the EmbeddingGroup for the raw SID JAGGED_SEQUENCE groups.

        Raw (passthrough) features carry no embedding tables, so this
        EmbeddingGroup holds no params (DMP-neutral); it exists purely to
        retrieve the raw SID sequences as flat ``(values, lengths)`` — the same
        path HSTU uses. The HF backbone still owns the token embeddings; SID ids
        flow through it directly (no embedding lookup here).
        """
        self.embedding_group = EmbeddingGroup(self._features, self._feature_groups)

    def build_input(self, batch: Batch) -> Dict[str, List[torch.Tensor]]:
        """Retrieve per-row SID token sequences via the EmbeddingGroup.

        ``embedding_group(batch)`` returns, per JAGGED_SEQUENCE group, the flat
        raw values ``"{group}.sequence"`` + ``"{group}.sequence_length"`` (the
        HSTU idiom). We map SID indices -> extended-vocab token ids and split to
        rows. The EmbeddingGroup output is keyed by GROUP name
        (``_history_group`` / ``_label_group``); the returned dict is keyed by
        FEATURE name (what the family ``predict`` consumes). The label is omitted
        in inference, where no ground truth is supplied.
        """
        g = self.embedding_group(batch)
        rows: Dict[str, List[torch.Tensor]] = {
            self._input_name: self._sid_token_rows(
                g[f"{self._history_group}.sequence"],
                g[f"{self._history_group}.sequence_length"],
                max_codes=self._max_seq_length,
            ),
        }
        if not self.is_inference:
            rows[self._label_name] = self._sid_token_rows(
                g[f"{self._label_group}.sequence"],
                g[f"{self._label_group}.sequence_length"],
                expected_width=self._num_levels,
            )
        return rows

    def _sid_token_rows(
        self,
        values: torch.Tensor,
        lengths: torch.Tensor,
        expected_width: Optional[int] = None,
        max_codes: Optional[int] = None,
    ) -> List[torch.Tensor]:
        """Map flat SID ``(values, lengths)`` -> per-row token-id tensors.

        ``build_input`` supplies a JAGGED_SEQUENCE group's flat
        ``"{group}.sequence"`` values + ``"{group}.sequence_length"``; ``values``
        may arrive as float / shape ``(N, 1)``. The whole batch is tokenized once
        on the backbone device, then split into rows.

        ``expected_width``, when set, enforces the sample contract: every row must
        have exactly that many codes (the answer = ``num_levels``).

        ``max_codes``, when set, caps each row to its most-recent whole items (the
        last ``floor(max_codes / num_levels) * num_levels`` codes, dropping the
        oldest head) so the pre-allocated pool covers every batch. Done on host
        views before the H2D copy, skipped unless a row overflows.
        """
        if values.dim() == 2 and values.size(-1) == 1:
            values = values.squeeze(-1)
        # host-side split bounds, read before the H2D copy below
        sizes = lengths.long().tolist()
        if expected_width is not None:
            bad = [i for i, n in enumerate(sizes) if n != expected_width]
            if bad:
                raise ValueError(
                    f"{type(self).__name__}: each SID item must be "
                    f"{expected_width} codes (len(codebook)); rows {bad} have "
                    f"{[sizes[i] for i in bad]} — anomalous sample(s)."
                )
        if max_codes:
            keep = (max_codes // self._num_levels) * self._num_levels
            if keep and any(n > keep for n in sizes):
                rows = torch.split(values, sizes)  # host views, no copy
                values = torch.cat([r[-keep:] for r in rows])  # keep recent tail
                sizes = [min(n, keep) for n in sizes]
        # one vectorized SID->token map over the whole batch, on the backbone device
        values = self._tokenize_sids(values.to(self.device).long())
        return list(torch.split(values, sizes))

    def predict(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """Family hook: build inputs, run the HF forward, return ``{"loss": ...}``.

        Architecture-specific — the decoder-only implementation lives in
        ``Qwen2RecLM``; other families (GPT-NeoX, Mamba, T5, …) need their own.
        See design §15.4/§16. Subclasses MUST implement this.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement predict "
            f"(GenerativeRecLM is abstract)."
        )

    def init_loss(self) -> None:
        """No-op: the loss is computed inside ``predict`` (HF loss_function)."""
        return

    def loss(
        self,
        predictions: Dict[str, torch.Tensor],
        batch: Batch,
    ) -> Dict[str, torch.Tensor]:
        """Surface the CE loss already computed in ``predict``."""
        return {"ce_loss": predictions["loss"]}

    # BaseModel declares only the eval-side metric hooks, but the train loop
    # calls both eval and train hooks, so both are overridden here.
    def init_metric(self) -> None:
        """Register a mean-CE metric for the eval loop."""
        self._metric_modules["ce_loss"] = torchmetrics.MeanMetric()

    def update_metric(
        self,
        predictions: Dict[str, torch.Tensor],
        batch: Batch,
        losses: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        """Update the mean-CE metric with this batch's loss."""
        self._metric_modules["ce_loss"].update(predictions["loss"].detach())

    def init_train_metric(self) -> None:
        """No-op: no train-time metric beyond the logged CE loss."""
        return

    def update_train_metric(
        self,
        predictions: Dict[str, torch.Tensor],
        batch: Batch,
        losses: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        """No-op: no train-time metric beyond the logged CE loss."""
        return
