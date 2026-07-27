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
* Streaming sample format: each row carries the history ``user_sequence`` (a
raw-int64 sequence feature) and the ``label`` answer (a ``data_config``
label_field), both holding local, 1-based per-level codes in
``[1, codebook[level]]``. Rows contain whole items in level order.
* The chat template is tokenised ONCE at ``__init__`` and cached as
non-persistent buffers, so per-batch encoding is integer arithmetic only
(no HF tokenizer in the hot path).
* Raw code -> token id by integer offsets:
``token = base_vocab + level_offset[level] + code - 1``. The model derives
``level_offsets`` from ``codebook``; sample writers must not pre-apply them.
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
        cfg = self._model_config
        common = cfg.common  # GenerativeRecLMConfig — shared by all families

        sid_atoms = self._read_common_config(common)

        self.lm = self._build_backbone()
        tokenizer, base = self._build_extended_tokenizer(sid_atoms)
        self._hf_tokenizer = tokenizer
        self._base_vocab = base

        self._pad_token_id = self._resolve_pad_token_id(tokenizer)

        self._build_prompt_tokens(tokenizer, cfg)

        self.init_input()

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
        # History = the single feature_group (its group_name keys the
        # EmbeddingGroup output); answer = the first data_config.label_field.
        hist = self._history_feature_group()
        self._history_group: str = hist.group_name
        self._input_name: str = hist.feature_names[0]
        self._label_name: str = self._labels[0] if self._labels else ""
        self._ignore_index: int = int(common.ignore_index)
        self._generated_sids_key: str = common.generated_sids_key
        param_dtype = self._DTYPE_BY_NAME.get(common.param_dtype)
        if param_dtype is None:
            raise ValueError(
                f"{type(self).__name__}: param_dtype must be one of "
                f"{list(self._DTYPE_BY_NAME)}, got {common.param_dtype!r}."
            )
        self._param_dtype: torch.dtype = param_dtype
        self._max_seq_length: int = int(common.max_sequence_length)
        codebook = list(common.codebook)
        if len(codebook) == 0:
            raise ValueError("GenerativeRecLM: codebook must be non-empty.")
        self._num_levels = len(codebook)
        # Level layout is derived from codebook and kept out of state_dict.
        lo, hi = self._sid_level_bands(codebook)
        self.register_buffer("_level_offsets", lo - 1, persistent=False)
        self.register_buffer("_codebook_sizes", hi - lo + 1, persistent=False)
        self._vocab_pad_mult = int(common.vocab_pad_to_multiple_of) or 128
        return sum(int(c) for c in codebook)

    def _history_feature_group(self) -> Any:
        """The history feature_group = the single declared feature_group.

        genrec declares exactly one feature_group — the JAGGED_SEQUENCE group
        carrying the history SID stream (the answer is a data_config.label_field,
        not a group). Its ``group_name`` keys the EmbeddingGroup output and its one
        member is the history feature. Fails loudly on a missing/empty group (vs a
        silent IndexError/KeyError later).
        """
        if not self._feature_groups:
            raise ValueError(
                f"{type(self).__name__}: no feature_group declared; genrec needs "
                f"one JAGGED_SEQUENCE group carrying the history SID stream."
            )
        g = self._feature_groups[0]
        if not g.feature_names:
            raise ValueError(
                f"{type(self).__name__}: history feature_group {g.group_name!r} "
                f"has no feature_names."
            )
        return g

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

    def hf_backbone(self):
        """The HF backbone module. Only use when export."""
        return self.lm

    def hf_tokenizer(self):
        """The extended tokenizer (base vocab + C0..C{sum-1}) to serialize.

        Only used at export (export_util.write_hf_assets).
        """
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

    def _tokenize_sids(
        self, codes: torch.Tensor, level_ids: torch.Tensor
    ) -> torch.Tensor:
        """Map local 1-based per-level codes to extended-vocab token ids.

        ``level_ids`` must broadcast against ``codes`` and identify each code's
        RQ level. Atom ``C{k}`` sits at ``base_vocab + k``, therefore:

        ``token = code + level_offsets[level] + base_vocab - 1``.
        """
        return codes + self._level_offsets[level_ids] + (self._base_vocab - 1)

    def _detokenize_sids(
        self, tokens: torch.Tensor, level_ids: torch.Tensor
    ) -> torch.Tensor:
        """Inverse of ``_tokenize_sids``: token ids to local 1-based codes."""
        return tokens - (self._base_vocab - 1) - self._level_offsets[level_ids]

    @staticmethod
    def _sid_level_bands(codebook: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-level flattened 1-based SID bands as two long tensors.

        Level ``j`` occupies a DISJOINT band ``[offset_j + 1, offset_j +
        codebook[j]]`` where ``offset_j = sum(codebook[:j])``. Single source of
        truth for the derived ``_level_offsets`` and ``_codebook_sizes`` buffers.
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

    def _sid_token_bands(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the inclusive token-id band for every SID level."""
        level_ids = torch.arange(self._num_levels, device=self.device)
        return (
            self._tokenize_sids(torch.ones_like(level_ids), level_ids),
            self._tokenize_sids(self._codebook_sizes, level_ids),
        )

    def _validate_sid_candidates(
        self, new_tokens: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        """Decode generated tokens to local 1-based codes and reject bad beams.

        ``new_tokens`` is the per-beam tail ``(B*num_return, w)`` (``w`` may be <
        ``num_levels`` when beams stop early). Returns ``(batch_size, num_return,
        num_levels)`` local codes. Every malformed candidate (early EOS /
        non-SID / wrong-level atom) is set to ``-1``, which cannot match a real
        1-based code.
        """
        level_ids = torch.arange(new_tokens.shape[1], device=new_tokens.device)
        codes = self._detokenize_sids(new_tokens, level_ids)
        # early-EOS beams return < num_levels tokens; pad to num_levels with -1.
        codes = F.pad(
            codes,
            (0, self._num_levels - codes.shape[1]),
            value=-1,
        )
        # any single out-of-band atom invalidates the WHOLE candidate (.any over
        # the levels -> masked_fill blanks the entire row to -1, matching no item).
        invalid = ((codes < 1) | (codes > self._codebook_sizes)).any(dim=1)
        codes = codes.masked_fill(invalid.unsqueeze(1), -1)
        # generate() returns rows batch-major ([b0_beam0, b0_beam1, ...]); group
        # the beams per user.
        return codes.view(batch_size, -1, self._num_levels)

    def init_input(self) -> None:
        """Build the EmbeddingGroup for the single raw SID JAGGED_SEQUENCE group.

        Raw (passthrough) features carry no embedding tables, so this
        EmbeddingGroup holds no params (DMP-neutral); it exists purely to
        retrieve the raw SID sequences as flat ``(values, lengths)`` — the same
        path HSTU uses. The HF backbone still owns the token embeddings; SID ids
        flow through it directly (no embedding lookup here).
        """
        self.embedding_group = EmbeddingGroup(self._features, self._feature_groups)

    def build_input(self, batch: Batch) -> Dict[str, List[torch.Tensor]]:
        """Retrieve per-row SID token sequences.

        HISTORY is a JAGGED_SEQUENCE feature_group: ``embedding_group(batch)``
        returns its flat raw values ``"{group}.sequence"`` +
        ``"{group}.sequence_length"`` (the HSTU idiom). The ANSWER is a
        data_config.label_field: its JaggedTensor comes from
        ``batch.jagged_labels[self._label_name]`` (so it can be absent at
        inference, where no ground truth is supplied — unlike a feature_group,
        which the EmbeddingGroup would require every forward). Both are mapped
        SID -> extended-vocab token ids and split to rows; the returned dict is
        keyed by FEATURE name (what the family ``predict`` consumes).
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
            jt = batch.jagged_labels[self._label_name]
            rows[self._label_name] = self._sid_token_rows(
                jt.values(),
                jt.lengths(),
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

        ``build_input`` supplies flat SID ``(values, lengths)``: the history from
        the JAGGED_SEQUENCE group's ``"{group}.sequence"`` /
        ``"{group}.sequence_length"``, and the answer from
        ``batch.jagged_labels[label].values()/.lengths()``. ``values`` may arrive
        as float / shape ``(N, 1)``. The whole batch is tokenized once on the
        backbone device, then split into rows.

        All values are local 1-based codes. Rows must contain whole items so the
        per-level offsets restart at level 0 for each row. ``expected_width``,
        when set, additionally requires exactly that many codes (the answer =
        ``num_levels``).

        ``max_codes``, when set, caps each row to its most-recent whole items (the
        last ``floor(max_codes / num_levels) * num_levels`` codes, dropping the
        oldest head) so the pre-allocated pool covers every batch. Done on host
        views before the H2D copy, skipped unless a row overflows.
        """
        if values.dim() == 2 and values.size(-1) == 1:
            values = values.squeeze(-1)
        sizes = lengths.long().tolist()
        misaligned = [i for i, n in enumerate(sizes) if n % self._num_levels != 0]
        if misaligned:
            raise ValueError(
                f"{type(self).__name__}: SID rows must contain whole "
                f"{self._num_levels}-level items; rows {misaligned} have lengths "
                f"{[sizes[i] for i in misaligned]}."
            )
        if expected_width is not None:
            bad = [i for i, n in enumerate(sizes) if n != expected_width]
            if bad:
                raise ValueError(
                    f"{type(self).__name__}: each SID item must be "
                    f"{expected_width} codes (len(codebook)); rows {bad} have "
                    f"{[sizes[i] for i in bad]} — anomalous sample(s)."
                )

        # TODO: The truncation logic should not be placed here, but should be
        # handled in FG. Since FG currently cannot control the truncation
        # direction (it keeps the HEAD), this may result in truncating the most
        # recent SIDs. Check it.
        if max_codes:
            keep = (max_codes // self._num_levels) * self._num_levels
            if keep and any(n > keep for n in sizes):
                rows = torch.split(values, sizes)
                values = torch.cat([r[-keep:] for r in rows])
                sizes = [min(n, keep) for n in sizes]
        codes = values.to(self.device).long()
        level_ids = torch.arange(codes.numel(), device=self.device) % self._num_levels
        invalid = (codes < 1) | (codes > self._codebook_sizes[level_ids])
        if invalid.any().item():
            raise ValueError(
                f"{type(self).__name__}: SID codes must be local 1-based values "
                f"in [1, codebook[level]]."
            )
        tokens = self._tokenize_sids(codes, level_ids)
        return list(torch.split(tokens, sizes))

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
