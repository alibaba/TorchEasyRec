# Copyright (c) 2026, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0

"""Generic generative-recommendation language-model base for TorchEasyRec.

Implements the FINAL design (see FINAL_DESIGN_GENERATIVE_REC_LM.md):

  * Per-family subclasses (design §2 / G4): ``GenerativeRecLM`` is the
    abstract base; each LLM family is a concrete subclass implementing the
    ``_build_prompt_tokens`` and ``predict`` hooks (e.g. ``Qwen2RecLM`` in
    ``tzrec/models/qwen2_rec_lm.py``). The pipeline config selects the family
    by its own oneof entry (``qwen2_rec_lm``), whose message-type name resolves
    directly to the same-named class via the BaseModel registry — no dispatch.
    Shared config lives in ``GenerativeRecLMConfig`` (the family message's
    ``common`` field); family-specific knobs sit on the family message.
  * Streaming sample format: each row carries two raw-int64 sequence features,
    ``user_sequence`` (list[int]) and ``label`` (list[int]), both holding raw
    SID indices in ``[1, sum(codebook)]``.
  * The chat template is tokenised ONCE at ``__init__`` and cached as
    ``nn.Module`` non-persistent buffers, so per-batch encoding is purely
    integer arithmetic + tensor concatenation (no HF tokenizer in the hot
    path).
  * SID → token id by integer offset: ``token = sid + base_vocab - 1`` (the
    SID atoms ``C0..C{sum-1}`` are added right after the original vocabulary,
    no [SEP] in between; matches algr's ``add_tokens`` layout).
  * Left padding with ``eos_token_id`` (L7 fix from §11 of the design doc) —
    real content sits at the END of every row so the suffix slice captures
    only ``[response + end_markers]`` and matches algr's pad-side exactly.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import torch
import torchmetrics
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from tzrec.datasets.utils import Batch
from tzrec.features.feature import BaseFeature
from tzrec.models.model import BaseModel
from tzrec.protos.model_pb2 import ModelConfig


class GenerativeRecLM(BaseModel):
    """Abstract base for HF-backed generative-recommendation LMs.

    The base owns the architecture-agnostic plumbing: model construction,
    SID vocab extension, the shared sample data-prep (``_sid_token_rows`` /
    ``_tokenize_sids`` — the streaming SID sample contract is the same for all
    families, design §1), loss, and metrics. The two architecture-specific
    pieces are abstract hooks that each family subclass implements (§15/§16):

        _build_prompt_tokens(tokenizer, cfg)  — cache the prompt template
        predict(batch)                        — build inputs + HF forward

    Family proto contract: every family message embeds
    ``GenerativeRecLMConfig common = 1`` (shared config the base reads) and
    supplies a backbone, by default via an ``hf_model_id`` field (overridable
    through ``_backbone_id``).

    ``Qwen2RecLM`` (``tzrec/models/qwen2_rec_lm.py``) provides the decoder-only
    chat implementation (ChatML splice + ``.model``/``.lm_head`` forward),
    reusable by Llama/Mistral/Gemma/Phi-style families; GPT-NeoX/RWKV/Mamba/T5
    each need their own. Each family registers directly (its oneof message-type
    name == the class name); there is no ``class_name`` dispatch.
    """

    # predictions key the inference branch emits generated SIDs under, stable
    # across families (PredictWrapper ``output_cols`` should reference it).
    GENERATED_SIDS_KEY = "generated_sids"

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

        # pad token for the left-padded splice (fall back to eos)
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        self._pad_token_id = int(pad_id)

        self._build_prompt_tokens(tokenizer, cfg)

        # one-shot debug dump of the first spliced batch
        self._smoke_log_once = os.environ.get("TZREC_GENRECLM_DEBUG", "0") == "1"
        self._first_predict = True

    def _read_common_config(self, common: Any) -> int:
        """Parse shared proto knobs into attributes; return the SID atom count."""
        self._input_name: str = common.user_sequence_feature_name
        self._label_name: str = common.label_feature_name
        self._ignore_index: int = int(common.ignore_index)
        # max history (SID codes) for activation pre-sizing = the user-sequence
        # feature's truncation length (the reader caps every row at it). 0 = off.
        self._max_seq_length: int = self._input_sequence_length()
        codebook = list(common.codebook)
        if len(codebook) == 0:
            raise ValueError(
                "GenerativeRecLM: codebook must be non-empty "
                "(see design §3 — required field)"
            )
        # len(codebook) = SID codes per item (answer width); sum = vocab atoms.
        self._num_levels = len(codebook)
        self._vocab_pad_mult = int(common.vocab_pad_to_multiple_of) or 128
        return sum(int(c) for c in codebook)

    def _build_backbone(self) -> Any:
        """Build the EMPTY extended architecture in bf16 — no weight download.

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
        # keep the stored dtype (bf16) so the DCP checkpoint's tensors match on
        # load_state_dict; the default upcasts to fp32 (2x memory on GPU).
        lm = AutoModelForCausalLM.from_config(
            hf_cfg, torch_dtype=hf_cfg.torch_dtype or torch.bfloat16
        )
        if next(lm.parameters()).dtype != torch.bfloat16:
            # Some configs/builders ignore torch_dtype; force bf16 to match DCP.
            lm = lm.to(torch.bfloat16)
        return lm

    def _build_extended_tokenizer(self, sid_atoms: int) -> tuple[Any, int]:
        """Add the SID atoms ``C0..C{sid_atoms-1}`` and resize ``self.lm``.

        Returns ``(tokenizer, base)`` where ``base`` is the tokenizer's next free
        id BEFORE adding the atoms — use ``len(tokenizer)``, NOT
        ``config.vocab_size`` (which counts reserved slots). The atoms append
        directly after the existing vocab (algr's layout), so the splice offset
        is ``token = base + (sid - 1)``.
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

        The SINGLE place ``AutoModelForCausalLM.from_pretrained`` runs. The
        PIPELINE (``tzrec/main.py``) calls this exactly once, only at COLD START
        (no checkpoint to resume). On resume/eval/export the empty arch built in
        ``__init__`` is weight-filled by native DCP ``load_state_dict`` instead,
        so the GB-scale download is skipped. Re-extends the vocab to the SAME
        target/pad as ``__init__`` so the module shapes stay identical.
        """
        # torch_dtype="auto" keeps the stored dtype (bf16); the default upcasts
        # to fp32 (2x memory on GPU).
        lm = AutoModelForCausalLM.from_pretrained(
            self._backbone_id(), torch_dtype="auto"
        )
        lm.resize_token_embeddings(
            self._target_vocab, pad_to_multiple_of=self._vocab_pad_mult
        )
        self.lm = lm

    def _input_sequence_length(self) -> int:
        """Truncation length (SID codes) of the user-sequence feature.

        The data reader caps every row's history at the feature's
        ``sequence_length``, so it is the guaranteed upper bound used to
        pre-size the activation pool (see ``Qwen2RecLM._predict_train``'s
        first-step padding). Returns 0 if the feature has no length cap, which
        disables pre-allocation.
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

        Called from ``__init__`` after vocab extension; the buffers it
        registers are consumed by the family's ``predict``. Architecture-
        specific — see design §15.1/§15.2. Subclasses MUST implement this.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _build_prompt_tokens "
            f"(GenerativeRecLM is abstract)."
        )

    def init_input(self) -> None:
        """Build the native sparse EmbeddingGroup only if declared.

        The HF backbone owns its own ``embed_tokens`` and SID token ids flow
        through it directly, so the dense-only GenerativeRecLM case has no
        sparse feature groups and keeps ``embedding_group = None`` (the dense
        forward never touches it). When a future model declares sparse
        ``feature_groups``, this builds the native ``EmbeddingGroup`` (same as
        ``RankModel.init_input``) so those params flow through the native
        planner/DMP/DCP path alongside the replicated backbone.
        """
        if self._feature_groups:
            # NOTE: enables the native sparse path for future dense+sparse models.
            from tzrec.modules.embedding import EmbeddingGroup

            self.embedding_group = EmbeddingGroup(
                self._features,
                self._feature_groups,
            )
        else:
            self.embedding_group = None

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

    def _sid_token_rows(
        self,
        jt,
        expected_width: Optional[int] = None,
        max_codes: Optional[int] = None,
    ) -> List[torch.Tensor]:
        """Read a SID jagged feature -> per-row token-id tensors.

        TER delivers the feature as a JaggedTensor (flat ``values`` +
        ``lengths``); ``values`` may arrive as float / shape ``(N, 1)``. The
        whole batch is tokenized once (``_tokenize_sids``) on the backbone
        device, then split into rows.

        ``expected_width``, when set, enforces the sample contract here at the
        data boundary: every row must have exactly that many codes (e.g. the
        answer = ``num_levels``); a deviation is an anomalous sample.

        ``max_codes``, when set, caps each row to its most-recent whole items —
        the last ``floor(max_codes / num_levels) * num_levels`` codes, dropping
        the oldest *head* (sequences are oldest->newest, so recent behaviour is
        preserved). FG_NONE does not truncate, so this is what actually enforces
        the feature's ``sequence_length`` — guaranteeing the pre-allocated pool
        covers every batch. Done on host views before the H2D copy, and skipped
        entirely unless some row overflows, so it's free in the common case and
        shrinks downstream work (and forward ``T``) when it fires.
        """
        values = jt.values()
        lengths = jt.lengths()
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
