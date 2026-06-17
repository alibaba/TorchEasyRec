# Copyright (c) 2026, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0

"""Qwen2/Qwen2.5 family subclass of ``GenerativeRecLM``.

Selected from the pipeline config by its own oneof entry (the message-type name
resolves directly to this class)::

    model_config {
        qwen2_rec_lm {
            common { hf_model_id: "..." codebook: 8192 ... }
            system_instruction: "..."
        }
    }

This subclass owns the decoder-only-chat implementation: the ChatML prompt
template, the causal-LM splice, and the ``.model``/``.lm_head`` forward. The
``GenerativeRecLM`` base owns the architecture-agnostic plumbing (vocab
extension, jagged->row, loss, metrics).

The splice/forward are generic to decoder-only families sharing Qwen2's
``.model``/``.lm_head`` layout (Llama/Mistral/Gemma/Phi); only ``QWEN2_TEMPLATE``
is Qwen2-specific.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence

from tzrec.datasets.utils import Batch
from tzrec.features.feature import BaseFeature
from tzrec.models.generative_rec_lm import GenerativeRecLM
from tzrec.protos.model_pb2 import ModelConfig


def _encode_no_special(tokenizer, text: str) -> List[int]:
    """Encode a fragment without prepending BOS / appending EOS specials.

    We're building the prompt manually from explicit ``<|im_start|>`` markers,
    so we must NOT let the tokenizer's BOS/EOS handling double-emit them.
    """
    return tokenizer.encode(text, add_special_tokens=False)

# Verbatim Qwen2 ChatML fragments.
QWEN2_TEMPLATE = {
    "system_prefix": "<|im_start|>system\n",
    "system_suffix": "<|im_end|>\n",
    "user_prefix": "<|im_start|>user\n",
    "user_suffix": "<|im_end|>\n",
    "asst_prefix": "<|im_start|>assistant\n",
    "asst_suffix": "<|im_end|>\n",
    "default_system_instruction": (
        "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    ),
}


class Qwen2RecLM(GenerativeRecLM):
    """Qwen2 / Qwen2.5 generative-recommendation LM."""

    CHAT_TEMPLATE = QWEN2_TEMPLATE

    def __init__(
        self,
        model_config: ModelConfig,
        features: List[BaseFeature],
        labels: List[str],
        sample_weights: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_config, features, labels, sample_weights, **kwargs)
        common = self._model_config.common
        # generation params, consumed only by this family's _generate.
        self._num_beams = int(common.num_beams)
        self._num_return = int(common.num_return_sequences)
        # worst-case spliced length for the first-step activation-pool pre-sizing.
        self._max_total_len = self._compute_max_total_length()
        self._pool_warmed = False
        # CE suffix width: the supervised tail [answer | asst_suffix | eos] is
        # fixed, so _forward_loss slices a constant suffix (no per-step sync).
        self._suffix_keep = self._num_levels + self.tpl_asst_suffix.numel() + 2

    def _compute_max_total_length(self) -> int:
        """Full spliced length at the max history (0 if pre-allocation is off).

        Fixed ChatML frame + ``self._max_seq_length`` history codes + the
        ``num_levels``-code answer: the ``T`` the activation pool is pre-sized to.
        """
        if self._max_seq_length <= 0:
            return 0
        frame = (
            self.tpl_system.numel()
            + self.tpl_user_prefix.numel()
            + self.tpl_user_suffix.numel()
            + self.tpl_asst_prefix.numel()
            + self.tpl_asst_suffix.numel()
            + self.tpl_eos.numel()
        )
        return int(frame + self._max_seq_length + self._num_levels)

    def _build_prompt_tokens(self, tokenizer, cfg) -> None:
        """Tokenise the family chat template once; cache as buffers.

        Composes the proto's optional ``system_instruction`` /
        ``user_prefix_text`` / ``user_suffix_text`` with the family's static
        fragments::

            tpl_system      = system_prefix + system_instruction + system_suffix
            tpl_user_prefix = user_prefix + user_prefix_text
            tpl_user_suffix = user_suffix_text + user_suffix
            tpl_asst_prefix / tpl_asst_suffix verbatim from the template

        Buffers are non-persistent: they move with ``model.to(...)`` but stay off
        the state_dict so HF safetensors round-tripping isn't polluted.
        """
        tpl = type(self).CHAT_TEMPLATE
        sys_text = cfg.system_instruction or tpl["default_system_instruction"]
        u_pre = cfg.user_prefix_text or ""
        u_suf = cfg.user_suffix_text or ""
        frags = {
            "system": tpl["system_prefix"] + sys_text + tpl["system_suffix"],
            "user_prefix": tpl["user_prefix"] + u_pre,
            "user_suffix": u_suf + tpl["user_suffix"],
            "asst_prefix": tpl["asst_prefix"],
            "asst_suffix": tpl["asst_suffix"],
        }
        for slot_name, frag_str in frags.items():
            ids = torch.tensor(
                _encode_no_special(tokenizer, frag_str), dtype=torch.long
            )
            self.register_buffer(f"tpl_{slot_name}", ids, persistent=False)
        # the trailing eos is a SUPERVISED token; cache it for the splice.
        self.register_buffer(
            "tpl_eos",
            torch.tensor([int(tokenizer.eos_token_id)], dtype=torch.long),
            persistent=False,
        )

    def _splice_input_ids(
        self,
        user_seq_rows: List[torch.Tensor],
        label_rows: List[torch.Tensor],
        pad_to: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build ``(input_ids, labels, attention_mask)``, each ``(B, T_max)``.

        Left-padded with ``eos_token_id``. ``attention_mask`` is essential —
        without it self-attention lets pad positions pollute real positions; CE
        is separately protected by ``-100`` labels at pad slots.

        Every answer is exactly ``self._num_levels`` SID codes, so the supervised
        tail ``[answer | asst_suffix | eos]`` has a FIXED width and lands in the
        same columns for every row after left-padding -> ``labels`` is one
        vectorized write. ``input_ids`` still varies per row (history length).

        ``user_seq_rows`` / ``label_rows`` already hold token ids on the model
        device (see ``_sid_token_rows``). ``pad_to`` left-extends every row for
        first-step pool pre-sizing; the supervised tail stays end-aligned.
        """
        assert len(user_seq_rows) == len(label_rows)
        A = self._num_levels

        # input_ids: assembled per row (user history length varies), then
        # left-padded into a (B, T) batch (real content right-aligned).
        rows_ids = [
            torch.cat([
                self.tpl_system, self.tpl_user_prefix, user_seq_rows[i],
                self.tpl_user_suffix, self.tpl_asst_prefix, label_rows[i],
                self.tpl_asst_suffix, self.tpl_eos,
            ])
            for i in range(len(user_seq_rows))
        ]
        input_ids, attention_mask = self._left_pad(rows_ids, pad_to=pad_to)

        # supervised tail is fixed-width -> same columns every row -> one write.
        # tail from the end: [answer(A) | asst_suffix(s) | eos(1)].
        B, T = input_ids.shape
        s = self.tpl_asst_suffix.numel()
        tail = A + s + 1
        labels = torch.full(
            (B, T), self._ignore_index, dtype=torch.long, device=self.device
        )
        labels[:, T - tail : T - tail + A] = torch.stack(label_rows)
        labels[:, -1] = self.tpl_eos[0]  # supervise the trailing eos
        return input_ids, labels, attention_mask

    def predict(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """Dispatch on the TER inference flag (``set_is_inference`` in main.py).

        Branch 1 (train / eval, ``not is_inference``) — teacher-forced forward +
        CE loss (the metric path).
        Branch 2 (inference, ``is_inference``) — beam-search the SID answer from
        the prompt.
        """
        if self.is_inference:
            return self._generate(batch)
        return self._predict_train(batch)

    def _predict_train(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """Branch 1: teacher-forced forward -> suffix-slice -> CE loss."""
        # SID indices -> token ids once at the data boundary (_sid_token_rows).
        u_rows = self._sid_token_rows(
            batch.sequence_dense_features[self._input_name],
            max_codes=self._max_seq_length,  # cap to most-recent items (drop oldest)
        )
        l_rows = self._sid_token_rows(
            batch.sequence_dense_features[self._label_name],
            expected_width=self._num_levels,  # answer = one item = num_levels codes
        )

        # One-shot pool pre-sizing: pad the FIRST train step to the worst-case
        # length so the allocator reserves its largest segments up front (no
        # mid-run growth). Extra positions are masked + -100 -> loss/grad unchanged.
        pad_to = 0
        if not self._pool_warmed and self._max_total_len > 0 and self.is_train:
            pad_to = self._max_total_len
            self._pool_warmed = True

        input_ids, labels, attention_mask = self._splice_input_ids(
            u_rows, l_rows, pad_to=pad_to
        )

        if self._smoke_log_once and self._first_predict:
            print(
                f"[GENRECLM_DEBUG] first batch: B={input_ids.shape[0]} "
                f"T={input_ids.shape[1]} pad_id={self._pad_token_id} "
                f"ign={self._ignore_index} dev={input_ids.device} "
                f"input_ids[0, -8:]={input_ids[0, -8:].tolist()} "
                f"labels[0, -8:]={labels[0, -8:].tolist()}",
                flush=True,
            )
            self._first_predict = False

        return self._forward_loss(input_ids, labels, attention_mask)

    def _forward_loss(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Teacher-forced forward over spliced ids -> suffix-slice -> CE loss."""
        outputs = self.lm.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # (B, T, D)

        # Slice the fixed-width supervised suffix (constant -> no per-step sync).
        # Outside it every label is -100 (CE unchanged); it also bounds the logits
        # to (B, suffix, V) — the full (B, T, vocab) + fp32 upcast would OOM.
        sl = slice(-self._suffix_keep, None)
        labels_sl = labels[:, sl]
        logits = self.lm.lm_head(hidden[:, sl, :])

        # HF ForCausalLMLoss: shift-by-one + CE with -100 ignore.
        loss = self.lm.loss_function(
            logits=logits,
            labels=labels_sl,
            vocab_size=self.lm.config.vocab_size,
        )
        return {"loss": loss, "logits": logits}

    def _generate(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """Branch 2: beam-search the SID answer (no ground truth supplied).

        Builds the prompt (no answer), generates up to ``num_levels`` new tokens
        per beam, and hands the generated tail to the base
        ``_validate_sid_candidates`` (token->SID, malformed beams -> ``-1``).
        Returns ``generated_sids`` of shape ``(B, num_return, num_levels)``.
        """
        u_rows = self._sid_token_rows(
            batch.sequence_dense_features[self._input_name],
            max_codes=self._max_seq_length,  # cap to most-recent items (drop oldest)
        )
        input_ids, attention_mask = self._splice_prompt_ids(u_rows)
        out = self.lm.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self._num_levels,
            num_beams=self._num_beams,
            num_return_sequences=self._num_return,
            do_sample=False,
            pad_token_id=self._pad_token_id,
        )
        new_tokens = out[:, input_ids.shape[1]:]  # the generated tail
        sids = self._validate_sid_candidates(new_tokens, input_ids.shape[0])
        return {self.GENERATED_SIDS_KEY: sids}

    def _splice_prompt_ids(
        self, user_seq_rows: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Assemble the answer-less prompt and left-pad into ``(B, T_max)``.

        Layout: ``[system | user_prefix | history | user_suffix | asst_prefix]``
        — everything up to (but not including) the answer, so generation
        continues from the assistant turn.
        """
        rows = [
            torch.cat([
                self.tpl_system, self.tpl_user_prefix, r,
                self.tpl_user_suffix, self.tpl_asst_prefix,
            ])
            for r in user_seq_rows
        ]
        return self._left_pad(rows)

    def _left_pad(
        self, rows: List[torch.Tensor], pad_to: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Left-pad token rows into ``(input_ids, attention_mask)``, ``(B, T_max)``.

        Real content is right-aligned, pad at the front. ``attention_mask`` is
        built from ``ones_like(row)`` (not ``!= pad``) so a real trailing eos is
        never masked when ``pad_token_id == eos``.

        ``pad_to`` left-extends the batch to at least that many columns (the
        first-step activation-pool pre-sizing). Extending on the LEFT keeps the
        end-aligned supervised tail in place, so labels/suffix-slice are intact.
        """
        input_ids = pad_sequence(
            rows, batch_first=True,
            padding_value=self._pad_token_id, padding_side="left",
        )
        attention_mask = pad_sequence(
            [torch.ones_like(r) for r in rows], batch_first=True,
            padding_value=0, padding_side="left",
        )
        if pad_to > input_ids.shape[1]:
            B, extra = input_ids.shape[0], pad_to - input_ids.shape[1]
            input_ids = torch.cat(
                [input_ids.new_full((B, extra), self._pad_token_id), input_ids],
                dim=1,
            )
            attention_mask = torch.cat(
                [attention_mask.new_zeros((B, extra)), attention_mask], dim=1
            )
        return input_ids, attention_mask
