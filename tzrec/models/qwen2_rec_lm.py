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

"""Qwen2/Qwen2.5 family subclass of ``GenerativeRecLM``.

Owns the decoder-only-chat implementation: the ChatML prompt template, the
causal-LM splice, and the ``.model``/``.lm_head`` forward.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizerBase

from tzrec.datasets.utils import Batch
from tzrec.features.feature import BaseFeature
from tzrec.models.generative_rec_lm import GenerativeRecLM
from tzrec.modules.escalating_beam import escalating_beam_search
from tzrec.protos.model_pb2 import ModelConfig
from tzrec.protos.models import generative_model_pb2


class Qwen2RecLM(GenerativeRecLM):
    """Qwen2 / Qwen2.5 generative-recommendation LM."""

    # ChatML frame. Family-specific; a subclass overrides it wholesale.
    CHAT_TEMPLATE = {
        "user_prefix": "<|im_start|>user\n",
        "user_suffix": "<|im_end|>\n",
        "asst_prefix": "<|im_start|>assistant\n",
        "asst_suffix": "<|im_end|>\n",
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
        common = self._model_config.common
        self._num_beams = int(common.num_beams)
        self._num_return = int(common.num_return_sequences)
        self._dynamic_beam = bool(common.dynamic_beam)
        if not self._dynamic_beam and self._num_return > self._num_beams:
            raise ValueError(
                f"{type(self).__name__}: num_return_sequences "
                f"({self._num_return}) must not exceed num_beams "
                f"({self._num_beams})."
            )
        self._max_total_len = self._compute_max_total_length()
        self._pool_warmed = False
        # +2 = trailing eos + HF's shift-by-one; constant width avoids a per-step sync.
        self._suffix_keep = self._num_levels + self.tpl_asst_suffix.numel() + 2

    def _compute_max_total_length(self) -> int:
        """Full spliced length at the max history (0 if pre-allocation is off).

        The ``T`` the activation pool is pre-sized to.
        """
        if self._max_seq_length <= 0:
            return 0
        frame = (
            sum(
                getattr(self, f"tpl_gap_{i}").numel()
                for i in range(self._num_slots + 1)
            )
            + self.tpl_asst_suffix.numel()
            + self.tpl_eos.numel()
        )
        return int(frame + self._max_seq_length * self._num_slots + self._num_levels)

    def _build_prompt_tokens(
        self,
        tokenizer: PreTrainedTokenizerBase,
        cfg: generative_model_pb2.Qwen2RecLM,
    ) -> None:
        """Tokenise the static prompt once, as the N+1 gaps around the N slots.

        The base resolves ``{{feature_name}}`` slots; this hook only frames them
        with ChatML and folds each slot feature's ``prefix_text`` / ``suffix_text``
        into the adjacent gap, so every gap is ONE contiguous string tokenized in
        one call. That keeps the encoding bit-identical to tokenizing the fully
        rendered prompt -- splitting the token ids instead would let a BPE merge
        span a seam. Splicing values between gaps is exact because the ``C*``
        atoms are added-vocab tokens, which HF fast tokenizers pre-split on.

        Buffers are non-persistent: they move with ``model.to(...)`` but stay off
        the state_dict so HF safetensors round-tripping isn't polluted.
        """
        tpl = type(self).CHAT_TEMPLATE
        gaps, features, groups = self._resolve_prompt_slots(cfg.prompt_template)
        self._slot_names = [f.name for f in features]
        self._slot_groups = groups
        self._num_slots = len(features)
        for i, gap in enumerate(gaps):
            head = tpl["user_prefix"] if i == 0 else features[i - 1].suffix_text
            tail = (
                features[i].prefix_text
                if i < self._num_slots
                else tpl["user_suffix"] + tpl["asst_prefix"]
            )
            # explicit <|im_start|> markers frame the prompt; no auto BOS/EOS.
            ids = torch.tensor(
                tokenizer.encode(head + gap + tail, add_special_tokens=False),
                dtype=torch.long,
            )
            self.register_buffer(f"tpl_gap_{i}", ids, persistent=False)
        # closes the assistant turn after the answer; not part of any gap.
        self.register_buffer(
            "tpl_asst_suffix",
            torch.tensor(
                tokenizer.encode(tpl["asst_suffix"], add_special_tokens=False),
                dtype=torch.long,
            ),
            persistent=False,
        )
        # the trailing eos is a SUPERVISED token; cache it for the splice.
        self.register_buffer(
            "tpl_eos",
            torch.tensor([int(tokenizer.eos_token_id)], dtype=torch.long),
            persistent=False,
        )

    def _prompt_rows(self, slot_rows: List[List[torch.Tensor]]) -> List[torch.Tensor]:
        """Per-row ``[gap_0 | slot_0 | gap_1 | ... | slot_N-1 | gap_N]``.

        Shared by the teacher-forced splice and the answer-less inference prompt.
        """
        gaps = [getattr(self, f"tpl_gap_{i}") for i in range(self._num_slots + 1)]
        rows = []
        for b in range(len(slot_rows[0])):
            parts: List[torch.Tensor] = []
            for i in range(self._num_slots):
                parts.append(gaps[i])
                parts.append(slot_rows[i][b])
            parts.append(gaps[self._num_slots])
            rows.append(torch.cat(parts))
        return rows

    def _slot_rows(
        self, rows: Dict[str, List[torch.Tensor]]
    ) -> List[List[torch.Tensor]]:
        """Per-slot token rows, in template order."""
        return [rows[name] for name in self._slot_names]

    def _splice_input_ids(
        self,
        slot_rows: List[List[torch.Tensor]],
        label_rows: List[torch.Tensor],
        pad_to: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build ``(input_ids, labels, attention_mask)``, each ``(B, T_max)``.

        Rows already hold on-device token ids (see ``_sid_token_rows``).

        Every answer is exactly ``self._num_levels`` SID codes, so the tail
        ``[answer | asst_suffix | eos]`` has a FIXED width and lands in the same
        columns for every row after left-padding -> ``labels`` is one vectorized
        write. Only the answer and the trailing eos are supervised: a decode
        emits exactly ``num_levels`` tokens and never has to produce
        ``asst_suffix``. ``pad_to`` left-extends every row for pool pre-sizing,
        keeping the supervised tail end-aligned.
        """
        if len(slot_rows[0]) != len(label_rows):
            raise ValueError(
                f"{type(self).__name__}: history/answer row count mismatch "
                f"({len(slot_rows[0])} vs {len(label_rows)})."
            )
        rows_ids = [
            torch.cat([prompt, label_rows[i], self.tpl_asst_suffix, self.tpl_eos])
            for i, prompt in enumerate(self._prompt_rows(slot_rows))
        ]
        input_ids, attention_mask = self._left_pad(rows_ids, pad_to=pad_to)

        B, T = input_ids.shape
        answer_width = self._num_levels
        tail = answer_width + self.tpl_asst_suffix.numel() + 1
        labels = torch.full(
            (B, T), self._ignore_index, dtype=torch.long, device=self.device
        )
        labels[:, T - tail : T - tail + answer_width] = torch.stack(label_rows)
        labels[:, -1] = self.tpl_eos[0]
        return input_ids, labels, attention_mask

    def predict(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """Dispatch on the TER inference flag (``set_is_inference`` in main.py)."""
        if self.is_inference:
            return self._generate(batch)
        return self._predict_train(batch)

    def _predict_train(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """Build the teacher-forced splice for a batch and return the CE loss."""
        rows = self.build_input(batch)

        # Pre-size the caching allocator on step 1; the extra columns are masked.
        pad_to = 0
        if not self._pool_warmed and self._max_total_len > 0 and self.is_train:
            pad_to = self._max_total_len
            self._pool_warmed = True

        input_ids, labels, attention_mask = self._splice_input_ids(
            self._slot_rows(rows), rows[self._label_name], pad_to=pad_to
        )
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

        # Bound the logits to the supervised suffix; a full (B, T, V) upcast OOMs.
        suffix = slice(-self._suffix_keep, None)
        logits = self.lm.lm_head(hidden[:, suffix, :])

        loss = self.lm.loss_function(
            logits=logits,
            labels=labels[:, suffix],
            vocab_size=self.lm.config.vocab_size,
            ignore_index=self._ignore_index,
        )
        return {"loss": loss}

    def _generate(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """Beam-search the SID answer (no ground truth supplied).

        Returns ``generated_sids`` of shape ``(B, C, num_levels)``, where ``C``
        is ``num_return_sequences`` on the HF path and the escalating beam's
        final width when ``dynamic_beam`` is set.
        """
        slot_rows = self._slot_rows(self.build_input(batch))
        input_ids, attention_mask = self._splice_prompt_ids(slot_rows)
        if self._dynamic_beam:
            new_tokens = self._dynamic_beam_search(input_ids, attention_mask)
        else:
            out = self.lm.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self._num_levels,
                num_beams=self._num_beams,
                num_return_sequences=self._num_return,
                do_sample=False,
                pad_token_id=self._pad_token_id,
            )
            new_tokens = out[:, input_ids.shape[1] :]
        sids = self._validate_sid_candidates(new_tokens, input_ids.shape[0])
        return {self._generated_sids_key: sids}

    def _dynamic_beam_search(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Escalating-beam decode; see ``escalating_beam_search`` for the schedule."""
        lo_tok, hi_tok = self._sid_token_bands()
        return escalating_beam_search(
            self.lm,
            input_ids,
            attention_mask,
            num_beams=self._num_beams,
            lo_tok=lo_tok,
            hi_tok=hi_tok,
        )

    def _splice_prompt_ids(
        self, slot_rows: List[List[torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Assemble the answer-less prompt and left-pad into ``(B, T_max)``."""
        return self._left_pad(self._prompt_rows(slot_rows))

    def _left_pad(
        self, rows: List[torch.Tensor], pad_to: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Left-pad token rows into ``(input_ids, attention_mask)``, ``(B, T_max)``.

        ``attention_mask`` is built from ``ones_like(row)`` (not ``!= pad``) so a
        real trailing eos is never masked when ``pad_token_id == eos``. ``pad_to``
        extends on the LEFT, keeping the end-aligned supervised tail in place.
        """
        input_ids = pad_sequence(
            rows,
            batch_first=True,
            padding_value=self._pad_token_id,
            padding_side="left",
        )
        attention_mask = pad_sequence(
            [torch.ones_like(r) for r in rows],
            batch_first=True,
            padding_value=0,
            padding_side="left",
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
