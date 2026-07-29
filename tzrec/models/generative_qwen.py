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

"""Qwen family subclass of ``BaseGenerativeModel``.

Owns the decoder-only-chat implementation: the ChatML prompt template, the
causal-LM splice, and the ``.model``/``.lm_head`` forward.
"""

from itertools import chain
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizerBase

from tzrec.datasets.utils import Batch
from tzrec.features.feature import BaseFeature
from tzrec.models.generative_model import BaseGenerativeModel
from tzrec.modules.dynamic_beam import dynamic_beam_search
from tzrec.protos.model_pb2 import ModelConfig
from tzrec.protos.models import generative_model_pb2


@torch.fx.wrap
def _fx_wrapped_generate(model: "GenerativeQwen", batch: Batch) -> torch.Tensor:
    """One opaque FX leaf spanning the whole decode.

    TorchRec's predict pipeline FX-traces the model; the decode is untraceable
    (per-row python lists, data-dependent beam widths). Wrapping the WHOLE
    decode is required -- a leaf returns one Proxy, so wrapping an inner helper
    just moves the failure to its caller. At run time this is a normal call.
    """
    return model._generate(batch)


class GenerativeQwen(BaseGenerativeModel):
    """Generative-recommendation LM on a Qwen backbone (Qwen2.5, Qwen3, ...)."""

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
        self._max_total_len = self._compute_max_total_length()
        self._pool_warmed = False
        # +2 = trailing eos + HF's shift-by-one; constant width avoids a per-step sync.
        self._suffix_keep = self._num_levels + self.tpl_asst_suffix.numel() + 2

    def _compute_max_total_length(self) -> int:
        """The ``T`` the activation pool pre-sizes to; 0 when disabled."""
        if self._max_seq_length <= 0:
            return 0
        frame = (
            sum(g.numel() for g in self._gaps)
            + self.tpl_asst_suffix.numel()
            + self.tpl_eos.numel()
        )
        return int(
            frame + self._max_seq_length * len(self._slot_names) + self._num_levels
        )

    @property
    def _gaps(self) -> List[torch.Tensor]:
        """The N+1 static prompt fragments around the N slots, template order.

        Re-read every time: ``.to()`` rebinds the buffer, so a cache goes stale.
        """
        return [getattr(self, f"tpl_gap_{i}") for i in range(len(self._slot_names) + 1)]

    def _build_prompt_tokens(
        self,
        tokenizer: PreTrainedTokenizerBase,
        cfg: generative_model_pb2.GenerativeQwen,
    ) -> None:
        """Tokenise the static prompt once, as the N+1 gaps around the N slots.

        Every gap is ONE string encoded in one call, so a BPE merge cannot span
        a seam. Splicing values between gaps is exact only because the ``C*``
        atoms are added-vocab tokens, which fast tokenizers pre-split on.
        Buffers are non-persistent: they follow ``.to()`` but stay off the
        state_dict.
        """
        tpl = self.CHAT_TEMPLATE
        gaps, features = self._resolve_prompt_slots(cfg.prompt_template)
        for i, gap in enumerate(gaps):
            head = tpl["user_prefix"] if i == 0 else features[i - 1].suffix_text
            tail = (
                features[i].prefix_text
                if i < len(features)
                else tpl["user_suffix"] + tpl["asst_prefix"]
            )
            # the template carries its own markers; no auto BOS/EOS.
            ids = torch.tensor(
                tokenizer.encode(head + gap + tail, add_special_tokens=False),
                dtype=torch.long,
            )
            self.register_buffer(f"tpl_gap_{i}", ids, persistent=False)
        self.register_buffer(
            "tpl_asst_suffix",
            torch.tensor(
                tokenizer.encode(tpl["asst_suffix"], add_special_tokens=False),
                dtype=torch.long,
            ),
            persistent=False,
        )
        # the trailing eos is SUPERVISED.
        self.register_buffer(
            "tpl_eos",
            torch.tensor([int(tokenizer.eos_token_id)], dtype=torch.long),
            persistent=False,
        )

    def _prompt_rows(self, slot_rows: List[List[torch.Tensor]]) -> List[torch.Tensor]:
        """Per-row ``[gap_0 | slot_0 | gap_1 | ... | slot_N-1 | gap_N]``."""
        gaps = self._gaps
        # zip pairs each slot with the gap before it; gaps[-1] closes the row.
        return [
            torch.cat([*chain.from_iterable(zip(gaps, slots)), gaps[-1]])
            for slots in zip(*slot_rows)
        ]

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

        The tail ``[answer | asst_suffix | eos]`` has a FIXED width, so after
        left padding it lands in the same columns for every row and ``labels``
        is one vectorized write. Only the answer and trailing eos are
        supervised. ``pad_to`` left-extends for pool pre-sizing.
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
            return {self._generated_sids_key: _fx_wrapped_generate(self, batch)}
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
        hidden = outputs.last_hidden_state

        # a full (B, T, V) upcast OOMs.
        suffix = slice(-self._suffix_keep, None)
        logits = self.lm.lm_head(hidden[:, suffix, :])

        loss = self.lm.loss_function(
            logits=logits,
            labels=labels[:, suffix],
            vocab_size=self.lm.config.vocab_size,
            ignore_index=self._ignore_index,
        )
        return {"loss": loss}

    def _generate(self, batch: Batch) -> torch.Tensor:
        """Beam-search the SID answer; returns ``(B, C, num_levels)`` best-first."""
        slot_rows = self._slot_rows(self.build_input(batch))
        input_ids, attention_mask = self._left_pad(self._prompt_rows(slot_rows))
        lo_tok, hi_tok = self._sid_token_bands()
        new_tokens = dynamic_beam_search(
            self.lm,
            input_ids,
            attention_mask,
            beam_widths=self._beam_widths,
            lo_tok=lo_tok,
            hi_tok=hi_tok,
        )
        sids = self._validate_sid_candidates(new_tokens, input_ids.shape[0])
        return sids[:, : self._num_return]

    def _left_pad(
        self, rows: List[torch.Tensor], pad_to: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Left-pad token rows into ``(input_ids, attention_mask)``, ``(B, T_max)``.

        The mask comes from ``ones_like``, not ``!= pad``, so a real trailing
        eos survives ``pad_token_id == eos``.
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
