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

"""Band-restricted beam SID decode (no tzrec deps).

The caller owns the width schedule; this module only enforces what each level
can supply.
"""

from typing import List, Sequence, Tuple

import torch
from transformers import PreTrainedModel


@torch.no_grad()
def dynamic_beam_search(
    model: PreTrainedModel,
    prompt_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    beam_widths: List[int],
    bands: Sequence[Tuple[int, int]],
) -> torch.Tensor:
    """Decode SID answers with a caller-supplied per-level beam width.

    Args:
        model: an HF causal LM exposing ``.model`` / ``.lm_head`` (Qwen layout).
        prompt_embeds: left-padded prompt embeddings ``(B, P, D)``. Embeddings
            rather than ids, because a projected slot has no vocabulary id.
        attention_mask: prompt mask ``(B, P)``.
        beam_widths: requested width per SID level; each is capped to what its
            band and the surviving prefixes supply.
        bands: inclusive ``(lo, hi)`` token-id edge per SID level. Host ints,
            because every use of them is a host-side slice bound.

    Returns:
        The SID token tail ``(B * W, num_levels)``, score-ordered best-first.
        The answer is fixed-length and EOS-free, so no beam bookkeeping.
    """
    device = prompt_embeds.device
    batch_size = prompt_embeds.shape[0]
    num_levels = len(bands)
    if len(beam_widths) != num_levels:
        raise ValueError(
            f"dynamic_beam_search: beam_widths has {len(beam_widths)} entries "
            f"but the bands describe {num_levels} SID levels."
        )
    if any(w < 1 for w in beam_widths):
        raise ValueError(
            f"dynamic_beam_search: beam_widths must be >= 1, got {list(beam_widths)}."
        )
    capped_widths: List[int] = []
    prev_width = 1
    for requested, (band_lo, band_hi) in zip(beam_widths, bands):
        capped_widths.append(min(requested, prev_width * (band_hi - band_lo + 1)))
        prev_width = capped_widths[-1]

    def _band_logp(logits: torch.Tensor, level: int) -> torch.Tensor:
        """Full-vocab log-probs, narrowed to ``level``'s band ``(rows, band)``.

        Normalize then slice: same ranking as a full-vocab log_softmax, 21x less
        memory at production vocab.
        """
        band_lo, band_hi = bands[level]
        log_z = torch.logsumexp(logits.float(), dim=-1, keepdim=True)
        return logits[:, band_lo : band_hi + 1].float() - log_z

    position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)
    outputs = model.model(
        inputs_embeds=prompt_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=True,
    )
    cache = outputs.past_key_values
    scores = _band_logp(model.lm_head(outputs.last_hidden_state[:, -1, :]), 0)
    beam_scores, in_band = scores.topk(capped_widths[0], dim=-1)
    seq = (in_band + bands[0][0]).reshape(-1, 1)
    beam_scores = beam_scores.reshape(-1)
    row_starts = torch.arange(batch_size, device=device)
    cache.reorder_cache(row_starts.repeat_interleave(capped_widths[0]))
    beam_mask = attention_mask.repeat_interleave(capped_widths[0], dim=0)
    width = capped_widths[0]

    for level in range(1, num_levels):
        beam_mask = torch.cat(
            [beam_mask, beam_mask.new_ones(batch_size * width, 1)], dim=1
        )
        # the row ends on the new token, so its position is the count before it.
        step_position = beam_mask.long().sum(-1, keepdim=True) - 1
        cache_position = torch.tensor([cache.get_seq_length()], device=device)
        outputs = model.model(
            input_ids=seq[:, -1:],
            attention_mask=beam_mask,
            position_ids=step_position,
            past_key_values=cache,
            use_cache=True,
            cache_position=cache_position,
        )
        band_lo, band_hi = bands[level]
        band_size = band_hi - band_lo + 1
        scores = _band_logp(model.lm_head(outputs.last_hidden_state[:, -1, :]), level)
        scores = scores + beam_scores[:, None]
        beam_scores, flat_idx = scores.view(batch_size, width * band_size).topk(
            capped_widths[level], dim=-1
        )
        next_token = band_lo + flat_idx % band_size
        parent = (flat_idx // band_size + row_starts[:, None] * width).reshape(-1)
        seq = torch.cat([seq[parent], next_token.reshape(-1, 1)], dim=1)
        beam_scores = beam_scores.reshape(-1)
        width = capped_widths[level]
        if level + 1 < num_levels:
            # the last level never reads the cache; skip the largest reorder copy.
            cache.reorder_cache(parent)
            beam_mask = beam_mask[parent]
    return seq
