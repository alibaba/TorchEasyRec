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

"""Escalating-beam SID decode (torch-only, no tzrec deps).

The beam width doubles at every SID level, so early levels are pruned hard.
"""

from typing import TYPE_CHECKING, List, Tuple

import torch

if TYPE_CHECKING:
    from transformers import PreTrainedModel


@torch.no_grad()
def escalating_beam_search(
    model: "PreTrainedModel",
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    num_beams: int,
    lo_tok: torch.Tensor,
    hi_tok: torch.Tensor,
) -> torch.Tensor:
    """Decode SID answers with the escalating beam.

    Args:
        model: an HF causal LM exposing ``.model`` / ``.lm_head`` (Qwen2 layout).
        input_ids: left-padded prompt ids ``(B, P)``.
        attention_mask: prompt mask ``(B, P)``.
        num_beams: base beam width; doubles per level.
        lo_tok: inclusive lower per-level token-space band edge, ``(num_levels,)``
            (``num_levels`` is inferred from its length).
        hi_tok: inclusive upper per-level token-space band edge, ``(num_levels,)``.

    Returns:
        The generated SID token tail ``(B * W, num_levels)`` score-ordered
        best-first per row, where ``W`` is ``num_beams * 2**num_levels`` capped
        to the number of distinct SIDs the codebook can supply. The answer is
        fixed-length and EOS-free, so no finished-beam bookkeeping is needed.
    """
    device = input_ids.device
    bsz = input_ids.shape[0]
    num_levels = lo_tok.shape[0]
    # Hoist the band edges to host once to keep the level loop sync-free.
    bands: List[Tuple[int, int]] = [
        (int(lo_tok[j]), int(hi_tok[j])) for j in range(num_levels)
    ]
    # Cap the doubling at what band x surviving prefixes supply (tiny codebooks).
    widths: List[int] = []
    prev = 1
    for j, (lo, hi) in enumerate(bands):
        widths.append(min(num_beams * (2 ** (j + 1)), prev * (hi - lo + 1)))
        prev = widths[-1]

    def _band_logp(logits: torch.Tensor, j: int) -> torch.Tensor:
        """Full-vocab log-probs, narrowed to level ``j``'s band ``(R, band)``.

        Slicing after normalizing keeps the exact cross-beam ranking of a
        full-vocab ``log_softmax`` without materializing one per level.
        """
        lo, hi = bands[j]
        log_z = torch.logsumexp(logits.float(), dim=-1, keepdim=True)
        return logits[:, lo : hi + 1].float() - log_z

    pos = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)
    h = model.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=pos,
        use_cache=True,
    )
    past = h.past_key_values
    scores = _band_logp(model.lm_head(h.last_hidden_state[:, -1, :]), 0)
    beam_scores, local = scores.topk(widths[0], dim=-1)  # (B, W0)
    seq = (local + bands[0][0]).reshape(-1, 1)
    beam_scores = beam_scores.reshape(-1)
    parent = torch.arange(bsz, device=device).repeat_interleave(widths[0])
    past.reorder_cache(parent)
    am = attention_mask.repeat_interleave(widths[0], dim=0)
    cur_w = widths[0]

    for j in range(1, num_levels):
        am = torch.cat([am, am.new_ones(bsz * cur_w, 1)], dim=1)
        step_pos = (am.long().cumsum(-1) - 1)[:, -1:].clamp(min=0)
        cache_pos = torch.tensor([past.get_seq_length()], device=device)
        h = model.model(
            input_ids=seq[:, -1:],
            attention_mask=am,
            position_ids=step_pos,
            past_key_values=past,
            use_cache=True,
            cache_position=cache_pos,
        )
        lo_j, hi_j = bands[j]
        band = hi_j - lo_j + 1
        scores = _band_logp(model.lm_head(h.last_hidden_state[:, -1, :]), j)
        scores = scores + beam_scores[:, None]  # (B*cur_w, band) cumulative
        beam_scores, idx = scores.view(bsz, cur_w * band).topk(widths[j], dim=-1)
        parent_local = torch.div(idx, band, rounding_mode="floor")
        tok = lo_j + idx % band
        row_base = torch.arange(bsz, device=device)[:, None] * cur_w
        parent = (parent_local + row_base).reshape(-1)
        seq = torch.cat([seq[parent], tok.reshape(-1, 1)], dim=1)
        beam_scores = beam_scores.reshape(-1)
        cur_w = widths[j]
        if j + 1 < num_levels:
            # the last level never reads the cache; skip the largest reorder copy.
            past.reorder_cache(parent)
            am = am[parent]
    return seq  # (B*cur_w, num_levels)
