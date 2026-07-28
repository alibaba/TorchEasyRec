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

"""Dynamic-width beam SID decode (no tzrec deps).

Backs ``dynamic_beam`` in ``GenerativeModelConfig``: the beam width varies per
SID level instead of staying fixed. The caller owns the schedule -- this module
only enforces what each level can actually supply.
"""

from typing import List, Tuple

import torch
from transformers import PreTrainedModel


@torch.no_grad()
def dynamic_beam_search(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    beam_widths: List[int],
    lo_tok: torch.Tensor,
    hi_tok: torch.Tensor,
) -> torch.Tensor:
    """Decode SID answers with a caller-supplied per-level beam width.

    Args:
        model: an HF causal LM exposing ``.model`` / ``.lm_head`` (Qwen layout).
        input_ids: left-padded prompt ids ``(B, P)``.
        attention_mask: prompt mask ``(B, P)``.
        beam_widths: requested width for each SID level, one entry per level.
            Any schedule is accepted -- doubling, flat, hand-tuned -- and each
            entry is capped to what its band and the surviving prefixes supply.
        lo_tok: inclusive lower per-level token-space band edge, ``(num_levels,)``.
        hi_tok: inclusive upper per-level token-space band edge, ``(num_levels,)``.

    Returns:
        The generated SID token tail ``(B * W, num_levels)`` score-ordered
        best-first per row, where ``W`` is the last capped width. The answer is
        fixed-length and EOS-free, so no finished-beam bookkeeping is needed.
    """
    device = input_ids.device
    bsz = input_ids.shape[0]
    num_levels = lo_tok.shape[0]
    if len(beam_widths) != num_levels:
        raise ValueError(
            f"dynamic_beam_search: beam_widths has {len(beam_widths)} entries "
            f"but the bands describe {num_levels} SID levels."
        )
    if any(w < 1 for w in beam_widths):
        raise ValueError(
            f"dynamic_beam_search: beam_widths must be >= 1, got {list(beam_widths)}."
        )
    # Hoist the band edges to host once to keep the level loop sync-free.
    bands: List[Tuple[int, int]] = [
        (int(lo_tok[j]), int(hi_tok[j])) for j in range(num_levels)
    ]
    # A level can only carry band x surviving prefixes, however much was asked.
    widths: List[int] = []
    prev = 1
    for w, (lo, hi) in zip(beam_widths, bands):
        widths.append(min(w, prev * (hi - lo + 1)))
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
    rows = torch.arange(bsz, device=device)
    past.reorder_cache(rows.repeat_interleave(widths[0]))
    am = attention_mask.repeat_interleave(widths[0], dim=0)
    cur_w = widths[0]

    for j in range(1, num_levels):
        am = torch.cat([am, am.new_ones(bsz * cur_w, 1)], dim=1)
        # the row always ends on the token just appended, so its position is
        # simply how many real tokens precede it.
        step_pos = am.long().sum(-1, keepdim=True) - 1
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
        tok = lo_j + idx % band
        parent = (idx // band + rows[:, None] * cur_w).reshape(-1)
        seq = torch.cat([seq[parent], tok.reshape(-1, 1)], dim=1)
        beam_scores = beam_scores.reshape(-1)
        cur_w = widths[j]
        if j + 1 < num_levels:
            # the last level never reads the cache; skip the largest reorder copy.
            past.reorder_cache(parent)
            am = am[parent]
    return seq  # (B*cur_w, num_levels)
