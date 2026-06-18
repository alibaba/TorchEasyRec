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

"""ALGR-style escalating-beam SID decode (torch-only, no tzrec deps).

A faithful port of ALGR's ``dynamic_beams`` schedule: the beam width doubles at
every SID level (``num_beams`` -> ``2*num_beams`` -> ...), keeping
``num_beams * 2**(j+1)`` candidates after level ``j`` and returning
``num_beams * 2**num_levels`` per row. The aggressive early pruning (only the
top ``2*num_beams`` level-0 prefixes survive) is what distinguishes it from a
fixed-width beam that keeps every level-0 code.

Lives in its own torch-only module so both the production path
(``Qwen2RecLM._dynamic_beam_search``) and the offline predict harness share one
tested implementation.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def escalating_beam_search(
    model,
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
        The generated SID token tail ``(B * num_beams * 2**num_levels,
        num_levels)``, score-ordered best-first per row. The SID answer is
        exactly ``num_levels`` codes with no in-answer EOS, so every beam emits
        exactly ``num_levels`` tokens — no finished-beam bookkeeping is needed,
        and band masking makes every candidate well-formed by construction.
    """
    device = input_ids.device
    bsz = input_ids.shape[0]
    num_levels = lo_tok.shape[0]
    # candidates kept after level j (doubling), capped to what the band + the
    # surviving prefixes can actually supply (guards tiny codebooks).
    widths, prev = [], 1
    for j in range(num_levels):
        avail = prev * int(hi_tok[j] - lo_tok[j] + 1)
        widths.append(min(num_beams * (2 ** (j + 1)), avail))
        prev = widths[-1]

    def _band_logp(logits: torch.Tensor, j: int) -> torch.Tensor:
        ids = torch.arange(logits.shape[-1], device=device)
        keep = (ids >= lo_tok[j]) & (ids <= hi_tok[j])
        logp = torch.log_softmax(logits.float(), dim=-1)
        return logp.masked_fill(~keep, float("-inf"))

    # 1. prompt forward (bsz beams) -> level-0 logits.
    pos = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)
    h = model.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=pos,
        use_cache=True,
    )
    past = h.past_key_values
    vocab = model.config.vocab_size
    scores = _band_logp(model.lm_head(h.last_hidden_state[:, -1, :]), 0)
    beam_scores, tok = scores.topk(widths[0], dim=-1)  # (B, W0)
    seq = tok.reshape(-1, 1)
    beam_scores = beam_scores.reshape(-1)
    parent = torch.arange(bsz, device=device).repeat_interleave(widths[0])
    past.reorder_cache(parent)
    am = attention_mask.repeat_interleave(widths[0], dim=0)
    cur_w = widths[0]

    # 2. levels 1..n-1: forward the last chosen atom with cache, escalate width.
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
        scores = _band_logp(model.lm_head(h.last_hidden_state[:, -1, :]), j)
        scores = scores + beam_scores[:, None]  # (B*cur_w, V) cumulative
        # global top-widths[j] per row over the (cur_w * V) continuations.
        beam_scores, idx = scores.view(bsz, cur_w * vocab).topk(widths[j], dim=-1)
        parent_local = torch.div(idx, vocab, rounding_mode="floor")  # in [0,cur_w)
        tok = idx % vocab
        row_base = torch.arange(bsz, device=device)[:, None] * cur_w
        parent = (parent_local + row_base).reshape(-1)
        past.reorder_cache(parent)
        am = am[parent]
        seq = torch.cat([seq[parent], tok.reshape(-1, 1)], dim=1)
        beam_scores = beam_scores.reshape(-1)
        cur_w = widths[j]
    return seq  # (B*cur_w, num_levels)
