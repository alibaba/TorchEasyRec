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

from typing import List, Optional, Sequence, Tuple

import torch
from transformers import PreTrainedModel
from transformers.cache_utils import Cache


def build_sid_prefix_masks(
    allowed_sids: torch.Tensor, bands: Sequence[Tuple[int, int]]
) -> List[torch.Tensor]:
    """Compile a dense SID-membership mask into per-level legal-prefix masks.

    Args:
        allowed_sids: flat boolean mask indexed by mixed-radix local SID codes.
        bands: inclusive token-band bounds, one pair per SID level.

    Returns:
        A flat boolean mask per depth on the input device. A prefix is legal
        exactly when at least one allowed complete SID extends it. Build once
        with the catalog, then reuse the masks across decode calls.
    """
    masks = [allowed_sids]
    for lo, hi in reversed(bands[1:]):
        masks.append(masks[-1].reshape(-1, hi - lo + 1).any(dim=-1))
    return list(reversed(masks))


def capped_beam_widths(
    beam_widths: Sequence[int],
    bands: Sequence[Tuple[int, int]],
    final_topk_per_prefix: bool = False,
) -> List[int]:
    """Cap each requested width to the candidates its band and prefixes supply.

    Args:
        beam_widths: requested width per SID level.
        bands: inclusive ``(lo, hi)`` token-id edge per level.
        final_topk_per_prefix: treat the last width as a per-parent token limit.

    Returns:
        The capped width per level.
    """
    capped: List[int] = []
    previous = 1
    for level, (requested, (band_lo, band_hi)) in enumerate(zip(beam_widths, bands)):
        band_size = band_hi - band_lo + 1
        if final_topk_per_prefix and level + 1 == len(bands):
            capped.append(previous * min(requested, band_size))
        else:
            capped.append(min(requested, previous * band_size))
        previous = capped[-1]
    return capped


@torch.no_grad()
def dynamic_beam_search(
    model: PreTrainedModel,
    prompt_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    capped_widths: Sequence[int],
    bands: Sequence[Tuple[int, int]],
    final_topk_per_prefix: bool = False,
    allowed_prefixes: Optional[Sequence[torch.Tensor]] = None,
    initial_cache: Optional[Cache] = None,
) -> torch.Tensor:
    """Decode SID answers with a caller-supplied per-level beam width.

    Args:
        model: an HF causal LM whose forward takes ``logits_to_keep`` and the
            modern ``Cache`` protocol; a legacy tuple cache is not supported.
        prompt_embeds: left-padded prompt embeddings ``(B, P, D)``. Embeddings
            rather than ids, because a projected slot has no vocabulary id.
        attention_mask: prompt mask ``(B, P)``.
        capped_widths: width per SID level, already capped to what its band and
            the surviving prefixes supply. The model derives it once at init.
        allowed_prefixes: optional masks from ``build_sid_prefix_masks``.
            Apply before TopK at every level, preserving the full-vocabulary
            normalizer.
        initial_cache: optional cache supplied to prefill, for specialized
            inference storage that implements the same branching protocol.
        final_topk_per_prefix: retain a local token TopK at the last level,
            then merge all retained candidates in cumulative-score order.
        bands: inclusive ``(lo, hi)`` token-id edge per SID level. Host ints,
            because every use of them is a host-side slice bound.

    Returns:
        The SID token tail ``(B * W, num_levels)``, score-ordered best-first.
        The fixed-length, EOS-free answer needs no finished-sequence bookkeeping.
    """
    device = prompt_embeds.device
    batch_size = prompt_embeds.shape[0]
    num_levels = len(bands)

    def _band_logp(logits: torch.Tensor, level: int) -> torch.Tensor:
        """Full-vocab log-probs, narrowed to ``level``'s band ``(rows, band)``.

        Normalize then slice: same ranking as a full-vocab log_softmax without
        materializing a second full-vocabulary log-probability tensor.
        """
        band_lo, band_hi = bands[level]
        log_z = torch.logsumexp(logits.float(), dim=-1, keepdim=True)
        return logits[:, band_lo : band_hi + 1].float() - log_z

    position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)
    cache_kwargs = (
        {"past_key_values": initial_cache} if initial_cache is not None else {}
    )
    outputs = model(
        inputs_embeds=prompt_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=True,
        logits_to_keep=1,
        **cache_kwargs,
    )
    cache = outputs.past_key_values
    scores = _band_logp(outputs.logits[:, -1, :], 0)
    if allowed_prefixes is not None:
        scores = scores.masked_fill(~allowed_prefixes[0].reshape(1, -1), float("-inf"))
    beam_scores, band_idx = scores.topk(capped_widths[0], dim=-1)
    seq = (band_idx + bands[0][0]).reshape(-1, 1)
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
        outputs = model(
            input_ids=seq[:, -1:],
            attention_mask=beam_mask,
            position_ids=step_position,
            past_key_values=cache,
            use_cache=True,
            cache_position=cache_position,
            logits_to_keep=1,
        )
        band_lo, band_hi = bands[level]
        band_size = band_hi - band_lo + 1
        scores = _band_logp(outputs.logits[:, -1, :], level)
        scores = scores + beam_scores[:, None]
        if allowed_prefixes is not None:
            level_mask = allowed_prefixes[level]
            prefix = seq[:, 0] - bands[0][0]
            for previous_level in range(1, level):
                lo, hi = bands[previous_level]
                prefix = prefix * (hi - lo + 1) + seq[:, previous_level] - lo
            allowed = level_mask.reshape(-1, band_size)[prefix]
            scores = scores.masked_fill(~allowed, float("-inf"))
        if final_topk_per_prefix and level + 1 == num_levels:
            local_k = capped_widths[level] // width
            local_scores, local_idx = scores.topk(local_k, dim=-1)
            beam_scores, merged_idx = local_scores.view(batch_size, -1).sort(
                dim=-1, descending=True
            )
            flat_idx = merged_idx // local_k * band_size + local_idx.view(
                batch_size, -1
            ).gather(1, merged_idx)
        else:
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
