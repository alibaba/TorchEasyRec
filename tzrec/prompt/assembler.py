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

"""The prompt assembler: one scripted walk with two call sites.

The collator runs it after parsing and the exported front-end runs the same
module at serving. ``PromptPlan`` unrolls into constant lists at construction,
leaving jagged integer arithmetic with no data-dependent control flow, which
is what ``torch.jit.script`` can carry into a runtime without tzrec source.
The walk is prompt structure only; ``hole_keys.py`` folds the prefix-cache
identity of each hole beside it.
"""

from dataclasses import dataclass
from typing import Dict, Final, List, Tuple

import torch
from torch import nn

from tzrec.prompt.types import (
    FillMode,
    PromptPlan,
    ResolvedSidSpace,
    SlotSeg,
    Static,
)
from tzrec.protos.model_pb2 import FeatureGroupType

INPUT_IDS = "input_ids"
CU_SEQLENS = "cu_seqlens"
HOLE_POSITIONS = "hole_positions"
HOLE_SLOT_COUNTS = "hole_slot_counts"
RESPONSE_LENGTHS = "response_lengths"
MAX_SEQLEN = "max_seqlen"
# every stream the walk emits; a caller under FX tracing indexes these rather
# than iterating the result, which is one opaque proxy there
OUTPUT_KEYS = (
    INPUT_IDS,
    CU_SEQLENS,
    HOLE_POSITIONS,
    HOLE_SLOT_COUNTS,
    RESPONSE_LENGTHS,
    MAX_SEQLEN,
)


@torch.jit.script
def _exclusive_cumsum(values: torch.Tensor) -> torch.Tensor:
    """Exclusive prefix sum along dim 0."""
    return torch.cumsum(values, dim=0) - values


@torch.jit.script
def _row_ids(lengths: torch.Tensor) -> torch.Tensor:
    """Row index of every element in a jagged buffer described by lengths."""
    return torch.repeat_interleave(
        torch.arange(lengths.numel(), dtype=torch.int64, device=lengths.device),
        lengths,
    )


@torch.jit.script
def _within_row_index(lengths: torch.Tensor) -> torch.Tensor:
    """Position of every element inside its own row."""
    total = int(torch.sum(lengths))
    starts = _exclusive_cumsum(lengths)
    return torch.arange(
        total, dtype=torch.int64, device=lengths.device
    ) - torch.repeat_interleave(starts, lengths)


@torch.jit.script
def batch_device(batch: Dict[str, torch.Tensor]) -> torch.device:
    """Device the batch already lives on, so nothing is built on the wrong one."""
    for value in batch.values():
        return value.device
    return torch.device("cpu")


@torch.jit.script
def _destinations(seg_start: torch.Tensor, seg_len: torch.Tensor) -> torch.Tensor:
    """Absolute index of every value of one segment in the packed stream."""
    return torch.repeat_interleave(seg_start, seg_len) + _within_row_index(seg_len)


class PromptAssembler(nn.Module):
    """Walks a compiled plan to build one batch's packed token stream.

    The collator calls it eagerly on the host; export scripts the same module
    into the serving front-end. It trusts the parsed batch as every scripted
    module does: the data contract belongs to the parser and to the request.

    Args:
        prompt_plan: the compiled walk order and its constants.
        sid_space: the resolved SID token space.
        include_response: whether to read and emit the supervised tail.
    """

    # TorchScript resolves a Final class attribute as a constant; a
    # module-level one it cannot see at all
    KIND_STATIC: Final[int] = 0
    KIND_INLINE: Final[int] = 1
    KIND_PROJECTED: Final[int] = 2

    kinds: List[int]
    static_tokens: List[List[int]]
    hole_slots: List[int]
    is_sequences: List[bool]
    member_names: List[List[str]]
    id_shifts: List[int]

    def __init__(
        self,
        prompt_plan: PromptPlan,
        sid_space: ResolvedSidSpace,
        include_response: bool = True,
    ) -> None:
        super().__init__()
        segments = tuple(prompt_plan.segments)
        self.num_body = len(segments)
        if include_response:
            segments = segments + tuple(prompt_plan.response_segments)

        # compile reserves a sentinel whenever a slot is PROJECTED, so -1 is
        # never written
        self.sentinel = -1
        if sid_space.sentinel_token_id is not None:
            self.sentinel = int(sid_space.sentinel_token_id)

        self.kinds = []
        self.static_tokens = []
        self.hole_slots = []
        self.is_sequences = []
        self.member_names = []
        self.id_shifts = []
        # the first slot member sizes the batch; an all-static plan reads the
        # batch_size the parser passes along
        self.anchor = ""
        # holes are grouped by projected occurrence in emission order, which is
        # the order of ``projected_slots``, of the front-end's projections and
        # of ``hole_keys``
        occurrences = 0
        for seg in segments:
            if isinstance(seg, Static):
                self._append(
                    self.KIND_STATIC, [int(t) for t in seg.token_ids], -1, False, [], 0
                )
                continue
            assert isinstance(seg, SlotSeg)
            if not self.anchor:
                self.anchor = seg.feature_names[0]
            is_sequence = seg.group_type == FeatureGroupType.JAGGED_SEQUENCE
            if seg.fill is FillMode.INLINE:
                self._append(
                    self.KIND_INLINE,
                    [],
                    -1,
                    is_sequence,
                    [seg.feature_names[0]],
                    int(seg.id_shift),
                )
            else:
                self._append(
                    self.KIND_PROJECTED,
                    [],
                    occurrences,
                    is_sequence,
                    list(seg.feature_names),
                    0,
                )
                occurrences += 1

        self.num_segments = len(self.kinds)
        self.num_hole_slots = occurrences

    def _append(
        self,
        kind: int,
        tokens: List[int],
        hole_slot: int,
        is_sequence: bool,
        members: List[str],
        id_shift: int,
    ) -> None:
        """Record one unrolled segment's constants."""
        self.kinds.append(kind)
        self.static_tokens.append(tokens)
        self.hole_slots.append(hole_slot)
        self.is_sequences.append(is_sequence)
        self.member_names.append(members)
        self.id_shifts.append(id_shift)

    def _batch_size(self, batch: Dict[str, torch.Tensor]) -> int:
        """Row count, from the anchor member.

        A sequence or a multi-value member carries ``lengths``, one per row; a
        dense member has one row per sample and no lengths.
        """
        if self.anchor == "":
            if "batch_size" in batch:
                return int(batch["batch_size"])
            return 0
        key = self.anchor + ".lengths"
        if key in batch:
            return int(batch[key].numel())
        return int(batch[self.anchor + ".values"].size(0))

    def _inline_counts(
        self, batch: Dict[str, torch.Tensor], index: int, batch_size: int
    ) -> torch.Tensor:
        """Token count each row contributes for one INLINE segment.

        For a multi-value sequence -- which is what a SID history is -- the row
        holds ``lengths`` items and each item holds ``key_lengths`` codes, so the
        count is a segmented sum rather than ``lengths`` itself.
        """
        member = self.member_names[index][0]
        lengths = batch[member + ".lengths"].to(torch.int64)
        key = member + ".key_lengths"
        if key in batch:
            key_lengths = batch[key].to(torch.int64).reshape(-1)
            counts = torch.zeros(batch_size, dtype=torch.int64, device=lengths.device)
            lengths = counts.index_add_(0, _row_ids(lengths), key_lengths)
        return lengths

    def _segment(
        self,
        batch: Dict[str, torch.Tensor],
        index: int,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-row length and row-major values of one segment."""
        kind = self.kinds[index]

        if kind == self.KIND_STATIC:
            run = torch.tensor(
                self.static_tokens[index], dtype=torch.int64, device=device
            )
            width = run.numel()
            seg_len = torch.full((batch_size,), width, dtype=torch.int64, device=device)
            return seg_len, run.unsqueeze(0).expand(batch_size, width).reshape(-1)

        member = self.member_names[index][0]
        if kind == self.KIND_INLINE:
            # offset SID codes need the base-vocab shift; tokenizer word ids none
            counts = self._inline_counts(batch, index, batch_size)
            values = batch[member + ".values"].to(torch.int64).reshape(-1)
            return counts, values + self.id_shifts[index]

        if self.is_sequences[index]:
            seg_len = batch[member + ".lengths"].to(torch.int64)
        else:
            seg_len = torch.ones(batch_size, dtype=torch.int64, device=device)
        total = int(torch.sum(seg_len))
        return seg_len, torch.full(
            (total,), self.sentinel, dtype=torch.int64, device=seg_len.device
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Assemble one batch.

        Args:
            batch: the parsed feature dict, keyed ``{feature}.values`` /
                ``.lengths`` / ``.key_lengths`` as the data parser emits it.

        Returns:
            ``input_ids``, ``cu_seqlens``, ``hole_positions``,
            ``hole_slot_counts``, ``response_lengths`` and ``max_seqlen``. Holes
            are grouped by projected occurrence in emission order, then by
            sample; the front-end's ``slot_embeds`` and ``hole_keys`` follow the
            same order.
        """
        batch_size = self._batch_size(batch)
        device = batch_device(batch)

        seg_lens: List[torch.Tensor] = []
        seg_values: List[torch.Tensor] = []
        for i in range(self.num_segments):
            length, values = self._segment(batch, i, batch_size, device)
            seg_lens.append(length)
            seg_values.append(values)

        stacked = torch.stack(seg_lens, dim=0)
        row_total = torch.sum(stacked, dim=0)
        row_start = _exclusive_cumsum(row_total)
        seg_offsets = torch.cumsum(stacked, dim=0) - stacked

        total_tokens = int(torch.sum(row_total))
        out = torch.zeros(total_tokens, dtype=torch.int64, device=device)

        # one entry per projected occurrence, and an empty stream under
        # Pattern I, where the concatenation below would otherwise have
        # nothing to join
        hole_parts: List[torch.Tensor] = [
            torch.zeros(0, dtype=torch.int64, device=device)
        ]
        for _ in range(self.num_hole_slots):
            hole_parts.append(torch.zeros(0, dtype=torch.int64, device=device))

        response_lengths = torch.zeros(batch_size, dtype=torch.int64, device=device)
        dests: List[torch.Tensor] = []
        for i in range(self.num_segments):
            dest = _destinations(row_start + seg_offsets[i], seg_lens[i])
            dests.append(dest)
            slot = self.hole_slots[i]
            if slot >= 0:
                hole_parts[slot + 1] = dest
            if i >= self.num_body:
                response_lengths = response_lengths + seg_lens[i]
        out.index_copy_(0, torch.cat(dests, dim=0), torch.cat(seg_values, dim=0))

        if self.num_segments > self.num_body:
            # the first response token is predicted by the one before it, so a
            # row that is nothing but its response supervises nothing
            response_lengths = response_lengths * (row_total > response_lengths)
            if not bool((response_lengths > 0).any()):
                raise ValueError(
                    "PromptAssembler: nothing in this batch can be supervised "
                    "-- every sample either carries no response or no token "
                    "before it. Give the template static text, or drop the "
                    "samples whose prompt features are all empty."
                )

        hole_positions = torch.cat(hole_parts, dim=0)
        # how many of those holes each projected occurrence owns, so a host can
        # cut the flat streams back into per-slot spans without the plan
        slot_counts = torch.zeros(self.num_hole_slots, dtype=torch.int64, device=device)
        for slot in range(self.num_hole_slots):
            slot_counts[slot] = hole_parts[slot + 1].numel()

        cu_seqlens = torch.cat(
            [
                torch.zeros(1, dtype=torch.int64, device=device),
                torch.cumsum(row_total, dim=0),
            ]
        )
        if batch_size > 0:
            max_seqlen = torch.max(row_total)
        else:
            max_seqlen = torch.zeros((), dtype=torch.int64, device=device)
        # literals rather than the module constants above: TorchScript cannot
        # see a module-level global. ``assembler_test`` pins the two together.
        return {
            "input_ids": out,
            "cu_seqlens": cu_seqlens.to(torch.int32),
            "hole_positions": hole_positions,
            "hole_slot_counts": slot_counts,
            "response_lengths": response_lengths,
            "max_seqlen": max_seqlen,
        }


@dataclass
class MultiTargetLayout:
    """Training-only SID branches that share each row's packed prompt prefix.

    Tokens keep the assembler's packed order. Every prefix and every target is
    one attention segment; a target's keys are its row's prefix followed by its
    own tokens, so causal attention aligned to the last key isolates targets.

    Args:
        position_ids: packed positions, restarting after the prefix per target.
        cu_seqlens_q: int32 query boundaries of the prefix and target segments.
        cu_seqlens_k: int32 key boundaries of the same segments.
        key_indices: packed token behind every segment key.
        max_seqlen: upper bound of any segment's query or key length.
        label_indices: packed indices of target tokens, shaped (targets, levels).
        score_indices: packed indices whose logits predict those labels.
        target_weights: inverse target count of each target's row.
        token_targets: target index of every packed token, -1 in a prefix.
    """

    position_ids: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    key_indices: torch.Tensor
    max_seqlen: int
    label_indices: torch.Tensor
    score_indices: torch.Tensor
    target_weights: torch.Tensor
    token_targets: torch.Tensor


def build_multi_target_layout(
    cu_seqlens: torch.Tensor, response_lengths: torch.Tensor, num_levels: int
) -> MultiTargetLayout:
    """Split each packed row into its shared prefix and independent SID targets.

    Args:
        cu_seqlens: packed row boundaries from the assembler.
        response_lengths: supervised SID token count of each row.
        num_levels: codebook levels per target SID.

    Returns:
        Positions, attention segments and scoring indices over packed tokens.
    """
    device = cu_seqlens.device
    cu_seqlens = cu_seqlens.to(torch.int64)
    response_lengths = response_lengths.to(torch.int64)
    row_lengths = torch.diff(cu_seqlens)
    prefix_lengths = row_lengths - response_lengths
    target_counts = response_lengths // num_levels
    segment_counts = target_counts + 1
    # one host read sizes every repeat below
    num_tokens, num_targets, num_keys, max_seqlen = torch.stack(
        (
            cu_seqlens[-1],
            target_counts.sum(),
            (prefix_lengths * segment_counts + response_lengths).sum(),
            row_lengths.max(),
        )
    ).tolist()
    row_ids = torch.arange(row_lengths.numel(), device=device)

    token_rows = torch.repeat_interleave(row_ids, row_lengths, output_size=num_tokens)
    token_offsets = torch.arange(num_tokens, device=device) - cu_seqlens[token_rows]
    token_prefix_lengths = prefix_lengths[token_rows]
    response_offsets = token_offsets - token_prefix_lengths
    in_prefix = response_offsets < 0
    position_ids = torch.where(
        in_prefix,
        token_offsets,
        token_prefix_lengths + response_offsets.remainder(num_levels),
    )
    token_targets = torch.where(
        in_prefix,
        -1,
        (target_counts.cumsum(0) - target_counts)[token_rows]
        + response_offsets.div(num_levels, rounding_mode="floor"),
    )

    num_segments = row_ids.numel() + num_targets
    segment_ids = torch.arange(num_segments, device=device)
    segment_rows = torch.repeat_interleave(
        row_ids, segment_counts, output_size=num_segments
    )
    # 0 for a row's prefix, i for its i-th target
    segment_ranks = (
        segment_ids - (segment_counts.cumsum(0) - segment_counts)[segment_rows]
    )
    segment_prefix_lengths = prefix_lengths[segment_rows]
    query_lengths = torch.where(segment_ranks > 0, num_levels, segment_prefix_lengths)
    key_lengths = segment_prefix_lengths + num_levels * (segment_ranks > 0)
    cu_seqlens_q = nn.functional.pad(query_lengths.cumsum(0), (1, 0))
    cu_seqlens_k = nn.functional.pad(key_lengths.cumsum(0), (1, 0))
    key_segments = torch.repeat_interleave(
        segment_ids, key_lengths, output_size=num_keys
    )
    key_offsets = torch.arange(num_keys, device=device) - cu_seqlens_k[key_segments]
    key_rows = segment_rows[key_segments]
    # past the prefix, a target's own keys skip the row's earlier targets
    key_indices = (
        cu_seqlens[key_rows]
        + key_offsets
        + (key_offsets >= prefix_lengths[key_rows])
        * (segment_ranks[key_segments] - 1)
        * num_levels
    )

    target_rows = torch.repeat_interleave(
        row_ids, target_counts, output_size=num_targets
    )
    target_segments = torch.arange(num_targets, device=device) + target_rows + 1
    label_indices = cu_seqlens_q[target_segments, None] + torch.arange(
        num_levels, device=device
    )
    score_indices = label_indices - 1
    score_indices[:, 0] = cu_seqlens[target_rows] + prefix_lengths[target_rows] - 1

    return MultiTargetLayout(
        position_ids=position_ids,
        cu_seqlens_q=cu_seqlens_q.to(torch.int32),
        cu_seqlens_k=cu_seqlens_k.to(torch.int32),
        key_indices=key_indices,
        max_seqlen=max_seqlen,
        label_indices=label_indices,
        score_indices=score_indices,
        target_weights=target_counts[target_rows].float().reciprocal(),
        token_targets=token_targets,
    )
