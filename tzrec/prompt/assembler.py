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
        self.id_shift = int(sid_space.base_vocab_size)

        self.kinds = []
        self.static_tokens = []
        self.hole_slots = []
        self.is_sequences = []
        self.member_names = []
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
                    self.KIND_STATIC, [int(t) for t in seg.token_ids], -1, False, []
                )
                continue
            assert isinstance(seg, SlotSeg)
            if not self.anchor:
                self.anchor = seg.feature_names[0]
            is_sequence = seg.group_type == FeatureGroupType.JAGGED_SEQUENCE
            if seg.fill is FillMode.INLINE:
                self._append(
                    self.KIND_INLINE, [], -1, is_sequence, [seg.feature_names[0]]
                )
            else:
                self._append(
                    self.KIND_PROJECTED,
                    [],
                    occurrences,
                    is_sequence,
                    list(seg.feature_names),
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
    ) -> None:
        """Record one unrolled segment's constants."""
        self.kinds.append(kind)
        self.static_tokens.append(tokens)
        self.hole_slots.append(hole_slot)
        self.is_sequences.append(is_sequence)
        self.member_names.append(members)

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
            # the data carries ``level_offsets[l] + code``; the LM vocabulary
            # needs one further uniform shift by ``base_vocab_size``
            counts = self._inline_counts(batch, index, batch_size)
            values = batch[member + ".values"].to(torch.int64).reshape(-1)
            return counts, values + self.id_shift

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
