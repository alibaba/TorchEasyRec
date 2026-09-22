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

"""The ``hole_keys`` fold: a serving-only prefix-cache identity per hole.

A PROJECTED slot writes the same sentinel at the same position for every
request, so an engine keying its prefix cache on token ids needs a per-hole
identity instead. ``HoleKeyBuilder`` folds each hole's input values, never the
projected vector, into one ``int64`` key per hole, row-aligned with the
assembler's ``hole_positions``. The fold is integer end to end, so a key cannot
depend on reduction order, device or batch split. No plan or model identity
enters a key: only the engine can namespace a shared cache.
"""

from typing import Dict, Final, List

import torch
from torch import nn

from tzrec.prompt.assembler import _row_ids, _within_row_index, batch_device
from tzrec.prompt.types import PromptPlan
from tzrec.protos.model_pb2 import FeatureGroupType

HOLE_KEYS = "hole_keys"


@torch.jit.script
def mix64(z: torch.Tensor) -> torch.Tensor:
    """A SplitMix64-shaped avalanche over int64, wrapping.

    torch's right shift on a signed integer is arithmetic, so every shift the
    mixer wants as logical is masked back. Getting that wrong is not a weaker
    hash, it is a different function on negative inputs.
    """
    z = z * (-7046029254386353131)
    z = (z ^ ((z >> 30) & 0x3FFFFFFFF)) * (-4658895280553007687)
    z = (z ^ ((z >> 27) & 0x1FFFFFFFFF)) * (-7723592293110705685)
    return z ^ ((z >> 31) & 0x1FFFFFFFF)


def _wrap64(value: int) -> int:
    """Reduce a Python int into the signed 64-bit range, wrapping.

    A salt is built on the host and handed to torch as a scalar, and torch
    refuses a scalar outside the tensor's dtype. Wrapping here is what makes
    "the fold wraps" true at the boundary as well as inside it.
    """
    value &= (1 << 64) - 1
    return value - (1 << 64) if value >= 1 << 63 else value


class HoleKeyBuilder(nn.Module):
    """Folds every projected slot's input values into one key per hole.

    Every value contributing to a hole is discriminated by slot, by member and
    by its index inside the hole; without those, two slots holding the same id,
    a two-member slot with ``(a, b)`` against ``(b, a)``, and a permuted
    multi-value item would all match silently. A dense member contributes its
    float32 bit pattern per row, the parsed input rather than a computed
    reduction. Response slots compile INLINE, so ``projected_slots`` is
    body-only and stays aligned with the assembler's holes.

    Args:
        prompt_plan: the compiled plan; its ``projected_slots`` fix the hole
            order.
    """

    # odd multipliers, written as signed int64 so torch takes them verbatim;
    # TorchScript resolves a Final class attribute as a constant
    C_SLOT: Final[int] = -7046029254386353131
    C_VALUE: Final[int] = -49064778989728563
    C_INDEX: Final[int] = -2960836687051489901
    # member index and position within a hole packed into one integer, wide
    # enough that a position cannot carry into the member index
    MEMBER_STRIDE: Final[int] = 1 << 32

    member_names: List[List[str]]
    is_sequences: List[bool]
    salts: List[int]

    def __init__(self, prompt_plan: PromptPlan) -> None:
        super().__init__()
        self.member_names = []
        self.is_sequences = []
        self.salts = []
        for seg in prompt_plan.projected_slots:
            self.member_names.append(list(seg.feature_names))
            self.is_sequences.append(seg.group_type == FeatureGroupType.JAGGED_SEQUENCE)
            # 1-based: slot 0 must not salt to zero, or its empty hole
            # would fold to zero as an unsalted accumulator did
            self.salts.append(_wrap64(self.C_SLOT * (int(seg.slot_id) + 1)))
        self.num_slots = len(self.member_names)

    def _fold_slot(self, batch: Dict[str, torch.Tensor], index: int) -> torch.Tensor:
        """One projected occurrence's keys, one per hole, in sample order."""
        salt = self.salts[index]
        members = self.member_names[index]
        is_sequence = self.is_sequences[index]
        # the assembler's hole count: one per item of a sequence slot, one per
        # sample of a DEEP slot; a dense member has one row per hole either way
        first = batch[members[0] + ".values"]
        if first.is_floating_point():
            num_holes = int(first.size(0))
        else:
            lengths = batch[members[0] + ".lengths"].to(torch.int64)
            num_holes = int(torch.sum(lengths)) if is_sequence else int(lengths.numel())
        # seeded with the salt, not zero: a hole no value reaches -- an empty
        # DEEP member, a zero-length item -- is still that slot's hole
        keys = torch.full((num_holes,), salt, dtype=torch.int64, device=first.device)
        for member_index in range(len(members)):
            member = members[member_index]
            raw = batch[member + ".values"]
            if raw.is_floating_point():
                rows = raw.size(0)
                width = raw.size(1)
                values = (
                    raw.to(torch.float32)
                    .contiguous()
                    .view(torch.int32)
                    .to(torch.int64)
                    .reshape(-1)
                    & 0xFFFFFFFF
                )
                hole = torch.arange(
                    rows, dtype=torch.int64, device=raw.device
                ).repeat_interleave(width)
                local = (
                    torch.arange(width, dtype=torch.int64, device=raw.device)
                    .unsqueeze(0)
                    .expand(rows, width)
                    .reshape(-1)
                )
            else:
                values = raw.to(torch.int64).reshape(-1)
                key_length_key = member + ".key_lengths"
                if not is_sequence:
                    lengths = batch[member + ".lengths"].to(torch.int64)
                    hole = _row_ids(lengths)
                    local = _within_row_index(lengths)
                elif key_length_key in batch:
                    key_lengths = batch[key_length_key].to(torch.int64).reshape(-1)
                    hole = _row_ids(key_lengths)
                    local = _within_row_index(key_lengths)
                else:
                    hole = torch.arange(
                        values.numel(), dtype=torch.int64, device=values.device
                    )
                    local = torch.zeros_like(hole)

            local = local + member_index * self.MEMBER_STRIDE
            mixed = mix64(values * self.C_VALUE + local * self.C_INDEX + salt)
            keys.index_add_(0, hole, mixed)
        return keys

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Fold one batch.

        Args:
            batch: the parsed feature dict, keyed ``{feature}.values`` /
                ``.lengths`` / ``.key_lengths`` as the data parser emits it.

        Returns:
            ``(total_holes,)`` int64, row-aligned with ``hole_positions``.
        """
        # an empty stream first, so a plan without a projected slot still has
        # something to join
        parts: List[torch.Tensor] = [
            torch.zeros(0, dtype=torch.int64, device=batch_device(batch))
        ]
        for i in range(self.num_slots):
            parts.append(self._fold_slot(batch, i))
        return torch.cat(parts, dim=0)
