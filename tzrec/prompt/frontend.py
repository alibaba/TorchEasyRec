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

"""The prompt front-end a serving runtime loads: assemble, look up, project.

The assembler is the collator's walk as tensor ops. ``PromptPlan`` is a
compile-time constant, so the segment loop unrolls into parallel constant
lists at construction and what remains is jagged integer arithmetic --
``cumsum``, ``repeat_interleave``, ``index_copy_`` -- with no data-dependent
control flow, which is what lets ``torch.jit.script`` carry it into a runtime
that has no tzrec source.

``hole_keys`` is integer end to end: ``int64`` addition is associative and
commutative and wraps deterministically, so the fold cannot depend on the order
``index_add_`` happens to reduce in, on the device, or on how the batch was
split. A float accumulator would satisfy "do not fold the projected vector" in
letter and reintroduce the variance in spirit.
"""

from typing import Dict, Final, List, Optional, Tuple

import torch
from torch import nn

from tzrec.prompt.types import (
    FillMode,
    FoldConstants,
    PromptPlan,
    ResolvedSidSpace,
    SlotSeg,
    Static,
)
from tzrec.protos.model_pb2 import FeatureGroupType

OUT_INPUT_IDS = "input_ids"
OUT_CU_SEQLENS = "cu_seqlens"
OUT_HOLE_POSITIONS = "hole_positions"
OUT_HOLE_KEYS = "hole_keys"
OUT_HOLE_SLOT_COUNTS = "hole_slot_counts"
OUT_SLOT_EMBEDS = "slot_embeds"
OUT_RESPONSE_LENGTHS = "response_lengths"


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
def _pick_device(batch: Dict[str, torch.Tensor]) -> torch.device:
    """Device the batch already lives on, so nothing is built on the wrong one."""
    for value in batch.values():
        return value.device
    return torch.device("cpu")


@torch.jit.script
def _destinations(seg_start: torch.Tensor, seg_len: torch.Tensor) -> torch.Tensor:
    """Absolute index of every value of one segment in the packed stream."""
    return torch.repeat_interleave(seg_start, seg_len) + _within_row_index(seg_len)


def _wrap64(value: int) -> int:
    """Reduce a Python int into the signed 64-bit range, wrapping.

    Every constant the fold mixes is built on the host and handed to torch as a
    scalar, and torch refuses a scalar outside the tensor's dtype. Wrapping here
    is what makes "the fold wraps" true at the boundary as well as inside it.
    """
    value &= (1 << 64) - 1
    return value - (1 << 64) if value >= 1 << 63 else value


def _plan_salt(fold: FoldConstants, plan_hash: str) -> int:
    """Low 64 bits of ``plan_hash``, as a signed multiple of the plan constant."""
    if not plan_hash:
        return 0
    return _wrap64(_wrap64(int(plan_hash[:16], 16)) * fold.plan)


def host_lengths_key(feature_name: str) -> str:
    """The per-row item count a host lookup stage emits beside its rows."""
    return f"{feature_name}__lengths"


class PromptAssembler(nn.Module):
    """Walks a compiled plan to build one batch's packed token stream.

    Args:
        plan: the compiled walk order and its constants.
        sid_space: resolved SID token space; required when a slot renders SID
            codes or is projected.
        features_are_dense: feature name to whether it arrives as floats.
        features_are_multi_valued: feature name to whether it carries
            ``key_lengths``, that is whether ``value_dim != 1``.
        plan_hash: the compiled plan's hash; its low bits salt every key, so
            two plans cannot cross-match in a shared prefix cache.
        include_response: whether to emit the supervised tail.
    """

    # TorchScript resolves a Final class attribute as a constant; a
    # module-level one it cannot see at all
    KIND_STATIC: Final[int] = 0
    KIND_INLINE: Final[int] = 1
    KIND_PROJECTED: Final[int] = 2
    # member index and position within a hole packed into one integer, wide
    # enough that a position cannot carry into the member index
    MEMBER_STRIDE: Final[int] = 1 << 32

    kinds: List[int]
    static_tokens: List[List[int]]
    value_keys: List[str]
    length_keys: List[str]
    host_length_keys: List[str]
    key_length_keys: List[str]
    hole_slots: List[int]
    slot_ids: List[int]
    is_sequences: List[bool]
    salts: List[int]
    member_value_keys: List[List[str]]
    member_length_keys: List[List[str]]
    member_host_length_keys: List[List[str]]
    member_key_length_keys: List[List[str]]
    member_is_dense: List[bool]

    def __init__(
        self,
        plan: PromptPlan,
        sid_space: Optional[ResolvedSidSpace] = None,
        features_are_dense: Optional[Dict[str, bool]] = None,
        features_are_multi_valued: Optional[Dict[str, bool]] = None,
        plan_hash: str = "",
        include_response: bool = True,
    ) -> None:
        super().__init__()
        dense = features_are_dense if features_are_dense is not None else {}
        multi = (
            features_are_multi_valued if features_are_multi_valued is not None else {}
        )
        segments = tuple(plan.segments)
        self.num_body = len(segments)
        if include_response:
            segments = segments + tuple(plan.response_segments)

        sentinel = -1
        if sid_space is not None and sid_space.sentinel_token_id is not None:
            sentinel = int(sid_space.sentinel_token_id)
        self.sentinel = sentinel
        id_shift = 0 if sid_space is None else int(sid_space.base_vocab_size)

        self.kinds = []
        self.static_tokens = []
        self.value_keys = []
        self.length_keys = []
        self.host_length_keys = []
        self.key_length_keys = []
        self.hole_slots = []
        self.slot_ids = []
        self.is_sequences = []
        self.member_value_keys = []
        self.member_length_keys = []
        self.member_host_length_keys = []
        self.member_key_length_keys = []
        self.member_is_dense = []

        for seg in segments:
            if isinstance(seg, Static):
                self._append(
                    PromptAssembler.KIND_STATIC,
                    [int(t) for t in seg.token_ids],
                    "",
                    "",
                    "",
                    "",
                    -1,
                    -1,
                    False,
                    [],
                    [],
                    [],
                    [],
                )
                continue

            assert isinstance(seg, SlotSeg)
            primary = seg.feature_names[0]
            is_sequence = seg.group_type == FeatureGroupType.JAGGED_SEQUENCE
            if seg.fill is FillMode.INLINE:
                if sid_space is None:
                    raise ValueError(
                        f"prompt slot [{seg.name}] renders INLINE, which means "
                        f"SID codes, but no sid_space was compiled."
                    )
                self._append(
                    PromptAssembler.KIND_INLINE,
                    [],
                    f"{primary}.values",
                    f"{primary}.lengths",
                    host_lengths_key(primary),
                    f"{primary}.key_lengths" if multi.get(primary, False) else "",
                    -1,
                    int(seg.slot_id),
                    is_sequence,
                    [],
                    [],
                    [],
                    [],
                )
                continue

            if sentinel < 0:
                raise ValueError(
                    f"prompt slot [{seg.name}] is PROJECTED but no sentinel token "
                    f"was compiled; a hole would be indistinguishable from content."
                )
            for name in seg.feature_names:
                if dense.get(name, False) and is_sequence:
                    raise ValueError(
                        f"prompt slot [{seg.name}] member [{name}] is a dense "
                        f"sequence; the fold has no per-item boundary for it."
                    )
            self._append(
                PromptAssembler.KIND_PROJECTED,
                [],
                "",
                f"{primary}.lengths",
                host_lengths_key(primary),
                "",
                int(plan.slot_index[seg.name]),
                int(seg.slot_id),
                is_sequence,
                [f"{name}.values" for name in seg.feature_names],
                [f"{name}.lengths" for name in seg.feature_names],
                [host_lengths_key(name) for name in seg.feature_names],
                [
                    f"{name}.key_lengths" if multi.get(name, False) else ""
                    for name in seg.feature_names
                ],
            )
            self.member_is_dense = self.member_is_dense + [
                dense.get(name, False) for name in seg.feature_names
            ]

        self.id_shift = id_shift
        self.num_segments = len(self.kinds)
        self.num_hole_slots = len(plan.projected_slots)
        self.fold_value = int(plan.fold.value)
        self.fold_index = int(plan.fold.index)
        plan_salt = _plan_salt(plan.fold, plan_hash)
        self.salts = [
            _wrap64(plan.fold.slot * slot_id + plan_salt) for slot_id in self.slot_ids
        ]

    def _append(
        self,
        kind: int,
        tokens: List[int],
        value_key: str,
        length_key: str,
        host_length_key: str,
        key_length_key: str,
        hole_slot: int,
        slot_id: int,
        is_sequence: bool,
        member_values: List[str],
        member_lengths: List[str],
        member_host_lengths: List[str],
        member_key_lengths: List[str],
    ) -> None:
        """Record one unrolled segment's constants."""
        self.kinds.append(kind)
        self.static_tokens.append(tokens)
        self.value_keys.append(value_key)
        self.length_keys.append(length_key)
        self.host_length_keys.append(host_length_key)
        self.key_length_keys.append(key_length_key)
        self.hole_slots.append(hole_slot)
        self.slot_ids.append(slot_id)
        self.is_sequences.append(is_sequence)
        self.member_value_keys.append(member_values)
        self.member_length_keys.append(member_lengths)
        self.member_host_length_keys.append(member_host_lengths)
        self.member_key_length_keys.append(member_key_lengths)

    def _lengths(
        self, batch: Dict[str, torch.Tensor], key: str, host_key: str
    ) -> torch.Tensor:
        """Per-row item count, from the parsed dict or from a host lookup stage."""
        if key in batch:
            return batch[key].to(torch.int64)
        return batch[host_key].to(torch.int64)

    def _batch_size(self, batch: Dict[str, torch.Tensor]) -> int:
        """Row count, from the first segment that names a per-row length."""
        for i in range(self.num_segments):
            key = self.length_keys[i]
            if key != "":
                if key in batch:
                    return int(batch[key].numel())
                host_key = self.host_length_keys[i]
                if host_key in batch:
                    return int(batch[host_key].numel())
        if "batch_size" in batch:
            return int(batch["batch_size"])
        return 0

    def _values_per_row(
        self, batch: Dict[str, torch.Tensor], index: int, batch_size: int
    ) -> torch.Tensor:
        """Token count each row contributes for one INLINE segment.

        For a multi-value sequence -- which is what a SID history is -- the row
        holds ``lengths`` items and each item holds ``key_lengths`` values, so
        the count is a segmented sum rather than ``lengths`` itself.
        """
        lengths = self._lengths(
            batch, self.length_keys[index], self.host_length_keys[index]
        )
        key = self.key_length_keys[index]
        if key == "":
            return lengths
        key_lengths = batch[key].to(torch.int64)
        counts = torch.zeros(batch_size, dtype=torch.int64, device=lengths.device)
        return counts.index_add_(0, _row_ids(lengths), key_lengths)

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

        if kind == self.KIND_INLINE:
            seg_len = self._values_per_row(batch, index, batch_size)
            values = batch[self.value_keys[index]].to(torch.int64).reshape(-1)
            return seg_len, values + self.id_shift

        if self.is_sequences[index]:
            seg_len = self._lengths(
                batch, self.length_keys[index], self.host_length_keys[index]
            )
        else:
            seg_len = torch.ones(batch_size, dtype=torch.int64, device=device)
        total = int(torch.sum(seg_len))
        return seg_len, torch.full(
            (total,), self.sentinel, dtype=torch.int64, device=seg_len.device
        )

    def _fold_segment(
        self,
        batch: Dict[str, torch.Tensor],
        index: int,
        batch_size: int,
        hole_base: int,
        keys: torch.Tensor,
    ) -> None:
        """Mix one projected segment's input values into ``keys``.

        Every member value that produces a hole contributes, discriminated by
        slot, by member and by its index inside the hole. Without the last two
        a two-member slot with values ``(a, b)`` would match one with
        ``(b, a)``, and a permuted multi-value item would match itself
        reordered -- both plausible, both wrong, and both silent.
        """
        salt = self.salts[index]
        member_values = self.member_value_keys[index]
        for member in range(len(member_values)):
            raw = batch[member_values[member]]
            lengths = self._lengths(
                batch,
                self.member_length_keys[index][member],
                self.member_host_length_keys[index][member],
            )
            key_length_key = self.member_key_length_keys[index][member]

            if not self.is_sequences[index]:
                # one hole per row; a dense member contributes its float32 bit
                # pattern verbatim, which is the parsed input and not a
                # computed reduction, so it is stable for a given request
                if raw.dtype == torch.float32:
                    width = raw.size(1)
                    values = (
                        raw.contiguous().view(torch.int32).to(torch.int64).reshape(-1)
                        & 0xFFFFFFFF
                    )
                    hole = torch.repeat_interleave(
                        torch.arange(batch_size, dtype=torch.int64, device=raw.device),
                        torch.full(
                            (batch_size,), width, dtype=torch.int64, device=raw.device
                        ),
                    )
                    local = (
                        torch.arange(width, dtype=torch.int64, device=raw.device)
                        .unsqueeze(0)
                        .expand(batch_size, width)
                        .reshape(-1)
                    )
                else:
                    values = raw.to(torch.int64).reshape(-1)
                    hole = _row_ids(lengths)
                    local = _within_row_index(lengths)
            else:
                values = raw.to(torch.int64).reshape(-1)
                if key_length_key == "":
                    hole = torch.arange(
                        values.numel(), dtype=torch.int64, device=values.device
                    )
                    local = torch.zeros_like(hole)
                else:
                    key_lengths = batch[key_length_key].to(torch.int64)
                    hole = _row_ids(key_lengths)
                    local = _within_row_index(key_lengths)

            local = local + member * self.MEMBER_STRIDE
            mixed = mix64(values * self.fold_value + local * self.fold_index + salt)
            keys.index_add_(0, hole + hole_base, mixed)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Assemble one batch.

        Args:
            batch: the parsed feature dict, keyed ``{feature}.values`` /
                ``.lengths`` / ``.key_lengths`` as both hosts already emit it.

        Returns:
            ``input_ids``, ``cu_seqlens``, ``hole_positions``, ``hole_keys``,
            ``hole_slot_counts`` and ``response_lengths``. The two hole-indexed
            outputs are row-aligned: entry ``k`` of each describes the same
            hole, in ``projected_slots`` order.
        """
        batch_size = self._batch_size(batch)
        device = _pick_device(batch)

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
        out = torch.zeros(total_tokens, dtype=torch.int64, device=row_total.device)

        # one entry per projected slot, and an empty stream under Pattern I,
        # where the concatenation below would otherwise have nothing to join
        hole_parts: List[torch.Tensor] = [
            torch.zeros(0, dtype=torch.int64, device=device)
        ]
        for _ in range(self.num_hole_slots):
            hole_parts.append(torch.zeros(0, dtype=torch.int64, device=device))

        response_lengths = torch.zeros(
            batch_size, dtype=torch.int64, device=row_total.device
        )
        for i in range(self.num_segments):
            dest = _destinations(row_start + seg_offsets[i], seg_lens[i])
            out.index_copy_(0, dest, seg_values[i])
            slot = self.hole_slots[i]
            if slot >= 0:
                hole_parts[slot + 1] = dest
            if i >= self.num_body:
                response_lengths = response_lengths + seg_lens[i]

        hole_positions = torch.cat(hole_parts, dim=0)
        # how many of those holes each projected slot owns, so a host can cut
        # the flat streams back into per-slot spans without re-deriving the plan
        slot_counts = torch.zeros(self.num_hole_slots, dtype=torch.int64, device=device)
        for slot in range(self.num_hole_slots):
            slot_counts[slot] = hole_parts[slot + 1].numel()
        keys = torch.zeros(hole_positions.numel(), dtype=torch.int64, device=out.device)
        hole_base = 0
        for slot in range(self.num_hole_slots):
            for i in range(self.num_segments):
                if self.hole_slots[i] == slot:
                    self._fold_segment(batch, i, batch_size, hole_base, keys)
                    hole_base = hole_base + hole_parts[slot + 1].numel()

        cu_seqlens = torch.cat(
            [
                torch.zeros(1, dtype=torch.int64, device=row_total.device),
                torch.cumsum(row_total, dim=0),
            ]
        )
        # literals rather than the module constants above: TorchScript cannot
        # see a module-level global. ``frontend_test`` pins the two together.
        return {
            "input_ids": out,
            "cu_seqlens": cu_seqlens.to(torch.int32),
            "hole_positions": hole_positions,
            "hole_keys": keys,
            "hole_slot_counts": slot_counts,
            "response_lengths": response_lengths,
        }


class SlotTable(nn.Module):
    """One projected slot's members, as plain embedding tables.

    Used when the host does not run an embedding stage of its own. The lookup
    has to live somewhere: either the host does it and hands over vectors, or
    the artifact carries the tables. Keeping both shapes behind one module means
    the walk, the fold and the projections do not change between them.

    Args:
        value_keys: each member's value key.
        length_keys: each member's per-row counts.
        key_length_keys: each member's per-item value counts, empty when the
            member holds one value per item.
        num_embeddings: each member's table size.
        dims: each member's embedding dimension.
        is_sequence: whether holes are items rather than rows.
        modes: each member's pooling, ``sum`` or ``mean``.
    """

    value_keys: List[str]
    length_keys: List[str]
    key_length_keys: List[str]

    def __init__(
        self,
        value_keys: List[str],
        length_keys: List[str],
        key_length_keys: List[str],
        num_embeddings: List[int],
        dims: List[int],
        is_sequence: bool,
        modes: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.value_keys = value_keys
        self.length_keys = length_keys
        self.key_length_keys = key_length_keys
        self.is_sequence = is_sequence
        modes = modes if modes is not None else ["sum"] * len(dims)
        self.tables = nn.ModuleList(
            [
                nn.EmbeddingBag(rows, dim, mode=mode, include_last_offset=True)
                for rows, dim, mode in zip(num_embeddings, dims, modes)
            ]
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Look every member up and concatenate along the feature axis.

        Returns:
            ``(num_holes, group_total_dim)``, ordered by hole.
        """
        parts: List[torch.Tensor] = []
        index = 0
        for table in self.tables:
            values = batch[self.value_keys[index]].to(torch.int64).reshape(-1)
            key = self.key_length_keys[index]
            if key != "":
                counts = batch[key].to(torch.int64)
            elif self.is_sequence:
                counts = torch.ones_like(values)
            else:
                counts = batch[self.length_keys[index]].to(torch.int64)
            offsets = torch.cat(
                [
                    torch.zeros(1, dtype=torch.int64, device=values.device),
                    torch.cumsum(counts, dim=0),
                ]
            )
            parts.append(table(values, offsets))
            index += 1
        return torch.cat(parts, dim=1)


class PromptFrontEnd(nn.Module):
    """The serving artifact: assemble, look up, project.

    Exported whether or not any slot is projected -- with none it degenerates
    to the assembler and three empty streams, which is cheap and keeps what a
    prompt happens to contain from deciding whether a serving-critical artifact
    exists.

    ``slot_embeds`` is ordered by ascending hole, matching ``hole_positions``
    entry for entry. That ordering is an obligation rather than a check: the
    engine scatters positionally, so a permuted source gives every hole a
    neighbour's embedding, the counts still agree, and nothing raises.

    Lookup is a stage of its own. Without a host embedding stage the artifact
    carries the tables and the batch holds raw ids. Under a host stage the
    embeddings are already in the batch when this runs, one entry per key in
    ``embed_keys`` for each slot, concatenated on the feature axis. Everything
    downstream of the lookup is identical either way.

    Args:
        assembler: the walk.
        projections: one per projected slot in ``projected_slots`` order; slots
            sharing a module appear more than once, by reference.
        embed_keys: per projected slot, the batch keys holding its members'
            looked-up rows, in member order. Ignored when ``tables`` is given.
        tables: one per projected slot, when the artifact carries the slot
            tables rather than receiving vectors from a host stage.
        vocab_hash: the compiled prompt's vocabulary digest, so a loader can
            refuse a front-end that does not pair with its ``prompt.json``.
        plan_hash: the compiled prompt's plan digest.
        bundle_uuid: identity of the SID bundle the prompt was compiled against.
    """

    embed_keys: List[List[str]]
    vocab_hash: str
    plan_hash: str
    bundle_uuid: str

    def __init__(
        self,
        assembler: PromptAssembler,
        projections: List[nn.Module],
        embed_keys: List[List[str]],
        tables: Optional[List[SlotTable]] = None,
        vocab_hash: str = "",
        plan_hash: str = "",
        bundle_uuid: str = "",
    ) -> None:
        super().__init__()
        self.assembler = assembler
        self.projections = nn.ModuleList(projections)
        self.embed_keys = embed_keys
        self.has_tables = tables is not None
        self.tables = nn.ModuleList(tables if tables is not None else [])
        self.vocab_hash = vocab_hash
        self.plan_hash = plan_hash
        self.bundle_uuid = bundle_uuid

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        """Assemble one batch and project its holes.

        The second argument is not decoration: a C++ host that runs this as its
        JIT stage calls ``forward(data, device)``, the same pair tzrec's own
        ScriptWrapper takes. A Python caller may omit it, in which case the
        batch stays where it is.

        Args:
            data: the parsed feature dict, plus looked-up rows under a host
                lookup stage.
            device: where to run; the batch is moved there first.

        Returns:
            The assembler's outputs plus ``slot_embeds``.
        """
        target = _pick_device(data) if device is None else device
        batch: Dict[str, torch.Tensor] = {}
        for key, value in data.items():
            batch[key] = value.to(target)
        out = self.assembler(batch)

        # gathered first, into a plain list: TorchScript indexes a ModuleList
        # only with a literal, so the two lists cannot be walked together
        features: List[torch.Tensor] = []
        if self.has_tables:
            for table in self.tables:
                features.append(table(batch))
        else:
            for keys in self.embed_keys:
                members: List[torch.Tensor] = []
                for key in keys:
                    members.append(batch[key])
                features.append(torch.cat(members, dim=1))

        parts: List[torch.Tensor] = []
        index = 0
        for projection in self.projections:
            parts.append(projection(features[index]))
            index += 1

        if len(parts) > 0:
            out["slot_embeds"] = torch.cat(parts, dim=0)
        else:
            out["slot_embeds"] = torch.zeros(
                0, 0, dtype=torch.float32, device=out["input_ids"].device
            )
        return out
