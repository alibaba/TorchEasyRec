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

"""Products of ``compile_prompt``.

This namespace disambiguates ``plan.SidSpace``, the resolved token space, from
``prompt_pb2.SidSpace``, the four knobs a user declares. Nothing here stores a
physical dimension: the model resolves those at ``__init__``.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Optional, Tuple, Union

from tzrec.protos.model_pb2 import FeatureGroupType
from tzrec.protos.prompt_pb2 import PromptProjection


class FillMode(Enum):
    """How a slot's value reaches the LM input space."""

    INLINE = "inline"
    PROJECTED = "projected"


class WidthKind(Enum):
    """Whether a segment's position count is known, bounded, or neither."""

    STATIC = "static"
    BOUNDED = "bounded"
    UNBOUNDED = "unbounded"


@dataclass(frozen=True)
class Width:
    """Position count of a slot.

    Args:
        kind: STATIC when the count is exact, BOUNDED when only a ceiling is
            known, UNBOUNDED when neither.
        n: the exact count or the ceiling; None when UNBOUNDED.
    """

    kind: WidthKind
    n: Optional[int] = None

    def __post_init__(self) -> None:
        """Reject a count that contradicts the kind."""
        if self.kind is WidthKind.UNBOUNDED:
            if self.n is not None:
                raise ValueError("UNBOUNDED width cannot carry a count.")
        elif self.n is None or self.n < 0:
            raise ValueError(
                f"{self.kind.name} width needs a count >= 0, got {self.n}."
            )


@dataclass(frozen=True)
class SidSpace:
    """The resolved SID token space, read by the data layer, model and serving.

    Three coordinate systems and the constants that convert between them: a
    local code in ``[0, codebook[l])``, a flat index ``level_offsets[l] + code``
    which is what the data carries, and an LM token id ``base_vocab + flat``
    which is what ``lm_head`` generates.

    Args:
        codebook: per-level vocabulary sizes.
        num_levels: codes per item; also the answer width.
        base_vocab: tokenizer size before the SID atoms were appended.
        level_offsets: ``cumsum(codebook) - codebook``.
        band_lo: inclusive lower token-id bound of each level.
        band_hi: inclusive upper token-id bound of each level.
        target_vocab: embedding rows after padding, what the LM resizes to.
        sentinel_token_id: id reserved for projected positions, None when no
            slot is projected.
        eos_token_id: end-of-sequence id of the extended tokenizer.
        pad_token_id: padding id of the extended tokenizer.
    """

    codebook: Tuple[int, ...]
    num_levels: int
    base_vocab: int
    level_offsets: Tuple[int, ...]
    band_lo: Tuple[int, ...]
    band_hi: Tuple[int, ...]
    target_vocab: int
    sentinel_token_id: Optional[int]
    eos_token_id: int
    pad_token_id: int

    @property
    def sid_vocab_size(self) -> int:
        """Atoms appended to the backbone vocabulary."""
        return sum(self.codebook)


@dataclass(frozen=True)
class Static:
    """A run of literal template tokens.

    Args:
        token_ids: the tokenized run.
        owner_slot_id: slot this run was folded into, so it vanishes when that
            slot is dropped. None when the run belongs to no slot.
    """

    token_ids: Tuple[int, ...]
    owner_slot_id: Optional[int]


@dataclass(frozen=True)
class SlotSeg:
    """One ``{{name}}`` position in the assembled stream.

    Args:
        slot_id: index into ``PromptPlan.projected_slots`` ordering.
        name: the placeholder name; also the derived group name.
        sources: member feature names.
        group_type: DEEP or JAGGED_SEQUENCE.
        output_key: "" for DEEP, ".sequence" otherwise.
        fill: INLINE writes token ids, PROJECTED writes sentinels and a hole.
        width: position count of this slot.
        droppable: whether an empty value removes the slot and its folded text.
    """

    slot_id: int
    name: str
    sources: Tuple[str, ...]
    group_type: "FeatureGroupType.ValueType"
    output_key: str
    fill: FillMode
    width: Width
    droppable: bool


Segment = Union[Static, SlotSeg]


@dataclass(frozen=True)
class PromptPlan:
    """The walk order the assembler follows, plus the ceilings derived from it.

    Args:
        segments: prompt body, in emission order.
        response_segments: supervised tail, in emission order.
        max_length: validation ceiling; an over-long row is an error.
        max_total_length: proven ceiling when every slot is bounded, else None.
        max_holes: per-row projected-position ceiling, not a runtime shape.
        suffix_keep: upper bound on the supervised logits window.
        static_prefix_len: leading positions that are request-invariant.
        length_buckets: sampler and graph-capture buckets.
        slot_index: slot name to its index in ``projected_slots``.
        projected_slots: fixes the order hole positions are written in.
    """

    segments: Tuple[Segment, ...]
    response_segments: Tuple[Segment, ...]
    max_length: int
    max_total_length: Optional[int]
    max_holes: int
    suffix_keep: Optional[int]
    static_prefix_len: int
    length_buckets: Tuple[int, ...]
    slot_index: Mapping[str, int]
    projected_slots: Tuple[SlotSeg, ...]


@dataclass(frozen=True)
class ModulePlan:
    """Projection topology. Model-only, never persisted.

    Args:
        projections: resolved module id to its configuration.
        slot_to_module: slot id to the module id it uses, so slots sharing a
            ``projection_name`` resolve to one module.
    """

    projections: Mapping[str, PromptProjection]
    slot_to_module: Mapping[int, str]


@dataclass(frozen=True)
class CompiledPrompt:
    """Everything ``compile_prompt`` produces.

    Args:
        sid_space: the resolved SID token space.
        prompt_plan: assembler walk order and ceilings.
        module_plan: projection topology.
        tokenizer_dir: where the extended tokenizer was written.
        vocab_hash: over sid_space and tokenizer.json; fatal on mismatch.
        plan_hash: over all four parts; warns on mismatch.
    """

    sid_space: Optional[SidSpace]
    prompt_plan: PromptPlan
    module_plan: ModulePlan
    tokenizer_dir: str
    vocab_hash: str
    plan_hash: str
