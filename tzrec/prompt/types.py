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

This namespace disambiguates ``plan.ResolvedSidSpace``, the resolved token space, from
``prompt_pb2.SidSpace``, the four knobs a user declares. Nothing here stores a
physical dimension: the model resolves those at ``__init__``.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Optional, Tuple, Union

from tzrec.protos.model_pb2 import FeatureGroupConfig, FeatureGroupType
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
        num_positions: the exact count or the ceiling; None when UNBOUNDED.
    """

    kind: WidthKind
    num_positions: Optional[int] = None

    def __post_init__(self) -> None:
        """Reject a count that contradicts the kind."""
        if self.kind is WidthKind.UNBOUNDED:
            if self.num_positions is not None:
                raise ValueError("UNBOUNDED width cannot carry a count.")
        elif self.num_positions is None or self.num_positions < 0:
            raise ValueError(
                f"{self.kind.name} width needs a count >= 0, got {self.num_positions}."
            )


@dataclass(frozen=True)
class ResolvedSidSpace:
    """The resolved SID token space, read by the data layer, model and serving.

    Three coordinate systems and the constants that convert between them: a
    local code in ``[0, codebook[l])``, a flat index ``level_offsets[l] + code``
    which is what the data carries, and an LM token id ``base_vocab_size + flat``
    which is what ``lm_head`` generates.

    Args:
        codebook: per-level vocabulary sizes.
        num_levels: codes per item; also the answer width.
        base_vocab_size: tokenizer size before the SID tokens were appended.
        level_offsets: ``cumsum(codebook) - codebook``.
        band_lo: inclusive lower token-id bound of each level.
        band_hi: inclusive upper token-id bound of each level.
        target_vocab_size: embedding rows after padding, what the LM resizes to.
        sentinel_token_id: id reserved for projected positions, None when no
            slot is projected.
        eos_token_id: end-of-sequence id of the extended tokenizer.
        pad_token_id: padding id of the extended tokenizer.
        bundle_uuid: identity of the SID bundle this space was compiled
            against, empty when no manifest was read. Serving refuses a
            catalog whose bundle differs: a copied artifact's path proves
            nothing.
    """

    codebook: Tuple[int, ...]
    num_levels: int
    base_vocab_size: int
    level_offsets: Tuple[int, ...]
    band_lo: Tuple[int, ...]
    band_hi: Tuple[int, ...]
    target_vocab_size: int
    sentinel_token_id: Optional[int]
    eos_token_id: int
    pad_token_id: int
    bundle_uuid: str = ""


@dataclass(frozen=True)
class Static:
    """A run of literal template tokens.

    Args:
        token_ids: the tokenized run.
    """

    token_ids: Tuple[int, ...]


@dataclass(frozen=True)
class SlotSeg:
    """One ``{{name}}`` position in the assembled stream.

    Args:
        slot_id: stable identifier assigned to the distinct prompt slot.
        name: the placeholder name; also the derived group name.
        feature_names: member feature names.
        group_type: DEEP or JAGGED_SEQUENCE.
        output_key: "" for DEEP, ".sequence" otherwise.
        fill: INLINE writes token ids, PROJECTED writes sentinels and a hole.
        width: position count of this slot.
    """

    slot_id: int
    name: str
    feature_names: Tuple[str, ...]
    group_type: "FeatureGroupType.ValueType"
    output_key: str
    fill: FillMode
    width: Width


Segment = Union[Static, SlotSeg]


@dataclass(frozen=True)
class FoldConstants:
    """Odd multipliers mixed into ``hole_keys``.

    Written as signed int64 so torch takes them verbatim: the fold wraps, and a
    host that had to convert them would be a second place to get it wrong.

    They live in the plan rather than in the host so the artifact, not the
    machine that runs it, decides the keys. Each closes a collision that would
    otherwise produce a correct-looking prefix-cache hit: two slots holding the
    same id, two members of one slot exchanging values, or a multi-value item
    permuted.

    Args:
        slot: multiplies ``slot_id`` into the per-hole salt.
        plan: multiplies the low 64 bits of ``plan_hash``, so keys are
            artifact-specific and a rolling upgrade cannot cross-match.
        value: multiplies each contributing value.
        index: multiplies the member-and-position index within a hole.
        position: multiplies the hole index in the per-item outer fold, without
            which a permuted history collides.
    """

    slot: int = -7046029254386353131
    plan: int = -4417276706812531889
    value: int = -49064778989728563
    index: int = -2960836687051489901
    position: int = -6752110988234923001


@dataclass(frozen=True)
class PromptPlan:
    """The walk order the assembler follows, plus the ceilings derived from it.

    Args:
        segments: prompt body, in emission order.
        response_segments: supervised tail, in emission order.
        max_length: validation ceiling; an over-long row is an error.
        max_total_length: proven ceiling when every slot is bounded, else None.
        max_holes: per-row projected-position ceiling, not a runtime shape.
        logits_suffix_len: upper bound on the supervised logits window.
        static_prefix_len: leading positions that are request-invariant.
        projected_slots: PROJECTED occurrences in emission order, which is also
            ascending hole position; nothing may reorder them by slot id or by
            shared module, because the serving scatter is positional.
        slot_index: slot name to its index in ``projected_slots``.
        fold: the constants ``hole_keys`` mixes in.
    """

    segments: Tuple[Segment, ...]
    response_segments: Tuple[Segment, ...]
    max_length: int
    max_total_length: Optional[int]
    max_holes: int
    logits_suffix_len: Optional[int]
    static_prefix_len: int
    projected_slots: Tuple[SlotSeg, ...]
    slot_index: Mapping[str, int] = field(default_factory=dict)
    fold: FoldConstants = field(default_factory=FoldConstants)


@dataclass(frozen=True)
class ProjectionPlan:
    """Projection topology. Model-only, never persisted.

    Args:
        projections: resolved module id to its configuration.
        slot_to_module: slot id to the module id it uses, so slots sharing a
            ``projection_name`` resolve to one module.
        feature_groups: one derived group per PROJECTED slot. Derived rather
            than declared: a prompt group is never shared with a model tower,
            and four of FeatureGroupConfig's six fields are meaningless here.
    """

    projections: Mapping[str, PromptProjection]
    slot_to_module: Mapping[int, str]
    feature_groups: Tuple[FeatureGroupConfig, ...] = ()


@dataclass(frozen=True)
class CompiledPrompt:
    """Everything ``compile_prompt`` produces.

    Args:
        sid_space: the resolved SID token space.
        prompt_plan: assembler walk order and ceilings.
        projection_plan: projection topology.
        vocab_hash: over sid_space and tokenizer.json; fatal on mismatch.
        plan_hash: over all four parts; warns on mismatch.
    """

    sid_space: Optional[ResolvedSidSpace]
    prompt_plan: PromptPlan
    projection_plan: ProjectionPlan
    vocab_hash: str
    plan_hash: str
