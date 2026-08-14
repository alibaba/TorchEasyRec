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

from dataclasses import dataclass
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
        projected_slots: PROJECTED occurrences in emission order.
    """

    segments: Tuple[Segment, ...]
    response_segments: Tuple[Segment, ...]
    max_length: int
    max_total_length: Optional[int]
    max_holes: int
    logits_suffix_len: Optional[int]
    static_prefix_len: int
    projected_slots: Tuple[SlotSeg, ...]


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
        tokenizer_dir: where the extended tokenizer was written.
        vocab_hash: over sid_space and tokenizer.json; fatal on mismatch.
        plan_hash: over all four parts; warns on mismatch.
    """

    sid_space: Optional[ResolvedSidSpace]
    prompt_plan: PromptPlan
    projection_plan: ProjectionPlan
    tokenizer_dir: str
    vocab_hash: str
    plan_hash: str
