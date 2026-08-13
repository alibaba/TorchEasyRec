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

"""Compiles a ``PromptConfig`` into the artifacts the rest of the stack reads.

Runs once, on the main process, and is the only tokenizer construction in an
entry point. It resolves no physical dimension: the model does that at
``__init__`` from ``group_total_dim``.
"""

import hashlib
import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tokenizers import Tokenizer

from tzrec.features.feature import BaseFeature
from tzrec.prompt.plan import (
    CompiledPrompt,
    FillMode,
    ProjectionPlan,
    PromptPlan,
    ResolvedSidSpace,
    Segment,
    SlotSeg,
    Static,
    Width,
    WidthKind,
)
from tzrec.protos.model_pb2 import FeatureGroupConfig, FeatureGroupType
from tzrec.protos.prompt_pb2 import (
    PromptConfig,
    PromptProjection,
    PromptSlot,
    SidSpace,
)
from tzrec.utils.logging_util import logger

_PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")


def _split_template(template: str) -> Tuple[List[str], List[str]]:
    """Split on ``{{name}}`` into static runs and the names between them.

    ``re.split`` with one capture group alternates literal, capture, literal,
    so a template with n placeholders yields n + 1 static runs. Both lists are
    returned; run i precedes name i.
    """
    parts = _PLACEHOLDER.split(template)
    return parts[0::2], parts[1::2]


def _ceil_to(value: int, multiple: int) -> int:
    """Round ``value`` up to a multiple, or return it unchanged when 0."""
    if multiple <= 1:
        return value
    return -(-value // multiple) * multiple


def _resolve_slot(
    name: str, declared_slots_by_name: Dict[str, PromptSlot]
) -> PromptSlot:
    """Return the declared slot, or an implicit single-feature slot."""
    if name in declared_slots_by_name:
        return declared_slots_by_name[name]
    implicit = PromptSlot(name=name)
    implicit.feature_names.append(name)
    return implicit


def _slot_width(
    members: Sequence[BaseFeature],
    group_type: "FeatureGroupType.ValueType",
    answer_levels: Optional[int] = None,
) -> Width:
    """Derive a slot's position count from its members.

    A DEEP slot pools to exactly one position. The answer is exactly one SID
    item, so its width is the codebook depth and needs no sequence_length. Any
    other sequence slot is bounded by its members' sequence_length, and
    unbounded when none declares one.
    """
    if answer_levels is not None:
        return Width(WidthKind.STATIC, answer_levels)
    if group_type == FeatureGroupType.DEEP:
        return Width(WidthKind.STATIC, 1)
    # BaseFeature.sequence_length, not config: a member of a SequenceFeature
    # group inherits the group's cap and never sets its own field.
    caps = [f.sequence_length for f in members if f.sequence_length]
    if not caps:
        return Width(WidthKind.UNBOUNDED)
    return Width(WidthKind.BOUNDED, max(caps))


def _derive_slot_layout(
    name: str, members: Sequence[BaseFeature]
) -> Tuple["FeatureGroupType.ValueType", FillMode]:
    """Derive how a prompt slot is grouped and emitted."""
    sequence_flags = {member.is_sequence for member in members}
    if len(sequence_flags) != 1:
        raise ValueError(
            f"prompt slot [{name}] mixes sequence and scalar features "
            f"{[member.name for member in members]}; a slot must be all one kind, "
            "or its group would carry a '.query' output the prompt cannot place."
        )
    is_sequence = sequence_flags.pop()
    group_type = (
        FeatureGroupType.JAGGED_SEQUENCE if is_sequence else FeatureGroupType.DEEP
    )
    fill_mode = (
        FillMode.INLINE
        if is_sequence and len(members) == 1 and not members[0].has_embedding
        else FillMode.PROJECTED
    )
    return group_type, fill_mode


def _atom_tokens(sid_space: SidSpace) -> List[str]:
    """Render the SID atom tokens, one per flat index."""
    fmt = sid_space.atom_token_format
    return [fmt.replace("{i}", str(i)) for i in range(sum(sid_space.codebook))]


def _read_manifest_codebook(path: str) -> List[int]:
    """Read ``codebook`` from a SID manifest."""
    if not os.path.exists(path):
        raise ValueError(f"sid_space.manifest_path [{path}] does not exist.")
    with open(path, "r") as f:
        return [int(c) for c in json.load(f)["codebook"]]


def _build_sid_space(
    cfg: PromptConfig, tok: Tokenizer, base_vocab: int, has_projection: bool
) -> Optional[ResolvedSidSpace]:
    """Extend the tokenizer with SID atoms and resolve the token space."""
    if not cfg.HasField("sid_space"):
        return None
    space = cfg.sid_space
    codebook = [int(c) for c in space.codebook]
    if not codebook:
        raise ValueError("sid_space.codebook is required; one size per SID level.")
    if any(c <= 0 for c in codebook):
        raise ValueError(f"every codebook size must be positive, got {codebook}.")

    if space.HasField("manifest_path"):
        declared = _read_manifest_codebook(space.manifest_path)
        if declared != codebook:
            raise ValueError(
                f"sid_space.codebook {codebook} does not match the manifest at "
                f"[{space.manifest_path}] which describes {declared}. The data "
                f"and the decode bands would disagree."
            )

    atoms = _atom_tokens(space)
    present = [a for a in atoms if tok.token_to_id(a) is not None]
    if present:
        raise ValueError(
            f"SID atoms are already in the base tokenizer, e.g. {present[:3]}; "
            f"change sid_space.atom_token_format."
        )
    tok.add_special_tokens(atoms)

    sentinel_id = None
    if has_projection:
        if tok.token_to_id(cfg.sentinel_token) is not None:
            raise ValueError(
                f"sentinel_token [{cfg.sentinel_token}] is already in the base "
                f"tokenizer; a projected position would be indistinguishable "
                f"from real content."
            )
        tok.add_special_tokens([cfg.sentinel_token])
        sentinel_id = tok.token_to_id(cfg.sentinel_token)

    offsets: List[int] = []
    running = 0
    for size in codebook:
        offsets.append(running)
        running += size
    lo = [base_vocab + o for o in offsets]
    hi = [lo[i] + codebook[i] - 1 for i in range(len(codebook))]

    return ResolvedSidSpace(
        codebook=tuple(codebook),
        num_levels=len(codebook),
        base_vocab=base_vocab,
        level_offsets=tuple(offsets),
        band_lo=tuple(lo),
        band_hi=tuple(hi),
        target_vocab=_ceil_to(
            tok.get_vocab_size(with_added_tokens=True),
            space.vocab_pad_to_multiple_of,
        ),
        sentinel_token_id=sentinel_id,
        eos_token_id=_special_id(tok, ("<|im_end|>", "<|endoftext|>")),
        pad_token_id=_special_id(tok, ("<|endoftext|>", "<|im_end|>")),
    )


def _special_id(tok: Tokenizer, candidates: Sequence[str]) -> int:
    """First candidate the tokenizer knows, so a family swap does not break."""
    for name in candidates:
        token_id = tok.token_to_id(name)
        if token_id is not None:
            return int(token_id)
    raise ValueError(
        f"none of {list(candidates)} is in the tokenizer; the prompt cannot "
        f"resolve its EOS/pad ids."
    )


def _hash(*parts: Any) -> str:
    """Stable sha256 over the given parts."""
    digest = hashlib.sha256()
    for part in parts:
        digest.update(repr(part).encode("utf-8"))
    return digest.hexdigest()


def compile_prompt(
    cfg: PromptConfig,
    features: Sequence[BaseFeature],
    model_dir: Optional[str] = None,
) -> CompiledPrompt:
    """Compile a prompt config into its plan, module and vocabulary artifacts.

    Args:
        cfg: the prompt config to compile.
        features: every feature the config may reference, already created.
        model_dir: where to write the extended tokenizer; skipped when None.

    Returns:
        The compiled prompt.
    """
    features_by_name = {feature.name: feature for feature in features}
    declared_slots_by_name = {slot.name: slot for slot in cfg.slots}

    body_runs, body_names = _split_template(cfg.prompt)
    resp_runs, resp_names = _split_template(cfg.response or "")
    resolved_slots_by_name = {
        name: _resolve_slot(name, declared_slots_by_name)
        for name in body_names + resp_names
    }

    unreferenced = set(declared_slots_by_name) - set(resolved_slots_by_name)
    if unreferenced:
        raise ValueError(
            f"declared prompt slots {sorted(unreferenced)} are never referenced "
            f"by a {{{{name}}}} placeholder."
        )

    members: Dict[str, List[BaseFeature]] = {}
    for name, slot in resolved_slots_by_name.items():
        missing_feature_names = [
            feature_name
            for feature_name in slot.feature_names
            if feature_name not in features_by_name
        ]
        if missing_feature_names:
            raise ValueError(
                f"prompt slot [{name}] names features {missing_feature_names} that "
                "are not in feature_configs."
            )
        members[name] = [
            features_by_name[feature_name] for feature_name in slot.feature_names
        ]

    response_slot_names = set(resp_names)
    group_types_by_slot_name: Dict[str, "FeatureGroupType.ValueType"] = {}
    fill_modes_by_slot_name: Dict[str, FillMode] = {}
    for name, slot_members in members.items():
        group_type, fill_mode = _derive_slot_layout(name, slot_members)
        if name in response_slot_names and fill_mode is FillMode.PROJECTED:
            raise ValueError(
                f"response slot [{name}] is PROJECTED; response slots must be "
                "INLINE because the LM generates them as vocabulary tokens."
            )
        group_types_by_slot_name[name] = group_type
        fill_modes_by_slot_name[name] = fill_mode
    for name, slot in resolved_slots_by_name.items():
        if fill_modes_by_slot_name[name] is FillMode.INLINE and slot.HasField(
            "projection"
        ):
            raise ValueError(
                f"prompt slot [{name}] is INLINE -- one sequence feature with no "
                f"embedding -- so it has no group to project; drop its projection."
            )

    tok = Tokenizer.from_file(cfg.tokenizer)
    base_vocab = tok.get_vocab_size(with_added_tokens=True)
    has_projection = any(
        fill_mode is FillMode.PROJECTED
        for fill_mode in fill_modes_by_slot_name.values()
    )
    sid_space = _build_sid_space(cfg, tok, base_vocab, has_projection)

    tokenizer_dir = ""
    if model_dir:
        tokenizer_dir = os.path.join(model_dir, "prompt", "tokenizer")
        os.makedirs(tokenizer_dir, exist_ok=True)
        tok.save(os.path.join(tokenizer_dir, "tokenizer.json"))

    slot_ids = {name: i for i, name in enumerate(resolved_slots_by_name)}
    segs: Dict[str, SlotSeg] = {}
    for name, slot in resolved_slots_by_name.items():
        group_type = group_types_by_slot_name[name]
        fill_mode = fill_modes_by_slot_name[name]
        levels = (
            sid_space.num_levels
            if sid_space is not None
            and name in response_slot_names
            and fill_mode is FillMode.INLINE
            else None
        )
        segs[name] = SlotSeg(
            slot_id=slot_ids[name],
            name=name,
            feature_names=tuple(slot.feature_names),
            group_type=group_type,
            output_key=(
                ".sequence" if group_type == FeatureGroupType.JAGGED_SEQUENCE else ""
            ),
            fill=fill_mode,
            width=_slot_width(members[name], group_type, levels),
        )

    body = _build_template_segments(body_runs, body_names, segs, tok)
    response = _build_template_segments(resp_runs, resp_names, segs, tok)

    projected = tuple(
        s
        for s in body + response
        if isinstance(s, SlotSeg) and s.fill is FillMode.PROJECTED
    )
    projection_plan = _build_projection_plan(projected, resolved_slots_by_name)

    plan = PromptPlan(
        segments=body,
        response_segments=response,
        max_length=int(cfg.max_length),
        max_total_length=_max_total_length(body + response),
        max_holes=_max_holes(projected),
        logits_suffix_len=_suffix_keep(response),
        static_prefix_len=_static_prefix_len(body),
        projected_slots=projected,
    )
    _validate(cfg, plan, sid_space)

    return CompiledPrompt(
        sid_space=sid_space,
        prompt_plan=plan,
        projection_plan=projection_plan,
        tokenizer_dir=tokenizer_dir,
        vocab_hash=_hash(sid_space, tok.to_str()),
        plan_hash=_hash(
            sid_space, plan, sorted(projection_plan.projections), tok.to_str()
        ),
    )


def _build_template_segments(
    static_text_parts: Sequence[str],
    slot_names: Sequence[str],
    slot_segments_by_name: Dict[str, SlotSeg],
    tokenizer: Tokenizer,
) -> Tuple[Segment, ...]:
    """Interleave tokenized static text parts with their slots."""
    segments: List[Segment] = []
    for index, static_text in enumerate(static_text_parts):
        if static_text:
            token_ids = tuple(
                tokenizer.encode(static_text, add_special_tokens=False).ids
            )
            segments.append(Static(token_ids=token_ids))
        if index < len(slot_names):
            segments.append(slot_segments_by_name[slot_names[index]])
    return tuple(segments)


def _build_projection_plan(
    projected: Sequence[SlotSeg], slots_by_name: Dict[str, PromptSlot]
) -> ProjectionPlan:
    """One module per distinct ``projection_name``, else one per slot."""
    projections: Dict[str, PromptProjection] = {}
    slot_to_module: Dict[int, str] = {}
    for seg in projected:
        slot = slots_by_name[seg.name]
        module_id = slot.projection_name or seg.name
        projection = (
            slot.projection if slot.HasField("projection") else PromptProjection()
        )
        if module_id in projections:
            if projections[module_id] != projection:
                raise ValueError(
                    f"prompt slots sharing projection_name [{module_id}] declare "
                    f"different projection bodies; they cannot share weights."
                )
        else:
            projections[module_id] = projection
        slot_to_module[seg.slot_id] = module_id

    groups = tuple(
        FeatureGroupConfig(
            group_name=seg.name,
            feature_names=list(seg.feature_names),
            group_type=seg.group_type,
        )
        for seg in projected
    )
    return ProjectionPlan(
        projections=projections,
        slot_to_module=slot_to_module,
        feature_groups=groups,
    )


def _max_total_length(segments: Sequence[Segment]) -> Optional[int]:
    """Provable position ceiling, or None when any slot is unbounded."""
    total = 0
    for seg in segments:
        if isinstance(seg, Static):
            total += len(seg.token_ids)
        elif seg.width.kind is WidthKind.UNBOUNDED:
            return None
        else:
            assert seg.width.num_positions is not None
            total += seg.width.num_positions
    return total


def _max_holes(projected: Sequence[SlotSeg]) -> int:
    """Per-row projected-position ceiling; unbounded slots cannot be counted."""
    total = 0
    for seg in projected:
        if seg.width.kind is WidthKind.UNBOUNDED:
            raise ValueError(
                f"prompt slot [{seg.name}] is PROJECTED and unbounded, so its "
                f"hole count is unknowable; give its members a sequence_length."
            )
        assert seg.width.num_positions is not None
        total += seg.width.num_positions
    return total


def _suffix_keep(response: Sequence[Segment]) -> Optional[int]:
    """Upper bound on the supervised window, or None when unbounded."""
    total = _max_total_length(response)
    # HF shifts logits by one, so the window opens one column before the first
    # supervised label.
    return None if total is None else total + 1


def _static_prefix_len(segments: Sequence[Segment]) -> int:
    """Leading positions that are request-invariant: static runs only."""
    total = 0
    for seg in segments:
        if not isinstance(seg, Static):
            break
        total += len(seg.token_ids)
    return total


def _validate(
    cfg: PromptConfig, plan: PromptPlan, sid_space: Optional[ResolvedSidSpace]
) -> None:
    """Apply the checks that need the whole plan."""
    if plan.max_length and plan.max_total_length is not None:
        if plan.max_total_length > plan.max_length:
            raise ValueError(
                f"the prompt can reach {plan.max_total_length} positions but "
                f"max_length is {plan.max_length}; a row could never be assembled."
            )
    if not plan.max_length and plan.max_total_length is None:
        logger.warning(
            "prompt has an unbounded slot and max_length is 0; graph-captured "
            "serving cannot size its buckets."
        )
    variable_slot_seen = False
    for seg in plan.segments:
        if not isinstance(seg, SlotSeg):
            continue
        if seg.width.kind is WidthKind.STATIC:
            if not variable_slot_seen:
                continue
            logger.warning(
                "a variable-width prompt slot precedes a fixed-width one; "
                f"static_prefix_len is {plan.static_prefix_len}, which bounds "
                "what a serving prefix cache may reuse."
            )
            break
        variable_slot_seen = True
    if plan.response_segments and plan.logits_suffix_len is None:
        raise ValueError(
            "the response has an unbounded slot, so the supervised logits "
            "window cannot be bounded. A decoder-only model would then "
            "materialize logits for every position, which is (batch x length x "
            "vocab) and will not fit. Give the response slot a fixed width."
        )
    if plan.static_prefix_len == 0:
        logger.warning(
            "static_prefix_len is 0: no leading run of the prompt is "
            "request-invariant, so a serving prefix cache can share nothing."
        )
    if sid_space is not None and cfg.HasField("response"):
        answer = [s for s in plan.response_segments if isinstance(s, SlotSeg)]
        for seg in answer:
            if (
                seg.width.kind is WidthKind.STATIC
                and seg.width.num_positions != sid_space.num_levels
            ):
                raise ValueError(
                    f"response slot [{seg.name}] is "
                    f"{seg.width.num_positions} positions but the codebook has "
                    f"{sid_space.num_levels} levels."
                )
