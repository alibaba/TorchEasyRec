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
    ModulePlan,
    PromptPlan,
    Segment,
    SidSpace,
    SlotSeg,
    Static,
    Width,
    WidthKind,
)
from tzrec.protos.model_pb2 import FeatureGroupConfig, FeatureGroupType
from tzrec.protos.prompt_pb2 import PromptConfig, PromptProjection, PromptSlot
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


def _resolve_slot(name: str, declared: Dict[str, PromptSlot]) -> PromptSlot:
    """Return the declared slot, or an implicit single-feature slot."""
    if name in declared:
        return declared[name]
    implicit = PromptSlot(name=name)
    implicit.feature_names.append(name)
    return implicit


def _slot_width(
    members: Sequence[BaseFeature], group_type: "FeatureGroupType.ValueType"
) -> Width:
    """Derive a slot's position count from its members.

    A DEEP slot pools to exactly one position. A sequence slot is bounded by
    the members' ``sequence_length``, and unbounded when none declares one.
    """
    if group_type == FeatureGroupType.DEEP:
        return Width(WidthKind.STATIC, 1)
    caps = [
        int(f.config.sequence_length)
        for f in members
        if f.config.HasField("sequence_length")
    ]
    if not caps:
        return Width(WidthKind.UNBOUNDED)
    return Width(WidthKind.BOUNDED, max(caps))


def _derive_fill(members: Sequence[BaseFeature]) -> FillMode:
    """INLINE only for a lone sequence member that declares no embedding."""
    if len(members) == 1 and members[0].is_sequence and not members[0].has_embedding:
        return FillMode.INLINE
    return FillMode.PROJECTED


def _group_type(
    name: str, members: Sequence[BaseFeature]
) -> "FeatureGroupType.ValueType":
    """JAGGED_SEQUENCE for sequence members, DEEP for scalars; never mixed."""
    kinds = {f.is_sequence for f in members}
    if len(kinds) != 1:
        raise ValueError(
            f"prompt slot [{name}] mixes sequence and scalar features "
            f"{[f.name for f in members]}; a slot must be all one kind, or its "
            f"group would carry a '.query' output the prompt cannot place."
        )
    return FeatureGroupType.JAGGED_SEQUENCE if kinds.pop() else FeatureGroupType.DEEP


def _atom_tokens(sid_space: Any) -> List[str]:
    """Render the SID atom tokens, one per flat index."""
    fmt = sid_space.atom_token_format
    return [fmt.replace("{i}", str(i)) for i in range(sum(sid_space.codebook))]


def _read_manifest_codebook(path: str) -> Optional[List[int]]:
    """Read ``codebook`` from a SID manifest, or None when there is no file."""
    if not os.path.exists(path):
        raise ValueError(f"sid_space.manifest_path [{path}] does not exist.")
    with open(path, "r") as f:
        return [int(c) for c in json.load(f)["codebook"]]


def _build_sid_space(
    cfg: PromptConfig, tok: Tokenizer, base_vocab: int, has_projection: bool
) -> Optional[SidSpace]:
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

    return SidSpace(
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
    by_name = {f.name: f for f in features}
    declared = {s.name: s for s in cfg.slots}

    body_runs, body_names = _split_template(cfg.prompt)
    resp_runs, resp_names = _split_template(cfg.response or "")
    slots = {n: _resolve_slot(n, declared) for n in body_names + resp_names}

    unreferenced = set(declared) - set(slots)
    if unreferenced:
        raise ValueError(
            f"declared prompt slots {sorted(unreferenced)} are never referenced "
            f"by a {{{{name}}}} placeholder."
        )

    members: Dict[str, List[BaseFeature]] = {}
    for name, slot in slots.items():
        missing = [f for f in slot.feature_names if f not in by_name]
        if missing:
            raise ValueError(
                f"prompt slot [{name}] names features {missing} that are not in "
                f"feature_configs."
            )
        members[name] = [by_name[f] for f in slot.feature_names]

    types = {n: _group_type(n, members[n]) for n in slots}
    fills = {n: _derive_fill(members[n]) for n in slots}
    has_projection = any(f is FillMode.PROJECTED for f in fills.values())

    for name, slot in slots.items():
        if fills[name] is FillMode.INLINE and slot.HasField("projection"):
            raise ValueError(
                f"prompt slot [{name}] is INLINE -- one sequence feature with no "
                f"embedding -- so it has no group to project; drop its projection."
            )

    tok = Tokenizer.from_file(cfg.tokenizer)
    base_vocab = tok.get_vocab_size(with_added_tokens=True)
    sid_space = _build_sid_space(cfg, tok, base_vocab, has_projection)

    tokenizer_dir = ""
    if model_dir:
        tokenizer_dir = os.path.join(model_dir, "prompt", "tokenizer")
        os.makedirs(tokenizer_dir, exist_ok=True)
        tok.save(os.path.join(tokenizer_dir, "tokenizer.json"))

    slot_ids = {n: i for i, n in enumerate(slots)}
    segs: Dict[str, SlotSeg] = {}
    for name, slot in slots.items():
        seq = types[name] == FeatureGroupType.JAGGED_SEQUENCE
        segs[name] = SlotSeg(
            slot_id=slot_ids[name],
            name=name,
            sources=tuple(slot.feature_names),
            group_type=types[name],
            output_key=".sequence" if seq else "",
            fill=fills[name],
            width=_slot_width(members[name], types[name]),
            droppable=bool(slot.drop_if_empty),
        )

    body = _weave(body_runs, body_names, segs, tok)
    response = _weave(resp_runs, resp_names, segs, tok)

    projected = tuple(
        s
        for s in body + response
        if isinstance(s, SlotSeg) and s.fill is FillMode.PROJECTED
    )
    module_plan = _build_module_plan(projected, slots)

    plan = PromptPlan(
        segments=body,
        response_segments=response,
        max_length=int(cfg.max_length),
        max_total_length=_max_total_length(body + response),
        max_holes=_max_holes(projected),
        suffix_keep=_suffix_keep(response),
        static_prefix_len=_static_prefix_len(body),
        length_buckets=tuple(int(b) for b in cfg.length_buckets),
        slot_index={s.name: i for i, s in enumerate(projected)},
        projected_slots=projected,
    )
    _validate(cfg, plan, sid_space)

    return CompiledPrompt(
        sid_space=sid_space,
        prompt_plan=plan,
        module_plan=module_plan,
        tokenizer_dir=tokenizer_dir,
        vocab_hash=_hash(sid_space, tok.to_str()),
        plan_hash=_hash(sid_space, plan, sorted(module_plan.projections), tok.to_str()),
    )


def _weave(
    runs: Sequence[str],
    names: Sequence[str],
    segs: Dict[str, SlotSeg],
    tok: Tokenizer,
) -> Tuple[Segment, ...]:
    """Interleave tokenized static runs with their slots, dropping empty runs."""
    out: List[Segment] = []
    for i, run in enumerate(runs):
        if run:
            ids = tuple(tok.encode(run, add_special_tokens=False).ids)
            out.append(Static(token_ids=ids, owner_slot_id=None))
        if i < len(names):
            out.append(segs[names[i]])
    return tuple(out)


def _build_module_plan(
    projected: Sequence[SlotSeg], slots: Dict[str, PromptSlot]
) -> ModulePlan:
    """One module per distinct ``projection_name``, else one per slot."""
    projections: Dict[str, PromptProjection] = {}
    slot_to_module: Dict[int, str] = {}
    for seg in projected:
        slot = slots[seg.name]
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
            feature_names=list(seg.sources),
            group_type=seg.group_type,
        )
        for seg in projected
    )
    return ModulePlan(
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
            assert seg.width.n is not None
            total += seg.width.n
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
        assert seg.width.n is not None
        total += seg.width.n
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
    cfg: PromptConfig, plan: PromptPlan, sid_space: Optional[SidSpace]
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
    if any(isinstance(s, SlotSeg) for s in plan.segments):
        first_slot = next(
            i for i, s in enumerate(plan.segments) if isinstance(s, SlotSeg)
        )
        later_static = any(
            isinstance(s, SlotSeg) and s.width.kind is WidthKind.STATIC
            for s in plan.segments[first_slot + 1 :]
        )
        if later_static:
            logger.warning(
                "a variable-width prompt slot precedes a fixed-width one; "
                f"static_prefix_len is {plan.static_prefix_len}, which bounds "
                "what a serving prefix cache may reuse."
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
                and seg.width.n != sid_space.num_levels
            ):
                raise ValueError(
                    f"response slot [{seg.name}] is {seg.width.n} positions but "
                    f"the codebook has {sid_space.num_levels} levels."
                )
