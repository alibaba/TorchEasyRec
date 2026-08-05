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

"""Builds the packed token stream a compiled prompt describes.

Runs in the dataloader worker, after the features are parsed and outside any
feature's ``_parse``. Pure integer arithmetic with no FG dependency, so the
same walk is portable to an online C++/Java processor.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from tzrec.prompt.plan import (
    CompiledPrompt,
    FillMode,
    PromptPlan,
    SidSpace,
    SlotSeg,
    Static,
)

PROMPT_INPUT_IDS = "prompt_input_ids"
PROMPT_CU_SEQLENS = "prompt_cu_seqlens"
PROMPT_HOLE_POSITIONS = "prompt_hole_positions"
PROMPT_LABELS = "prompt_labels"
PROMPT_MAX_SEQLEN = "prompt_max_seqlen"


@dataclass
class AssembledPrompt:
    """One batch of assembled prompts, in packed varlen form.

    Args:
        input_ids: every row's tokens concatenated, ``(total_tokens,)``.
        cu_seqlens: row boundaries into ``input_ids``, ``(batch_size + 1,)``.
        hole_positions: absolute indices the projected embeddings overwrite,
            in ``PromptPlan.projected_slots`` order within each row.
        labels: ``ignore_index`` outside the response span.
    """

    input_ids: np.ndarray
    cu_seqlens: np.ndarray
    hole_positions: np.ndarray
    labels: np.ndarray

    @property
    def max_seqlen(self) -> int:
        """Widest row, computed on the host so the model never derives it."""
        if self.cu_seqlens.size < 2:
            return 0
        return int(np.max(np.diff(self.cu_seqlens)))


class PromptAssembler:
    """Walks a ``PromptPlan`` to build token streams.

    Args:
        plan: the compiled walk order.
        sid_space: resolved SID token space; required when a slot renders SIDs.
        ignore_index: label value outside the supervised span.
    """

    def __init__(
        self,
        plan: PromptPlan,
        sid_space: Optional[SidSpace] = None,
        ignore_index: int = -100,
    ) -> None:
        self._plan = plan
        self._sid = sid_space
        self._ignore_index = ignore_index
        inline = [
            s
            for s in plan.segments + plan.response_segments
            if isinstance(s, SlotSeg) and s.fill is FillMode.INLINE
        ]
        if inline and sid_space is None:
            raise ValueError(
                f"prompt slots {[s.name for s in inline]} render INLINE, which "
                f"means SID codes, but no sid_space was compiled."
            )

    def _inline_tokens(self, name: str, values: np.ndarray) -> np.ndarray:
        """Validate offset SID codes against their bands and shift to token ids.

        The data carries ``level_offsets[l] + code``; the LM vocabulary needs
        one further uniform shift by ``base_vocab``.
        """
        assert self._sid is not None
        levels = self._sid.num_levels
        if values.size % levels:
            raise ValueError(
                f"prompt slot [{name}]: {values.size} values is not a whole "
                f"number of {levels}-level items."
            )
        by_level = values.reshape(-1, levels)
        lo = np.asarray(self._sid.level_offsets, dtype=np.int64)
        hi = lo + np.asarray(self._sid.codebook, dtype=np.int64)
        if np.any(by_level < lo) or np.any(by_level >= hi):
            raise ValueError(
                f"prompt slot [{name}]: SID values must already carry their "
                f"level offset, so level l lies in "
                f"[level_offsets[l], level_offsets[l] + codebook[l]). Read the "
                f"offset_codebook column, not codebook or origin_codebook."
            )
        return values.astype(np.int64, copy=False) + self._sid.base_vocab

    def _emit_row(
        self,
        segments: Sequence[object],
        row: int,
        values: Dict[str, List[np.ndarray]],
        counts: Dict[str, np.ndarray],
        out: List[int],
        holes: List[int],
        base: int,
    ) -> None:
        """Append one row's tokens for one segment list, recording holes."""
        for seg in segments:
            if isinstance(seg, Static):
                out.extend(seg.token_ids)
                continue
            assert isinstance(seg, SlotSeg)
            if seg.fill is FillMode.INLINE:
                out.extend(self._inline_tokens(seg.name, values[seg.name][row]))
            else:
                assert self._sid is not None
                width = int(counts[seg.name][row])
                holes.extend(range(base + len(out), base + len(out) + width))
                out.extend([self._sid.sentinel_token_id] * width)

    def assemble(
        self,
        values: Dict[str, List[np.ndarray]],
        counts: Optional[Dict[str, np.ndarray]] = None,
        batch_size: Optional[int] = None,
    ) -> AssembledPrompt:
        """Assemble one batch.

        Args:
            values: INLINE slot name to its per-row value arrays.
            counts: PROJECTED slot name to its per-row position count.
            batch_size: row count; inferred from ``values`` when omitted.

        Returns:
            The packed streams.
        """
        counts = counts or {}
        if batch_size is None:
            if not values:
                raise ValueError("batch_size is required when no INLINE slot exists.")
            batch_size = len(next(iter(values.values())))

        ids: List[int] = []
        labels: List[int] = []
        holes: List[int] = []
        cu = [0]
        for row in range(batch_size):
            row_ids: List[int] = []
            self._emit_row(
                self._plan.segments, row, values, counts, row_ids, holes, len(ids)
            )
            prompt_len = len(row_ids)
            self._emit_row(
                self._plan.response_segments,
                row,
                values,
                counts,
                row_ids,
                holes,
                len(ids),
            )
            # supervision covers the response span only; the prompt is context.
            row_labels = [self._ignore_index] * prompt_len + row_ids[prompt_len:]
            if self._plan.max_length and len(row_ids) > self._plan.max_length:
                raise ValueError(
                    f"assembled row {row} is {len(row_ids)} tokens, over "
                    f"max_length {self._plan.max_length}. Rows are never "
                    f"truncated: cap the source features instead."
                )
            ids.extend(row_ids)
            labels.extend(row_labels)
            cu.append(len(ids))

        return AssembledPrompt(
            input_ids=np.asarray(ids, dtype=np.int64),
            cu_seqlens=np.asarray(cu, dtype=np.int64),
            hole_positions=np.asarray(holes, dtype=np.int64),
            labels=np.asarray(labels, dtype=np.int64),
        )


def assemble_into(
    prompt: CompiledPrompt,
    parsed: Dict[str, "np.ndarray"],
    ignore_index: int = -100,
) -> Dict[str, np.ndarray]:
    """Run the assembler over one parsed batch and key it for the batch.

    Args:
        prompt: the compiled prompt.
        parsed: ``{feature}.values`` / ``{feature}.lengths`` as the data parser
            emits them.
        ignore_index: label value outside the supervised span.

    Returns:
        The five streams, keyed as ``additional_infos`` expects them.
    """
    plan = prompt.prompt_plan
    values: Dict[str, List[np.ndarray]] = {}
    counts: Dict[str, np.ndarray] = {}
    batch_size = 0
    for seg in plan.segments + plan.response_segments:
        if not isinstance(seg, SlotSeg):
            continue
        source = seg.sources[0]
        lengths = np.asarray(parsed[f"{source}.lengths"])
        batch_size = max(batch_size, int(lengths.size))
        if seg.fill is FillMode.INLINE:
            flat = np.asarray(parsed[f"{source}.values"])
            bounds = np.concatenate(([0], np.cumsum(lengths)))
            values[seg.name] = [
                flat[bounds[i] : bounds[i + 1]] for i in range(lengths.size)
            ]
        else:
            counts[seg.name] = lengths

    out = PromptAssembler(plan, prompt.sid_space, ignore_index).assemble(
        values, counts, batch_size=batch_size
    )
    return {
        PROMPT_INPUT_IDS: out.input_ids,
        PROMPT_CU_SEQLENS: out.cu_seqlens,
        PROMPT_HOLE_POSITIONS: out.hole_positions,
        PROMPT_LABELS: out.labels,
        PROMPT_MAX_SEQLEN: np.asarray(out.max_seqlen, dtype=np.int64),
    }
