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
from typing import Dict, List, Optional

import numpy as np

from tzrec.prompt.types import (
    FillMode,
    PromptPlan,
    ResolvedSidSpace,
    SlotSeg,
    Static,
)
from tzrec.protos.model_pb2 import FeatureGroupType

PROMPT_INPUT_IDS = "prompt_input_ids"
PROMPT_CU_SEQLENS = "prompt_cu_seqlens"
PROMPT_HOLE_POSITIONS = "prompt_hole_positions"
PROMPT_MAX_SEQLEN = "prompt_max_seqlen"
PROMPT_RESPONSE_LENGTHS = "prompt_response_lengths"


@dataclass
class AssembledPrompt:
    """One batch of assembled prompts, in packed varlen form.

    Args:
        input_ids: every sample's tokens concatenated, ``(total_tokens,)``.
        cu_seqlens: sample boundaries into ``input_ids``, ``(batch_size + 1,)``.
        hole_positions: absolute indices the projected embeddings overwrite,
            grouped by included projected occurrence in
            ``PromptPlan.projected_slots`` order, then by sample.
        response_lengths: number of response tokens in each sample.
    """

    input_ids: np.ndarray
    cu_seqlens: np.ndarray
    hole_positions: np.ndarray
    response_lengths: np.ndarray

    @property
    def max_seqlen(self) -> int:
        """Widest sample, computed on the host so the model never derives it."""
        if self.cu_seqlens.size < 2:
            return 0
        return int(np.max(np.diff(self.cu_seqlens)))


def _run_positions(starts: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Absolute positions of concatenated variable-length runs.

    Run i occupies ``lengths[i]`` positions from ``starts[i]``; the result is
    every position, runs back to back.
    """
    total = int(lengths.sum())
    within = np.arange(total, dtype=np.int64) - np.repeat(
        np.cumsum(lengths) - lengths, lengths
    )
    return np.repeat(starts, lengths) + within


class PromptAssembler:
    """Walks a ``PromptPlan`` to build token streams.

    Args:
        prompt_plan: the compiled walk order.
        sid_space: resolved SID token space; required when a slot renders SIDs.
        include_response: whether to read and emit the supervised response.
    """

    def __init__(
        self,
        prompt_plan: PromptPlan,
        sid_space: Optional[ResolvedSidSpace] = None,
        include_response: bool = True,
    ) -> None:
        self._prompt_plan = prompt_plan
        self._sid_space = sid_space
        self._response_segments = (
            prompt_plan.response_segments if include_response else ()
        )
        if sid_space is not None:
            self._flat_lo = np.asarray(sid_space.level_offsets, dtype=np.int64)
            self._flat_hi = self._flat_lo + np.asarray(
                sid_space.codebook, dtype=np.int64
            )
        inline = [
            s
            for s in prompt_plan.segments + self._response_segments
            if isinstance(s, SlotSeg) and s.fill is FillMode.INLINE
        ]
        if inline and sid_space is None:
            raise ValueError(
                f"prompt slots {[s.name for s in inline]} render INLINE, which "
                f"means SID codes, but no sid_space was compiled."
            )

    def _inline_tokens(
        self,
        name: str,
        flat: np.ndarray,
        lengths: np.ndarray,
        exact_width: Optional[int] = None,
    ) -> np.ndarray:
        """Validate offset SID codes against their bands and shift to token ids.

        The data carries ``level_offsets[l] + code``; the LM vocabulary needs
        one further uniform shift by ``base_vocab_size``.

        Args:
            name: the slot, for the message.
            flat: the slot's offset SID codes for the whole batch.
            lengths: per-sample position counts into ``flat``.
            exact_width: required position count, for the response; any whole
                number of items otherwise.
        """
        assert self._sid_space is not None
        levels = self._sid_space.num_levels
        partial = np.nonzero(lengths % levels)[0]
        if partial.size:
            sample = int(partial[0])
            raise ValueError(
                f"prompt slot [{name}]: sample {sample} has "
                f"{int(lengths[sample])} values, not a whole number of "
                f"{levels}-level items."
            )
        if exact_width is not None:
            wrong = np.nonzero(lengths != exact_width)[0]
            if wrong.size:
                sample = int(wrong[0])
                raise ValueError(
                    f"prompt slot [{name}]: sample {sample} has "
                    f"{int(lengths[sample])} values, but the compiled width is "
                    f"{exact_width}. The loss window is sized from that width, "
                    f"so a wider row would be supervised only in part."
                )
        by_level = flat.reshape(-1, levels)
        if np.any(by_level < self._flat_lo) or np.any(by_level >= self._flat_hi):
            raise ValueError(
                f"prompt slot [{name}]: SID values must already carry their "
                f"level offset, so level l lies in "
                f"[level_offsets[l], level_offsets[l] + codebook[l]). Read the "
                f"offset_codebook column, not codebook or origin_codebook."
            )
        return flat.astype(np.int64, copy=False) + self._sid_space.base_vocab_size

    def _pack(
        self,
        inline_flat: Dict[str, np.ndarray],
        inline_lengths: Dict[str, np.ndarray],
        projected_lengths: Dict[str, np.ndarray],
        batch_size: int,
    ) -> AssembledPrompt:
        """Pack every sample in one pass over the compiled segments.

        The loop is over segments, which the plan fixes, never over the batch:
        each segment writes its whole column of the flat buffer at once.

        Args:
            inline_flat: INLINE slot name to its batch-wide value stream.
            inline_lengths: INLINE slot name to its per-sample position counts.
            projected_lengths: PROJECTED slot name to its per-sample counts.
            batch_size: sample count.

        Returns:
            The packed streams.
        """
        segments = self._prompt_plan.segments + self._response_segments
        body_count = len(self._prompt_plan.segments)

        seg_lengths = np.empty((len(segments), batch_size), dtype=np.int64)
        for index, seg in enumerate(segments):
            if isinstance(seg, Static):
                seg_lengths[index] = len(seg.token_ids)
            elif seg.fill is FillMode.INLINE:
                seg_lengths[index] = inline_lengths[seg.name]
            else:
                seg_lengths[index] = projected_lengths[seg.name]

        row_lengths = seg_lengths.sum(axis=0)
        cu_seqlens = np.concatenate(([0], np.cumsum(row_lengths))).astype(np.int64)
        max_length = self._prompt_plan.max_length
        if max_length:
            over = np.nonzero(row_lengths > max_length)[0]
            if over.size:
                sample = int(over[0])
                raise ValueError(
                    f"assembled sample {sample} is {int(row_lengths[sample])} "
                    f"tokens, over max_length {max_length}. Samples are never "
                    f"truncated: cap the source features instead."
                )

        # row start plus the exclusive prefix sum down the segment axis
        seg_starts = cu_seqlens[:-1] + (np.cumsum(seg_lengths, axis=0) - seg_lengths)

        input_ids = np.empty(int(cu_seqlens[-1]), dtype=np.int64)
        holes: List[np.ndarray] = []
        for index, seg in enumerate(segments):
            lengths = seg_lengths[index]
            destinations = _run_positions(seg_starts[index], lengths)
            if isinstance(seg, Static):
                input_ids[destinations] = np.tile(
                    np.asarray(seg.token_ids, dtype=np.int64), batch_size
                )
            elif seg.fill is FillMode.INLINE:
                input_ids[destinations] = self._inline_tokens(
                    seg.name,
                    inline_flat[seg.name],
                    lengths,
                    exact_width=(
                        seg.width.num_positions if index >= body_count else None
                    ),
                )
            else:
                assert self._sid_space is not None
                input_ids[destinations] = self._sid_space.sentinel_token_id
                holes.append(destinations)

        response_lengths = (
            seg_lengths[body_count:].sum(axis=0)
            if body_count < len(segments)
            else np.zeros(batch_size, dtype=np.int64)
        )
        return AssembledPrompt(
            input_ids=input_ids,
            cu_seqlens=cu_seqlens,
            # occurrence-major then sample-major: the model concatenates the
            # projected embeddings in projected_slots order
            hole_positions=(
                np.concatenate(holes) if holes else np.empty(0, dtype=np.int64)
            ),
            response_lengths=response_lengths.astype(np.int64),
        )

    def forward(
        self, parsed_features: Dict[str, "np.ndarray"]
    ) -> Dict[str, np.ndarray]:
        """Reshape one parsed batch, assemble it, and key it for the batch.

        Args:
            parsed_features: ``{feature}.values`` / ``{feature}.lengths`` as
                the data parser emits them.

        Returns:
            The five streams, keyed as ``additional_infos`` expects them.
        """
        inline_flat: Dict[str, np.ndarray] = {}
        inline_lengths: Dict[str, np.ndarray] = {}
        projected_lengths: Dict[str, np.ndarray] = {}
        batch_size: Optional[int] = None
        for seg in self._prompt_plan.segments + self._response_segments:
            if not isinstance(seg, SlotSeg):
                continue
            sources = (
                seg.feature_names
                if seg.fill is FillMode.PROJECTED
                else seg.feature_names[:1]
            )
            member_lengths: List[tuple[str, np.ndarray]] = []
            for source in sources:
                lengths_key = f"{source}.lengths"
                if seg.group_type == FeatureGroupType.JAGGED_SEQUENCE:
                    lengths = np.asarray(parsed_features[lengths_key])
                    slot_batch_size = int(lengths.size)
                    member_lengths.append((source, lengths))
                elif lengths_key in parsed_features:
                    slot_batch_size = int(np.asarray(parsed_features[lengths_key]).size)
                else:
                    values = np.asarray(parsed_features[f"{source}.values"])
                    slot_batch_size = int(values.shape[0])
                if batch_size is None:
                    batch_size = slot_batch_size
                elif slot_batch_size != batch_size:
                    raise ValueError(
                        f"prompt slot [{seg.name}] has {slot_batch_size} samples, "
                        f"expected {batch_size}."
                    )
            if seg.fill is FillMode.INLINE:
                source, lengths = member_lengths[0]
                # a dense sequence feature emits (total, value_dim); the stream
                # is one code per position, so value_dim is always 1 here
                inline_flat[seg.name] = np.asarray(
                    parsed_features[f"{source}.values"]
                ).reshape(-1)
                inline_lengths[seg.name] = np.asarray(lengths, dtype=np.int64)
            elif seg.group_type == FeatureGroupType.JAGGED_SEQUENCE:
                source, lengths = member_lengths[0]
                for other_source, other_lengths in member_lengths[1:]:
                    if not np.array_equal(lengths, other_lengths):
                        raise ValueError(
                            f"prompt slot [{seg.name}] PROJECTED features "
                            f"[{source}] and [{other_source}] have different "
                            "per-sample lengths."
                        )
                projected_lengths[seg.name] = lengths
            else:
                assert batch_size is not None
                projected_lengths[seg.name] = np.ones(batch_size, dtype=np.int64)

        out = self._pack(
            inline_flat,
            inline_lengths,
            projected_lengths,
            batch_size if batch_size is not None else 0,
        )
        return {
            PROMPT_INPUT_IDS: out.input_ids,
            PROMPT_CU_SEQLENS: out.cu_seqlens,
            PROMPT_HOLE_POSITIONS: out.hole_positions,
            PROMPT_MAX_SEQLEN: np.asarray(out.max_seqlen, dtype=np.int64),
            PROMPT_RESPONSE_LENGTHS: out.response_lengths,
        }
