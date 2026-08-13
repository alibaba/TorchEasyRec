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
PROMPT_LABELS = "prompt_labels"
PROMPT_MAX_SEQLEN = "prompt_max_seqlen"


@dataclass
class AssembledPrompt:
    """One batch of assembled prompts, in packed varlen form.

    Args:
        input_ids: every sample's tokens concatenated, ``(total_tokens,)``.
        cu_seqlens: sample boundaries into ``input_ids``, ``(batch_size + 1,)``.
        hole_positions: absolute indices the projected embeddings overwrite,
            grouped by included projected occurrence in
            ``PromptPlan.projected_slots`` order, then by sample.
        labels: ``ignore_index`` outside the response span.
    """

    input_ids: np.ndarray
    cu_seqlens: np.ndarray
    hole_positions: np.ndarray
    labels: np.ndarray

    @property
    def max_seqlen(self) -> int:
        """Widest sample, computed on the host so the model never derives it."""
        if self.cu_seqlens.size < 2:
            return 0
        return int(np.max(np.diff(self.cu_seqlens)))


class PromptAssembler:
    """Walks a ``PromptPlan`` to build token streams.

    Args:
        prompt_plan: the compiled walk order.
        sid_space: resolved SID token space; required when a slot renders SIDs.
        ignore_index: label value outside the supervised span.
        include_response: whether to read and emit the supervised response.
    """

    def __init__(
        self,
        prompt_plan: PromptPlan,
        sid_space: Optional[ResolvedSidSpace] = None,
        ignore_index: int = -100,
        include_response: bool = True,
    ) -> None:
        self._prompt_plan = prompt_plan
        self._sid_space = sid_space
        self._ignore_index = ignore_index
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

    def _inline_tokens(self, name: str, values: np.ndarray) -> np.ndarray:
        """Validate offset SID codes against their bands and shift to token ids.

        The data carries ``level_offsets[l] + code``; the LM vocabulary needs
        one further uniform shift by ``base_vocab``.
        """
        assert self._sid_space is not None
        levels = self._sid_space.num_levels
        if values.size % levels:
            raise ValueError(
                f"prompt slot [{name}]: {values.size} values is not a whole "
                f"number of {levels}-level items."
            )
        by_level = values.reshape(-1, levels)
        if np.any(by_level < self._flat_lo) or np.any(by_level >= self._flat_hi):
            raise ValueError(
                f"prompt slot [{name}]: SID values must already carry their "
                f"level offset, so level l lies in "
                f"[level_offsets[l], level_offsets[l] + codebook[l]). Read the "
                f"offset_codebook column, not codebook or origin_codebook."
            )
        return values.astype(np.int64, copy=False) + self._sid_space.base_vocab

    def _emit_sample(
        self,
        segments: Sequence[object],
        sample_index: int,
        inline_values: Dict[str, List[np.ndarray]],
        projected_lengths: Dict[str, np.ndarray],
        out: List[int],
        holes_by_occurrence: List[List[int]],
        projected_occurrence_index: int,
        sample_start: int,
    ) -> int:
        """Append one sample's tokens for one segment list, recording holes."""
        for seg in segments:
            if isinstance(seg, Static):
                out.extend(seg.token_ids)
                continue
            assert isinstance(seg, SlotSeg)
            if seg.fill is FillMode.INLINE:
                out.extend(
                    self._inline_tokens(seg.name, inline_values[seg.name][sample_index])
                )
            else:
                assert self._sid_space is not None
                width = int(projected_lengths[seg.name][sample_index])
                holes_by_occurrence[projected_occurrence_index].extend(
                    range(
                        sample_start + len(out),
                        sample_start + len(out) + width,
                    )
                )
                projected_occurrence_index += 1
                out.extend([self._sid_space.sentinel_token_id] * width)
        return projected_occurrence_index

    def assemble(
        self,
        inline_values: Dict[str, List[np.ndarray]],
        projected_lengths: Optional[Dict[str, np.ndarray]] = None,
        batch_size: Optional[int] = None,
    ) -> AssembledPrompt:
        """Assemble one batch.

        Args:
            inline_values: INLINE slot name to its per-sample value arrays.
            projected_lengths: PROJECTED slot name to its per-sample position count.
            batch_size: sample count; inferred from ``inline_values`` when omitted.

        Returns:
            The packed streams.
        """
        projected_lengths = projected_lengths or {}
        if batch_size is None:
            if not inline_values:
                raise ValueError("batch_size is required when no INLINE slot exists.")
            first_inline_values = next(iter(inline_values.values()))
            batch_size = len(first_inline_values)

        jagged_token_ids: List[int] = []
        labels: List[int] = []
        holes_by_occurrence = [
            []
            for seg in self._prompt_plan.segments + self._response_segments
            if isinstance(seg, SlotSeg) and seg.fill is FillMode.PROJECTED
        ]
        cu_seqlens = [0]
        for sample_index in range(batch_size):
            sample_token_ids: List[int] = []
            projected_occurrence_index = self._emit_sample(
                self._prompt_plan.segments,
                sample_index,
                inline_values,
                projected_lengths,
                sample_token_ids,
                holes_by_occurrence,
                0,
                len(jagged_token_ids),
            )
            prompt_len = len(sample_token_ids)
            self._emit_sample(
                self._response_segments,
                sample_index,
                inline_values,
                projected_lengths,
                sample_token_ids,
                holes_by_occurrence,
                projected_occurrence_index,
                len(jagged_token_ids),
            )
            # supervision covers the response span only; the prompt is context.
            sample_labels = [self._ignore_index] * prompt_len + sample_token_ids[
                prompt_len:
            ]
            if (
                self._prompt_plan.max_length
                and len(sample_token_ids) > self._prompt_plan.max_length
            ):
                raise ValueError(
                    f"assembled sample {sample_index} is "
                    f"{len(sample_token_ids)} tokens, over "
                    f"max_length {self._prompt_plan.max_length}. Samples are "
                    f"never truncated: cap the source features instead."
                )
            jagged_token_ids.extend(sample_token_ids)
            labels.extend(sample_labels)
            cu_seqlens.append(len(jagged_token_ids))

        holes = [
            position
            for occurrence_positions in holes_by_occurrence
            for position in occurrence_positions
        ]
        return AssembledPrompt(
            input_ids=np.asarray(jagged_token_ids, dtype=np.int64),
            cu_seqlens=np.asarray(cu_seqlens, dtype=np.int64),
            hole_positions=np.asarray(holes, dtype=np.int64),
            labels=np.asarray(labels, dtype=np.int64),
        )

    def assemble_batch(
        self, parsed_features: Dict[str, "np.ndarray"]
    ) -> Dict[str, np.ndarray]:
        """Reshape one parsed batch, assemble it, and key it for the batch.

        Args:
            parsed_features: ``{feature}.values`` / ``{feature}.lengths`` as
                the data parser emits them.

        Returns:
            The five streams, keyed as ``additional_infos`` expects them.
        """
        inline_values: Dict[str, List[np.ndarray]] = {}
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
                flat = np.asarray(parsed_features[f"{source}.values"]).reshape(-1)
                bounds = np.concatenate(([0], np.cumsum(lengths)))
                inline_values[seg.name] = [
                    flat[bounds[i] : bounds[i + 1]] for i in range(lengths.size)
                ]
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

        out = self.assemble(
            inline_values,
            projected_lengths,
            batch_size=batch_size if batch_size is not None else 0,
        )
        return {
            PROMPT_INPUT_IDS: out.input_ids,
            PROMPT_CU_SEQLENS: out.cu_seqlens,
            PROMPT_HOLE_POSITIONS: out.hole_positions,
            PROMPT_LABELS: out.labels,
            PROMPT_MAX_SEQLEN: np.asarray(out.max_seqlen, dtype=np.int64),
        }
