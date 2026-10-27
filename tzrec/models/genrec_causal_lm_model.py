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

"""Decoder-only forward and decode over an HF causal LM.

This subclass asks the backbone's own forward for a suffix window of logits, so
a full (batch, length, vocab) tensor is never materialized, and it decodes by
prefilling once and stepping a self-attention cache.

``init_backbone`` checks the interfaces both paths need. A backbone whose
forward has no ``logits_to_keep``, or which returns a legacy tuple cache, is
not supported; Qwen2.5/Qwen3 are what CI covers.
"""

from typing import Any, Dict, List, Optional, Tuple, cast

import torch

from tzrec.datasets.utils import Batch
from tzrec.features.feature import BaseFeature
from tzrec.models.genrec_model import BaseGenRecModel
from tzrec.modules.dynamic_beam import capped_beam_widths, dynamic_beam_search
from tzrec.prompt.assembler import (
    CU_SEQLENS,
    INPUT_IDS,
    MAX_SEQLEN,
    RESPONSE_LENGTHS,
)
from tzrec.prompt.types import CompiledPrompt, SlotSeg
from tzrec.protos.model_pb2 import ModelConfig
from tzrec.protos.models.genrec_model_pb2 import GenRecModelConfig


class GenRecCausalLMModel(BaseGenRecModel):
    """An HF causal LM driven by a compiled prompt.

    Args:
        model_config: the model oneof.
        features: every created feature.
        labels: data_config label fields.
        sample_weights: optional sample weight fields.
        compiled_prompt: the compiled prompt; required.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        features: List[BaseFeature],
        labels: List[str],
        sample_weights: Optional[List[str]] = None,
        compiled_prompt: Optional[CompiledPrompt] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_config,
            features,
            labels,
            sample_weights,
            compiled_prompt=compiled_prompt,
            **kwargs,
        )
        common = self._model_config.common
        self._generated_sids_key = common.generated_sids_key
        self._read_beam_config(common)

    def _read_beam_config(self, common: GenRecModelConfig) -> None:
        """Parse the decode knobs; the schedule must match the codebook.

        Args:
            common: the shared model config.
        """
        space = self._prompt.sid_space
        self._num_return_sequences = int(common.num_return_sequences)
        beam_widths = list(common.beam_widths)
        if not beam_widths:
            raise ValueError(
                f"{type(self).__name__}: beam_widths is required; give one "
                f"width per SID level, e.g. [50, 50, 50] or [100, 200, 400]."
            )
        if len(beam_widths) != space.num_levels:
            raise ValueError(
                f"{type(self).__name__}: beam_widths has "
                f"{len(beam_widths)} entries but the codebook has "
                f"{space.num_levels} levels; give one width per level."
            )
        if any(width < 1 for width in beam_widths):
            raise ValueError(
                f"{type(self).__name__}: beam_widths must be >= 1, got {beam_widths}."
            )
        if self._num_return_sequences < 1:
            raise ValueError(
                f"{type(self).__name__}: num_return_sequences must be >= 1, got "
                f"{self._num_return_sequences}."
            )
        self._bands: List[Tuple[int, int]] = list(zip(space.band_lo, space.band_hi))
        self._capped_widths = capped_beam_widths(beam_widths, self._bands)

        if self._num_return_sequences > self._capped_widths[-1]:
            raise ValueError(
                f"{type(self).__name__}: num_return_sequences "
                f"({self._num_return_sequences}) must not exceed the final capped "
                f"beam width ({self._capped_widths[-1]})."
            )

    def predict(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """Run teacher-forced loss or inference decode over the assembled stream.

        Args:
            batch: carries the packed prompt in ``additional_infos``.

        Returns:
            The response-window logits and labels when training, the decoded
            SIDs otherwise.
        """
        # outside the leaf on both paths, so the pipeline can prefetch it
        embeds = self.build_input(batch)
        if self.is_inference:
            return {self._generated_sids_key: _fx_wrapped_generate(self, embeds, batch)}
        logits, labels, parents, counts = _fx_wrapped_forward(self, embeds, batch)
        return {
            "logits": logits,
            "labels": labels,
            "target_parents": parents,
            "target_counts": counts,
        }

    def _training_predictions(
        self, embeds: torch.Tensor, batch: Batch
    ) -> Dict[str, torch.Tensor]:
        """Select shared-prefix scoring for a response containing multiple SIDs.

        Args:
            embeds: assembled prompt and response embeddings.
            batch: packed row boundaries and response lengths.

        Returns:
            Logits and labels, plus parent indices for a multi-target response.
        """
        response = self._prompt.prompt_plan.response_segments
        levels = self._prompt.sid_space.num_levels
        if len(response) == 1 and isinstance(response[0], SlotSeg):
            lengths = batch.additional_infos[RESPONSE_LENGTHS].to(torch.int64)
            if bool((lengths.remainder(levels) != 0).any()):
                raise ValueError(
                    "SID response length must be a multiple of codebook levels."
                )
            if bool((lengths > levels).any()):
                return self._multi_target_predictions(embeds, batch)
            logits, labels = self._forward(embeds, batch)
            return {
                "logits": logits,
                "labels": labels,
                "target_parents": torch.arange(lengths.numel(), device=embeds.device),
                "target_counts": lengths // levels,
            }
        logits, labels = self._forward(embeds, batch)
        return {"logits": logits, "labels": labels}

    def _multi_target_predictions(
        self, embeds: torch.Tensor, batch: Batch
    ) -> Dict[str, torch.Tensor]:
        """Score independent SID branches over one differentiable prompt prefix.

        Args:
            embeds: packed embeddings, retaining one prefix per parent sample.
            batch: existing assembler row boundaries and response lengths.

        Returns:
            Per-target logits, labels, parent indices and target counts.
        """
        from torch.nn.attention.flex_attention import create_block_mask

        levels = self._prompt.sid_space.num_levels
        infos = batch.additional_infos
        response_lengths = infos[RESPONSE_LENGTHS].to(torch.int64)
        if self.lm.is_gradient_checkpointing:
            raise ValueError(
                "Shared SID training requires gradient checkpointing to be disabled."
            )

        padded, _ = self._left_pad(embeds, batch)
        rows, width = padded.shape[:2]
        lengths = torch.diff(infos[CU_SEQLENS]).to(torch.int64)
        starts = width - lengths
        prefix_ends = width - response_lengths
        positions = torch.arange(width, device=embeds.device)
        relative = positions[None, :] - prefix_ends[:, None]
        position_ids = torch.where(
            relative < 0,
            positions[None, :] - starts[:, None],
            prefix_ends[:, None] - starts[:, None] + relative.remainder(levels),
        ).clamp(min=0)

        def mask_mod(b, h, q, kv):
            """Allow causal prefix attention and same-branch response attention."""
            prefix_end = prefix_ends[b]
            q_relative = q - prefix_end
            kv_relative = kv - prefix_end
            prefix_key = (kv < prefix_end) & (kv <= q)
            branch_key = (
                (q >= prefix_end)
                & (kv >= prefix_end)
                & (q_relative // levels == kv_relative // levels)
                & (kv <= q)
            )
            return (
                (q >= starts[b])
                & (q < width)
                & (kv >= starts[b])
                & (kv < width)
                & (prefix_key | branch_key)
            )

        block_mask = create_block_mask(
            mask_mod, rows, None, width, width, device=embeds.device, _compile=True
        )
        previous_attention = self.lm.config._attn_implementation
        try:
            self.lm.set_attn_implementation("flex_attention")
            hidden = self.lm.base_model(
                inputs_embeds=padded,
                attention_mask={"full_attention": block_mask},
                position_ids=position_ids,
                use_cache=False,
            ).last_hidden_state
        finally:
            self.lm.set_attn_implementation(previous_attention)

        counts = response_lengths // levels
        parents = torch.repeat_interleave(
            torch.arange(rows, device=embeds.device), counts
        )
        parent_starts = counts.cumsum(0) - counts
        target_indices = torch.arange(parents.numel(), device=embeds.device)
        target_starts = (
            prefix_ends[parents] + (target_indices - parent_starts[parents]) * levels
        )
        offsets = torch.arange(levels, device=embeds.device)
        score_positions = target_starts[:, None] + offsets[None, :] - 1
        score_positions[:, 0] = prefix_ends[parents] - 1
        score_indices = parents[:, None] * width + score_positions
        unique_scores, inverse = score_indices.reshape(-1).unique(return_inverse=True)
        logits = self.lm.get_output_embeddings()(
            hidden.reshape(-1, hidden.shape[-1])[unique_scores]
        )[inverse].reshape(parents.numel(), levels, -1)

        input_ids, _ = self._left_pad(infos[INPUT_IDS].unsqueeze(-1), batch)
        labels = input_ids.squeeze(-1)[
            parents[:, None], target_starts[:, None] + offsets[None, :]
        ]
        return {
            "logits": logits,
            "labels": labels,
            "target_parents": parents,
            "target_counts": counts,
        }

    def _forward(
        self, embeds: torch.Tensor, batch: Batch
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the LM over the assembled embeddings and cut the response window.

        Body and head are called separately so logits cover the supervised
        window only: a full (batch, length, vocab) upcast does not fit.

        Args:
            embeds: the assembled prompt embeddings, packed.
            batch: carries the row boundaries and the collator's width.

        Returns:
            Logits and labels over the same window, so the shift ``loss``
            applies lands on the pairs the window was sized for.
        """
        infos = batch.additional_infos
        cu_seqlens = infos[CU_SEQLENS]
        suffix = cast(int, self._prompt.prompt_plan.logits_suffix_len)
        suffix_offsets = torch.arange(-suffix, 0, device=embeds.device)
        # (rows, suffix): the window each row is supervised over, the same
        # absolute positions whichever layout the kernel is given
        keep = cu_seqlens[1:, None] + suffix_offsets
        labels = infos[INPUT_IDS][keep].masked_fill(
            suffix_offsets[None, :] < -infos[RESPONSE_LENGTHS][:, None],
            self._ignore_index,
        )
        if self._attn_kernel == "flash_attention_2":
            logits = self._varlen_logits(embeds, batch, keep)
        else:
            logits = self._padded_logits(embeds, batch, suffix)
        return logits, labels

    def _varlen_logits(
        self, embeds: torch.Tensor, batch: Batch, keep: torch.Tensor
    ) -> torch.Tensor:
        """Score every row in one varlen call, with no padding at all.

        Only the flash varlen kernel reads ``cu_seq_lens``; every other kernel
        would rebuild the row boundaries as a dense ``(1, 1, T, T)`` mask and
        pay ``(sum L)^2`` instead of ``sum L^2``.

        Args:
            embeds: the assembled prompt embeddings, packed.
            batch: carries the row boundaries and the collator's width.
            keep: ``(rows, suffix)`` absolute positions of the response window.

        Returns:
            ``(rows, suffix, vocab)`` logits over the response window.
        """
        infos = batch.additional_infos
        cu_seqlens = infos[CU_SEQLENS]
        row_starts = torch.repeat_interleave(
            cu_seqlens[:-1], torch.diff(cu_seqlens), output_size=embeds.shape[0]
        )
        position_ids = (
            torch.arange(embeds.shape[0], device=embeds.device) - row_starts
        ).unsqueeze(0)
        # varlen flash-attention wants int32 boundaries
        flash_cu_seqlens = cu_seqlens.to(torch.int32)
        max_seqlen = int(infos[MAX_SEQLEN])
        outputs = self.lm(
            inputs_embeds=embeds.unsqueeze(0),
            attention_mask=None,
            position_ids=position_ids,
            use_cache=False,
            logits_to_keep=keep.reshape(-1),
            cu_seq_lens_q=flash_cu_seqlens,
            cu_seq_lens_k=flash_cu_seqlens,
            max_length_q=max_seqlen,
            max_length_k=max_seqlen,
        )
        return outputs.logits.reshape(keep.shape[0], keep.shape[1], -1)

    def _padded_logits(
        self, embeds: torch.Tensor, batch: Batch, suffix: int
    ) -> torch.Tensor:
        """Score left-padded rows, which every kernel can serve.

        Args:
            embeds: the assembled prompt embeddings, packed.
            batch: carries the row boundaries and the collator's width.
            suffix: width of the supervised window.

        Returns:
            ``(rows, suffix, vocab)`` logits over the response window.
        """
        padded, mask = self._left_pad(embeds, batch)
        if padded.shape[1] < suffix:
            raise ValueError(
                f"{type(self).__name__}: no sample reaches the supervised "
                f"window -- the padded width is {padded.shape[1]}, the window "
                f"is {suffix}, and logits_to_keep would silently return the "
                f"narrower one."
            )
        # left padding offsets every row, so spell the positions out rather
        # than letting HF assign arange(max_seqlen) over the padding too
        position_ids = (mask.cumsum(-1) - 1).clamp(min=0)
        outputs = self.lm(
            inputs_embeds=padded,
            attention_mask=mask,
            position_ids=position_ids,
            use_cache=False,
            logits_to_keep=suffix,
        )
        return outputs.logits

    def _generate(self, embeds: torch.Tensor, batch: Batch) -> torch.Tensor:
        """Beam-search the SID answer.

        Args:
            embeds: the assembled prompt embeddings, packed.
            batch: carries the packed prompt and the padded width.

        Returns:
            ``(B, num_return, num_levels)`` local codes, best first.
        """
        padded, mask = self._left_pad(embeds, batch)
        tokens = dynamic_beam_search(
            self.lm, padded, mask, self._capped_widths, self._bands
        )
        codes = self._tokens_to_local_codes(tokens, padded.shape[0])
        return codes[:, : self._num_return_sequences, :]

    def _left_pad(
        self,
        embeds: torch.Tensor,
        batch: Batch,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pad the packed rows into a rectangle, newest token last.

        Used by beam decode and by the padded training forward.

        Args:
            embeds: packed embeddings, ``(total_tokens, hidden)``.
            batch: carries the packed prompt metadata.

        Returns:
            Padded embeddings and attention mask.
        """
        infos = batch.additional_infos
        cu_seqlens = infos[CU_SEQLENS]
        max_seqlen = int(infos[MAX_SEQLEN])
        starts = cu_seqlens[:-1]
        lengths = cu_seqlens[1:] - starts
        batch_size = lengths.numel()
        hidden = embeds.shape[-1]

        columns = torch.arange(max_seqlen, device=embeds.device)
        mask = columns[None, :] >= (max_seqlen - lengths)[:, None]

        padded = embeds.new_zeros((batch_size, max_seqlen, hidden))
        # mask selects row-major, which is how embeds and input_ids are packed
        padded[mask] = embeds
        return padded, mask.long()


@torch.fx.wrap
def _fx_wrapped_forward(
    model: "GenRecCausalLMModel", embeds: torch.Tensor, batch: Batch
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hide the packed forward from FX.

    ``TrainPipelineSparseDist`` symbolically traces the model whenever a
    sharded module exists, and the LM call reads ``max_seqlen`` as a host int.

    Args:
        model: the model whose response window to compute.
        embeds: the assembled prompt embeddings, packed.
        batch: the batch being scored.

    Returns:
        Training predictions and optional multi-target parent metadata.
    """
    predictions = model._training_predictions(embeds, batch)
    labels = predictions["labels"]
    return (
        predictions["logits"],
        labels,
        predictions.get("target_parents", labels.new_empty(0)),
        predictions.get("target_counts", labels.new_empty(0)),
    )


@torch.fx.wrap
def _fx_wrapped_generate(
    model: "GenRecCausalLMModel", embeds: torch.Tensor, batch: Batch
) -> torch.Tensor:
    """Hide the decode loop from FX.

    ``PredictPipelineSparseDist`` FX-traces the model, and beam decode reads
    host ints and branches on them. Wrapping an inner helper only moves the
    failure to the next such read, so the whole loop is one leaf.

    Args:
        model: the model whose decode loop to run.
        embeds: the assembled prompt embeddings, packed.
        batch: the batch to decode.

    Returns:
        The decoded local codes.
    """
    return model._generate(embeds, batch)
