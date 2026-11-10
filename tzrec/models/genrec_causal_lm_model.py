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

from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward

from tzrec.datasets.utils import Batch
from tzrec.features.feature import BaseFeature
from tzrec.models.genrec_model import BaseGenRecModel
from tzrec.modules.dynamic_beam import capped_beam_widths, dynamic_beam_search
from tzrec.prompt.assembler import (
    CU_SEQLENS,
    INPUT_IDS,
    MAX_SEQLEN,
    RESPONSE_LENGTHS,
    MultiTargetLayout,
    build_multi_target_layout,
)
from tzrec.prompt.types import CompiledPrompt, SlotSeg
from tzrec.protos.model_pb2 import ModelConfig
from tzrec.protos.models.genrec_model_pb2 import GenRecModelConfig


def _shared_prefix_attention(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Any,
    scaling: Optional[float] = None,
    dropout: float = 0.0,
    *,
    multi_target_layout: MultiTargetLayout,
    **kwargs: Any,
) -> Tuple[torch.Tensor, None]:
    """Run one FA2 varlen call over the prefix and target segments.

    Every target segment reads its row's prefix keys before its own, so the
    causal mask, aligned to the last key, hides the row's other targets.

    Args:
        module: the calling attention module; unused.
        query: packed queries shaped (1, query heads, total tokens, head dim).
        key: packed keys, retaining the backbone's grouped KV heads.
        value: packed values with the same layout as keys.
        attention_mask: required by the HF attention interface; always None
            here, the segments carry the visibility.
        scaling: backbone attention scale.
        dropout: attention dropout probability, 0 outside training.
        multi_target_layout: batch-local segments and key gather indices.
        **kwargs: unused backbone attention options.

    Returns:
        Packed attention output and no attention weights.
    """
    from flash_attn import flash_attn_varlen_func

    layout = multi_target_layout
    # rotary embedding may promote q/k to fp32; v keeps the projection dtype
    query, key = query.to(value.dtype), key.to(value.dtype)
    output = flash_attn_varlen_func(
        query[0].transpose(0, 1),
        key[0].transpose(0, 1).index_select(0, layout.key_indices),
        value[0].transpose(0, 1).index_select(0, layout.key_indices),
        cu_seqlens_q=layout.cu_seqlens_q,
        cu_seqlens_k=layout.cu_seqlens_k,
        max_seqlen_q=layout.max_seqlen,
        max_seqlen_k=layout.max_seqlen,
        dropout_p=dropout,
        softmax_scale=scaling,
        causal=True,
    )
    return output.unsqueeze(0), None


_SHARED_PREFIX_ATTENTION = "genrec_shared_prefix_fa2"
AttentionInterface.register(_SHARED_PREFIX_ATTENTION, _shared_prefix_attention)


def _padded_sdpa_attention(*args: Any, **kwargs: Any) -> Tuple[torch.Tensor, None]:
    """Run HF's SDPA attention without cuDNN for a left-padded multi-target batch.

    cuDNN's masked SDPA backward returns NaN dq at lengths that are odd
    multiples of 64 (torch 2.14, cuDNN 9.24).

    Args:
        *args: HF attention interface arguments.
        **kwargs: HF attention interface options.

    Returns:
        Attention output and no attention weights.
    """
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
        return sdpa_attention_forward(*args, **kwargs)


_PADDED_SDPA_ATTENTION = "genrec_padded_sdpa"
AttentionInterface.register(_PADDED_SDPA_ATTENTION, _padded_sdpa_attention)


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
        response = self._prompt.prompt_plan.response_segments
        self._sid_only_response = len(response) == 1 and isinstance(
            response[0], SlotSeg
        )

    def _read_beam_config(self, common: GenRecModelConfig) -> None:
        """Parse the decode knobs; the schedule must match the codebook.

        Args:
            common: the shared model config.
        """
        space = self._prompt.sid_space
        self._bands: List[Tuple[int, int]] = list(zip(space.band_lo, space.band_hi))
        self._capped_widths, self._num_return_sequences = resolve_beam_config(
            common, self._bands
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
        predictions = _fx_wrapped_forward(self, embeds, batch)
        keys = ("logits", "labels")
        if self._sid_only_response:
            keys += ("target_weights",)
        return {key: predictions[key] for key in keys}

    def _multi_target_predictions(
        self, embeds: torch.Tensor, batch: Batch
    ) -> Dict[str, torch.Tensor]:
        """Score independent SID branches over one prompt per row.

        FA2 keeps the batch packed; other kernels run left-padded SDPA whose
        per-row mask hides each target from its siblings.

        Args:
            embeds: packed embeddings, retaining one prefix per row.
            batch: existing assembler row boundaries and response lengths.

        Returns:
            Per-target logits and labels, and each target's row weight.
        """
        infos = batch.additional_infos
        layout = build_multi_target_layout(
            infos[CU_SEQLENS],
            infos[RESPONSE_LENGTHS],
            self._prompt.sid_space.num_levels,
        )
        if self._attn_kernel == "flash_attention_2":
            attention, attention_mask = _SHARED_PREFIX_ATTENTION, None
            inputs_embeds = embeds.unsqueeze(0)
            position_ids = layout.position_ids.unsqueeze(0)
            token_mask = None
        else:
            attention = _PADDED_SDPA_ATTENTION
            inputs_embeds, padding_mask = _left_pad(
                embeds, infos[CU_SEQLENS], layout.max_seqlen
            )
            token_mask = padding_mask.bool()
            position_ids = torch.zeros_like(padding_mask)
            position_ids[token_mask] = layout.position_ids
            # -2 marks left padding, which causality keeps attending to itself
            targets = torch.full_like(padding_mask, -2)
            targets[token_mask] = layout.token_targets
            key_targets = targets[:, None, :]
            attention_mask = (
                ((key_targets == -1) | (key_targets == targets[:, :, None]))
                .tril()
                .unsqueeze(1)
            )
        previous_attention = self.lm.config._attn_implementation
        self.lm.set_attn_implementation(attention)
        hidden = self.lm.base_model(
            inputs_embeds=inputs_embeds,
            attention_mask={"full_attention": attention_mask},
            position_ids=position_ids,
            use_cache=False,
            multi_target_layout=layout,
        ).last_hidden_state
        self.lm.set_attn_implementation(previous_attention)
        hidden = hidden[0] if token_mask is None else hidden[token_mask]
        return {
            "logits": self.lm.get_output_embeddings()(hidden[layout.score_indices]),
            "labels": infos[INPUT_IDS][layout.label_indices],
            "target_weights": layout.target_weights,
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
        infos = batch.additional_infos
        padded, mask = _left_pad(embeds, infos[CU_SEQLENS], int(infos[MAX_SEQLEN]))
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
        infos = batch.additional_infos
        return generate_codes(
            self.lm,
            embeds,
            infos[CU_SEQLENS],
            int(infos[MAX_SEQLEN]),
            self._capped_widths,
            self._bands,
            self._num_return_sequences,
        )


def resolve_beam_config(
    common: GenRecModelConfig, bands: Sequence[Tuple[int, int]]
) -> Tuple[List[int], int]:
    """Validate the decode knobs against the codebook and cap the beam widths.

    Args:
        common: the shared model config.
        bands: inclusive ``(lo, hi)`` token-id edge per SID level.

    Returns:
        The capped width per level, and ``num_return_sequences``.
    """
    num_return_sequences = int(common.num_return_sequences)
    beam_widths = list(common.beam_widths)
    if not beam_widths:
        raise ValueError(
            "beam_widths is required; give one width per SID level, e.g. "
            "[50, 50, 50] or [100, 200, 400]."
        )
    if len(beam_widths) != len(bands):
        raise ValueError(
            f"beam_widths has {len(beam_widths)} entries but the codebook has "
            f"{len(bands)} levels; give one width per level."
        )
    if any(width < 1 for width in beam_widths):
        raise ValueError(f"beam_widths must be >= 1, got {beam_widths}.")
    if num_return_sequences < 1:
        raise ValueError(
            f"num_return_sequences must be >= 1, got {num_return_sequences}."
        )
    capped_widths = capped_beam_widths(beam_widths, bands)
    if num_return_sequences > capped_widths[-1]:
        raise ValueError(
            f"num_return_sequences ({num_return_sequences}) must not exceed the "
            f"final capped beam width ({capped_widths[-1]})."
        )
    return capped_widths, num_return_sequences


def _left_pad(
    embeds: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad packed rows into a rectangle, newest token last.

    Args:
        embeds: packed embeddings, ``(total_tokens, hidden)``.
        cu_seqlens: row boundaries, ``(rows + 1,)``.
        max_seqlen: the widest row.

    Returns:
        Padded embeddings and attention mask.
    """
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


def generate_codes(
    lm: torch.nn.Module,
    embeds: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    capped_widths: Sequence[int],
    bands: Sequence[Tuple[int, int]],
    num_return_sequences: int,
) -> torch.Tensor:
    """Beam-search the SID answer of packed prompts.

    Args:
        lm: the causal LM.
        embeds: the assembled prompt embeddings, packed.
        cu_seqlens: row boundaries, ``(rows + 1,)``.
        max_seqlen: the widest row.
        capped_widths: beam width per SID level, already capped.
        bands: inclusive ``(lo, hi)`` token-id edge per SID level.
        num_return_sequences: beams to return per row.

    Returns:
        ``(rows, num_return_sequences, num_levels)`` local codes, best first.
    """
    padded, mask = _left_pad(embeds, cu_seqlens, max_seqlen)
    tokens = dynamic_beam_search(lm, padded, mask, capped_widths, bands)
    # a band starts at base_vocab_size + level_offset, so this undoes both shifts
    codes = tokens - tokens.new_tensor([lo for lo, _ in bands])
    codes = codes.view(padded.shape[0], -1, len(bands))
    return codes[:, :num_return_sequences, :]


@torch.fx.wrap
def _fx_wrapped_forward(
    model: "GenRecCausalLMModel", embeds: torch.Tensor, batch: Batch
) -> Dict[str, torch.Tensor]:
    """Hide the packed forward and its per-batch routing from FX.

    ``TrainPipelineSparseDist`` symbolically traces the model whenever a
    sharded module exists, and the LM call reads ``max_seqlen`` as a host int.
    A SID-only response whose batch has a row with several SIDs takes the
    multi-target forward; every other batch keeps the response window.

    Args:
        model: the model whose response window to compute.
        embeds: the assembled prompt embeddings, packed.
        batch: the batch being scored.

    Returns:
        Logits and labels, plus per-target row weights for a SID-only
        response.
    """
    if not model._sid_only_response:
        logits, labels = model._forward(embeds, batch)
        return {"logits": logits, "labels": labels}
    lengths = batch.additional_infos[RESPONSE_LENGTHS]
    if int(lengths.max()) > model._prompt.sid_space.num_levels:
        return model._multi_target_predictions(embeds, batch)
    logits, labels = model._forward(embeds, batch)
    return {
        "logits": logits,
        "labels": labels,
        "target_weights": (lengths > 0).float(),
    }


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
