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
from transformers import AttentionInterface

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
    *,
    shared_prefix_layout: MultiTargetLayout,
    **kwargs: Any,
) -> Tuple[torch.Tensor, None]:
    """Run packed causal prefix FA2 followed by isolated response FlexAttention.

    Args:
        module: backbone attention module after projection and rotary embedding.
        query: packed queries shaped (1, query heads, total tokens, head dim).
        key: packed keys, retaining the backbone's grouped KV heads.
        value: packed values with the same layout as keys.
        attention_mask: response-only BlockMask over gathered per-parent keys.
        scaling: backbone attention scale.
        shared_prefix_layout: batch-local packing and scatter indices.
        **kwargs: backbone attention options forwarded to the Flex adapter.

    Returns:
        Packed attention output and no attention weights.
    """
    from flash_attn import flash_attn_varlen_func
    from transformers.integrations.flex_attention import flex_attention_forward

    layout = shared_prefix_layout
    if query.dtype == torch.float32:
        dtype = (
            torch.get_autocast_dtype("cuda")
            if torch.is_autocast_enabled("cuda")
            else module.q_proj.weight.dtype
        )
        query, key, value = query.to(dtype), key.to(dtype), value.to(dtype)
    _, heads, _, head_dim = query.shape
    flat_query = query[0].transpose(0, 1)
    flat_key = key[0].transpose(0, 1)
    flat_value = value[0].transpose(0, 1)
    prefix_tokens = layout.prefix_tokens
    prefix_output = flash_attn_varlen_func(
        flat_query[:prefix_tokens],
        flat_key[:prefix_tokens],
        flat_value[:prefix_tokens],
        cu_seqlens_q=layout.prefix_cu_seqlens,
        cu_seqlens_k=layout.prefix_cu_seqlens,
        max_seqlen_q=layout.max_prefix_length,
        max_seqlen_k=layout.max_prefix_length,
        dropout_p=0.0,
        softmax_scale=scaling,
        causal=True,
    )
    rows = layout.prefix_cu_seqlens.numel() - 1
    response_tokens = flat_query[prefix_tokens:]
    response_query, padded_key, padded_value = (
        tokens.new_zeros(rows * width, tokens.shape[1], head_dim)
        .index_copy(0, indices, tokens)
        .reshape(rows, width, tokens.shape[1], head_dim)
        .transpose(1, 2)
        for tokens, width, indices in (
            (response_tokens, layout.response_width, layout.response_indices),
            (flat_key, layout.key_width, layout.key_indices),
            (flat_value, layout.key_width, layout.key_indices),
        )
    )
    response_output, _ = flex_attention_forward(
        module,
        response_query,
        padded_key,
        padded_value,
        attention_mask,
        scaling=scaling,
        **kwargs,
    )
    output = torch.cat(
        (
            prefix_output,
            response_output.reshape(-1, heads, head_dim).index_select(
                0, layout.response_indices
            ),
        )
    )
    return output.unsqueeze(0), None


_SHARED_PREFIX_ATTENTION = "genrec_shared_prefix_fa2_flex"
AttentionInterface.register(_SHARED_PREFIX_ATTENTION, _shared_prefix_attention)


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
        if self.lm.is_gradient_checkpointing:
            raise ValueError(
                "Shared SID training requires gradient checkpointing to be disabled."
            )

        hybrid = self._attn_kernel == "flash_attention_2"
        rows = infos[RESPONSE_LENGTHS].numel()
        width = int(infos[MAX_SEQLEN])
        layout = build_multi_target_layout(
            infos[CU_SEQLENS],
            infos[RESPONSE_LENGTHS],
            width,
            levels,
            pack_prefix=hybrid,
        )
        if hybrid:
            config = self.lm.config
            if (
                config.model_type != "qwen2"
                or config.attention_dropout != 0
                or config.use_sliding_window
                or embeds.device.type != "cuda"
                or (
                    embeds.dtype not in (torch.bfloat16, torch.float16)
                    and not torch.is_autocast_enabled("cuda")
                )
            ):
                raise ValueError(
                    "Shared-prefix FA2 requires Qwen2 full attention on CUDA, "
                    "BF16/FP16 computation and zero attention dropout."
                )
            inputs = embeds[layout.packed_indices].unsqueeze(0)
        else:
            inputs, _ = self._left_pad(embeds, batch)
        block_mask = create_block_mask(
            layout.mask_mod,
            rows,
            None,
            layout.response_width if hybrid else width,
            width,
            device=embeds.device,
            _compile=True,
        )
        previous_attention = self.lm.config._attn_implementation
        try:
            self.lm.set_attn_implementation(
                _SHARED_PREFIX_ATTENTION if hybrid else "flex_attention"
            )
            hidden = self.lm.base_model(
                inputs_embeds=inputs,
                attention_mask={"full_attention": block_mask},
                position_ids=layout.position_ids,
                use_cache=False,
                **({"shared_prefix_layout": layout} if hybrid else {}),
            ).last_hidden_state
        finally:
            self.lm.set_attn_implementation(previous_attention)

        logits = self.lm.get_output_embeddings()(
            hidden.reshape(-1, hidden.shape[-1])[layout.logit_indices]
        )[layout.logit_inverse].reshape(layout.target_parents.numel(), levels, -1)
        return {
            "logits": logits,
            "labels": infos[INPUT_IDS][layout.label_indices],
            "target_parents": layout.target_parents,
            "target_counts": layout.target_counts,
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
