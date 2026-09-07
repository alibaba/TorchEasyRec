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

from typing import Any, Dict, List, Optional, Tuple

import torch

from tzrec.datasets.utils import Batch
from tzrec.features.feature import BaseFeature
from tzrec.models.genrec_model import BaseGenrecModel
from tzrec.modules.dynamic_beam import capped_beam_widths, dynamic_beam_search
from tzrec.prompt.assembler import (
    PROMPT_CU_SEQLENS,
    PROMPT_INPUT_IDS,
    PROMPT_MAX_SEQLEN,
    PROMPT_RESPONSE_LENGTHS,
)
from tzrec.prompt.types import CompiledPrompt
from tzrec.protos.model_pb2 import ModelConfig
from tzrec.protos.models.genrec_model_pb2 import GenrecModelConfig


class GenrecCausalLMModel(BaseGenrecModel):
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

    def _read_beam_config(self, common: GenrecModelConfig) -> None:
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
        logits, labels = _fx_wrapped_forward(self, embeds, batch)
        return {"logits": logits, "labels": labels}

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
        padded, mask, labels = self._left_pad_packed_inputs(
            embeds, batch, build_labels=True
        )
        suffix = self._prompt.prompt_plan.logits_suffix_len
        outputs = self.lm(
            inputs_embeds=padded,
            attention_mask=mask,
            use_cache=False,
            logits_to_keep=suffix,
        )
        return outputs.logits, labels[:, -suffix:]

    def _generate(self, embeds: torch.Tensor, batch: Batch) -> torch.Tensor:
        """Beam-search the SID answer.

        Args:
            embeds: the assembled prompt embeddings, packed.
            batch: carries the packed prompt and the padded width.

        Returns:
            ``(B, num_return, num_levels)`` local codes, best first.
        """
        padded, mask, _ = self._left_pad_packed_inputs(embeds, batch)
        tokens = dynamic_beam_search(
            self.lm, padded, mask, self._capped_widths, self._bands
        )
        codes = self._tokens_to_local_codes(tokens, padded.shape[0])
        return codes[:, : self._num_return_sequences, :]

    def _left_pad_packed_inputs(
        self,
        embeds: torch.Tensor,
        batch: Batch,
        build_labels: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Left-pad packed prompt embeddings for the causal LM.

        Args:
            embeds: packed embeddings, ``(total_tokens, hidden)``.
            batch: carries the packed prompt metadata.
            build_labels: whether to build response-only training labels.

        Returns:
            Padded embeddings, attention mask and optional labels.
        """
        infos = batch.additional_infos
        cu_seqlens = infos[PROMPT_CU_SEQLENS]
        max_seqlen = int(infos[PROMPT_MAX_SEQLEN])
        starts = cu_seqlens[:-1]
        lengths = cu_seqlens[1:] - starts
        batch_size = lengths.numel()
        hidden = embeds.shape[-1]

        columns = torch.arange(max_seqlen, device=embeds.device)
        mask = columns[None, :] >= (max_seqlen - lengths)[:, None]

        padded = embeds.new_zeros((batch_size, max_seqlen, hidden))
        # mask selects row-major, which is how embeds and input_ids are packed
        padded[mask] = embeds
        if not build_labels:
            return padded, mask.long(), None

        input_ids = infos[PROMPT_INPUT_IDS]
        response_lengths = infos[PROMPT_RESPONSE_LENGTHS]
        labels = torch.full(
            (batch_size, max_seqlen),
            self._ignore_index,
            dtype=input_ids.dtype,
            device=embeds.device,
        )
        labels[mask] = input_ids
        response_mask = columns[None, :] >= (max_seqlen - response_lengths)[:, None]
        labels[~response_mask] = self._ignore_index
        return padded, mask.long(), labels


@torch.fx.wrap
def _fx_wrapped_forward(
    model: "GenrecCausalLMModel", embeds: torch.Tensor, batch: Batch
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Hide the padded forward from FX.

    ``TrainPipelineSparseDist`` symbolically traces the model whenever a
    sharded module exists, and ``_left_pad_packed_inputs`` reads
    ``max_seqlen`` as a host int.

    Args:
        model: the model whose response window to compute.
        embeds: the assembled prompt embeddings, packed.
        batch: the batch being scored.

    Returns:
        The response-window logits and labels.
    """
    return model._forward(embeds, batch)


@torch.fx.wrap
def _fx_wrapped_generate(
    model: "GenrecCausalLMModel", embeds: torch.Tensor, batch: Batch
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
