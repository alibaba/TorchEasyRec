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

"""Decoder-only forward and decode over a Qwen backbone.

Everything backbone-agnostic is in ``BasePromptGenerativeModel``. What is here
is what a decoder-only family does differently: it reaches past ``lm(...)`` into
``lm.model`` and ``lm.lm_head`` so logits are materialized for a suffix window
only, and it decodes by prefilling once and stepping a self-attention cache.

The layout is Qwen's, and Llama and Mistral share it exactly.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch

from tzrec.datasets.utils import Batch
from tzrec.features.feature import BaseFeature
from tzrec.models.prompt_generative_model import BasePromptGenerativeModel
from tzrec.modules.dynamic_beam import dynamic_beam_search
from tzrec.prompt.assembler import (
    PROMPT_CU_SEQLENS,
    PROMPT_LABELS,
    PROMPT_MAX_SEQLEN,
)
from tzrec.prompt.plan import CompiledPrompt
from tzrec.protos.model_pb2 import ModelConfig
from tzrec.protos.models.prompt_model_pb2 import PromptModelConfig


class PromptGenerativeQwen(BasePromptGenerativeModel):
    """Qwen family (Qwen2.5, Qwen3, ...) driven by a compiled prompt.

    Args:
        model_config: the model oneof.
        features: every created feature.
        labels: data_config label fields.
        sample_weights: optional sample weight fields.
        prompt: the compiled prompt; required.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        features: List[BaseFeature],
        labels: List[str],
        sample_weights: Optional[List[str]] = None,
        prompt: Optional[CompiledPrompt] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_config, features, labels, sample_weights, prompt, **kwargs
        )
        common = self._model_config.common
        self._ignore_index = int(common.ignore_index)
        self._generated_sids_key = common.generated_sids_key
        self._read_beam_config(common)

    def _read_beam_config(self, common: PromptModelConfig) -> None:
        """Parse the decode knobs; the schedule must match the codebook.

        Args:
            common: the shared model config.
        """
        space = self._prompt.sid_space
        self._num_return_sequences = int(common.num_return_sequences)
        self._beam_widths: List[int] = list(common.beam_widths)
        if not self._beam_widths:
            raise ValueError(
                f"{type(self).__name__}: beam_widths is required; give one "
                f"width per SID level, e.g. [50, 50, 50] or [100, 200, 400]."
            )
        if len(self._beam_widths) != space.num_levels:
            raise ValueError(
                f"{type(self).__name__}: beam_widths has "
                f"{len(self._beam_widths)} entries but the codebook has "
                f"{space.num_levels} levels; give one width per level."
            )
        if self._num_return_sequences > self._beam_widths[-1]:
            raise ValueError(
                f"{type(self).__name__}: num_return_sequences "
                f"({self._num_return_sequences}) must not exceed the final beam width "
                f"({self._beam_widths[-1]})."
            )

    def predict(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """Teacher-forced forward over the assembled stream.

        Args:
            batch: carries the packed prompt in ``additional_infos``.

        Returns:
            The loss when training, the decoded SIDs otherwise.
        """
        if self.is_inference:
            return {self._generated_sids_key: _fx_wrapped_generate(self, batch)}
        # the embedding lookup stays traceable so the train pipeline can still
        # see the sharded module and prefetch it; only the padding and the LM,
        # which read the collator's width as a host int, are hidden.
        return _fx_wrapped_loss(self, self._prompt_embeds(batch), batch)

    def _forward_loss(
        self, embeds: torch.Tensor, batch: Batch
    ) -> Dict[str, torch.Tensor]:
        """Run the LM over the assembled embeddings and score the response.

        Body and head are called separately so logits cover the supervised
        window only: a full (batch, length, vocab) upcast does not fit.

        Args:
            embeds: the assembled prompt embeddings, packed.
            batch: carries the row boundaries and the collator's width.

        Returns:
            The loss.
        """
        infos = batch.additional_infos
        padded, mask, labels = _unpack(
            embeds,
            infos[PROMPT_CU_SEQLENS],
            infos[PROMPT_LABELS],
            int(infos[PROMPT_MAX_SEQLEN]),
            self._ignore_index,
        )
        outputs = self.lm.model(inputs_embeds=padded, attention_mask=mask)

        suffix = self._prompt.prompt_plan.logits_suffix_len
        window = slice(-suffix, None) if suffix else slice(None)
        logits = self.lm.lm_head(outputs.last_hidden_state[:, window, :])
        loss = self.lm.loss_function(
            logits=logits,
            labels=labels[:, window],
            vocab_size=self.lm.config.vocab_size,
            ignore_index=self._ignore_index,
        )
        return {"loss": loss}

    def _generate(self, batch: Batch) -> torch.Tensor:
        """Beam-search the SID answer.

        Args:
            batch: carries the packed prompt and the collator's width.

        Returns:
            ``(B, num_return, num_levels)`` local codes, best first.
        """
        embeds = self._prompt_embeds(batch)
        infos = batch.additional_infos
        padded, mask, _ = _unpack(
            embeds,
            infos[PROMPT_CU_SEQLENS],
            None,
            int(infos[PROMPT_MAX_SEQLEN]),
            self._ignore_index,
        )
        space = self._prompt.sid_space
        tokens = dynamic_beam_search(
            self.lm,
            padded,
            mask,
            self._beam_widths,
            list(zip(space.band_lo, space.band_hi)),
        )
        codes = self._tokens_to_local_codes(tokens, padded.shape[0])
        return codes[:, : self._num_return_sequences, :]


def _unpack(
    embeds: torch.Tensor,
    cu_seqlens: torch.Tensor,
    labels: Optional[torch.Tensor],
    max_seqlen: int,
    ignore_index: int,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Left-pad a packed varlen batch at the LM boundary.

    Padding lives in this one adapter. ``max_seqlen`` is the collator's, not
    ``lengths.max()``: deriving it here would sync the device to the host every
    step, which the design forbids.

    Pads go on the left so that every row ends on a real token. Both consumers
    index from the right -- the loss keeps a fixed-width suffix and decode
    prefills from ``[:, -1, :]`` -- so right-padding would hand a short row its
    padding instead of its answer.

    Args:
        embeds: packed embeddings, ``(total_tokens, hidden)``.
        cu_seqlens: row boundaries, ``(batch_size + 1,)``.
        labels: packed labels, or None at inference where nothing is scored.
        max_seqlen: the collator's padded width.
        ignore_index: label value for padding.

    Returns:
        Padded embeddings, attention mask and labels.
    """
    starts = cu_seqlens[:-1]
    lengths = cu_seqlens[1:] - starts
    batch_size = lengths.numel()
    hidden = embeds.shape[-1]

    columns = torch.arange(max_seqlen, device=embeds.device)
    mask = columns[None, :] >= (max_seqlen - lengths)[:, None]

    padded = embeds.new_zeros((batch_size, max_seqlen, hidden))
    # mask selects row-major, which is the order embeds and labels are packed in
    padded[mask] = embeds
    if labels is None:
        return padded, mask.long(), None

    out_labels = torch.full(
        (batch_size, max_seqlen),
        ignore_index,
        dtype=labels.dtype,
        device=embeds.device,
    )
    out_labels[mask] = labels
    return padded, mask.long(), out_labels


@torch.fx.wrap
def _fx_wrapped_loss(
    model: "PromptGenerativeQwen", embeds: torch.Tensor, batch: Batch
) -> Dict[str, torch.Tensor]:
    """Hide the padded forward from FX.

    ``TrainPipelineSparseDist`` symbolically traces the model whenever a
    sharded module exists, and ``_unpack`` reads ``max_seqlen`` as a host int.

    Args:
        model: the model whose loss to compute.
        embeds: the assembled prompt embeddings, packed.
        batch: the batch being scored.

    Returns:
        The loss.
    """
    return model._forward_loss(embeds, batch)


@torch.fx.wrap
def _fx_wrapped_generate(model: "PromptGenerativeQwen", batch: Batch) -> torch.Tensor:
    """Hide the decode loop from FX.

    ``PredictPipelineSparseDist`` FX-traces the model, and beam decode reads
    host ints and branches on them. Wrapping an inner helper only moves the
    failure to the next such read, so the whole loop is one leaf.

    Args:
        model: the model whose decode loop to run.
        batch: the batch to decode.

    Returns:
        The decoded local codes.
    """
    return model._generate(batch)
