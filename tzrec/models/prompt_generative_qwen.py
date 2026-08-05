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

"""Prompt-native generative recommendation over a Qwen backbone.

The model reads ``target_vocab``, the module plan and the decode bands from a
``CompiledPrompt``, and never reads the prompt's structure. Assembly happens in
the dataloader worker; what arrives here is already a packed token stream plus
the positions the projected slots must overwrite.
"""

from typing import Any, Dict, List, Optional

import torch
import torchmetrics
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM

from tzrec.datasets.utils import Batch
from tzrec.features.feature import BaseFeature
from tzrec.models.model import BaseModel
from tzrec.modules.dynamic_beam import dynamic_beam_search
from tzrec.modules.embedding import EmbeddingGroup
from tzrec.modules.prompt_projection import PromptProjection
from tzrec.prompt.assembler import (
    PROMPT_CU_SEQLENS as _PROMPT_CU_SEQLENS,
)
from tzrec.prompt.assembler import (
    PROMPT_HOLE_POSITIONS as _PROMPT_HOLE_POSITIONS,
)
from tzrec.prompt.assembler import (
    PROMPT_INPUT_IDS as _PROMPT_INPUT_IDS,
)
from tzrec.prompt.assembler import (
    PROMPT_LABELS as _PROMPT_LABELS,
)
from tzrec.prompt.assembler import (
    PROMPT_MAX_SEQLEN as _PROMPT_MAX_SEQLEN,
)
from tzrec.prompt.persist import save_prompt_assets
from tzrec.prompt.plan import CompiledPrompt, SlotSeg
from tzrec.protos.model_pb2 import ModelConfig
from tzrec.protos.models.prompt_model_pb2 import PromptModelConfig
from tzrec.utils.logging_util import logger

_PARAM_DTYPE: Dict[int, torch.dtype] = {
    PromptModelConfig.FP32: torch.float32,
    PromptModelConfig.BF16: torch.bfloat16,
    PromptModelConfig.FP16: torch.float16,
}


class PromptGenerativeQwen(BaseModel):
    """Qwen backbone driven by a compiled prompt.

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
        super().__init__(model_config, features, labels, sample_weights, **kwargs)
        if prompt is None:
            raise ValueError(
                f"{type(self).__name__} needs a compiled prompt; call "
                f"compile_prompt(pipeline_config.prompt_config, features) and "
                f"pass it to _create_model."
            )
        self._prompt = prompt
        cfg = self._model_config
        common = cfg.common

        self._ignore_index = int(common.ignore_index)
        self._generated_sids_key = common.generated_sids_key
        self._read_beam_config(common)

        self.lm = self._build_backbone(cfg.hf_model_id, common.param_dtype)
        self.lm.resize_token_embeddings(
            prompt.sid_space.target_vocab, mean_resizing=True
        )
        self.embedding_group = EmbeddingGroup(
            self._features, list(self._prompt.module_plan.feature_groups)
        )
        self._build_projections()

    def _read_beam_config(self, common: PromptModelConfig) -> None:
        """Parse the decode knobs; the schedule must match the codebook."""
        space = self._prompt.sid_space
        if space is None:
            raise ValueError(
                f"{type(self).__name__}: prompt_config declares no sid_space, "
                f"so there is nothing to decode."
            )
        self._num_return = int(common.num_return_sequences)
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
        if self._num_return > self._beam_widths[-1]:
            raise ValueError(
                f"{type(self).__name__}: num_return_sequences "
                f"({self._num_return}) must not exceed the final beam width "
                f"({self._beam_widths[-1]})."
            )

    def _build_backbone(self, hf_model_id: str, param_dtype: int) -> nn.Module:
        """Build the LM empty, so HF weights load only on cold start."""
        config = AutoConfig.from_pretrained(hf_model_id)
        model = AutoModelForCausalLM.from_config(config)
        return model.to(_PARAM_DTYPE[param_dtype])

    def _build_projections(self) -> None:
        """One module per resolved id, aligned with ``plan.projected_slots``.

        Slots sharing a ``projection_name`` share a module by reference, so
        they must agree on ``group_total_dim``.
        """
        plan = self._prompt.prompt_plan
        modules = self._prompt.module_plan
        hidden = int(self.lm.config.hidden_size)

        built: Dict[str, PromptProjection] = {}
        aligned: List[PromptProjection] = []
        for seg in plan.projected_slots:
            module_id = modules.slot_to_module[seg.slot_id]
            in_dim = self._slot_in_dim(seg)
            if module_id not in built:
                built[module_id] = PromptProjection(
                    modules.projections[module_id], in_dim, hidden
                )
            elif built[module_id].in_dim != in_dim:
                raise ValueError(
                    f"prompt slots sharing projection_name [{module_id}] have "
                    f"different group widths ({built[module_id].in_dim} vs "
                    f"{in_dim}); they cannot share a module."
                )
            aligned.append(built[module_id])
        self.projections = nn.ModuleDict(built)
        # zipped with plan.projected_slots; shared modules appear by reference
        self._slot_projections = aligned

    def _slot_in_dim(self, seg: SlotSeg) -> int:
        """Total group output width of a projected slot."""
        return self.embedding_group.group_total_dim(seg.name + seg.output_key)

    def hf_backbone(self) -> nn.Module:
        """The HF module export and checkpointing reach for."""
        return self.lm

    def predict(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """Teacher-forced forward over the assembled stream.

        Args:
            batch: carries the packed prompt in ``additional_infos``.

        Returns:
            The loss when training, the decoded SIDs otherwise.
        """
        if self.is_inference:
            return {self._generated_sids_key: _fx_wrapped_generate(self, batch)}
        return self._forward_loss(self._prompt_embeds(batch), batch)

    def _sid_token_bands(self) -> "tuple[torch.Tensor, torch.Tensor]":
        """Inclusive token-id band of every SID level, as device tensors."""
        space = self._prompt.sid_space
        device = self.lm.get_input_embeddings().weight.device
        return (
            torch.tensor(space.band_lo, device=device),
            torch.tensor(space.band_hi, device=device),
        )

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
            infos[_PROMPT_CU_SEQLENS],
            infos[_PROMPT_LABELS],
            int(infos[_PROMPT_MAX_SEQLEN]),
            self._ignore_index,
        )
        lo_tok, hi_tok = self._sid_token_bands()
        tokens = dynamic_beam_search(
            self.lm, padded, mask, self._beam_widths, lo_tok, hi_tok
        )
        return self._detokenize(tokens, padded.shape[0])

    def _detokenize(self, tokens: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Undo both shifts: token id back to a local 0-based code."""
        space = self._prompt.sid_space
        offsets = torch.tensor(space.level_offsets, device=tokens.device)
        codes = tokens - space.base_vocab - offsets
        codes = codes.view(batch_size, -1, space.num_levels)
        return codes[:, : self._num_return, :]

    def _prompt_embeds(self, batch: Batch) -> torch.Tensor:
        """Gather the token stream, then overwrite the projected positions."""
        ids = batch.additional_infos[_PROMPT_INPUT_IDS]
        embeds = self.lm.get_input_embeddings()(ids)

        plan = self._prompt.prompt_plan
        if not plan.projected_slots:
            return embeds

        grouped = self.embedding_group(batch)
        hidden = embeds.shape[-1]
        parts = [
            proj(grouped[seg.name + seg.output_key]).reshape(-1, hidden)
            for seg, proj in zip(plan.projected_slots, self._slot_projections)
        ]
        # out of place: embeds carries grad from the embedding lookup
        return embeds.index_copy(
            0, batch.additional_infos[_PROMPT_HOLE_POSITIONS], torch.cat(parts)
        )

    def _forward_loss(
        self, embeds: torch.Tensor, batch: Batch
    ) -> Dict[str, torch.Tensor]:
        """Run the LM over the assembled embeddings and score the response."""
        infos = batch.additional_infos
        padded, mask, labels = _unpack(
            embeds,
            infos[_PROMPT_CU_SEQLENS],
            infos[_PROMPT_LABELS],
            int(infos[_PROMPT_MAX_SEQLEN]),
            self._ignore_index,
        )
        outputs = self.lm.model(inputs_embeds=padded, attention_mask=mask)

        suffix = self._prompt.prompt_plan.suffix_keep
        window = slice(-suffix, None) if suffix else slice(None)
        logits = self.lm.lm_head(outputs.last_hidden_state[:, window, :])
        loss = self.lm.loss_function(
            logits=logits,
            labels=labels[:, window],
            vocab_size=self.lm.config.vocab_size,
            ignore_index=self._ignore_index,
        )
        return {"loss": loss}

    def init_loss(self) -> None:
        """No-op: the LM computes its own CE inside ``predict``."""
        return

    def loss(
        self, predictions: Dict[str, torch.Tensor], batch: Batch
    ) -> Dict[str, torch.Tensor]:
        """Surface the CE already computed in ``predict``.

        Args:
            predictions: what ``predict`` returned.
            batch: the batch, unused.

        Returns:
            The named loss.
        """
        return {"ce_loss": predictions["loss"]}

    def init_metric(self) -> None:
        """Register a mean-CE metric for the eval loop."""
        self._metric_modules["ce_loss"] = torchmetrics.MeanMetric()

    def update_metric(
        self,
        predictions: Dict[str, torch.Tensor],
        batch: Batch,
        losses: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        """Update the mean-CE metric with this batch's loss.

        Args:
            predictions: what ``predict`` returned.
            batch: the batch, unused.
            losses: the named losses, unused.
        """
        self._metric_modules["ce_loss"].update(predictions["loss"].detach())

    def update_train_metric(
        self, predictions: Dict[str, torch.Tensor], batch: Batch
    ) -> None:
        """No-op: nothing beyond the logged CE.

        Args:
            predictions: what ``predict`` returned.
            batch: the batch, unused.
        """
        return

    def save_assets(self, target_dir: str) -> None:
        """Co-locate the prompt contract, so the checkpoint is self-describing.

        Args:
            target_dir: the checkpoint or export directory.
        """
        save_prompt_assets(self._prompt, target_dir)

    def init_from_pretrained(self) -> None:
        """Load HF weights once, on a cold start only."""
        source = self._model_config.hf_model_id
        logger.info(f"loading pretrained weights from [{source}].")
        pretrained = AutoModelForCausalLM.from_pretrained(source)
        pretrained.resize_token_embeddings(
            self._prompt.sid_space.target_vocab, mean_resizing=True
        )
        self.lm.load_state_dict(pretrained.state_dict())
        del pretrained


def _unpack(
    embeds: torch.Tensor,
    cu_seqlens: torch.Tensor,
    labels: torch.Tensor,
    max_seqlen: int,
    ignore_index: int,
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
    """Pad a packed varlen batch at the LM boundary.

    Padding lives in this one adapter. ``max_seqlen`` is the collator's, not
    ``lengths.max()``: deriving it here would sync the device to the host every
    step, which §7.4 of the design forbids.
    """
    starts = cu_seqlens[:-1]
    lengths = cu_seqlens[1:] - starts
    batch_size = lengths.numel()
    hidden = embeds.shape[-1]

    columns = torch.arange(max_seqlen, device=embeds.device)
    mask = columns[None, :] < lengths[:, None]

    padded = embeds.new_zeros((batch_size, max_seqlen, hidden))
    out_labels = torch.full(
        (batch_size, max_seqlen),
        ignore_index,
        dtype=labels.dtype,
        device=embeds.device,
    )
    # mask selects row-major, which is the order embeds and labels are packed in
    padded[mask] = embeds
    out_labels[mask] = labels
    return padded, mask.long(), out_labels


@torch.fx.wrap
def _fx_wrapped_generate(model: "PromptGenerativeQwen", batch: Batch) -> torch.Tensor:
    """Hide the decode loop from FX.

    ``PredictPipelineSparseDist`` FX-traces the model, and beam decode reads
    host ints and branches on them. Wrapping an inner helper only moves the
    failure to the next such read, so the whole loop is one leaf.
    """
    return model._generate(batch)
