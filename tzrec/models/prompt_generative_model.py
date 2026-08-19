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

"""Shared causal-LM plumbing for prompt-native generative models.

This layer builds an empty causal LM, resizes its vocabulary, wires slot
projections, converts SID coordinate systems and implements checkpoint hooks.
A family subclass owns its forward and decode path.
"""

from typing import Any, Dict, List, Optional

import torch
import torchmetrics
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM

from tzrec.datasets.utils import Batch
from tzrec.features.feature import BaseFeature
from tzrec.models.model import BaseModel
from tzrec.modules.embedding import EmbeddingGroup
from tzrec.modules.prompt_projection import PromptProjection
from tzrec.prompt.assembler import (
    PROMPT_HOLE_POSITIONS,
    PROMPT_INPUT_IDS,
)
from tzrec.prompt.persist import save_prompt_assets
from tzrec.prompt.plan import CompiledPrompt
from tzrec.protos.model_pb2 import ModelConfig
from tzrec.protos.models.prompt_model_pb2 import PromptModelConfig
from tzrec.utils.logging_util import logger

_PARAM_DTYPE: Dict[int, torch.dtype] = {
    PromptModelConfig.FP32: torch.float32,
    PromptModelConfig.BF16: torch.bfloat16,
    PromptModelConfig.FP16: torch.float16,
}


class BasePromptGenerativeModel(BaseModel):
    """An HF backbone driven by a compiled prompt.

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
        super().__init__(model_config, features, labels, sample_weights, **kwargs)
        if compiled_prompt is None:
            raise ValueError(
                f"{type(self).__name__} needs a compiled prompt; call "
                f"compile_prompt(pipeline_config.prompt_config, features) and "
                f"pass it to _create_model."
            )
        if compiled_prompt.sid_space is None:
            raise ValueError(
                f"{type(self).__name__}: prompt_config declares no sid_space, "
                f"so there is no SID vocabulary to extend or decode."
            )
        self._prompt = compiled_prompt
        cfg = self._model_config

        self.lm = self._build_backbone(
            cfg.hf_model_name_or_path, cfg.common.lm_parameter_dtype
        )
        # Every run replaces this initialization from pretrained or DCP weights.
        self.lm.resize_token_embeddings(
            compiled_prompt.sid_space.target_vocab_size, mean_resizing=False
        )
        self.init_input()

        # decode subtracts these every step; a buffer follows the module's device
        self.register_buffer(
            "_level_offsets",
            torch.tensor(compiled_prompt.sid_space.level_offsets),
            persistent=False,
        )

    def init_input(self) -> None:
        """Build the projected-slot embedding groups and projection modules."""
        self.embedding_group = EmbeddingGroup(
            self._features, list(self._prompt.projection_plan.feature_groups)
        )
        self._build_projections()

    def _build_backbone(
        self, hf_model_name_or_path: str, lm_parameter_dtype: int
    ) -> nn.Module:
        """Build the LM from config, so HF weights load only on cold start.

        Args:
            hf_model_name_or_path: hub id or local directory naming the
                architecture and cold-start weights.
            lm_parameter_dtype: dtype of the LM parameters.

        Returns:
            A randomly initialized backbone with the requested parameter dtype.
        """
        config = AutoConfig.from_pretrained(hf_model_name_or_path)
        model = AutoModelForCausalLM.from_config(config)
        return model.to(_PARAM_DTYPE[lm_parameter_dtype])

    def _build_projections(self) -> None:
        """One module per resolved id, aligned with ``prompt_plan.projected_slots``.

        Slots sharing a ``projection_name`` share a module by reference, so
        they must agree on ``group_total_dim``.
        """
        prompt_plan = self._prompt.prompt_plan
        projection_plan = self._prompt.projection_plan
        hidden_size = int(self.lm.config.hidden_size)

        modules_by_id: Dict[str, PromptProjection] = {}
        in_dims: Dict[str, int] = {}
        aligned_modules: List[PromptProjection] = []
        for seg in prompt_plan.projected_slots:
            module_id = projection_plan.slot_to_module[seg.slot_id]
            in_dim = self.embedding_group.group_total_dim(seg.name + seg.output_key)
            if module_id not in modules_by_id:
                modules_by_id[module_id] = PromptProjection(
                    projection_plan.projections[module_id], in_dim, hidden_size
                )
                in_dims[module_id] = in_dim
            elif in_dims[module_id] != in_dim:
                raise ValueError(
                    f"prompt slots sharing projection_name [{module_id}] have "
                    f"different group dims ({in_dims[module_id]} vs "
                    f"{in_dim}); they cannot share a module."
                )
            aligned_modules.append(modules_by_id[module_id])
        self.projections = nn.ModuleDict(modules_by_id)
        self._slot_projections = aligned_modules

    def hf_backbone(self) -> nn.Module:
        """The HF module export and checkpointing reach for."""
        return self.lm

    def build_input(self, batch: Batch) -> torch.Tensor:
        """Build packed LM input embeddings and fill projected positions.

        Args:
            batch: carries the packed prompt in ``additional_infos``.

        Returns:
            ``(total_tokens, hidden_size)``.
        """
        ids = batch.additional_infos[PROMPT_INPUT_IDS]
        embeds = self.lm.get_input_embeddings()(ids)

        prompt_plan = self._prompt.prompt_plan
        if not prompt_plan.projected_slots:
            return embeds

        grouped = self.embedding_group(batch)
        hidden_size = embeds.shape[-1]
        projected_embeddings = [
            proj(grouped[seg.name + seg.output_key]).reshape(-1, hidden_size)
            for seg, proj in zip(prompt_plan.projected_slots, self._slot_projections)
        ]
        # The assembler records holes in this projected-occurrence-major order.
        # out of place: embeds carries grad from the embedding lookup
        return embeds.index_copy(
            0,
            batch.additional_infos[PROMPT_HOLE_POSITIONS],
            torch.cat(projected_embeddings).to(embeds.dtype),
        )

    def _tokens_to_local_codes(
        self, tokens: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        """Undo both shifts: token id back to a local 0-based code.

        Args:
            tokens: generated token ids, ``(batch_size * beams, num_levels)``.
            batch_size: rows in the batch.

        Returns:
            ``(batch_size, beams, num_levels)`` local codes.
        """
        space = self._prompt.sid_space
        codes = tokens - space.base_vocab_size - self._level_offsets
        return codes.view(batch_size, -1, space.num_levels)

    def init_loss(self) -> None:
        """No-op: an LM computes its own CE inside ``predict``."""
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
        source = self._model_config.hf_model_name_or_path
        logger.info(f"loading pretrained weights from [{source}].")
        pretrained = AutoModelForCausalLM.from_pretrained(source)
        pretrained.resize_token_embeddings(
            self._prompt.sid_space.target_vocab_size, mean_resizing=True
        )
        self.lm.load_state_dict(pretrained.state_dict())
        del pretrained
