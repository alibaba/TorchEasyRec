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

"""Shared causal-LM plumbing for generative recommendation models.

This layer builds an empty causal LM, resizes its vocabulary, wires slot
projections, converts SID coordinate systems, scores the response window and
supplies the digests a checkpoint records. A family subclass owns its forward
and decode path.

``GenrecFrontEnd`` is the half of the model tzrec serves: the assembled prompt
and the projected slots, everything before the LM's embedding gather. It is
exported like any tzrec model; the LM itself is handed to an LLM engine as the
HuggingFace weights beside it.
"""

import inspect
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torchmetrics
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM

from tzrec.acc import utils as acc_utils
from tzrec.datasets.utils import Batch
from tzrec.features.feature import BaseFeature
from tzrec.models.model import BaseModel
from tzrec.modules.embedding import EmbeddingGroup
from tzrec.modules.prompt_projection import PromptProjection
from tzrec.prompt.assembler import (
    CU_SEQLENS,
    HOLE_KEYS,
    HOLE_POSITIONS,
    HOLE_SLOT_COUNTS,
    INPUT_IDS,
    PROMPT_CU_SEQLENS,
    PROMPT_HOLE_KEYS,
    PROMPT_HOLE_POSITIONS,
    PROMPT_HOLE_SLOT_COUNTS,
    PROMPT_INPUT_IDS,
)
from tzrec.prompt.compile import compile_prompt
from tzrec.prompt.persist import PROMPT_DIR, TOKENIZER_DIR, write_serving_contract
from tzrec.prompt.types import CompiledPrompt, PromptPlan
from tzrec.protos.model_pb2 import FeatureGroupConfig, ModelConfig
from tzrec.protos.models.genrec_model_pb2 import GenrecModelConfig
from tzrec.protos.pipeline_pb2 import EasyRecConfig
from tzrec.utils import config_util, env_util
from tzrec.utils.hf_export_util import dcp_to_hf, write_composite_config
from tzrec.utils.logging_util import logger

SLOT_EMBEDS = "slot_embeds"

_PARAM_DTYPE: Dict[int, torch.dtype] = {
    GenrecModelConfig.FP32: torch.float32,
    GenrecModelConfig.BF16: torch.bfloat16,
    GenrecModelConfig.FP16: torch.float16,
}

_REQUIRED_LM_ATTRS: Tuple[str, ...] = (
    "loss_function",
    "get_input_embeddings",
    "resize_token_embeddings",
)


class BaseGenrecModel(BaseModel):
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
        if compiled_prompt.prompt_plan.logits_suffix_len is None:
            raise ValueError(
                f"{type(self).__name__}: the response is unbounded, so the "
                f"supervised window cannot be sized and predict would retain a "
                f"full (batch, length, vocab) logits tensor."
            )
        self._prompt = compiled_prompt
        cfg = self._model_config

        self._ignore_index = int(cfg.common.ignore_index)
        self.lm: nn.Module
        self.init_backbone(cfg.hf_model_name_or_path, cfg.common.lm_parameter_dtype)
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
        self.init_projections()

    def init_backbone(
        self, hf_model_name_or_path: str, lm_parameter_dtype: int
    ) -> None:
        """Assign ``self.lm`` from config, so HF weights load only on cold start.

        Args:
            hf_model_name_or_path: hub id or local directory naming the
                architecture and cold-start weights.
            lm_parameter_dtype: dtype of the LM parameters.
        """
        config = AutoConfig.from_pretrained(hf_model_name_or_path)
        model = AutoModelForCausalLM.from_config(config)
        self.lm = model.to(_PARAM_DTYPE[lm_parameter_dtype])
        self._check_backbone_interfaces(hf_model_name_or_path)

    def _check_backbone_interfaces(self, hf_model_name_or_path: str) -> None:
        """Reject a backbone this model cannot drive.

        Args:
            hf_model_name_or_path: what named the architecture, for the message.

        Raises:
            ValueError: the backbone lacks an interface the forward or the
                banded decode needs.
        """
        missing = [name for name in _REQUIRED_LM_ATTRS if not hasattr(self.lm, name)]
        for name in ("vocab_size", "hidden_size"):
            if not hasattr(self.lm.config, name):
                missing.append(f"config.{name}")
        if "logits_to_keep" not in inspect.signature(type(self.lm).forward).parameters:
            missing.append("forward(logits_to_keep)")
        if missing:
            raise ValueError(
                f"{type(self).__name__}: {hf_model_name_or_path} builds "
                f"{type(self.lm).__name__}, which is missing {sorted(missing)}."
            )

    def init_projections(self) -> None:
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

    @property
    def compiled_prompt(self) -> CompiledPrompt:
        """The prompt this model was built against."""
        return self._prompt

    def build_input(self, batch: Batch) -> torch.Tensor:
        """Build packed LM input embeddings and fill projected positions.

        Args:
            batch: carries the packed prompt in ``additional_infos``.

        Returns:
            ``(total_tokens, hidden_size)``.
        """
        ids = batch.additional_infos[PROMPT_INPUT_IDS]
        embeds = self.lm.get_input_embeddings()(ids)
        if not self._prompt.prompt_plan.projected_slots:
            return embeds
        projected = project_slots(
            self.embedding_group,
            self._prompt.prompt_plan,
            self._slot_projections,
            batch,
            embeds.shape[-1],
        )
        # out of place: embeds carries grad from the embedding lookup
        return embeds.index_copy(
            0, batch.additional_infos[PROMPT_HOLE_POSITIONS], projected.to(embeds.dtype)
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
        """No-op: the backbone owns the causal-LM loss."""
        return

    def loss(
        self, predictions: Dict[str, torch.Tensor], batch: Batch
    ) -> Dict[str, torch.Tensor]:
        """Score the response window with the backbone's own causal-LM loss.

        Args:
            predictions: the response-window logits and labels.
            batch: the batch, unused.

        Returns:
            The named loss.
        """
        return {
            "ce_loss": self.lm.loss_function(
                logits=predictions["logits"],
                labels=predictions["labels"],
                vocab_size=self.lm.config.vocab_size,
                ignore_index=self._ignore_index,
            )
        }

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
            predictions: what ``predict`` returned, unused.
            batch: the batch, unused.
            losses: the named losses the eval loop already computed.
        """
        if losses is not None:
            self._metric_modules["ce_loss"].update(losses["ce_loss"].detach())

    def update_train_metric(
        self, predictions: Dict[str, torch.Tensor], batch: Batch
    ) -> None:
        """No-op: nothing beyond the logged CE.

        Args:
            predictions: what ``predict`` returned.
            batch: the batch, unused.
        """
        return

    def prompt_digests(self) -> Dict[str, str]:
        """The contract digests the checkpoint records, for restore checking."""
        return {
            "vocab_hash": self._prompt.vocab_hash,
            "plan_hash": self._prompt.plan_hash,
        }

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


def project_slots(
    embedding_group: EmbeddingGroup,
    prompt_plan: PromptPlan,
    slot_projections: Sequence[nn.Module],
    batch: Batch,
    hidden_size: int,
) -> torch.Tensor:
    """Look every projected slot up and project it into the LM input space.

    Args:
        embedding_group: the model's prompt groups.
        prompt_plan: fixes the slot order.
        slot_projections: one module per projected slot, in the same order.
        batch: the batch to look up.
        hidden_size: the LM hidden size.

    Returns:
        ``(total_holes, hidden_size)`` in the order the assembler records holes:
        projected occurrence first, then sample.
    """
    grouped = embedding_group(batch)
    parts = [
        proj(grouped[seg.name + seg.output_key]).reshape(-1, hidden_size)
        for seg, proj in zip(prompt_plan.projected_slots, slot_projections)
    ]
    return parts[0] if len(parts) == 1 else torch.cat(parts)


class GenrecFrontEnd(nn.Module):
    """The served half of a genrec model, exported like any tzrec model.

    It shares the model's embedding group and projections, so under the
    inference wrapper their state-dict names are the checkpoint's and the LM is
    never loaded at export. ``predict`` returns the assembled prompt and the
    projected slot embeddings; an LLM engine gathers the LM's own table, scatters
    ``slot_embeds`` at ``hole_positions`` and decodes.

    Args:
        model: the genrec model to serve.
    """

    def __init__(self, model: BaseGenrecModel) -> None:
        super().__init__()
        if acc_utils.is_aot() or acc_utils.is_trt() or env_util.use_rtp():
            raise ValueError(
                "the genrec front-end is exported with TorchScript only: its "
                "prompt walk has data-dependent shapes, which AOT, TRT and RTP "
                "export cannot capture. Unset ENABLE_AOT / ENABLE_TRT / USE_RTP."
            )
        self.embedding_group = model.embedding_group
        self.projections = model.projections
        self._slot_projections = list(model._slot_projections)
        self._prompt = model.compiled_prompt
        self._features = list(model.features)
        self._hidden_size = int(model.lm.config.hidden_size)

    @property
    def features(self) -> List[BaseFeature]:
        """The features the served prompt reads."""
        return self._features

    @property
    def feature_groups(self) -> List[FeatureGroupConfig]:
        """The groups derived for the projected slots."""
        return list(self._prompt.projection_plan.feature_groups)

    @property
    def compiled_prompt(self) -> CompiledPrompt:
        """The prompt the inference wrapper assembles before ``predict``."""
        return self._prompt

    def predict(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """Return the assembled streams and the projected slot embeddings.

        Args:
            batch: carries the assembled prompt in ``additional_infos``.

        Returns:
            The serving contract's outputs.
        """
        infos = batch.additional_infos
        out = {
            INPUT_IDS: infos[PROMPT_INPUT_IDS],
            CU_SEQLENS: infos[PROMPT_CU_SEQLENS],
            HOLE_POSITIONS: infos[PROMPT_HOLE_POSITIONS],
            HOLE_KEYS: infos[PROMPT_HOLE_KEYS],
            HOLE_SLOT_COUNTS: infos[PROMPT_HOLE_SLOT_COUNTS],
        }
        if self._prompt.prompt_plan.projected_slots:
            out[SLOT_EMBEDS] = project_slots(
                self.embedding_group,
                self._prompt.prompt_plan,
                self._slot_projections,
                batch,
                self._hidden_size,
            )
        else:
            out[SLOT_EMBEDS] = torch.zeros(
                0, 0, dtype=torch.float32, device=infos[PROMPT_INPUT_IDS].device
            )
        return out

    def export_assets(
        self, pipeline_config: EasyRecConfig, checkpoint_path: str, save_dir: str
    ) -> None:
        """Write what an LLM engine reads beside the scripted front-end.

        The HuggingFace weights and composite config, the extended tokenizer and
        the SID space. Called by the export on rank 0, inside its save dir.

        Args:
            pipeline_config: the pipeline being exported.
            checkpoint_path: the checkpoint the weights come from.
            save_dir: the export directory.
        """
        if config_util.use_dense_ema(
            pipeline_config.export_config, pipeline_config.train_config
        ):
            raise ValueError(
                "HF export: dcp_to_hf reads <checkpoint>/model, so it cannot "
                "serve Dense EMA parameters. Set export_config.use_dense_ema to "
                "false to export the raw weights."
            )
        dcp_to_hf(checkpoint_path, save_dir)
        write_composite_config(save_dir)
        compile_prompt(
            pipeline_config.prompt_config,
            self._features,
            list(pipeline_config.data_config.label_fields),
            tokenizer_dir=os.path.join(save_dir, PROMPT_DIR, TOKENIZER_DIR),
        )
        write_serving_contract(self._prompt, save_dir)
