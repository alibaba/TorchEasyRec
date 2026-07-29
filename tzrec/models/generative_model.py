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

"""Architecture-agnostic base for HF-backed generative-recommendation LMs.

A family subclass (e.g. ``GenerativeQwen``) supplies ``_build_prompt_tokens`` and
``predict``; ``GenerativeModelConfig`` holds the shared config and the sample
contract.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torchmetrics
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from tzrec.datasets.utils import Batch
from tzrec.features.feature import BaseFeature
from tzrec.features.sid_feature import SidFeature
from tzrec.models.model import BaseModel
from tzrec.modules.embedding import EmbeddingGroup
from tzrec.protos import model_pb2
from tzrec.protos.model_pb2 import ModelConfig
from tzrec.protos.models import generative_model_pb2


class BaseGenerativeModel(BaseModel):
    """Model construction, SID vocab extension, data-prep, loss and metrics."""

    # flat width used when beam_widths is empty; a family may override it
    _default_beam_width = 50

    _PARAM_DTYPE: Dict[int, torch.dtype] = {
        generative_model_pb2.FP32: torch.float32,
        generative_model_pb2.BF16: torch.bfloat16,
        generative_model_pb2.FP16: torch.float16,
    }

    def __init__(
        self,
        model_config: ModelConfig,
        features: List[BaseFeature],
        labels: List[str],
        sample_weights: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_config, features, labels, sample_weights, **kwargs)
        cfg = self._model_config
        sid_atoms = self._read_common_config(cfg.common)

        self.lm = self._build_backbone()
        tokenizer, base = self._build_extended_tokenizer(sid_atoms)
        self._hf_tokenizer = tokenizer
        self._base_vocab = base
        self._pad_token_id = self._resolve_pad_token_id(tokenizer)

        self._build_prompt_tokens(tokenizer, cfg)
        self.init_input()

    @staticmethod
    def _resolve_pad_token_id(tokenizer: PreTrainedTokenizerBase) -> int:
        """Pad id for the left-padded splice, falling back to eos."""
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        if pad_id is None:
            raise ValueError(
                "BaseGenerativeModel: tokenizer has neither pad_token_id nor "
                "eos_token_id; cannot choose a pad id for the left-padded splice."
            )
        return int(pad_id)

    def _read_common_config(
        self, common: generative_model_pb2.GenerativeModelConfig
    ) -> int:
        """Parse shared proto knobs into attributes; return the SID atom count."""
        self._label_name: str = self._labels[0] if self._labels else ""
        self._ignore_index: int = int(common.ignore_index)
        self._generated_sids_key: str = common.generated_sids_key
        self._param_dtype: torch.dtype = self._PARAM_DTYPE[common.param_dtype]
        self._max_seq_length: int = int(common.max_sequence_length)
        codebook = self._shared_sid_space()
        self._num_levels = len(codebook)
        if 0 < self._max_seq_length < self._num_levels:
            raise ValueError(
                f"{type(self).__name__}: max_sequence_length "
                f"({self._max_seq_length}) cannot hold one {self._num_levels}"
                f"-level item; use 0 to disable the budget or a multiple of "
                f"{self._num_levels}."
            )
        sizes = torch.tensor(codebook, dtype=torch.long)
        self.register_buffer("_codebook_sizes", sizes, persistent=False)
        # only the decode path needs these; SidFeature folds them into inputs.
        self.register_buffer(
            "_level_offsets", torch.cumsum(sizes, 0) - sizes, persistent=False
        )
        self._vocab_pad_mult = int(common.vocab_pad_to_multiple_of)
        self._read_beam_config(common)
        return sum(codebook)

    def _read_beam_config(
        self, common: generative_model_pb2.GenerativeModelConfig
    ) -> None:
        """Parse the decode knobs; the width schedule must match the codebook."""
        self._num_return = int(common.num_return_sequences)
        self._beam_widths: List[int] = (
            list(common.beam_widths) or [self._default_beam_width] * self._num_levels
        )
        if len(self._beam_widths) != self._num_levels:
            raise ValueError(
                f"{type(self).__name__}: beam_widths has "
                f"{len(self._beam_widths)} entries but the codebook has "
                f"{self._num_levels} levels; give one width per level."
            )
        if self._num_return > self._beam_widths[-1]:
            raise ValueError(
                f"{type(self).__name__}: num_return_sequences "
                f"({self._num_return}) must not exceed the final beam width "
                f"({self._beam_widths[-1]})."
            )

    def _shared_sid_space(self) -> List[int]:
        """The one codebook every SID feature declares."""
        spaces = {
            f.name: tuple(f.codebook)
            for f in self._features
            if isinstance(f, SidFeature)
        }
        if not spaces:
            raise ValueError(
                f"{type(self).__name__}: no SID feature declares a codebook; "
                f"genrec needs at least one sequence_sid_feature."
            )
        if len(set(spaces.values())) != 1:
            raise ValueError(
                f"{type(self).__name__}: all SID features must share one "
                f"codebook, got {dict(sorted(spaces.items()))}."
            )
        return list(next(iter(spaces.values())))

    def _slot_group_names(self) -> Dict[str, str]:
        """{feature_name: group_name} for every declared feature_group.

        One JAGGED_SEQUENCE feature per group: EmbeddingGroup interleaves a
        group's members into one sequence that cannot be split back apart.
        """
        by_feature: Dict[str, str] = {}
        for group in self._feature_groups:
            if group.group_type != model_pb2.JAGGED_SEQUENCE:
                raise ValueError(
                    f"{type(self).__name__}: feature_group {group.group_name!r} "
                    f"must be JAGGED_SEQUENCE, got "
                    f"{model_pb2.FeatureGroupType.Name(group.group_type)}."
                )
            if len(group.feature_names) != 1:
                raise ValueError(
                    f"{type(self).__name__}: feature_group {group.group_name!r} "
                    f"must hold exactly one feature (its members are interleaved "
                    f"into one sequence), got {list(group.feature_names)}."
                )
            name = group.feature_names[0]
            if name in by_feature:
                raise ValueError(
                    f"{type(self).__name__}: feature {name!r} is claimed by both "
                    f"feature_group {by_feature[name]!r} and "
                    f"{group.group_name!r}; only one can fill its prompt slot."
                )
            by_feature[name] = group.group_name
        return by_feature

    def _resolve_prompt_slots(
        self, template: str
    ) -> Tuple[List[str], List["SidFeature"]]:
        """Split a ``{{feature_name}}`` template into N+1 gaps and N features.

        Records ``_slot_names`` / ``_slot_groups`` here rather than in the family
        hook, because ``build_input`` reads them.
        """
        parts = re.split(r"\{\{(\w+)\}\}", template)
        gaps, names = parts[0::2], parts[1::2]
        if not names:
            raise ValueError(
                f"{type(self).__name__}: prompt_template needs at least one "
                f"{{{{feature_name}}}} slot naming a declared feature."
            )
        group_of = self._slot_group_names()
        by_name = {f.name: f for f in self._features}
        features = []
        for name in names:
            if name not in group_of:
                raise ValueError(
                    f"{type(self).__name__}: prompt_template slot {{{{{name}}}}} "
                    f"names no feature_group; declared: {sorted(group_of)}."
                )
            feature = by_name.get(name)
            if feature is None:
                raise ValueError(
                    f"{type(self).__name__}: prompt_template slot {{{{{name}}}}} "
                    f"has no feature_config."
                )
            if not isinstance(feature, SidFeature):
                raise ValueError(
                    f"{type(self).__name__}: prompt slot {{{{{name}}}}} names a "
                    f"{type(feature).__name__}; only a SidFeature can fill a slot."
                )
            features.append(feature)
        unused = sorted(set(group_of) - set(names))
        if unused:
            raise ValueError(
                f"{type(self).__name__}: feature_group(s) {unused} are declared "
                f"but never referenced by a prompt_template slot."
            )
        self._slot_names = names
        self._slot_groups = [group_of[n] for n in names]
        return gaps, features

    def _build_backbone(self) -> PreTrainedModel:
        """Build the EMPTY architecture; weights arrive later from HF or DCP."""
        hf_model_id = self._model_config.hf_model_id
        if not hf_model_id:
            raise ValueError(f"{type(self).__name__}: empty hf_model_id.")
        hf_cfg = AutoConfig.from_pretrained(hf_model_id)
        lm = AutoModelForCausalLM.from_config(hf_cfg, torch_dtype=self._param_dtype)
        # no-op when from_config already honoured torch_dtype; not all do.
        return lm.to(self._param_dtype)

    def _build_extended_tokenizer(
        self, sid_atoms: int
    ) -> Tuple[PreTrainedTokenizerBase, int]:
        """Add the SID atoms ``C0..C{sid_atoms-1}`` and resize ``self.lm``.

        Returns ``(tokenizer, base)`` where ``base`` is the tokenizer's next free
        id BEFORE adding the atoms -- use ``len(tokenizer)``, NOT
        ``config.vocab_size`` (which counts reserved slots).
        """
        tokenizer = AutoTokenizer.from_pretrained(
            self._model_config.hf_model_id, use_fast=True
        )
        base = len(tokenizer)
        added = tokenizer.add_tokens([f"C{i}" for i in range(sid_atoms)])
        if added != sid_atoms:
            raise RuntimeError(
                f"BaseGenerativeModel: tokenizer was expected to grow by "
                f"{sid_atoms} new atoms, only added {added}. "
                f"Aborting to avoid silent SID-token mismatch."
            )
        # stash the resize target so init_from_pretrained re-extends identically.
        self._target_vocab = base + sid_atoms
        self.lm.resize_token_embeddings(
            self._target_vocab, pad_to_multiple_of=self._vocab_pad_mult or None
        )
        c0_id = tokenizer.convert_tokens_to_ids("C0")
        if c0_id != base:
            raise RuntimeError(
                f"BaseGenerativeModel: SID atom layout mismatch -- expected "
                f"C0 at token id {base}, got {c0_id}. "
                f"Splice arithmetic would produce wrong token ids."
            )
        return tokenizer, base

    def init_from_pretrained(self) -> None:
        """Load the pretrained HF weights and re-extend to ``__init__``'s vocab.

        The new SID rows differ per rank; DDP's ``_sync_module_states``
        broadcast from rank 0 reconciles them.
        """
        # drop the empty arch first: holding both peaks at 2x model host RAM.
        self.lm = None
        lm = AutoModelForCausalLM.from_pretrained(
            self._model_config.hf_model_id,
            torch_dtype=self._param_dtype,
            low_cpu_mem_usage=True,
        )
        lm.resize_token_embeddings(
            self._target_vocab, pad_to_multiple_of=self._vocab_pad_mult or None
        )
        self.lm = lm

    def hf_backbone(self) -> PreTrainedModel:
        """The HF backbone module, for checkpoint/export asset writing."""
        return self.lm

    def hf_tokenizer(self) -> PreTrainedTokenizerBase:
        """The extended tokenizer (base vocab + C0..C{sum-1}) to serialize."""
        return self._hf_tokenizer

    def _build_prompt_tokens(
        self, tokenizer: PreTrainedTokenizerBase, cfg: Any
    ) -> None:
        """Family hook: cache the tokenised prompt template as buffers.

        Called from ``__init__`` after vocab extension; consumed by ``predict``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _build_prompt_tokens "
            f"(BaseGenerativeModel is abstract)."
        )

    @property
    def device(self) -> torch.device:
        """Device the HF backbone runs on."""
        return self.lm.device

    def _tokenize_sids(self, flat: torch.Tensor) -> torch.Tensor:
        """Map flat SID indices to extended-vocab token ids."""
        return flat + self._base_vocab

    def _detokenize_sids(
        self, tokens: torch.Tensor, level_ids: torch.Tensor
    ) -> torch.Tensor:
        """Inverse of ``_tokenize_sids``: token ids to local 0-based codes."""
        return tokens - self._base_vocab - self._level_offsets[level_ids]

    def _sid_token_bands(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the inclusive token-id band for every SID level."""
        return (
            self._tokenize_sids(self._level_offsets),
            self._tokenize_sids(self._level_offsets + self._codebook_sizes - 1),
        )

    def _validate_sid_candidates(
        self, new_tokens: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        """Decode the per-beam tail ``(B*C, w)`` to ``(B, C, num_levels)`` codes.

        Any malformed candidate becomes all ``-1``, which no real code matches.
        """
        level_ids = torch.arange(new_tokens.shape[1], device=new_tokens.device)
        codes = self._detokenize_sids(new_tokens, level_ids)
        codes = F.pad(codes, (0, self._num_levels - codes.shape[1]), value=-1)
        invalid = ((codes < 0) | (codes >= self._codebook_sizes)).any(dim=1)
        codes = codes.masked_fill(invalid.unsqueeze(1), -1)
        # decoders return rows batch-major; group per user.
        return codes.view(batch_size, -1, self._num_levels)

    def init_input(self) -> None:
        """Build the EmbeddingGroup; passthrough features hold no params."""
        self.embedding_group = EmbeddingGroup(self._features, self._feature_groups)

    def build_input(self, batch: Batch) -> Dict[str, List[torch.Tensor]]:
        """Retrieve per-row SID token sequences, keyed by feature name.

        The answer is a label_field so it can be absent at inference.
        """
        g = self.embedding_group(batch)
        rows: Dict[str, List[torch.Tensor]] = {
            name: self._sid_token_rows(
                g[f"{group}.sequence"],
                g[f"{group}.sequence_length"],
                max_codes=self._max_seq_length,
            )
            for name, group in zip(self._slot_names, self._slot_groups)
        }
        if not self.is_inference:
            if not self._label_name:
                raise ValueError(
                    f"{type(self).__name__}: training needs the answer SIDs; "
                    f"declare it as the first data_config.label_field."
                )
            jt = batch.jagged_labels[self._label_name]
            rows[self._label_name] = self._answer_token_rows(jt.values(), jt.lengths())
        return rows

    def _sid_token_rows(
        self,
        values: torch.Tensor,
        lengths: torch.Tensor,
        max_codes: Optional[int] = None,
    ) -> List[torch.Tensor]:
        """Map a feature's flat SID stream to per-row token-id tensors.

        ``max_codes`` caps each row to its most-recent WHOLE items.
        """
        values = values.reshape(-1)  # value_dim 1 arrives as (N,) or (N, 1)
        sizes = lengths.long().tolist()
        # TODO(shuqi): move truncation into FG once FG can keep the TAIL, not the HEAD.
        if max_codes:
            keep = (max_codes // self._num_levels) * self._num_levels
            if keep and any(n > keep for n in sizes):
                rows = torch.split(values, sizes)
                values = torch.cat([r[-keep:] for r in rows])
                sizes = [min(n, keep) for n in sizes]
        tokens = self._tokenize_sids(values.to(self.device).long())
        return list(torch.split(tokens, sizes))

    def _answer_token_rows(
        self, values: torch.Tensor, lengths: torch.Tensor
    ) -> List[torch.Tensor]:
        """Map the answer label to token ids; every row is ``num_levels`` codes.

        A label_field is not offset by SidFeature, so the fold-in happens here.
        """
        values = values.reshape(-1)  # value_dim 1 arrives as (N,) or (N, 1)
        sizes = lengths.long().tolist()
        bad = [i for i, n in enumerate(sizes) if n != self._num_levels]
        if bad:
            raise ValueError(
                f"{type(self).__name__}: each answer must be "
                f"{self._num_levels} codes (len(codebook)); rows {bad} have "
                f"{[sizes[i] for i in bad]} -- anomalous sample(s)."
            )
        codes = values.to(self.device).long()
        level_ids = torch.arange(codes.numel(), device=self.device) % self._num_levels
        invalid = (codes < 0) | (codes >= self._codebook_sizes[level_ids])
        if invalid.any():
            raise ValueError(
                f"{type(self).__name__}: answer SID codes must be local 0-based "
                f"values in [0, codebook[level])."
            )
        tokens = self._tokenize_sids(codes + self._level_offsets[level_ids])
        return list(torch.split(tokens, sizes))

    def init_loss(self) -> None:
        """No-op: the loss is computed inside ``predict`` (HF loss_function)."""
        return

    def loss(
        self,
        predictions: Dict[str, torch.Tensor],
        batch: Batch,
    ) -> Dict[str, torch.Tensor]:
        """Surface the CE loss already computed in ``predict``."""
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
        """Update the mean-CE metric with this batch's loss."""
        self._metric_modules["ce_loss"].update(predictions["loss"].detach())

    # NOTE: BaseModel declares no such hook, but the train loop calls it.
    def update_train_metric(
        self,
        predictions: Dict[str, torch.Tensor],
        batch: Batch,
    ) -> None:
        """No-op: no train-time metric beyond the logged CE loss."""
        return
