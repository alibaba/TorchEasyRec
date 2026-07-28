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
    """Model construction, SID vocab extension, data-prep, loss and metrics.

    The family's proto message must carry ``common`` (a
    ``GenerativeModelConfig``) and ``hf_model_id``; this base reads both.
    """

    # See `common.param_dtype` in the proto for why FP32 is the default.
    # The enum is closed, so protobuf rejects anything not listed here.
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
        # the budget truncates to WHOLE items, so anything under one item's width
        # would floor to zero and silently leave the history uncapped.
        if 0 < self._max_seq_length < self._num_levels:
            raise ValueError(
                f"{type(self).__name__}: max_sequence_length "
                f"({self._max_seq_length}) cannot hold one {self._num_levels}"
                f"-level item; use 0 to disable the budget or a multiple of "
                f"{self._num_levels}."
            )
        sizes = torch.tensor(codebook, dtype=torch.long)
        self.register_buffer("_codebook_sizes", sizes, persistent=False)
        # only the decode path still needs per-level offsets: SidFeature folds
        # them into the input stream, but generated tokens must be split back
        # into per-level codes and no feature is involved in generation.
        self.register_buffer(
            "_level_offsets", torch.cumsum(sizes, 0) - sizes, persistent=False
        )
        self._vocab_pad_mult = int(common.vocab_pad_to_multiple_of)
        return sum(codebook)

    def _shared_sid_space(self) -> List[int]:
        """The one codebook every SID feature declares.

        One extended vocabulary and one answer width, so adding a feature must
        never resize ``lm_head``; disagreement is a typo, not a reshape.
        """
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
        group's members into one ``{group}.sequence``, which no longer splits
        back into separate prompt slots. The map is keyed by feature, so a
        feature claimed by two groups is rejected rather than silently
        resolving to whichever group came last.
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

        Slots and declared feature_groups must correspond exactly, and each slot
        must name a ``SidFeature``, so a misspelt or unused feature fails here
        instead of vanishing from the prompt. ``_slot_names`` / ``_slot_groups``
        are recorded here, not in the family hook, because ``build_input`` reads
        them and would otherwise fail far from the cause.
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
        """Build the EMPTY architecture -- shapes only, no weight download.

        Weights arrive from ``init_from_pretrained`` (cold start) or DCP.
        """
        hf_model_id = self._model_config.hf_model_id
        if not hf_model_id:
            raise ValueError(f"{type(self).__name__}: empty hf_model_id.")
        hf_cfg = AutoConfig.from_pretrained(hf_model_id)
        lm = AutoModelForCausalLM.from_config(hf_cfg, torch_dtype=self._param_dtype)
        # a no-op when from_config already honoured torch_dtype, which not every
        # architecture does.
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
            # a pre-existing Cxxx token would shift the atoms off `base`.
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

        Every rank resizes identically or DDP's shape check fails. The new SID
        rows come from the global RNG and so differ per rank; DDP's
        ``_sync_module_states`` broadcast from rank 0 is what reconciles them.
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
        """Map flat SID indices to extended-vocab token ids.

        ``SidFeature`` folded the offsets in and codes are 0-based, so the flat
        index IS the atom index; the model only owns the vocabulary shift.
        """
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

        ``w`` may be < ``num_levels`` when beams stop early. Any malformed
        candidate (early EOS, non-SID or wrong-level atom) becomes all ``-1``,
        which no real 0-based code can match.
        """
        level_ids = torch.arange(new_tokens.shape[1], device=new_tokens.device)
        codes = self._detokenize_sids(new_tokens, level_ids)
        codes = F.pad(codes, (0, self._num_levels - codes.shape[1]), value=-1)
        invalid = ((codes < 0) | (codes >= self._codebook_sizes)).any(dim=1)
        codes = codes.masked_fill(invalid.unsqueeze(1), -1)
        # decoders return rows batch-major ([b0_c0, b0_c1, ...]); group per user.
        return codes.view(batch_size, -1, self._num_levels)

    def init_input(self) -> None:
        """Build the EmbeddingGroup over the raw SID JAGGED_SEQUENCE groups.

        Passthrough features own no tables, so this holds no params (DMP-neutral)
        and only retrieves the flat ``(values, lengths)``.
        """
        self.embedding_group = EmbeddingGroup(self._features, self._feature_groups)

    def build_input(self, batch: Batch) -> Dict[str, List[torch.Tensor]]:
        """Retrieve per-row SID token sequences, keyed by feature name.

        The answer is a ``data_config.label_field`` rather than a feature_group
        so it can be absent at inference, where no ground truth is supplied.
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

        ``SidFeature._parse`` already validated and offset the codes, so only the
        vocabulary shift and the model-owned budget are left. ``max_codes`` caps
        each row to its most-recent WHOLE items, dropping the oldest head.
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

        The answer is a label_field, not a feature, so nothing has offset it --
        the one place the model still owns the per-level fold-in.
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
