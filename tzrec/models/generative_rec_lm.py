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

"""Generic generative-recommendation language-model base for TorchEasyRec.

``GenerativeRecLM`` owns the architecture-agnostic plumbing; a concrete family
subclass (e.g. ``Qwen2RecLM``) implements ``_build_prompt_tokens`` and
``predict``. Shared config and the sample contract live in
``GenerativeRecLMConfig`` (see ``protos/models/generative_model.proto``).
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
from tzrec.models.model import BaseModel
from tzrec.modules.embedding import EmbeddingGroup
from tzrec.protos import model_pb2
from tzrec.protos.model_pb2 import ModelConfig
from tzrec.protos.models import generative_model_pb2


class GenerativeRecLM(BaseModel):
    """Abstract base for HF-backed generative-recommendation LMs.

    Owns model construction, SID vocab extension, sample data-prep, loss and
    metrics; subclasses implement ``_build_prompt_tokens`` and ``predict``. The
    family's proto message must supply ``common`` (``GenerativeRecLMConfig``)
    and an ``hf_model_id`` field, both read directly by this base.
    """

    # See `common.param_dtype` in the proto for why fp32 is the default.
    _DTYPE_BY_NAME = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
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
                "GenerativeRecLM: tokenizer has neither pad_token_id nor "
                "eos_token_id; cannot choose a pad id for the left-padded splice."
            )
        return int(pad_id)

    def _read_common_config(
        self, common: generative_model_pb2.GenerativeRecLMConfig
    ) -> int:
        """Parse shared proto knobs into attributes; return the SID atom count."""
        self._label_name: str = self._labels[0] if self._labels else ""
        self._ignore_index: int = int(common.ignore_index)
        self._generated_sids_key: str = common.generated_sids_key
        param_dtype = self._DTYPE_BY_NAME.get(common.param_dtype)
        if param_dtype is None:
            raise ValueError(
                f"{type(self).__name__}: param_dtype must be one of "
                f"{list(self._DTYPE_BY_NAME)}, got {common.param_dtype!r}."
            )
        self._param_dtype: torch.dtype = param_dtype
        self._max_seq_length: int = int(common.max_sequence_length)
        codebook = self._shared_sid_space()
        self._num_levels = len(codebook)
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

        SID features share a single space: the model has one extended vocabulary
        and one answer width, so adding a feature must not resize ``lm_head``.
        Requiring the declarations to agree makes that invariant explicit and
        turns a typo into an error instead of a silent reshape.
        """
        spaces = {}
        for feature in self._features:
            codebook = getattr(feature, "codebook", None)
            if codebook is not None:
                spaces[feature.name] = tuple(codebook)
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

        Each group must be JAGGED_SEQUENCE and hold exactly ONE feature: the
        EmbeddingGroup column-interleaves the members of a group into a single
        ``{group}.sequence``, so two features in one group could not be spliced
        into separate prompt slots.
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
            by_feature[group.feature_names[0]] = group.group_name
        return by_feature

    def _resolve_prompt_slots(
        self, template: str
    ) -> Tuple[List[str], List[BaseFeature], List[str]]:
        """Split a prompt template into its static gaps and its slot features.

        ``template`` carries ``{{feature_name}}`` slots; returns ``N+1`` static
        gap strings, the ``N`` features they are interleaved with in template
        order, and those features' feature_group names. Every slot must name a
        declared feature, every declared feature_group must be referenced by a
        slot, and every slot feature must expose the prompt-text interface
        (``prefix_text`` / ``suffix_text``) -- so a misspelt or unused feature
        fails here rather than silently vanishing from the prompt.
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
            if not hasattr(feature, "prefix_text") or not hasattr(
                feature, "suffix_text"
            ):
                raise ValueError(
                    f"{type(self).__name__}: feature {name!r} is a "
                    f"{type(feature).__name__}, which does not expose the prompt "
                    f"text interface (prefix_text/suffix_text); use a SidFeature."
                )
            features.append(feature)
        unused = sorted(set(group_of) - set(names))
        if unused:
            raise ValueError(
                f"{type(self).__name__}: feature_group(s) {unused} are declared "
                f"but never referenced by a prompt_template slot."
            )
        return gaps, features, [group_of[n] for n in names]

    def _build_backbone(self) -> PreTrainedModel:
        """Build the EMPTY extended architecture -- no weight download.

        Only the module shapes matter here; the weights arrive from
        ``init_from_pretrained`` (cold start) or DCP (restore/eval).
        """
        hf_model_id = self._model_config.hf_model_id
        if not hf_model_id:
            raise ValueError(f"{type(self).__name__}: empty hf_model_id.")
        hf_cfg = AutoConfig.from_pretrained(hf_model_id)
        lm = AutoModelForCausalLM.from_config(hf_cfg, torch_dtype=self._param_dtype)
        if next(lm.parameters()).dtype != self._param_dtype:
            lm = lm.to(self._param_dtype)
        return lm

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
                f"GenerativeRecLM: tokenizer was expected to grow by "
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
                f"GenerativeRecLM: SID atom layout mismatch -- expected "
                f"C0 at token id {base}, got {c0_id}. "
                f"Splice arithmetic would produce wrong token ids."
            )
        return tokenizer, base

    def init_from_pretrained(self) -> None:
        """Load the pretrained HF backbone weights into ``self.lm``.

        Re-extends the vocab to ``__init__``'s target so the shapes match.

        Every rank runs this, so every rank must apply the identical resize or
        DDP's parameter-shape check fails. The newly-created SID embedding rows
        are drawn from the global RNG and therefore differ per rank; DDP's
        ``_sync_module_states`` broadcast from rank 0 is what makes them agree.
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
            f"(GenerativeRecLM is abstract)."
        )

    @property
    def device(self) -> torch.device:
        """Device the HF backbone runs on."""
        return self.lm.device

    def _tokenize_sids(self, flat: torch.Tensor) -> torch.Tensor:
        """Map flat SID indices to extended-vocab token ids.

        ``SidFeature`` has already folded the per-level offsets in, and codes are
        0-based, so the flat index IS the atom index: the model only owns the
        shift into its own vocabulary.
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
        """Decode generated tokens to local 0-based codes and reject bad beams.

        ``new_tokens`` is the per-beam tail ``(B*C, w)`` (``w`` may be <
        ``num_levels`` when beams stop early). Returns ``(batch_size, C,
        num_levels)`` local codes. Every malformed candidate (early EOS /
        non-SID / wrong-level atom) is set to ``-1``, which cannot match a real
        0-based code.
        """
        level_ids = torch.arange(new_tokens.shape[1], device=new_tokens.device)
        codes = self._detokenize_sids(new_tokens, level_ids)
        codes = F.pad(codes, (0, self._num_levels - codes.shape[1]), value=-1)
        # one out-of-band atom invalidates the whole candidate row.
        invalid = ((codes < 0) | (codes >= self._codebook_sizes)).any(dim=1)
        codes = codes.masked_fill(invalid.unsqueeze(1), -1)
        # decoders return rows batch-major ([b0_c0, b0_c1, ...]); group per user.
        return codes.view(batch_size, -1, self._num_levels)

    def init_input(self) -> None:
        """Build the EmbeddingGroup for the single raw SID JAGGED_SEQUENCE group.

        Raw passthrough features carry no embedding tables, so this group holds
        no params (DMP-neutral); it only retrieves the flat ``(values, lengths)``.
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

        ``SidFeature._parse`` has already validated the codes and folded in the
        per-level offsets, so this only applies the model-owned budget and the
        shift into the extended vocabulary.

        ``max_codes``, when set, caps each row to its most-recent whole items
        (the last ``floor(max_codes / num_levels) * num_levels`` codes, dropping
        the oldest head). Skipped unless a row overflows.
        """
        if values.dim() == 2 and values.size(-1) == 1:
            values = values.squeeze(-1)
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
        """Map the answer label to token ids.

        The answer is a ``data_config.label_field``, not a feature, so nothing
        has offset it: this is the one place the model still owns the per-level
        fold-in. Every row must be exactly ``num_levels`` codes.
        """
        if values.dim() == 2 and values.size(-1) == 1:
            values = values.squeeze(-1)
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
        if invalid.any().item():
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
