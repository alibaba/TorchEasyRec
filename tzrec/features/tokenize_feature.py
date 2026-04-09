# Copyright (c) 2024, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Any, Dict, List

import numpy as np
import pyarrow as pa
import pyfg

from tzrec.datasets.utils import SequenceSparseData
from tzrec.features.feature import _parse_fg_encoded_sparse_feature_impl
from tzrec.features.id_feature import IdFeature
from tzrec.protos.data_pb2 import FgMode
from tzrec.protos.feature_pb2 import FeatureConfig, TextNormalizeOption

NORM_OPTION_MAPPING = {
    TextNormalizeOption.TEXT_LOWER2UPPER: 2,
    TextNormalizeOption.TEXT_UPPER2LOWER: 4,
    TextNormalizeOption.TEXT_SBC2DBC: 8,
    TextNormalizeOption.TEXT_CHT2CHS: 16,
    TextNormalizeOption.TEXT_FILTER: 32,
    TextNormalizeOption.TEXT_SPLITCHRS: 512,
}


class TokenizeFeature(IdFeature):
    """TokenizeFeature class.

    Args:
        feature_config (FeatureConfig): a instance of feature config.
    """

    def __init__(
        self,
        feature_config: FeatureConfig,
        **kwargs,
    ) -> None:
        # pre-set attributes referenced by BaseFeature.__del__ so that a
        # failure in the assertions below does not raise an unraisable
        # AttributeError during garbage collection.
        self._fg_op = None
        self._tok_fg_op = None
        fc_type = feature_config.WhichOneof("feature")
        cfg = getattr(feature_config, fc_type)
        self._tokens_as_sequence = bool(getattr(cfg, "tokens_as_sequence", False))
        if self._tokens_as_sequence:
            assert fc_type == "tokenize_feature", (
                "tokens_as_sequence is only valid with the `tokenize_feature` "
                "oneof entry, not `sequence_tokenize_feature`."
            )
            # Auto-enable sequence mode so BaseFeature picks up
            # sequence_delim / sequence_length from the config block.
            kwargs["is_sequence"] = True
        super().__init__(feature_config, **kwargs)

    @property
    def num_embeddings(self) -> int:
        """Get embedding row count."""
        if len(self.vocab_file) > 0:
            if self._tok_fg_op is None:
                self.init_fg()
            num_embeddings = self._tok_fg_op.vocab_size()
        else:
            raise ValueError(
                f"{self.__class__.__name__}[{self.name}] must set vocab_file"
            )
        return num_embeddings

    @property
    def value_dim(self) -> int:
        """Fg value dimension of the feature."""
        # When consumed as a token sequence, each sequence element is a single
        # token id (value_dim=1), matching how sequence_id_feature flows into
        # SequenceEmbeddingGroupImpl (no segment_reduce, EC + to_padded_dense).
        if self._tokens_as_sequence:
            return 1
        return 0

    @property
    def vocab_file(self) -> str:
        """Vocab file."""
        if self.config.HasField("vocab_file"):
            # for tokenize feature, tokenize info already in vocab model,
            # we do not need check default_bucketize_value
            vocab_file = self.config.vocab_file
            if self.config.HasField("asset_dir"):
                vocab_file = os.path.join(self.config.asset_dir, vocab_file)
            return vocab_file
        else:
            return ""

    @property
    def stop_char_file(self) -> str:
        """Stop char file."""
        stop_char_file = ""
        if self.config.HasField("text_normalizer"):
            norm_cfg = self.config.text_normalizer
            if norm_cfg.HasField("stop_char_file"):
                stop_char_file = norm_cfg.stop_char_file
                if self.config.HasField("asset_dir"):
                    stop_char_file = os.path.join(self.config.asset_dir, stop_char_file)
        return stop_char_file

    def init_fg(self) -> None:
        """Init fg op."""
        super().init_fg()
        # for get vocab_size
        fg_cfgs = self.fg_json()
        # pyre-ignore [16]
        self._tok_fg_op = pyfg.FeatureFactory.create(fg_cfgs[-1], False)

    def _fg_json(self) -> List[Dict[str, Any]]:
        """Get fg json config impl."""
        fg_cfgs = []
        expression = self.config.expression
        norm_fg_cfg = None
        if self.config.HasField("text_normalizer"):
            norm_cfg = self.config.text_normalizer
            norm_fg_name = self.config.feature_name + "__text_norm"
            expression = "feature:" + norm_fg_name
            norm_fg_cfg = {
                "feature_type": "text_normalizer",
                "feature_name": norm_fg_name,
                "expression": self.config.expression,
                "is_gbk_input": False,
                "is_gbk_output": False,
                "stub_type": True,
            }
            if norm_cfg.HasField("max_length"):
                norm_fg_cfg["max_length"] = norm_cfg.max_length
            if len(self.stop_char_file) > 0:
                norm_fg_cfg["stop_char_file"] = self.stop_char_file
            if len(norm_cfg.norm_options) > 0:
                parameter = 0
                for norm_option in norm_cfg.norm_options:
                    if norm_option in NORM_OPTION_MAPPING:
                        parameter += NORM_OPTION_MAPPING[norm_option]
                    if norm_option == TextNormalizeOption.TEXT_REMOVE_SPACE:
                        norm_fg_cfg["remove_space"] = True
                norm_fg_cfg["parameter"] = parameter

            if self.is_grouped_sequence and len(self.config.sequence_fields) > 0:
                norm_fg_cfg["sequence_fields"] = list(self.config.sequence_fields)
            fg_cfgs.append(norm_fg_cfg)

        assert self.config.tokenizer_type in [
            "bpe",
            "sentencepiece",
        ], "tokenizer_type only support [bpe, sentencepiece] now."
        fg_cfg = {
            "feature_type": "tokenize_feature",
            "feature_name": self.config.feature_name,
            "default_value": self.default_value,
            "vocab_file": self.vocab_file,
            "expression": expression,
            "tokenizer_type": self.config.tokenizer_type,
            "output_type": "word_id",
            "output_delim": self._fg_encoded_multival_sep,
        }
        if self.config.HasField("stub_type"):
            fg_cfg["stub_type"] = self.config.stub_type
        if self.is_grouped_sequence:
            if norm_fg_cfg is None:
                if len(self.config.sequence_fields) > 0:
                    fg_cfg["sequence_fields"] = list(self.config.sequence_fields)
            else:
                fg_cfg["sequence_fields"] = [norm_fg_name]

        fg_cfgs.append(fg_cfg)
        return fg_cfgs

    def fg_json(self) -> List[Dict[str, Any]]:
        """Get fg json config.

        For ``tokens_as_sequence`` mode, we bypass the sequence-wrapper logic
        in ``BaseFeature.fg_json`` so that fg emits a plain
        ``tokenize_feature`` entry whose output ``(values, lengths)`` already
        describes a per-sample token sequence. The sequence semantics are
        applied downstream (in ``_parse`` and ``SequenceEmbeddingGroupImpl``)
        instead of through a ``sequence_tokenize_feature`` fg wrapper (which
        interprets the input as a delimited list of texts).
        """
        if not self._tokens_as_sequence:
            return super().fg_json()
        fg_cfgs = self._fg_json()
        for fg_cfg in fg_cfgs:
            if not fg_cfg.get("default_value"):
                fg_cfg["default_value"] = "0"
        return fg_cfgs

    def _parse(self, input_data: Dict[str, pa.Array]):
        """Parse input data for the feature impl.

        In ``tokens_as_sequence`` mode, run the normal (non-sequence) fg path
        to obtain ``(values, lengths)``, then wrap it as ``SequenceSparseData``
        with ``key_lengths = ones`` (one id per sequence element) and
        ``seq_lengths = lengths`` (per-sample token count). Because
        ``value_dim == 1``, ``SequenceEmbeddingGroupImpl`` will skip the
        multi-value ``segment_reduce`` branch and feed the per-token
        embeddings straight into ``to_padded_dense``.
        """
        if not self._tokens_as_sequence:
            return super()._parse(input_data)

        if self.fg_mode == FgMode.FG_NORMAL:
            # pyre-ignore [16]
            fgout, status = self._fg_op.process_arrow(input_data)
            assert status.ok(), status.message()
            feat_data = fgout[self.name]
            values = feat_data.np_values
            seq_lengths = feat_data.np_lengths
        elif self.fg_mode == FgMode.FG_NONE:
            feat = input_data[self.name]
            sparse = _parse_fg_encoded_sparse_feature_impl(
                self.name,
                feat,
                **self._fg_encoded_kwargs,
            )
            values = sparse.values
            seq_lengths = sparse.lengths
        else:
            raise ValueError(
                f"fg_mode: {self.fg_mode} is not supported for "
                "tokens_as_sequence TokenizeFeature."
            )

        return SequenceSparseData(
            name=self.name,
            values=values,
            key_lengths=np.ones(values.shape[0], dtype=np.int32),
            seq_lengths=seq_lengths,
        )

    def assets(self) -> Dict[str, str]:
        """Asset file paths."""
        assets = {"vocab_file": self.vocab_file}
        if len(self.stop_char_file) > 0:
            assets["text_normalizer.stop_char_file"] = self.stop_char_file
        return assets
