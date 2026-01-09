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

import pyfg

from tzrec.features.id_feature import IdFeature
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
        super().__init__(feature_config, **kwargs)
        self._tok_fg_op = None

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
        if self.config.HasField("text_normalizer"):
            fg_cfgs = self.fg_json()
            fg_cfg = None
            for fg_cfg in fg_cfgs:
                if fg_cfg["feature_name"] == self.name:
                    break
            assert fg_cfg is not None
            # pyre-ignore [16]
            self._tok_fg_op = pyfg.FeatureFactory.create(fg_cfg, False)
        else:
            self._tok_fg_op = self._fg_op

    def _fg_json(self) -> List[Dict[str, Any]]:
        """Get fg json config impl."""
        fg_cfgs = []
        expression = self.config.expression
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
            fg_cfgs.append(norm_fg_cfg)

        assert self.config.tokenizer_type in [
            "bpe",
            "sentencepiece",
        ], "tokenizer_type only support [bpe, sentencepiece] now."
        fg_cfg = {
            "feature_type": "tokenize_feature",
            "feature_name": self.config.feature_name,
            "default_value": self.config.default_value,
            "vocab_file": self.vocab_file,
            "expression": expression,
            "tokenizer_type": self.config.tokenizer_type,
            "output_type": "word_id",
            "output_delim": self._fg_encoded_multival_sep,
        }
        if self.config.HasField("stub_type"):
            fg_cfg["stub_type"] = self.config.stub_type
        fg_cfgs.append(fg_cfg)
        return fg_cfgs

    def assets(self) -> Dict[str, str]:
        """Asset file paths."""
        assets = {"vocab_file": self.vocab_file}
        if len(self.stop_char_file) > 0:
            assets["text_normalizer.stop_char_file"] = self.stop_char_file
        return assets
