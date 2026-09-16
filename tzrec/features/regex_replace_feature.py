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

from typing import Any, Dict, List

from tzrec.features.feature import (
    MAX_HASH_BUCKET_SIZE,
)
from tzrec.features.id_feature import IdFeature


class RegexReplaceFeature(IdFeature):
    """RegexReplaceFeature class.

    Args:
        feature_config (FeatureConfig): a instance of feature config.
    """

    @property
    def value_dim(self) -> int:
        """Fg value dimension of the feature."""
        # fg types the output column as array<string> unless value_dim is 1, and
        # tokenize_feature rejects an array input, so we default to 1 instead of
        # IdFeature's 0. it has to be the property, the model side and the fg
        # json would disagree otherwise.
        if self.config.HasField("value_dim"):
            return self.config.value_dim
        else:
            return 1

    def fg_json(self) -> List[Dict[str, Any]]:
        """Get fg json config."""
        if len(self.config.regex_pattern) == 0:
            # fg compiles an empty pattern list into `(?:)`, which matches the
            # empty string everywhere and inserts replacement between every char
            raise ValueError(
                f"{self.__class__.__name__}[{self.name}] must set regex_pattern."
            )
        # fg has no sequence_regex_replace_feature, the sequence version is
        # activated by is_sequence, so we do not use _fg_json here.
        fg_cfg = {
            "feature_type": "regex_replace_feature",
            "feature_name": self.config.feature_name,
            "default_value": self.default_value,
            "expression": self.config.expression,
            "regex_pattern": list(self.config.regex_pattern),
            "replacement": self.config.replacement,
        }
        if not self.config.replace_all:
            fg_cfg["replace_all"] = False
        if self.config.icase:
            fg_cfg["icase"] = True
        if self.config.separator != "\x1d":
            fg_cfg["separator"] = self.config.separator
        if self.config.HasField("zch") or self.config.HasField("dynamicemb"):
            fg_cfg["hash_bucket_size"] = MAX_HASH_BUCKET_SIZE
        elif self.config.HasField("hash_bucket_size"):
            fg_cfg["hash_bucket_size"] = self.config.hash_bucket_size
        elif len(self.vocab_list) > 0:
            fg_cfg["vocab_list"] = self.vocab_list
            fg_cfg["default_bucketize_value"] = self.default_bucketize_value
        elif len(self.vocab_dict) > 0:
            fg_cfg["vocab_dict"] = self.vocab_dict
            fg_cfg["default_bucketize_value"] = self.default_bucketize_value
        elif len(self.vocab_file) > 0:
            fg_cfg["vocab_file"] = self.vocab_file
            fg_cfg["default_bucketize_value"] = self.default_bucketize_value
        elif self.config.HasField("num_buckets"):
            fg_cfg["num_buckets"] = self.config.num_buckets
        fg_cfg["value_dim"] = self.value_dim
        if self.config.HasField("stub_type"):
            fg_cfg["stub_type"] = self.config.stub_type

        if self.is_sequence:
            if self.is_grouped_sequence:
                if len(self.config.sequence_fields) > 0:
                    fg_cfg["sequence_fields"] = list(self.config.sequence_fields)
            else:
                fg_cfg["sequence_delim"] = self.sequence_delim
                fg_cfg["sequence_length"] = self.sequence_length
            fg_cfg["is_sequence"] = True

        return [fg_cfg]
