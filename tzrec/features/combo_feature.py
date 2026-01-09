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

from typing import Any, Dict, List, Optional, Tuple

from tzrec.datasets.utils import (
    CROSS_NEG_DATA_GROUP,
)
from tzrec.features.feature import (
    MAX_HASH_BUCKET_SIZE,
)
from tzrec.features.id_feature import IdFeature
from tzrec.protos.feature_pb2 import FeatureConfig


class ComboFeature(IdFeature):
    """ComboFeature class.

    Args:
        feature_config (FeatureConfig): a instance of feature config.
    """

    def __init__(
        self,
        feature_config: FeatureConfig,
        **kwargs,
    ) -> None:
        super().__init__(feature_config, **kwargs)

    # pyre-ignore [56]
    @IdFeature.is_neg.setter
    def is_neg(self, value: bool) -> None:
        """Feature is negative sampled or not."""
        self._is_neg = value
        self._data_group = CROSS_NEG_DATA_GROUP

    @property
    def num_embeddings(self) -> int:
        """Get embedding row count."""
        if self.config.HasField("zch"):
            num_embeddings = self.config.zch.zch_size
        elif self.config.HasField("dynamicemb"):
            num_embeddings = self.config.dynamicemb.max_capacity
        elif self.config.HasField("hash_bucket_size"):
            num_embeddings = self.config.hash_bucket_size
        elif len(self.vocab_list) > 0:
            num_embeddings = len(self.vocab_list)
        elif len(self.vocab_dict) > 0:
            num_embeddings = max(list(self.vocab_dict.values())) + 1
        elif len(self.vocab_file) > 0:
            self.init_fg()
            num_embeddings = self.vocab_file_size
        else:
            raise ValueError(
                f"{self.__class__.__name__}[{self.name}] must set hash_bucket_size"
                " or vocab_list or vocab_dict or zch.zch_size"
            )
        return num_embeddings

    def _build_side_inputs(self) -> Optional[List[Tuple[str, str]]]:
        """Input field names with side."""
        if len(self.config.expression) > 0:
            return [tuple(x.split(":")) for x in self.config.expression]
        else:
            return None

    def _fg_json(self) -> List[Dict[str, Any]]:
        """Get fg json config impl."""
        fg_cfg = {
            "feature_type": "combo_feature",
            "feature_name": self.config.feature_name,
            "default_value": self.config.default_value,
            "expression": list(self.config.expression),
            "value_type": "string",
            "need_prefix": False,
            "value_dim": self.config.value_dim,
        }
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
        if self.config.HasField("stub_type"):
            fg_cfg["stub_type"] = self.config.stub_type

        if self.is_grouped_sequence and len(self.config.sequence_fields) > 0:
            fg_cfg["sequence_fields"] = list(self.config.sequence_fields)
        return [fg_cfg]
