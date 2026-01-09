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

from tzrec.features.feature import (
    MAX_HASH_BUCKET_SIZE,
    BaseFeature,
)
from tzrec.protos import feature_pb2
from tzrec.protos.feature_pb2 import FeatureConfig


class IdFeature(BaseFeature):
    """IdFeature class.

    Args:
        feature_config (FeatureConfig): a instance of feature config.
    """

    def __init__(
        self,
        feature_config: FeatureConfig,
        **kwargs,
    ) -> None:
        super().__init__(feature_config, **kwargs)

        if isinstance(self.config, feature_pb2.IdFeature) and self.config.HasField(
            "weighted"
        ):
            self._is_weighted = self.config.weighted

    @property
    def value_dim(self) -> int:
        """Fg value dimension of the feature."""
        if self.config.HasField("value_dim"):
            return self.config.value_dim
        elif self.is_sequence:
            # for sequence, single value is default setting.
            return 1
        else:
            return 0

    @property
    def output_dim(self) -> int:
        """Output dimension of the feature after embedding."""
        return self.config.embedding_dim

    @property
    def is_sparse(self) -> bool:
        """Feature is sparse or dense."""
        if self._is_sparse is None:
            self._is_sparse = True
        return self._is_sparse

    @property
    def num_embeddings(self) -> int:
        """Get embedding row count."""
        if self.config.HasField("zch"):
            num_embeddings = self.config.zch.zch_size
        elif self.config.HasField("dynamicemb"):
            num_embeddings = self.config.dynamicemb.max_capacity
        elif self.config.HasField("hash_bucket_size"):
            num_embeddings = self.config.hash_bucket_size
        elif self.config.HasField("num_buckets"):
            num_embeddings = self.config.num_buckets
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
                " or num_buckets or vocab_list or vocab_dict or zch.zch_size"
            )
        return num_embeddings

    def _build_side_inputs(self) -> Optional[List[Tuple[str, str]]]:
        """Input field names with side."""
        if self.config.HasField("expression"):
            return [tuple(self.config.expression.split(":"))]
        else:
            return None

    def _fg_json(self) -> List[Dict[str, Any]]:
        """Get fg json config impl."""
        fg_cfg = {
            "feature_type": "id_feature",
            "feature_name": self.config.feature_name,
            "default_value": self.config.default_value,
            "expression": self.config.expression,
            "value_type": "string",
            "need_prefix": False,
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
        elif self.config.HasField("num_buckets"):
            fg_cfg["num_buckets"] = self.config.num_buckets
        if self.config.weighted:
            fg_cfg["weighted"] = True
        fg_cfg["value_dim"] = self.value_dim
        if self.config.HasField("fg_value_type"):
            fg_cfg["value_type"] = self.config.fg_value_type
        if self.config.HasField("stub_type"):
            fg_cfg["stub_type"] = self.config.stub_type
        return [fg_cfg]

    def assets(self) -> Dict[str, str]:
        """Asset file paths."""
        assets = {}
        if len(self.vocab_file) > 0:
            assets["vocab_file"] = self.vocab_file
        return assets
