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
    BaseFeature,
)
from tzrec.protos.feature_pb2 import FeatureConfig


class LookupFeature(BaseFeature):
    """LookupFeature class.

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
    @BaseFeature.is_neg.setter
    def is_neg(self, value: bool) -> None:
        """Feature is negative sampled or not."""
        self._is_neg = value
        self._data_group = CROSS_NEG_DATA_GROUP

    @property
    def value_dim(self) -> int:
        """Fg value dimension of the feature."""
        if self.config.HasField("value_dim"):
            return self.config.value_dim
        else:
            return 1

    @property
    def output_dim(self) -> int:
        """Output dimension of the feature after embedding."""
        if self.has_embedding:
            return self.config.embedding_dim
        else:
            return max(self.config.value_dim, 1)

    @property
    def is_sparse(self) -> bool:
        """Feature is sparse or dense."""
        if self._is_sparse is None:
            self._is_sparse = (
                self.config.HasField("zch")
                or self.config.HasField("hash_bucket_size")
                or self.config.HasField("num_buckets")
                or len(self.vocab_list) > 0
                or len(self.vocab_dict) > 0
                or len(self.vocab_file) > 0
                or len(self.config.boundaries) > 0
            )
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
            num_embeddings = len(self.config.boundaries) + 1
        return num_embeddings

    @property
    def _dense_emb_type(self) -> Optional[str]:
        return self.config.WhichOneof("dense_emb")

    def _build_side_inputs(self) -> Optional[List[Tuple[str, str]]]:
        """Input field names with side."""
        if self.config.HasField("map") and self.config.HasField("key"):
            return [tuple(x.split(":")) for x in [self.config.map, self.config.key]]
        else:
            return None

    def _fg_json(self) -> List[Dict[str, Any]]:
        """Get fg json config impl."""
        fg_cfg = {
            "feature_type": "lookup_feature",
            "feature_name": self.config.feature_name,
            "map": self.config.map,
            "key": self.config.key,
            "default_value": self.config.default_value,
            "value_type": "float",
            "needDiscrete": self.config.need_discrete,
            "needKey": self.config.need_key,
            "combiner": self.config.combiner.lower(),
        }
        raw_fg_cfg = None
        if self.config.value_dim > 1:
            fg_cfg["feature_name"] = self.config.feature_name + "__lookup"
            fg_cfg["default_value"] = ""
            fg_cfg["value_type"] = "string"
            fg_cfg["value_dim"] = 1
            fg_cfg["needDiscrete"] = True
            fg_cfg["combiner"] = ""
            fg_cfg["stub_type"] = True
            raw_fg_cfg = {
                "feature_type": "raw_feature",
                "feature_name": self.config.feature_name,
                "default_value": self.config.default_value,
                "expression": "feature:" + fg_cfg["feature_name"],
                "separator": self.config.value_separator,
                "value_dim": self.config.value_dim,
                "value_type": "float",
            }
            if self.config.HasField("normalizer"):
                raw_fg_cfg["normalizer"] = self.config.normalizer
            if len(self.config.boundaries) > 0:
                raw_fg_cfg["boundaries"] = list(self.config.boundaries)
        else:
            if self.config.separator != "\x1d":
                fg_cfg["separator"] = self.config.separator
            if self.config.HasField("normalizer"):
                fg_cfg["normalizer"] = self.config.normalizer
            if self.config.HasField("zch") or self.config.HasField("dynamicemb"):
                fg_cfg["hash_bucket_size"] = MAX_HASH_BUCKET_SIZE
                fg_cfg["value_type"] = "string"
            elif self.config.HasField("hash_bucket_size"):
                fg_cfg["hash_bucket_size"] = self.config.hash_bucket_size
                fg_cfg["value_type"] = "string"
            elif self.config.HasField("num_buckets"):
                fg_cfg["num_buckets"] = self.config.num_buckets
                fg_cfg["value_type"] = "string"
            elif len(self.vocab_list) > 0:
                fg_cfg["vocab_list"] = self.vocab_list
                fg_cfg["default_bucketize_value"] = self.default_bucketize_value
                fg_cfg["value_type"] = "string"
            elif len(self.vocab_dict) > 0:
                fg_cfg["vocab_dict"] = self.vocab_dict
                fg_cfg["default_bucketize_value"] = self.default_bucketize_value
                fg_cfg["value_type"] = "string"
            elif len(self.vocab_file) > 0:
                fg_cfg["vocab_file"] = self.vocab_file
                fg_cfg["default_bucketize_value"] = self.default_bucketize_value
                fg_cfg["value_type"] = "string"
            elif len(self.config.boundaries) > 0:
                fg_cfg["boundaries"] = list(self.config.boundaries)

            if self.config.HasField("fg_value_type"):
                fg_cfg["value_type"] = self.config.fg_value_type
            if fg_cfg["value_type"] == "string":
                fg_cfg["needDiscrete"] = True
            if fg_cfg["needDiscrete"]:
                fg_cfg["combiner"] = ""
            if fg_cfg["combiner"] == "":
                fg_cfg["value_dim"] = self.value_dim
            if self.config.HasField("stub_type"):
                fg_cfg["stub_type"] = self.config.stub_type

        if self.is_grouped_sequence:
            if len(self.config.sequence_fields) > 0:
                fg_cfg["sequence_fields"] = list(self.config.sequence_fields)
            if raw_fg_cfg is not None:
                raw_fg_cfg["sequence_fields"] = [self.config.feature_name + "__lookup"]

        fg_cfgs = [fg_cfg]
        if raw_fg_cfg is not None:
            fg_cfgs.append(raw_fg_cfg)

        return fg_cfgs

    def assets(self) -> Dict[str, str]:
        """Asset file paths."""
        assets = {}
        if len(self.vocab_file) > 0:
            assets["vocab_file"] = self.vocab_file
        return assets
