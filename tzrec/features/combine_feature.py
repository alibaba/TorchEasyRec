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

from typing import Any, Dict, List, Optional, Tuple

from tzrec.features.feature import (
    BaseFeature,
)
from tzrec.protos.feature_pb2 import FeatureConfig


class CombineFeature(BaseFeature):
    """CombineFeature class.

    Combines input field values using value_map for mapping and
    supports both sparse (with boundaries/num_buckets) and dense output modes.

    Args:
        feature_config (FeatureConfig): a instance of feature config.
    """

    def __init__(
        self,
        feature_config: FeatureConfig,
        **kwargs,
    ) -> None:
        super().__init__(feature_config, **kwargs)

    @property
    def value_dim(self) -> int:
        """Fg value dimension of the feature."""
        return 1

    @property
    def output_dim(self) -> int:
        """Output dimension of the feature after embedding."""
        if self.has_embedding:
            return self.config.embedding_dim
        else:
            return 1

    @property
    def is_sparse(self) -> bool:
        """Feature is sparse or dense."""
        if self._is_sparse is None:
            self._is_sparse = (
                self.config.HasField("num_buckets") or len(self.config.boundaries) > 0
            )
        return self._is_sparse

    @property
    def num_embeddings(self) -> int:
        """Get embedding row count."""
        if self.config.HasField("num_buckets"):
            num_embeddings = self.config.num_buckets
        else:
            num_embeddings = len(self.config.boundaries) + 1
        return num_embeddings

    @property
    def _dense_emb_type(self) -> Optional[str]:
        return self.config.WhichOneof("dense_emb")

    def _build_side_inputs(self) -> Optional[List[Tuple[str, str]]]:
        """Input field names with side."""
        if self.config.HasField("expression"):
            return [tuple(self.config.expression.split(":"))]
        else:
            return None

    def _fg_json(self) -> List[Dict[str, Any]]:
        """Get fg json config impl."""
        fg_cfg = {
            "feature_type": "combine_feature",
            "feature_name": self.config.feature_name,
            "default_value": self.config.default_value,
            "expression": self.config.expression,
            "value_type": "float",
            "need_prefix": False,
            "combiner": self.config.combiner.lower(),
        }
        if self.config.separator != "\x1d":
            fg_cfg["separator"] = self.config.separator
        if self.config.HasField("normalizer"):
            fg_cfg["normalizer"] = self.config.normalizer
        if len(self.config.value_map) > 0:
            fg_cfg["value_map"] = dict(self.config.value_map)
        if self.config.HasField("num_buckets"):
            fg_cfg["num_buckets"] = self.config.num_buckets
            fg_cfg["value_type"] = "int64"
        elif len(self.config.boundaries) > 0:
            fg_cfg["boundaries"] = list(self.config.boundaries)
        if self.is_sparse:
            fg_cfg["value_dim"] = 1
        if self.config.HasField("stub_type"):
            fg_cfg["stub_type"] = self.config.stub_type
        if self.is_grouped_sequence and len(self.config.sequence_fields) > 0:
            fg_cfg["sequence_fields"] = list(self.config.sequence_fields)
        return [fg_cfg]

    def assets(self) -> Dict[str, str]:
        """Asset file paths."""
        return {}
