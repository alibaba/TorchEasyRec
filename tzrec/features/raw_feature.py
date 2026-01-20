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
    BaseFeature,
)
from tzrec.protos.feature_pb2 import FeatureConfig


class RawFeature(BaseFeature):
    """RawFeature class.

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
            return self.value_dim

    @property
    def is_sparse(self) -> bool:
        """Feature is sparse or dense."""
        if self._is_sparse is None:
            self._is_sparse = len(self.config.boundaries) > 0
        return self._is_sparse

    @property
    def num_embeddings(self) -> int:
        """Get embedding row count."""
        return len(self.config.boundaries) + 1

    @property
    def _dense_emb_type(self) -> Optional[str]:
        if self._is_sequence:
            # sequence feature not support dense emb now.
            return None
        else:
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
            "feature_type": "raw_feature",
            "feature_name": self.config.feature_name,
            "default_value": self.config.default_value,
            "expression": self.config.expression,
            "value_type": "float",
        }
        if self.config.value_dim > 1:
            if self.config.separator != "\x1d":
                fg_cfg["separator"] = self.config.separator
            fg_cfg["value_dim"] = self.config.value_dim
        if self.config.normalizer != "":
            fg_cfg["normalizer"] = self.config.normalizer
        if len(self.config.boundaries) > 0:
            fg_cfg["boundaries"] = list(self.config.boundaries)
        if self.config.HasField("stub_type"):
            fg_cfg["stub_type"] = self.config.stub_type

        if self.is_grouped_sequence and len(self.config.sequence_fields) > 0:
            fg_cfg["sequence_fields"] = list(self.config.sequence_fields)
        return [fg_cfg]
