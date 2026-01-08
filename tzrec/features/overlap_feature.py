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
from tzrec.features.raw_feature import RawFeature


class OverlapFeature(RawFeature):
    """OverlapFeature class."""

    # pyre-ignore [56]
    @RawFeature.is_neg.setter
    def is_neg(self, value: bool) -> None:
        """Feature is negative sampled or not."""
        self._is_neg = value
        self._data_group = CROSS_NEG_DATA_GROUP

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

    def _build_side_inputs(self) -> Optional[List[Tuple[str, str]]]:
        """Input field names with side."""
        if self.config.HasField("query") and self.config.HasField("title"):
            return [tuple(x.split(":")) for x in [self.config.query, self.config.title]]
        else:
            return None

    def _fg_json(self) -> List[Dict[str, Any]]:
        """Get fg json config impl."""
        fg_cfg = {
            "feature_type": "overlap_feature",
            "feature_name": self.name,
            "query": self.config.query,
            "title": self.config.title,
            "method": self.config.method,
            "value_type": "float",
        }
        fg_cfg["separator"] = self.config.separator
        if self.config.normalizer != "":
            fg_cfg["normalizer"] = self.config.normalizer
        if len(self.config.boundaries) > 0:
            fg_cfg["boundaries"] = list(self.config.boundaries)
        if self.config.HasField("stub_type"):
            fg_cfg["stub_type"] = self.config.stub_type
        return [fg_cfg]
