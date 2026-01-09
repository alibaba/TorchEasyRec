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
from tzrec.protos.feature_pb2 import FeatureConfig


class ExprFeature(RawFeature):
    """ExprFeature class.

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
    @RawFeature.is_neg.setter
    def is_neg(self, value: bool) -> None:
        """Feature is negative sampled or not."""
        self._is_neg = value
        self._data_group = CROSS_NEG_DATA_GROUP

    def _build_side_inputs(self) -> Optional[List[Tuple[str, str]]]:
        """Input field names with side."""
        if len(self.config.variables) > 0:
            return [tuple(x.split(":")) for x in self.config.variables]
        else:
            return None

    def _fg_json(self) -> List[Dict[str, Any]]:
        """Get fg json config impl."""
        fg_cfg = {
            "feature_type": "expr_feature",
            "feature_name": self.config.feature_name,
            "default_value": self.config.default_value,
            "expression": self.config.expression,
            "variables": list(self.config.variables),
            "value_type": "float",
        }
        if self.config.separator != "\x1d":
            fg_cfg["separator"] = self.config.separator
        if self.config.HasField("fill_missing"):
            fg_cfg["fill_missing"] = self.config.fill_missing
        if len(self.config.boundaries) > 0:
            fg_cfg["boundaries"] = list(self.config.boundaries)
        if self.config.HasField("value_dim"):
            fg_cfg["value_dim"] = self.config.value_dim
        if self.config.HasField("stub_type"):
            fg_cfg["stub_type"] = self.config.stub_type

        if self.is_grouped_sequence and len(self.config.sequence_fields) > 0:
            fg_cfg["sequence_fields"] = list(self.config.sequence_fields)
        return [fg_cfg]
