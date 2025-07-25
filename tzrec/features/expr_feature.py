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

import pyarrow as pa

from tzrec.datasets.utils import (
    CROSS_NEG_DATA_GROUP,
    DenseData,
    ParsedData,
    SparseData,
)
from tzrec.features.feature import (
    FgMode,
    _parse_fg_encoded_dense_feature_impl,
    _parse_fg_encoded_sparse_feature_impl,
)
from tzrec.features.raw_feature import RawFeature
from tzrec.protos.feature_pb2 import FeatureConfig


class ExprFeature(RawFeature):
    """ExprFeature class.

    Args:
        feature_config (FeatureConfig): a instance of feature config.
        fg_mode (FgMode): input data fg mode.
        fg_encoded_multival_sep (str, optional): multival_sep when fg_mode=FG_NONE
    """

    def __init__(
        self,
        feature_config: FeatureConfig,
        fg_mode: FgMode = FgMode.FG_NONE,
        fg_encoded_multival_sep: Optional[str] = None,
    ) -> None:
        super().__init__(feature_config, fg_mode, fg_encoded_multival_sep)

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
            return self._embedding_dim
        else:
            return 1

    def _build_side_inputs(self) -> Optional[List[Tuple[str, str]]]:
        """Input field names with side."""
        if len(self.config.variables) > 0:
            return [tuple(x.split(":")) for x in self.config.variables]
        else:
            return None

    def _parse(self, input_data: Dict[str, pa.Array]) -> ParsedData:
        """Parse input data for the feature impl.

        Args:
            input_data (dict): raw input feature data.

        Return:
            parsed feature data.
        """
        if self.fg_mode == FgMode.FG_NONE:
            # input feature is already lookuped
            feat = input_data[self.name]
            if self.is_sparse:
                parsed_feat = _parse_fg_encoded_sparse_feature_impl(
                    self.name, feat, **self._fg_encoded_kwargs
                )
            else:
                parsed_feat = _parse_fg_encoded_dense_feature_impl(
                    self.name, feat, **self._fg_encoded_kwargs
                )
        elif self.fg_mode == FgMode.FG_NORMAL:
            input_feats = [input_data[x].tolist() for x in self.inputs]
            if self.is_sparse:
                values, lengths = self._fg_op.to_bucketized_jagged_tensor(input_feats)
                parsed_feat = SparseData(name=self.name, values=values, lengths=lengths)
            else:
                values = self._fg_op.transform(input_feats)
                parsed_feat = DenseData(name=self.name, values=values)
        else:
            raise ValueError(
                f"fg_mode: {self.fg_mode} is not supported without fg handler."
            )
        return parsed_feat

    def fg_json(self) -> List[Dict[str, Any]]:
        """Get fg json config."""
        fg_cfg = {
            "feature_type": "expr_feature",
            "feature_name": self.name,
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
        return [fg_cfg]
