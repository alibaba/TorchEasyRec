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
    DenseData,
    ParsedData,
    SparseData,
)
from tzrec.features.feature import (
    BaseFeature,
    FgMode,
    _parse_fg_encoded_dense_feature_impl,
    _parse_fg_encoded_sparse_feature_impl,
)
from tzrec.protos.feature_pb2 import FeatureConfig


class RawFeature(BaseFeature):
    """RawFeature class.

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

    @property
    def name(self) -> str:
        """Feature name."""
        return self.config.feature_name

    @property
    def output_dim(self) -> int:
        """Output dimension of the feature."""
        if self.is_sparse:
            if self.config.HasField("atd"):
                return self.config.atd.embedding_dim
            else:
                return self.config.embedding_dim
        else:
            return self.config.value_dim

    @property
    def is_sparse(self) -> bool:
        """Feature is sparse or dense."""
        if self._is_sparse is None:
            self._is_sparse = len(self.config.boundaries) > 0
        return self._is_sparse

    @property
    def num_embeddings(self) -> int:
        """Get embedding row count."""
        if self.config.HasField("atd"):
            return self.config.atd.num_bins
        return len(self.config.boundaries) + 1

    def _build_side_inputs(self) -> List[Tuple[str, str]]:
        """Input field names with side."""
        return [tuple(self.config.expression.split(":"))]

    def _parse(self, input_data: Dict[str, pa.Array]) -> ParsedData:
        """Parse input data for the feature impl.

        Args:
            input_data (dict): raw input feature data.

        Return:
            parsed feature data.
        """
        if self.fg_mode == FgMode.FG_NONE:
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
            input_feat = input_data[self.inputs[0]]
            if pa.types.is_list(input_feat.type):
                input_feat = input_feat.fill_null([])
            input_feat = input_feat.tolist()
            if self._fg_op.is_sparse:
                values, lengths = self._fg_op.to_bucketized_jagged_tensor(input_feat)
                parsed_feat = SparseData(name=self.name, values=values, lengths=lengths)
            else:
                values = self._fg_op.transform(input_feat)
                parsed_feat = DenseData(name=self.name, values=values)
        else:
            raise ValueError(
                f"fg_mode: {self.fg_mode} is not supported without fg handler."
            )
        return parsed_feat

    def fg_json(self) -> List[Dict[str, Any]]:
        """Get fg json config."""
        fg_cfg = {
            "feature_type": "raw_feature",
            "feature_name": self.name,
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
        return [fg_cfg]
