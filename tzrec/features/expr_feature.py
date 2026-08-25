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

    @property
    def is_sparse(self) -> bool:
        """Feature is sparse or dense."""
        if self._is_sparse is None:
            self._is_sparse = (
                self.config.HasField("hash_bucket_size")
                or self.config.HasField("num_buckets")
                or len(self.config.boundaries) > 0
            )
        return self._is_sparse

    @property
    def num_embeddings(self) -> int:
        """Get embedding row count."""
        if self.config.HasField("hash_bucket_size"):
            num_embeddings = self.config.hash_bucket_size
        elif self.config.HasField("num_buckets"):
            num_embeddings = self.config.num_buckets
        else:
            num_embeddings = len(self.config.boundaries) + 1
        return num_embeddings

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
            "default_value": self.default_value,
            "expression": self.config.expression,
            "variables": list(self.config.variables),
            "value_type": "float",
        }
        if self.config.separator != "\x1d":
            fg_cfg["separator"] = self.config.separator
        if self.config.HasField("fill_missing"):
            fg_cfg["fill_missing"] = self.config.fill_missing
        if self.config.HasField("hash_bucket_size"):
            fg_cfg["hash_bucket_size"] = self.config.hash_bucket_size
        elif self.config.HasField("num_buckets"):
            fg_cfg["num_buckets"] = self.config.num_buckets
            fg_cfg["value_type"] = "int64"
        elif len(self.config.boundaries) > 0:
            fg_cfg["boundaries"] = list(self.config.boundaries)
        if self.config.HasField("value_dim"):
            fg_cfg["value_dim"] = self.config.value_dim
        elif self.is_sequence:
            # pyfg requires an explicit value_dim for sequence sub-features
            # consumed by bool_mask_feature.
            fg_cfg["value_dim"] = self.value_dim
        if self.config.HasField("fg_value_type"):
            assert not self.config.fg_value_type.startswith("int") or (
                len(self.config.boundaries) == 0
                and not self.config.HasField("hash_bucket_size")
            ), (
                f"expr feature[{self.name}]: int fg_value_type is not supported "
                "with boundaries or hash_bucket_size."
            )
            fg_cfg["value_type"] = self.config.fg_value_type
        if self.config.HasField("stub_type"):
            fg_cfg["stub_type"] = self.config.stub_type

        if self.is_grouped_sequence and len(self.config.sequence_fields) > 0:
            fg_cfg["sequence_fields"] = list(self.config.sequence_fields)
        return [fg_cfg]
