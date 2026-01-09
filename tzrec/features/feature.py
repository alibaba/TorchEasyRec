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

import hashlib
import os
import shutil
from collections import OrderedDict
from copy import copy
from functools import partial  # NOQA
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pyarrow as pa
import pyfg
import torch
from torch import nn  # NOQA
from torchrec.distributed.planner.types import ParameterConstraints
from torchrec.modules.embedding_configs import (
    BaseEmbeddingConfig,
    EmbeddingBagConfig,
    EmbeddingConfig,
    PoolingType,
)
from torchrec.modules.mc_modules import (
    DistanceLFU_EvictionPolicy,
    LFU_EvictionPolicy,
    LRU_EvictionPolicy,
    ManagedCollisionModule,
    MCHManagedCollisionModule,
    average_threshold_filter,  # NOQA
    dynamic_threshold_filter,  # NOQA
    probabilistic_threshold_filter,  # NOQA
)
from torchrec.types import DataType

from tzrec.datasets.utils import (
    BASE_DATA_GROUP,
    C_NEG_SAMPLE_MASK,
    C_SAMPLE_MASK,
    NEG_DATA_GROUP,
    DenseData,
    ParsedData,
    SequenceDenseData,
    SequenceSparseData,
    SparseData,
)
from tzrec.modules.dense_embedding_collection import (
    AutoDisEmbeddingConfig,
    DenseEmbeddingConfig,
    MLPDenseEmbeddingConfig,
)
from tzrec.protos import feature_pb2
from tzrec.protos.data_pb2 import FgMode
from tzrec.protos.feature_pb2 import FeatureConfig, SequenceFeature
from tzrec.utils import config_util, dynamicemb_util, env_util
from tzrec.utils.load_class import get_register_class_meta
from tzrec.utils.logging_util import logger

_FEATURE_CLASS_MAP = {}
_meta_cls = get_register_class_meta(_FEATURE_CLASS_MAP)


MAX_HASH_BUCKET_SIZE = 2**63 - 1


def _parse_fg_encoded_sparse_feature_impl(
    name: str,
    feat: pa.Array,
    multival_sep: str = chr(3),
    default_value: Optional[List[int]] = None,
    is_weighted: bool = False,
) -> SparseData:
    """Parse fg encoded sparse feature.

    Args:
        name (str): feature name.
        feat (pa.Array): input feature data.
        multival_sep (str): string separator for multi-val data.
        default_value (list): default value.
        is_weighted (bool): input feature is weighted or not.

    Returns:
        an instance of SparseData.
    """
    weight_values = None
    if (
        pa.types.is_string(feat.type)
        or pa.types.is_list(feat.type)
        or pa.types.is_map(feat.type)
    ):
        weight = None
        if pa.types.is_string(feat.type) or pa.types.is_list(feat.type):
            if pa.types.is_string(feat.type):
                # dtype = string
                is_empty = pa.compute.equal(feat, pa.scalar(""))
                nulls = pa.nulls(len(feat))
                feat = pa.compute.if_else(is_empty, nulls, feat)
                feat = pa.compute.split_pattern(feat, multival_sep)
            elif pa.types.is_list(feat.type):
                # dtype = list<int> or others can cast to list<int>
                if default_value is not None:
                    is_empty = pa.compute.equal(pa.compute.list_value_length(feat), 0)
                    nulls = pa.nulls(len(feat))
                    feat = pa.compute.if_else(is_empty, nulls, feat)
            if is_weighted:
                assert pa.types.is_string(feat.values.type)
                fw = pa.compute.split_pattern(feat.values, ":")
                weight = pa.ListArray.from_arrays(
                    feat.offsets, fw.values[1::2], mask=feat.is_null()
                )
                feat = pa.ListArray.from_arrays(
                    feat.offsets, fw.values[::2], mask=feat.is_null()
                )
        else:
            # dtype = map<int,float> or others can cast to map<int,float>
            weight = pa.ListArray.from_arrays(
                feat.offsets, feat.items, mask=feat.is_null()
            )
            feat = pa.ListArray.from_arrays(
                feat.offsets, feat.keys, mask=feat.is_null()
            )

        feat = feat.cast(pa.list_(pa.int64()), safe=False)
        if weight is not None:
            weight = weight.cast(pa.list_(pa.float32()), safe=False)

        if default_value is not None:
            feat = feat.fill_null(default_value)
            if weight:
                weight = weight.fill_null([1.0])

        feat_values = feat.values.to_numpy()
        feat_offsets = feat.offsets.to_numpy()
        feat_lengths = feat_offsets[1:] - feat_offsets[:-1]
        if weight is not None:
            weight_values = weight.values.to_numpy()
    elif pa.types.is_integer(feat.type):
        assert not is_weighted
        # dtype = int
        if default_value is not None:
            feat = feat.cast(pa.int64()).fill_null(default_value[0])
            feat_values = feat.to_numpy()
            feat_lengths = np.ones_like(feat_values, np.int32)
        else:
            feat_values = feat.drop_null().cast(pa.int64()).to_numpy()
            feat_lengths = 1 - feat.is_null().cast(pa.int32()).to_numpy()
    else:
        raise ValueError(
            f"{name} only support str|int|list<int>|map<int,double> dtype input, "
            f"but get {feat.type}."
        )
    return SparseData(name, feat_values, feat_lengths, weights=weight_values)


def _parse_fg_encoded_dense_feature_impl(
    name: str,
    feat: pa.Array,
    multival_sep: str = chr(3),
    default_value: Optional[List[float]] = None,
) -> DenseData:
    """Parse fg encoded dense feature.

    Args:
        name (str): feature name.
        feat (npt.NDArray): input feature data.
        multival_sep (str): string separator for multi-val data.
        default_value (list): default value.

    Returns:
        an instance of DenseData.
    """
    if pa.types.is_string(feat.type):
        # dtype = string
        if default_value is not None:
            is_empty = pa.compute.equal(feat, pa.scalar(""))
            feat = pa.compute.if_else(is_empty, pa.nulls(len(feat)), feat)
            feat = feat.fill_null(multival_sep.join(map(str, default_value)))
        list_feat = pa.compute.split_pattern(feat, multival_sep)
        list_feat = list_feat.cast(pa.list_(pa.float32()), safe=False)
        feat_values = np.stack(list_feat.to_numpy(zero_copy_only=False))
    elif pa.types.is_list(feat.type):
        # dtype = list<float> or others can cast to list<float>
        feat = feat.cast(pa.list_(pa.float32()), safe=False)
        if default_value is not None:
            is_empty = pa.compute.equal(pa.compute.list_value_length(feat), 0)
            feat = pa.compute.if_else(is_empty, pa.nulls(len(feat)), feat)
            feat = feat.fill_null(default_value)
        feat_values = np.stack(feat.to_numpy(zero_copy_only=False))
    elif pa.types.is_integer(feat.type) or pa.types.is_floating(feat.type):
        # dtype = int or float
        feat = feat.cast(pa.float32(), safe=False)
        if default_value is not None:
            feat = feat.fill_null(default_value[0])
        feat_values = feat.to_numpy()[:, np.newaxis]
    else:
        raise ValueError(
            f"{name} only support str|int|float|list<float> dtype input,"
            f" but get {feat.type}."
        )
    return DenseData(name, feat_values)


def _parse_fg_encoded_sequence_sparse_feature_impl(
    name: str,
    feat: pa.Array,
    sequence_delim: str = ";",
    multival_sep: str = chr(3),
    default_value: Optional[List[int]] = None,
) -> SequenceSparseData:
    """Parse fg encoded sequence sparse feature.

    Args:
        name (str): feature name.
        feat (pa.Array): input feature data.
        sequence_delim (str): sequence delimiter.
        multival_sep (str): string separator for multi-val data.
        default_value (int): default value.

    Returns:
        an instance of SequenceSparseFeature.
    """
    if pa.types.is_string(feat.type):
        # dtype = string
        is_empty = pa.compute.equal(feat, pa.scalar(""))
        feat = pa.compute.if_else(is_empty, pa.nulls(len(feat)), feat)
        if default_value is not None:
            feat = feat.fill_null(multival_sep.join(map(str, default_value)))
        list_seq_feat = pa.compute.split_pattern(feat, sequence_delim)
        list_feat = pa.compute.split_pattern(list_seq_feat.values, multival_sep)
        seq_offsets = list_seq_feat.offsets.to_numpy()
        seq_lengths = seq_offsets[1:] - seq_offsets[:-1]
        feat_values = list_feat.values.cast(pa.int64()).to_numpy()
        feat_offsets = list_feat.offsets.to_numpy()
        feat_lengths = feat_offsets[1:] - feat_offsets[:-1]
    elif pa.types.is_list(feat.type):
        if pa.types.is_list(feat.type.value_type):
            # dtype = list<list<int>> or others can cast to list<list<int>>
            feat = feat.cast(pa.list_(pa.list_(pa.int64())), safe=False)
            if default_value is not None:
                is_empty = pa.compute.equal(pa.compute.list_value_length(feat), 0)
                feat = pa.compute.if_else(is_empty, pa.nulls(len(feat)), feat)
                feat = feat.fill_null([default_value])
            seq_offsets = feat.offsets.to_numpy()
            seq_lengths = seq_offsets[1:] - seq_offsets[:-1]
            feat_values = feat.values.values.to_numpy()
            feat_offsets = feat.values.offsets.to_numpy()
            feat_lengths = feat_offsets[1:] - feat_offsets[:-1]
        else:
            # dtype = list<int> or others can cast to list<int>
            feat = feat.cast(pa.list_(pa.int64()), safe=False)
            if default_value is not None:
                is_empty = pa.compute.equal(pa.compute.list_value_length(feat), 0)
                feat = pa.compute.if_else(is_empty, pa.nulls(len(feat)), feat)
                feat = feat.fill_null(default_value)
            seq_offsets = feat.offsets.to_numpy()
            seq_lengths = seq_offsets[1:] - seq_offsets[:-1]
            feat_values = feat.values.to_numpy()
            feat_lengths = np.ones_like(feat_values, dtype=np.int32)
    else:
        raise ValueError(
            f"{name} only support str|list<int>|list<list<int>> dtype input,"
            f" but get {feat.type}."
        )
    return SequenceSparseData(name, feat_values, feat_lengths, seq_lengths)


def _parse_fg_encoded_sequence_dense_feature_impl(
    name: str,
    feat: pa.Array,
    sequence_delim: str = ";",
    multival_sep: str = chr(3),
    value_dim: int = 1,
    default_value: Optional[List[float]] = None,
) -> SequenceDenseData:
    """Parse fg encoded sequence dense feature.

    Args:
        name (str): feature name.
        feat (pa.Array): input feature data.
        sequence_delim (str): sequence delimiter.
        multival_sep (str): string separator for multi-val data.
        value_dtype (pa.DataType): value dtype.
        value_dim (int): value dimension.
        default_value (list): default value.

    Returns:
        an instance of SequenceSparseFeature.
    """
    if pa.types.is_string(feat.type):
        is_empty = pa.compute.equal(feat, pa.scalar(""))
        feat = pa.compute.if_else(is_empty, pa.nulls(len(feat)), feat)
        if default_value is not None:
            feat = feat.fill_null(multival_sep.join(map(str, default_value)))
        list_seq_feat = pa.compute.split_pattern(feat, sequence_delim)
        list_feat = pa.compute.split_pattern(list_seq_feat.values, multival_sep)
        seq_offsets = list_seq_feat.offsets.to_numpy()
        seq_lengths = seq_offsets[1:] - seq_offsets[:-1]
        feat_values = (
            list_feat.values.cast(pa.float32(), safe=False)
            .to_numpy()
            .reshape(-1, value_dim)
        )
    elif pa.types.is_list(feat.type):
        if pa.types.is_list(feat.type.value_type):
            # dtype = list<list<float>> or others can cast to list<list<float>>
            feat = feat.cast(pa.list_(pa.list_(pa.float32())), safe=False)
            if default_value is not None:
                is_empty = pa.compute.equal(pa.compute.list_value_length(feat), 0)
                feat = pa.compute.if_else(is_empty, pa.nulls(len(feat)), feat)
                feat = feat.fill_null([default_value])
            seq_offsets = feat.offsets.to_numpy()
            seq_lengths = seq_offsets[1:] - seq_offsets[:-1]
            feat_values = feat.values.values.to_numpy().reshape(-1, value_dim)
        else:
            # dtype = list<float> or others can cast to list<float>
            feat = feat.cast(pa.list_(pa.float32()), safe=False)
            if default_value is not None:
                is_empty = pa.compute.equal(pa.compute.list_value_length(feat), 0)
                feat = pa.compute.if_else(is_empty, pa.nulls(len(feat)), feat)
                feat = feat.fill_null(default_value)
            seq_offsets = feat.offsets.to_numpy()
            seq_lengths = seq_offsets[1:] - seq_offsets[:-1]
            feat_values = feat.values.to_numpy().reshape(-1, value_dim)
    else:
        raise ValueError(
            f"{name} only support str|list<float>|list<float<float>> dtype input,"
            f" but get {feat.type}."
        )
    return SequenceDenseData(name, feat_values, seq_lengths)


def _dtype_str_to_data_type(data_type_str: str) -> DataType:
    if data_type_str == "FP32":
        data_type = DataType.FP32
    elif data_type_str == "FP16":
        data_type = DataType.FP16
    else:
        raise ValueError(
            "Embedding only support FP32 and FP16 now, "
            f"[{data_type_str}] is not supported."
        )
    return data_type


def build_embedding_constraints(
    constraints_cfg: feature_pb2.ParameterConstraints,
) -> ParameterConstraints:
    """Build ParameterConstraints for embedding parameter."""
    constraints = ParameterConstraints(
        sharding_types=list(constraints_cfg.sharding_types)
        if len(constraints_cfg.sharding_types) > 0
        else None,
        compute_kernels=list(constraints_cfg.compute_kernels)
        if len(constraints_cfg.compute_kernels) > 0
        else None,
    )
    return constraints


class InvalidFgInputError(Exception):
    """Invalid Feature side inputs exception."""

    pass


class BaseFeature(object, metaclass=_meta_cls):
    """Base feature class.

    Args:
        feature_config (FeatureConfig): a instance of feature config.
        fg_mode (FgMode): input data fg mode.
        fg_encoded_multival_sep (str, optional): multival_sep when fg_mode=FG_NONE
        sequence_name (str): sequence group name.
        sequence_delim (str): separator for sequence feature.
        sequence_length (int): max sequence length.
        sequence_pk (str): sequence primary key name for serving.
    """

    def __init__(
        self,
        feature_config: FeatureConfig,
        fg_mode: FgMode = FgMode.FG_NONE,
        fg_encoded_multival_sep: Optional[str] = None,
        is_sequence: bool = False,
        sequence_name: Optional[str] = None,
        sequence_delim: Optional[str] = None,
        sequence_length: Optional[int] = None,
        sequence_pk: Optional[str] = None,
        **kwargs,
    ) -> None:
        fc_type = feature_config.WhichOneof("feature")
        self._feature_config = feature_config
        self.config = getattr(self._feature_config, fc_type)

        self.fg_mode = fg_mode

        self._fg_op = None
        self._is_neg = False
        self._is_sparse = None
        self._is_weighted = False
        self._is_user_feat = None
        self._data_group = BASE_DATA_GROUP
        self._inputs = None
        self._side_inputs = None
        self._vocab_list = None
        self._vocab_dict = None
        self._is_sequence = is_sequence
        self._is_grouped_seq = False

        # for sequence feature
        self._underline = "_" if env_util.use_rtp() else "__"
        self.sequence_name = None
        self.sequence_delim = None
        self.sequence_length = None
        self.sequence_pk = None
        if is_sequence:
            if sequence_name is None:
                self.sequence_delim = self.config.sequence_delim
                self.sequence_length = self.config.sequence_length
            else:
                self._is_sequence = True
                self._is_grouped_seq = True
                self.sequence_name = sequence_name
                self.sequence_delim = sequence_delim
                self.sequence_length = sequence_length
                if not sequence_pk:
                    self.sequence_pk = f"user:{sequence_name}"
                else:
                    self.sequence_pk = sequence_pk

        # for fg encoded data
        self._fg_encoded_kwargs = {}
        self._fg_encoded_multival_sep = fg_encoded_multival_sep or chr(3)
        if self.fg_mode == FgMode.FG_NONE:
            if self.config.HasField("fg_encoded_default_value"):
                self._fg_encoded_kwargs["default_value"] = (
                    self.fg_encoded_default_value()
                )
            elif self.config.use_mask:
                try:
                    self._fg_encoded_kwargs["default_value"] = (
                        self.fg_encoded_default_value()
                    )
                except Exception:
                    raise RuntimeError(
                        f"when use mask, you should set fg_encoded_default_value"
                        f" for {self.name}"
                    ) from None
            self._fg_encoded_kwargs["multival_sep"] = self._fg_encoded_multival_sep

        if self.fg_mode == FgMode.FG_NORMAL:
            self.init_fg()

    @property
    def name(self) -> str:
        """Feature name."""
        if self._is_grouped_seq:
            return f"{self.sequence_name}{self._underline}{self.config.feature_name}"
        else:
            return self.config.feature_name

    @property
    def is_neg(self) -> bool:
        """Feature is negative sampled or not."""
        return self._is_neg

    @is_neg.setter
    def is_neg(self, value: bool) -> None:
        """Feature is negative sampled or not."""
        self._is_neg = value
        self._data_group = NEG_DATA_GROUP

    @property
    def data_group(self) -> str:
        """Data group for the feature."""
        return self._data_group

    @data_group.setter
    def data_group(self, data_group: str) -> None:
        """Data group for the feature."""
        self._data_group = data_group

    @property
    def feature_config(self) -> FeatureConfig:
        """Feature config for the feature."""
        return self._feature_config

    @feature_config.setter
    def feature_config(self, feature_config: FeatureConfig) -> None:
        """Feature config for the feature."""
        fc_type = feature_config.WhichOneof("feature")
        self._feature_config = feature_config
        self.config = getattr(self._feature_config, fc_type)

    @property
    def is_user_feat(self) -> bool:
        """Feature is user side or not."""
        if self._is_user_feat is None:
            # legacy without dag, we may not set is_user_feat
            if self.is_grouped_sequence:
                return True
            for side, _ in self.side_inputs:
                if side != "user":
                    return False
            return True
        else:
            return self._is_user_feat

    @is_user_feat.setter
    def is_user_feat(self, value: bool) -> None:
        """Feature is user side or not."""
        self._is_user_feat = value

    @property
    def value_dim(self) -> int:
        """Fg value dimension of the feature."""
        raise NotImplementedError

    @property
    def output_dim(self) -> int:
        """Output dimension of the feature after embedding."""
        raise NotImplementedError

    @property
    def is_sparse(self) -> bool:
        """Feature is sparse or dense."""
        if self._is_sparse is None:
            self._is_sparse = False
        return self._is_sparse

    @property
    def is_sequence(self) -> bool:
        """Feature is sequence or not."""
        return self._is_sequence

    @property
    def is_grouped_sequence(self) -> bool:
        """Feature is grouped sequence or not."""
        return self._is_grouped_seq

    @property
    def is_weighted(self) -> bool:
        """Feature is weighted id feature or not."""
        return self._is_weighted

    @property
    def has_embedding(self) -> bool:
        """Feature has embedding or not."""
        if self.is_sparse:
            return True
        else:
            return self._dense_emb_type is not None

    @property
    def pooling_type(self) -> PoolingType:
        """Get embedding pooling type."""
        pooling_type = self.config.pooling.upper()
        assert pooling_type in {"SUM", "MEAN"}, "available pooling type is SUM | MEAN"
        return getattr(PoolingType, pooling_type)

    @property
    def num_embeddings(self) -> int:
        """Get embedding row count."""
        raise NotImplementedError

    @property
    def _embedding_dim(self) -> int:
        if self.has_embedding:
            assert self.config.embedding_dim > 0, (
                f"embedding_dim of {self.__class__.__name__}[{self.name}] "
                "should be greater than 0."
            )
        return self.config.embedding_dim

    @property
    def _dense_emb_type(self) -> Optional[str]:
        return None

    @property
    def emb_bag_config(self) -> Optional[EmbeddingBagConfig]:
        """Get EmbeddingBagConfig of the feature."""
        if self.is_sparse:
            embedding_name = self.config.embedding_name or f"{self.name}_emb"
            init_fn = None
            if self.config.HasField("init_fn"):
                init_fn = eval(f"partial({self.config.init_fn})")
            emb_bag_config = EmbeddingBagConfig(
                num_embeddings=self.num_embeddings,
                embedding_dim=self._embedding_dim,
                name=embedding_name,
                feature_names=[self.name],
                pooling=self.pooling_type,
                init_fn=init_fn,
                data_type=_dtype_str_to_data_type(self.config.data_type),
            )
            # pyre-ignore [16]
            emb_bag_config.trainable = self.config.trainable
            return emb_bag_config
        else:
            return None

    @property
    def emb_config(self) -> Optional[EmbeddingConfig]:
        """Get EmbeddingConfig of the feature."""
        if self.is_sparse:
            embedding_name = self.config.embedding_name or f"{self.name}_emb"
            init_fn = None
            if self.config.HasField("init_fn"):
                init_fn = eval(f"partial({self.config.init_fn})")
            emb_config = EmbeddingConfig(
                num_embeddings=self.num_embeddings,
                embedding_dim=self._embedding_dim,
                name=embedding_name,
                feature_names=[self.name],
                init_fn=init_fn,
                data_type=_dtype_str_to_data_type(self.config.data_type),
            )
            # pyre-ignore [16]
            emb_config.trainable = self.config.trainable
            return emb_config
        else:
            return None

    @property
    def dense_emb_config(
        self,
    ) -> Optional[DenseEmbeddingConfig]:
        """Get DenseEmbeddingConfig of the feature."""
        if self._dense_emb_type:
            dense_emb_config = getattr(self.config, self._dense_emb_type)

            if self._dense_emb_type == "autodis":
                assert self.value_dim <= 1, (
                    "autodis embedding do not support"
                    f" feature [{self.name}] with value_dim > 1 now."
                )
                return AutoDisEmbeddingConfig(
                    embedding_dim=self._embedding_dim,
                    n_channels=dense_emb_config.num_channels,
                    temperature=dense_emb_config.temperature,
                    keep_prob=dense_emb_config.keep_prob,
                    feature_names=[self.name],
                )
            elif self._dense_emb_type == "mlp":
                return MLPDenseEmbeddingConfig(
                    embedding_dim=self._embedding_dim,
                    feature_names=[self.name],
                    value_dim=self.value_dim,
                )

        return None

    def mc_module(self, device: torch.device) -> Optional[ManagedCollisionModule]:
        """Get ManagedCollisionModule."""
        if self.is_sparse:
            if hasattr(self.config, "zch") and self.config.HasField("zch"):
                evict_type = self.config.zch.WhichOneof("eviction_policy")
                evict_config = getattr(self.config.zch, evict_type)
                threshold_filtering_func = None
                if self.config.zch.HasField("threshold_filtering_func"):
                    threshold_filtering_func = eval(
                        self.config.zch.threshold_filtering_func
                    )
                if evict_type == "lfu":
                    eviction_policy = LFU_EvictionPolicy(
                        threshold_filtering_func=threshold_filtering_func
                    )
                elif evict_type == "lru":
                    eviction_policy = LRU_EvictionPolicy(
                        decay_exponent=evict_config.decay_exponent,
                        threshold_filtering_func=threshold_filtering_func,
                    )
                elif evict_type == "distance_lfu":
                    eviction_policy = DistanceLFU_EvictionPolicy(
                        decay_exponent=evict_config.decay_exponent,
                        threshold_filtering_func=threshold_filtering_func,
                    )
                else:
                    raise ValueError("Unknown evict policy type: {evict_type}")
                return MCHManagedCollisionModule(
                    zch_size=self.config.zch.zch_size,
                    device=device,
                    eviction_interval=self.config.zch.eviction_interval,
                    eviction_policy=eviction_policy,
                )
        return None

    @property
    def inputs(self) -> List[str]:
        """Input field names."""
        if not self._inputs:
            if self.fg_mode in [FgMode.FG_NONE, FgMode.FG_BUCKETIZE]:
                self._inputs = [self.name]
            else:
                self._inputs = [v for _, v in self.side_inputs]
        return self._inputs

    def _need_seq_prefix(self, side: str, name: str) -> bool:
        """Check input fields should add prefix of group sequence or not."""
        if self._is_grouped_seq:
            if (
                hasattr(self.config, "sequence_fields")
                and len(self.config.sequence_fields) > 0
            ):
                return name in self.config.sequence_fields
            else:
                return side == "item"
        else:
            return False

    @property
    def side_inputs(self) -> List[Tuple[str, str]]:
        """Input field names with side."""
        if self._side_inputs is None:
            side_inputs = self._build_side_inputs()
            if not side_inputs:
                raise InvalidFgInputError(
                    f"{self.__class__.__name__}[{self.name}] must have fg "
                    f"input names, e.g., item:cat_a."
                )
            self._side_inputs = []
            for x in side_inputs:
                if not (
                    len(x) == 2
                    and x[0] in ["user", "item", "context", "feature", "const"]
                ):
                    raise InvalidFgInputError(
                        f"{self.__class__.__name__}[{self.name}] must have valid fg "
                        f"input names, e.g., item:cat_a, but got {x}."
                    )
                side, name = x[0], x[1]
                seq_prefix = (
                    f"{self.sequence_name}{self._underline}"
                    if self._need_seq_prefix(side, name)
                    else ""
                )
                self._side_inputs.append((side, f"{seq_prefix}{name}"))
        return self._side_inputs

    @property
    def stub_type(self) -> bool:
        """Only used as fg dag intermediate result or not."""
        if hasattr(self.config, "stub_type") and self.config.HasField("stub_type"):
            return self.config.stub_type
        return False

    def parameter_constraints(
        self, emb_config: Optional[BaseEmbeddingConfig]
    ) -> Optional[ParameterConstraints]:
        """Embedding parameter constraints."""
        if self.config.HasField("embedding_constraints"):
            return build_embedding_constraints(self.config.embedding_constraints)
        elif hasattr(self.config, "dynamicemb") and self.config.HasField("dynamicemb"):
            emb_config = emb_config if emb_config is not None else self.emb_config
            assert emb_config is not None
            return dynamicemb_util.build_dynamicemb_constraints(
                self.config.dynamicemb, emb_config
            )
        else:
            return None

    def _build_side_inputs(self) -> Optional[List[Tuple[str, str]]]:
        """Build input field names with side."""
        return NotImplemented

    def _parse(self, input_data: Dict[str, pa.Array]) -> ParsedData:
        """Parse input data for the feature impl.

        Args:
            input_data (dict): raw input feature data.

        Return:
            parsed feature data.
        """
        raise NotImplementedError

    def parse(
        self, input_data: Dict[str, pa.Array], is_training: bool = False
    ) -> ParsedData:
        """Parse input data for the feature.

        Args:
            input_data (dict): raw input feature data.
            is_training (bool): is training or not.

        Return:
            parsed feature data.
        """
        if is_training and self.config.use_mask:
            t_input_data = {}
            i = 0
            for name in self.inputs:
                data = input_data[name]
                if i == 0 and not pa.types.is_map(data.type):
                    mask = (
                        input_data[C_NEG_SAMPLE_MASK]
                        if self.is_neg
                        else input_data[C_SAMPLE_MASK]
                    )
                    data = pa.compute.if_else(mask, pa.nulls(len(data)), data)
                    i += 1
                t_input_data[name] = data
        else:
            t_input_data = input_data

        parsed_data = self._parse(t_input_data)
        return parsed_data

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
            if self.is_sequence:
                if self.is_sparse:
                    parsed_feat = _parse_fg_encoded_sequence_sparse_feature_impl(
                        self.name,
                        feat,
                        sequence_delim=self.sequence_delim,
                        **self._fg_encoded_kwargs,
                    )
                else:
                    parsed_feat = _parse_fg_encoded_sequence_dense_feature_impl(
                        self.name,
                        feat,
                        sequence_delim=self.sequence_delim,
                        value_dim=self.config.value_dim,
                        **self._fg_encoded_kwargs,
                    )
            else:
                if self.is_sparse:
                    parsed_feat = _parse_fg_encoded_sparse_feature_impl(
                        self.name,
                        feat,
                        is_weighted=self._is_weighted,
                        **self._fg_encoded_kwargs,
                    )
                else:
                    parsed_feat = _parse_fg_encoded_dense_feature_impl(
                        self.name, feat, **self._fg_encoded_kwargs
                    )
        elif self.fg_mode == FgMode.FG_NORMAL:
            fgout, status = self._fg_op.process_arrow(input_data)
            assert status.ok(), status.message()
            feat_data = fgout[self.name]
            if self.is_sequence:
                if self.is_sparse:
                    parsed_feat = SequenceSparseData(
                        name=self.name,
                        values=feat_data.np_values,
                        key_lengths=feat_data.np_key_lengths,
                        seq_lengths=feat_data.np_lengths,
                    )
                else:
                    parsed_feat = SequenceDenseData(
                        name=self.name,
                        values=feat_data.dense_values,
                        seq_lengths=feat_data.np_lengths,
                    )
            else:
                if self.is_sparse:
                    parsed_feat = SparseData(
                        name=self.name,
                        values=feat_data.np_values,
                        lengths=feat_data.np_lengths,
                        weights=feat_data.np_weights if self._is_weighted else None,
                    )
                else:
                    parsed_feat = DenseData(
                        name=self.name, values=feat_data.dense_values
                    )
        else:
            raise ValueError(
                f"fg_mode: {self.fg_mode} is not supported without fg handler."
            )
        return parsed_feat

    def init_fg(self) -> None:
        """Init fg op."""
        if self._fg_op is None:
            cfgs = self.fg_json()
            if self._is_grouped_seq:
                self._fg_op = pyfg.FgArrowHandler(
                    {
                        "features": [
                            {
                                "sequence_name": self.sequence_name,
                                "sequence_length": self.sequence_length,
                                "sequence_delim": self.sequence_delim,
                                "sequence_pk": self.sequence_pk,
                                "features": cfgs,
                            }
                        ]
                    },
                    1,
                )
            else:
                # pyre-ignore [16]
                self._fg_op = pyfg.FgArrowHandler({"features": cfgs}, 1)

    def fg_json(self) -> List[Dict[str, Any]]:
        """Get fg json config."""
        fg_cfgs = self._fg_json()
        if self.is_sequence:
            for fg_cfg in fg_cfgs:
                if self.config.default_value == "":
                    logger.warning(
                        f"Sequence{self.__class__.__name__}[{self.name}]  "
                        "not support empty default value now. reset to zero."
                    )
                    fg_cfg["default_value"] = "0"
                if not self._is_grouped_seq:
                    fg_cfg["sequence_delim"] = self.sequence_delim
                    fg_cfg["sequence_length"] = self.sequence_length
                    fg_cfg["feature_type"] = f"sequence_{fg_cfg['feature_type']}"
        return fg_cfgs

    def _fg_json(self) -> List[Dict[str, Any]]:
        """Get fg json config impl."""
        raise NotImplementedError

    def fg_encoded_default_value(self) -> Optional[Union[List[int], List[float]]]:
        """Get fg encoded default value."""
        if self.config.HasField("fg_encoded_default_value"):
            if self.config.fg_encoded_default_value == "":
                return None
            if self.is_sparse:
                return list(
                    map(
                        int,
                        self.config.fg_encoded_default_value.split(
                            self._fg_encoded_multival_sep
                        ),
                    )
                )
            else:
                return list(
                    map(
                        float,
                        self.config.fg_encoded_default_value.split(
                            self._fg_encoded_multival_sep
                        ),
                    )
                )
        else:
            # we try to initialize fg to get fg_encoded_default_value
            self.init_fg()
            # pyre-ignore [16]
            output, status = self._fg_op({x: [None] for _, x in self.side_inputs})
            assert status.ok(), status.message()
            default_value = output[self.name][0]
            self._fg_op.reset_executor()
            return default_value

    @property
    def vocab_list(self) -> List[str]:
        """Vocab list."""
        if self._vocab_list is None:
            if len(self.config.vocab_list) > 0:
                if self.config.HasField("default_bucketize_value"):
                    # when set default_bucketize_value, we do not add additional
                    # `default_value` and <OOV> vocab to vocab_list
                    assert self.config.default_bucketize_value < len(
                        self.config.vocab_list
                    ), (
                        "default_bucketize_value should be less than len(vocab_list) "
                        f"in {self.__class__.__name__}[{self.name}]"
                    )
                    self._vocab_list = list(self.config.vocab_list)
                else:
                    self._vocab_list = [self.config.default_value, "<OOV>"] + list(
                        self.config.vocab_list
                    )
            else:
                self._vocab_list = []
        return self._vocab_list

    @property
    def vocab_dict(self) -> Dict[str, int]:
        """Vocab dict."""
        if self._vocab_dict is None:
            if len(self.config.vocab_dict) > 0:
                vocab_dict = OrderedDict(self.config.vocab_dict.items())
                if self.config.HasField("default_bucketize_value"):
                    # when set default_bucketize_value, we do not add additional
                    # `default_value` and <OOV> vocab to vocab_dict
                    self._vocab_dict = vocab_dict
                else:
                    is_rank_zero = os.environ.get("RANK", "0") == "0"
                    if min(list(self.config.vocab_dict.values())) <= 1 and is_rank_zero:
                        logger.warn(
                            "min index of vocab_dict in "
                            f"{self.__class__.__name__}[{self.name}] should "
                            "start from 2. index0 is default_value, index1 is <OOV>."
                        )
                    vocab_dict[self.config.default_value] = 0
                    self._vocab_dict = vocab_dict
            else:
                self._vocab_dict = {}
        return self._vocab_dict

    @property
    def vocab_file(self) -> str:
        """Vocab file."""
        if self.config.HasField("vocab_file"):
            if not self.config.HasField("default_bucketize_value"):
                raise ValueError(
                    "default_bucketize_value must be set when use vocab_file."
                )
            vocab_file = self.config.vocab_file
            if self.config.HasField("asset_dir"):
                vocab_file = os.path.join(self.config.asset_dir, vocab_file)
            return vocab_file
        else:
            return ""

    @property
    def vocab_file_size(self) -> int:
        """Vocab file size."""
        if len(self.vocab_file) > 0:
            vocab_dict = dict()
            has_value = False
            with open(self.vocab_file) as f:
                for line in f.readlines():
                    tokens = line.strip().split(maxsplit=1)
                    if len(tokens) > 1:
                        vocab_dict[tokens[0]] = int(tokens[1])
                        has_value = True
                    else:
                        vocab_dict[tokens[0]] = 0
            if has_value:
                return max(list(vocab_dict.values())) + 1
            else:
                return len(vocab_dict)
        else:
            return ""

    @property
    def default_bucketize_value(self) -> int:
        """Default bucketize value."""
        if self.config.HasField("default_bucketize_value"):
            return self.config.default_bucketize_value
        else:
            return 1

    def assets(self) -> Dict[str, str]:
        """Asset file paths."""
        return {}

    def __del__(self) -> None:
        # pyre-ignore [16]
        if self._fg_op and isinstance(self._fg_op, pyfg.FgArrowHandler):
            self._fg_op.reset_executor()


def create_features(
    feature_configs: List[FeatureConfig],
    fg_mode: FgMode = FgMode.FG_NONE,
    neg_fields: Optional[List[str]] = None,
    fg_encoded_multival_sep: Optional[str] = None,
    force_base_data_group: bool = False,
) -> List[BaseFeature]:
    """Build feature list from feature config.

    Args:
        feature_configs (list): list of feature_config.
        fg_mode (FgMode): input data fg mode.
        neg_fields (list, optional): negative sampled input fields.
        fg_encoded_multival_sep (str, optional): multival_sep when fg_mode=FG_NONE
        force_base_data_group (bool): force padding data into same
            data group with same batch_size.

    Return:
        features: list of Feature.
    """
    features = []
    for feat_config in feature_configs:
        feat_type = feat_config.WhichOneof("feature")
        oneof_feat_config = getattr(feat_config, feat_type)
        feat_cls_name = oneof_feat_config.__class__.__name__
        if feat_cls_name == "SequenceFeature":
            sequence_name = oneof_feat_config.sequence_name
            sequence_delim = oneof_feat_config.sequence_delim
            sequence_length = oneof_feat_config.sequence_length
            sequence_pk = oneof_feat_config.sequence_pk
            for sub_feat_config in oneof_feat_config.features:
                sub_feat_cls_name = config_util.which_msg(sub_feat_config, "feature")
                # pyre-ignore [16]
                feature = BaseFeature.create_class(sub_feat_cls_name)(
                    sub_feat_config,
                    fg_mode=fg_mode,
                    fg_encoded_multival_sep=fg_encoded_multival_sep,
                    is_sequence=True,
                    sequence_name=sequence_name,
                    sequence_delim=sequence_delim,
                    sequence_length=sequence_length,
                    sequence_pk=sequence_pk,
                )
                features.append(feature)
        else:
            feature = BaseFeature.create_class(feat_cls_name)(
                feat_config,
                fg_mode=fg_mode,
                fg_encoded_multival_sep=fg_encoded_multival_sep,
                is_sequence=feat_type.startswith("sequence_"),
            )
            features.append(feature)

    has_dag = False
    for feature in features:
        if neg_fields:
            if len(set(feature.inputs) & set(neg_fields)):
                feature.is_neg = True
        if force_base_data_group:
            feature.data_group = BASE_DATA_GROUP
        try:
            side_inputs = feature.side_inputs
            for k, _ in side_inputs:
                if k == "feature":
                    has_dag = True
                    break
        except InvalidFgInputError:
            pass

    if has_dag:
        fg_json = create_fg_json(features)
        # pyre-ignore [16]
        fg_handler = pyfg.FgArrowHandler(fg_json, 1)
        user_feats = fg_handler.user_features() | set(
            fg_handler.sequence_feature_to_name().keys()
        )
        for feature in features:
            feature.is_user_feat = feature.name in user_feats

    return features


def _copy_assets(
    feature: BaseFeature,
    asset_dir: Optional[str] = None,
    use_relative_asset_dir: bool = False,
) -> BaseFeature:
    if asset_dir and len(feature.assets()) > 0:
        # deepcopy feature config
        feature_config = type(feature.feature_config)()
        feature_config.CopyFrom(feature.feature_config)
        feature = copy(feature)
        feature.feature_config = feature_config
        for k, v in feature.assets().items():
            with open(v, "rb") as f:
                fhash = hashlib.md5(f.read()).hexdigest()
            fprefix, fext = os.path.splitext(os.path.basename(v))
            fname = f"{fprefix}_{fhash}{fext}"
            fpath = os.path.join(asset_dir, fname)
            if not os.path.exists(fpath):
                shutil.copy(v, fpath)
            config_util.edit_config(feature.config, {k: fname})
            if not use_relative_asset_dir:
                feature.config.asset_dir = asset_dir
            else:
                feature.config.ClearField("asset_dir")
    return feature


def _remove_one_feature_bucketizer(fg_json: Dict[str, Any]) -> Dict[str, Any]:
    fg_json.pop("hash_bucket_size", None)
    fg_json.pop("vocab_dict", None)
    fg_json.pop("vocab_list", None)
    fg_json.pop("boundaries", None)
    fg_json.pop("num_buckets", None)
    if fg_json["feature_type"] != "tokenize_feature":
        fg_json.pop("vocab_file", None)
    return fg_json


def create_fg_json(
    features: List[BaseFeature],
    asset_dir: Optional[str] = None,
    remove_bucketizer: bool = False,
) -> Dict[str, Any]:
    """Create feature generate config for features."""
    results = []
    seq_to_idx = {}
    for feature in features:
        feature = _copy_assets(feature, asset_dir, use_relative_asset_dir=True)
        if feature.is_grouped_sequence:
            # pyre-ignore [16]
            if feature.sequence_name not in seq_to_idx:
                results.append(
                    {
                        "sequence_name": feature.sequence_name,
                        "sequence_length": feature.sequence_length,  # pyre-ignore [16]
                        "sequence_delim": feature.sequence_delim,  # pyre-ignore [16]
                        "sequence_pk": feature.sequence_pk,  # pyre-ignore [16]
                        "features": [],
                    }
                )
                seq_to_idx[feature.sequence_name] = len(results) - 1
            fg_json = feature.fg_json()
            if remove_bucketizer:
                fg_json = [_remove_one_feature_bucketizer(x) for x in fg_json]
            idx = seq_to_idx[feature.sequence_name]
            results[idx]["features"].extend(fg_json)
        else:
            fg_json = feature.fg_json()
            if remove_bucketizer:
                fg_json = [_remove_one_feature_bucketizer(x) for x in fg_json]
            results.extend(fg_json)
    return {"features": results}


def create_feature_configs(
    features: List[BaseFeature], asset_dir: Optional[str] = None
) -> List[FeatureConfig]:
    """Create feature configs for features."""
    results = OrderedDict()
    for feature in features:
        feature = _copy_assets(feature, asset_dir)
        if feature.is_grouped_sequence:
            # pyre-ignore [16]
            if feature.sequence_name not in results:
                results[feature.sequence_name] = FeatureConfig(
                    sequence_feature=SequenceFeature(
                        sequence_name=feature.sequence_name,
                        sequence_length=feature.sequence_length,  # pyre-ignore [16]
                        sequence_delim=feature.sequence_delim,  # pyre-ignore [16]
                        sequence_pk=feature.sequence_pk,  # pyre-ignore [16]
                    )
                )
            results[feature.sequence_name].sequence_feature.features.append(
                feature.feature_config
            )
        else:
            results[feature.name] = feature.feature_config
    return list(results.values())
