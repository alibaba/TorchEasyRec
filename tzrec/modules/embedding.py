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

from collections import OrderedDict, defaultdict
from typing import Dict, List, Optional, Union

import torch
from torch import nn
from torchrec.modules.embedding_configs import EmbeddingBagConfig, EmbeddingConfig
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
    EmbeddingCollection,
)
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor, KeyedTensor

from tzrec.acc.utils import is_input_tile, is_input_tile_emb
from tzrec.datasets.utils import Batch
from tzrec.features.feature import BaseFeature
from tzrec.modules.sequence import create_seq_encoder
from tzrec.protos import model_pb2
from tzrec.protos.model_pb2 import FeatureGroupConfig, SeqGroupConfig


@torch.fx.wrap
def _update_dict_tensor(
    dict1: Dict[str, torch.Tensor], dict2: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    for key, value in dict2.items():
        if key in dict1:
            dict1[key] = torch.cat([dict1[key], value], dim=-1)
        else:
            dict1[key] = value
    return dict1


@torch.fx.wrap
def _merge_list_of_tensor_dict(
    list_of_tensor_dict: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    result = {}
    for tensor_dict in list_of_tensor_dict:
        result.update(tensor_dict)
    return result


@torch.fx.wrap
def _merge_list_of_jt_dict(
    list_of_jt_dict: List[Dict[str, JaggedTensor]],
) -> Dict[str, JaggedTensor]:
    result: Dict[str, JaggedTensor] = {}
    for jt_dict in list_of_jt_dict:
        result.update(jt_dict)
    return result


@torch.fx.wrap
def _cast_away_kjt_optional(arg: Optional[KeyedJaggedTensor]) -> KeyedJaggedTensor:
    assert arg is not None
    return arg


@torch.fx.wrap
def _cast_away_kt_optional(arg: Optional[KeyedTensor]) -> KeyedTensor:
    assert arg is not None
    return arg


@torch.fx.wrap
def _cast_away_jt_dict_optional(
    arg: Optional[Dict[str, JaggedTensor]],
) -> Dict[str, JaggedTensor]:
    assert arg is not None
    return arg


@torch.fx.wrap
def _int_item(x: torch.Tensor) -> int:
    return int(x.item())


class EmbeddingGroup(nn.Module):
    """Applies embedding lookup transformation for feature group.

    Args:
        features (list): list of features.
        feature_groups (list): list of feature group config.
        wide_embedding_dim (int, optional): wide group feature embedding dim.
        device (torch.device): embedding device, default is meta.
    """

    def __init__(
        self,
        features: List[BaseFeature],
        feature_groups: List[FeatureGroupConfig],
        wide_embedding_dim: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        if device is None:
            device = torch.device("meta")
        self._features = features
        self._feature_groups = feature_groups
        self._name_to_feature = {x.name: x for x in features}
        self._name_to_feature_group = {x.group_name: x for x in feature_groups}

        self.emb_impls = nn.ModuleDict()
        self.seq_emb_impls = nn.ModuleDict()
        self.seq_encoders = nn.ModuleDict()

        self._impl_key_to_feat_groups = defaultdict(list)
        self._impl_key_to_seq_groups = defaultdict(list)
        self._group_name_to_impl_key = dict()
        self._group_name_to_seq_encoder_configs = defaultdict(list)

        for feature_group in feature_groups:
            group_name = feature_group.group_name
            self._respect_and_supplement_feature_group(feature_group)
            self._add_feature_group_sign_for_sequence_groups(feature_group)
            features_data_group = defaultdict(list)
            for feature_name in feature_group.feature_names:
                feature = self._name_to_feature[feature_name]
                features_data_group[feature.data_group].append(feature_name)
            for sequence_group in feature_group.sequence_groups:
                for feature_name in sequence_group.feature_names:
                    feature = self._name_to_feature[feature_name]
                    features_data_group[feature.data_group].append(feature_name)

            if len(features_data_group) > 1:
                error_info = [",".join(v) for v in features_data_group.values()]
                raise ValueError(
                    f"Feature {error_info} should not belong to same feature group."
                )
            impl_key = list(features_data_group.keys())[0]
            self._group_name_to_impl_key[group_name] = impl_key
            if feature_group.group_type == model_pb2.SEQUENCE:
                self._impl_key_to_seq_groups[impl_key].append(feature_group)
            else:
                self._impl_key_to_feat_groups[impl_key].append(feature_group)
                if len(feature_group.sequence_groups) > 0:
                    self._impl_key_to_seq_groups[impl_key].extend(
                        list(feature_group.sequence_groups)
                    )
                if len(feature_group.sequence_encoders) > 0:
                    self._group_name_to_seq_encoder_configs[group_name] = list(
                        feature_group.sequence_encoders
                    )

        for k, v in self._impl_key_to_feat_groups.items():
            self.emb_impls[k] = EmbeddingGroupImpl(
                features,
                feature_groups=v,
                wide_embedding_dim=wide_embedding_dim,
                device=device,
            )

        for k, v in self._impl_key_to_seq_groups.items():
            self.seq_emb_impls[k] = SequenceEmbeddingGroupImpl(
                features, feature_groups=v, device=device
            )
        self._group_name_to_seq_encoders = nn.ModuleDict()
        for (
            group_name,
            seq_encoder_configs,
        ) in self._group_name_to_seq_encoder_configs.items():
            impl_key = self._group_name_to_impl_key[group_name]
            seq_emb = self.seq_emb_impls[impl_key]
            group_seq_encoders = nn.ModuleList()
            for seq_encoder_config in seq_encoder_configs:
                seq_encoder = create_seq_encoder(
                    seq_encoder_config, seq_emb.all_group_total_dim()
                )
                group_seq_encoders.append(seq_encoder)
            self._group_name_to_seq_encoders[group_name] = group_seq_encoders

        self._group_feature_dims = OrderedDict()
        for feature_group in feature_groups:
            group_name = feature_group.group_name
            if feature_group.group_type != model_pb2.SEQUENCE:
                feature_dim = OrderedDict()
                impl_key = self._group_name_to_impl_key[group_name]
                feature_emb = self.emb_impls[impl_key]
                feature_dim.update(feature_emb.group_feature_dims(group_name))
                if group_name in self._group_name_to_seq_encoders:
                    seq_encoders = self._group_name_to_seq_encoders[group_name]
                    for i, seq_encoder in enumerate(seq_encoders):
                        feature_dim[f"{group_name}_seq_encoder_{i}"] = (
                            seq_encoder.output_dim()
                        )
                self._group_feature_dims[group_name] = feature_dim

    def _respect_and_supplement_feature_group(
        self, feature_group: FeatureGroupConfig
    ) -> None:
        """Respect feature group sequence_groups and sequence_encoders."""
        group_name = feature_group.group_name
        sequence_groups = list(feature_group.sequence_groups)
        sequence_encoders = list(feature_group.sequence_encoders)
        is_deep = feature_group.group_type == model_pb2.DEEP
        if is_deep:
            if len(sequence_groups) == 0 and sequence_encoders == 0:
                return
            elif len(sequence_groups) > 0 and sequence_encoders == 0:
                raise ValueError(
                    f"{group_name} group has sequence_groups,but no sequence_encoders "
                )
            elif len(sequence_groups) == 0 and len(sequence_encoders) > 0:
                raise ValueError(
                    f"{group_name} group has sequence_encoders,but no sequence_groups "
                )

            if len(sequence_groups) > 1:
                for sequence_group in sequence_groups:
                    if not sequence_group.HasField("group_name"):
                        raise ValueError(
                            f"{group_name} has many sequence_groups, "
                            f"every sequence_group must has group_name"
                        )
            elif len(sequence_groups) == 1 and not sequence_groups[0].HasField(
                "group_name"
            ):
                sequence_groups[0].group_name = group_name

            group_has_encoder = {
                sequence_group.group_name: False for sequence_group in sequence_groups
            }
            for sequence_encoder in sequence_encoders:
                seq_type = sequence_encoder.WhichOneof("seq_module")
                seq_config = getattr(sequence_encoder, seq_type)
                if not seq_config.HasField("input") and len(sequence_groups) == 1:
                    seq_config.input = sequence_groups[0].group_name
                if not seq_config.HasField("input"):
                    raise ValueError(
                        f"{group_name} group has multi sequence_groups, "
                        f"so sequence_encoders must has input"
                    )
                if seq_config.input not in group_has_encoder:
                    raise ValueError(
                        f"{group_name} sequence_encoder input {seq_config.input} "
                        f"not in sequence_groups"
                    )
                else:
                    group_has_encoder[seq_config.input] = True
            for k, v in group_has_encoder.items():
                if not v:
                    raise ValueError(
                        f"{group_name} sequence_groups {k} not has seq_encoder"
                    )
        else:
            if len(sequence_groups) > 0 or len(sequence_encoders) > 0:
                raise ValueError(
                    f"{group_name} group group_type is not DEEP, "
                    f"sequence_groups and sequence_encoders must configured in DEEP"
                )

    def _add_feature_group_sign_for_sequence_groups(
        self, feature_group: FeatureGroupConfig
    ) -> None:
        """Assign sequence_groups and sequence_encoder relation group name."""
        group_name = feature_group.group_name
        sequence_groups = list(feature_group.sequence_groups)
        sequence_encoders = list(feature_group.sequence_encoders)
        if len(sequence_groups) > 0:
            for sequence_group in sequence_groups:
                sequence_group.group_name = (
                    group_name + "___" + sequence_group.group_name
                )
            for sequence_encoder in sequence_encoders:
                seq_type = sequence_encoder.WhichOneof("seq_module")
                seq_config = getattr(sequence_encoder, seq_type)
                seq_config.input = group_name + "___" + seq_config.input

    def group_names(self) -> List[str]:
        """Feature group names."""
        return list(self._name_to_feature_group.keys())

    def group_dims(self, group_name: str) -> List[int]:
        """Output dimension of each feature in a feature group.

        Args:
            group_name (str): feature group name, when group type is sequence,
                should use {group_name}.query or {group_name}.sequence.

        Return:
            group_dims (list): output dimension of each feature.
        """
        true_name = group_name.split(".")[0] if "." in group_name else group_name
        feature_group = self._name_to_feature_group[true_name]
        impl_key = self._group_name_to_impl_key[true_name]
        if feature_group.group_type == model_pb2.SEQUENCE:
            return self.seq_emb_impls[impl_key].group_dims(group_name)
        else:
            dims = self.emb_impls[impl_key].group_dims(group_name)
            if group_name in self._group_name_to_seq_encoders:
                for seq_encoder in self._group_name_to_seq_encoders[group_name]:
                    dims.append(seq_encoder.output_dim())
            return dims

    def group_total_dim(self, group_name: str) -> int:
        """Total output dimension of a feature group.

        Args:
            group_name (str): feature group name, when group type is sequence,
                should use {group_name}.query or {group_name}.sequence.

        Return:
            total_dim (int): total dimension of feature group.
        """
        true_name = group_name.split(".")[0] if "." in group_name else group_name
        feature_group = self._name_to_feature_group[true_name]
        impl_key = self._group_name_to_impl_key[true_name]
        if feature_group.group_type == model_pb2.SEQUENCE:
            return self.seq_emb_impls[impl_key].group_total_dim(group_name)
        else:
            return sum(self._group_feature_dims[group_name].values())

    def group_feature_dims(self, group_name: str) -> Dict[str, int]:
        """Every feature group each feature dim."""
        true_name = group_name.split(".")[0] if "." in group_name else group_name
        feature_group = self._name_to_feature_group[true_name]
        if feature_group.group_type == model_pb2.SEQUENCE:
            raise ValueError("not support sequence group")
        return self._group_feature_dims[group_name]

    def has_group(self, group_name: str) -> bool:
        """Check the feature group exist or not."""
        true_name = group_name.split(".")[0] if "." in group_name else group_name
        return true_name in self._name_to_feature_group.keys()

    def forward(
        self,
        batch: Batch,
    ) -> Dict[str, torch.Tensor]:
        """Forward the module.

        Args:
            batch (Batch): a instance of Batch with features.

        Returns:
            group_features (dict): dict of feature_group to embedded tensor.
        """
        result_dicts = []
        input_tile_emb = is_input_tile_emb()
        input_tile = is_input_tile()
        if input_tile:
            # cat->H2D->tile user dense feat
            emb_keys = list(self.emb_impls.keys())
            seq_emb_keys = list(self.seq_emb_impls.keys())
            unique_keys = list(set(emb_keys + seq_emb_keys))

            for key in unique_keys:
                if key + "_user" in batch.dense_features.keys():
                    dense_feat_kt_user = batch.dense_features[key + "_user"]
                    values_tile = dense_feat_kt_user.values().tile(batch.batch_size, 1)
                    batch.dense_features[key + "_user"] = KeyedTensor(
                        keys=dense_feat_kt_user.keys(),
                        length_per_key=dense_feat_kt_user.length_per_key(),
                        values=values_tile,
                    )

        for key, emb_impl in self.emb_impls.items():
            sparse_feat_kjt = None
            sparse_feat_kjt_user = None
            dense_feat_kt = None
            dense_feat_kt_user = None

            if emb_impl.has_dense:
                if input_tile:
                    dense_feat_kt = batch.dense_features[key + "_item"]
                else:
                    dense_feat_kt = batch.dense_features[key]

            if emb_impl.has_dense_user:
                dense_feat_kt_user = batch.dense_features[key + "_user"]

            if emb_impl.has_sparse:
                if input_tile_emb:
                    sparse_feat_kjt = batch.sparse_features[key + "_item"]
                else:
                    sparse_feat_kjt = batch.sparse_features[key]

            if emb_impl.has_sparse_user:
                sparse_feat_kjt_user = batch.sparse_features[key + "_user"]
            result_dicts.append(
                emb_impl(
                    sparse_feat_kjt,
                    dense_feat_kt,
                    sparse_feat_kjt_user,
                    dense_feat_kt_user,
                    batch.batch_size,
                )
            )

        for key, seq_emb_impl in self.seq_emb_impls.items():
            sparse_feat_kjt = None
            sparse_feat_kjt_user = None
            dense_feat_kt = None
            dense_feat_kt_user = None
            if seq_emb_impl.has_dense:
                if input_tile:
                    dense_feat_kt = batch.dense_features[key + "_item"]
                else:
                    dense_feat_kt = batch.dense_features[key]

            if seq_emb_impl.has_dense_user:
                dense_feat_kt_user = batch.dense_features[key + "_user"]

            if seq_emb_impl.has_sparse:
                if input_tile_emb:
                    sparse_feat_kjt = batch.sparse_features[key + "_item"]
                else:
                    sparse_feat_kjt = batch.sparse_features[key]

            if seq_emb_impl.has_sparse_user:
                sparse_feat_kjt_user = batch.sparse_features[key + "_user"]

            result_dicts.append(
                seq_emb_impl(
                    sparse_feat_kjt,
                    dense_feat_kt,
                    batch.sequence_dense_features,
                    sparse_feat_kjt_user,
                    dense_feat_kt_user,
                    batch.batch_size,
                )
            )

        result = _merge_list_of_tensor_dict(result_dicts)
        seq_feature_dict = {}
        for group_name, seq_encoders in self._group_name_to_seq_encoders.items():
            new_feature = []
            for seq_encoder in seq_encoders:
                new_feature.append(seq_encoder(result))
            seq_feature_dict[group_name] = torch.cat(new_feature, dim=-1)
        return _update_dict_tensor(result, seq_feature_dict)


def add_embedding_bag_config(
    emb_bag_configs: Dict[str, EmbeddingBagConfig], emb_bag_config: EmbeddingBagConfig
) -> None:
    """Add embedding bag config .

    Args:
        emb_bag_configs: Dict[str, EmbeddingBagConfig]: a dict contains emb_bag_configs
        emb_bag_config: EmbeddingBagConfig

    Returns:
        None
    """
    if emb_bag_config.name in emb_bag_configs:
        existed_emb_bag_config = emb_bag_configs[emb_bag_config.name]
        assert (
            emb_bag_config.num_embeddings == existed_emb_bag_config.num_embeddings
            and emb_bag_config.embedding_dim == existed_emb_bag_config.embedding_dim
            and emb_bag_config.pooling == existed_emb_bag_config.pooling
            and repr(emb_bag_config.init_fn) == repr(existed_emb_bag_config.init_fn)
        ), (
            f"there is a mismatch between {emb_bag_config} and "
            f"{existed_emb_bag_config}, can not share embedding."
        )
        for feature_name in emb_bag_config.feature_names:
            if feature_name not in existed_emb_bag_config.feature_names:
                existed_emb_bag_config.feature_names.append(feature_name)
    else:
        emb_bag_configs[emb_bag_config.name] = emb_bag_config


def add_embedding_config(
    emb_configs: Dict[str, EmbeddingConfig], emb_config: EmbeddingConfig
) -> None:
    """Add embedding config .

    Args:
        emb_configs: Dict[str, EmbeddingConfig]: a dict contains emb_configs
        emb_config: EmbeddingConfig

    Returns:
        None
    """
    if emb_config.name in emb_configs:
        existed_emb_config = emb_configs[emb_config.name]
        assert (
            emb_config.num_embeddings == existed_emb_config.num_embeddings
            and emb_config.embedding_dim == existed_emb_config.embedding_dim
            and repr(emb_config.init_fn) == repr(existed_emb_config.init_fn)
        ), (
            f"there is a mismatch between {emb_config} and "
            f"{existed_emb_config}, can not share embedding."
        )
        for feature_name in emb_config.feature_names:
            if feature_name not in existed_emb_config.feature_names:
                existed_emb_config.feature_names.append(feature_name)
    else:
        emb_configs[emb_config.name] = emb_config


class EmbeddingGroupImpl(nn.Module):
    """Applies embedding lookup transformation for feature group.

    Args:
        features (list): list of features.
        feature_groups (list): list of feature group config.
        wide_embedding_dim (int, optional): wide group feature embedding dim.
        device (torch.device): embedding device, default is meta.
    """

    def __init__(
        self,
        features: List[BaseFeature],
        feature_groups: List[FeatureGroupConfig],
        wide_embedding_dim: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        if device is None:
            device = torch.device("meta")
        name_to_feature = {x.name: x for x in features}
        emb_bag_configs = OrderedDict()

        self.has_sparse = False
        self.has_sparse_user = False
        self.has_dense = False
        self.has_dense_user = False

        self._group_to_feature_names = OrderedDict()
        self._group_to_shared_feature_names = OrderedDict()
        self._group_total_dim = dict()
        self._group_feature_output_dims = dict()

        feat_to_group_to_emb_name = defaultdict(dict)
        for feature_group in feature_groups:
            group_name = feature_group.group_name
            for feature_name in feature_group.feature_names:
                feature = name_to_feature[feature_name]
                if feature.is_sparse:
                    emb_bag_config = feature.emb_bag_config
                    # pyre-ignore [16]
                    emb_name = emb_bag_config.name
                    if feature_group.group_type == model_pb2.WIDE:
                        emb_name = emb_name + "_wide"
                    feat_to_group_to_emb_name[feature_name][group_name] = emb_name

        shared_feature_flag = dict()
        for feature_name, group_to_emb_name in feat_to_group_to_emb_name.items():
            if len(set(group_to_emb_name.values())) > 1:
                shared_feature_flag[feature_name] = True
            else:
                shared_feature_flag[feature_name] = False

        input_tile_emb = is_input_tile_emb()
        input_tile = is_input_tile()
        emb_bag_configs_user = OrderedDict()
        emb_bag_configs_item = OrderedDict()

        for feature_group in feature_groups:
            total_dim = 0
            feature_output_dims = OrderedDict()
            group_name = feature_group.group_name
            feature_names = list(feature_group.feature_names)
            shared_feature_names = []

            is_wide = feature_group.group_type == model_pb2.WIDE
            for name in feature_names:
                shared_name = name
                feature = name_to_feature[name]
                if feature.is_sparse:
                    output_dim = feature.output_dim
                    emb_bag_config = feature.emb_bag_config
                    assert emb_bag_config is not None
                    if is_wide:
                        # TODO(hongsheng.jhs): change to embedding_dim to 1
                        # when fbgemm support embedding_dim=1
                        emb_bag_config.embedding_dim = output_dim = (
                            wide_embedding_dim or 4
                        )
                    # we may modify ebc name at feat_to_group_to_emb_name, e.g., wide
                    emb_bag_config.name = feat_to_group_to_emb_name[name][group_name]

                    if input_tile_emb:
                        if feature.is_user_feat:
                            add_embedding_bag_config(
                                emb_bag_configs=emb_bag_configs_user,
                                emb_bag_config=emb_bag_config,
                            )
                        else:
                            add_embedding_bag_config(
                                emb_bag_configs=emb_bag_configs_item,
                                emb_bag_config=emb_bag_config,
                            )
                    else:
                        add_embedding_bag_config(
                            emb_bag_configs=emb_bag_configs,
                            emb_bag_config=emb_bag_config,
                        )
                    if input_tile_emb and feature.is_user_feat:
                        self.has_sparse_user = True
                    else:
                        self.has_sparse = True

                    if shared_feature_flag[name]:
                        shared_name = shared_name + "@" + emb_bag_config.name
                else:
                    output_dim = feature.output_dim
                    # TODO (hongsheng.jhs) make dense feature support embedding_dim
                    if is_wide:
                        raise ValueError(
                            "dense feature should not be configured in wide group."
                        )
                    else:
                        if input_tile and feature.is_user_feat:
                            self.has_dense_user = True
                        else:
                            self.has_dense = True
                total_dim += output_dim
                feature_output_dims[name] = output_dim
                shared_feature_names.append(shared_name)
            self._group_to_feature_names[group_name] = feature_names
            if len(shared_feature_names) > 0:
                self._group_to_shared_feature_names[group_name] = shared_feature_names
            self._group_total_dim[group_name] = total_dim
            self._group_feature_output_dims[group_name] = feature_output_dims

        if input_tile_emb:
            self.ebc_user = EmbeddingBagCollection(
                list(emb_bag_configs_user.values()), device=device
            )
            self.ebc_item = EmbeddingBagCollection(
                list(emb_bag_configs_item.values()), device=device
            )
        else:
            self.ebc = EmbeddingBagCollection(
                list(emb_bag_configs.values()), device=device
            )

    def group_dims(self, group_name: str) -> List[int]:
        """Output dimension of each feature in a feature group."""
        return list(self._group_feature_output_dims[group_name].values())

    def group_feature_dims(self, group_name: str) -> Dict[str, int]:
        """Output dimension of each feature in a feature group."""
        return self._group_feature_output_dims[group_name]

    def group_total_dim(self, group_name: str) -> int:
        """Total output dimension of a feature group."""
        return self._group_total_dim[group_name]

    def forward(
        self,
        sparse_feature: KeyedJaggedTensor,
        dense_feature: KeyedTensor,
        sparse_feature_user: KeyedJaggedTensor,
        dense_feature_user: KeyedTensor,
        batch_size: int = -1,
    ) -> Dict[str, torch.Tensor]:
        """Forward the module."""
        kts: List[KeyedTensor] = []
        if self.has_sparse:
            if is_input_tile_emb():
                kts.append(self.ebc_item(sparse_feature))
            else:
                kts.append(self.ebc(sparse_feature))

        if self.has_sparse_user:
            keyed_tensor_user = self.ebc_user(sparse_feature_user)
            values_tile = keyed_tensor_user.values().tile(batch_size, 1)
            keyed_tensor_user_tile = KeyedTensor(
                keys=keyed_tensor_user.keys(),
                length_per_key=keyed_tensor_user.length_per_key(),
                values=values_tile,
            )
            kts.append(keyed_tensor_user_tile)

        if self.has_dense:
            kts.append(dense_feature)

        if self.has_dense_user:
            kts.append(dense_feature_user)

        group_tensors = KeyedTensor.regroup_as_dict(
            kts,
            list(self._group_to_shared_feature_names.values()),
            list(self._group_to_shared_feature_names.keys()),
        )

        return group_tensors


class SequenceEmbeddingGroupImpl(nn.Module):
    """Applies embedding lookup transformation for sequence feature group.

    Args:
        features (list): list of features.
        feature_groups (list): list of feature group config or seq group config.
        device (torch.device): embedding device, default is meta.
    """

    def __init__(
        self,
        features: List[BaseFeature],
        feature_groups: List[Union[FeatureGroupConfig, SeqGroupConfig]],
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        if device is None:
            device = torch.device("meta")
        name_to_feature = {x.name: x for x in features}
        dim_to_emb_configs = defaultdict(OrderedDict)

        self.has_sparse = False
        self.has_sparse_user = False
        self.has_dense = False
        self.has_dense_user = False
        self.has_sequence_dense = False

        self._group_to_shared_query = OrderedDict()
        self._group_to_shared_sequence = OrderedDict()
        self._group_total_dim = dict()
        self._group_output_dims = dict()

        feat_to_group_to_emb_name = defaultdict(dict)
        for feature_group in feature_groups:
            group_name = feature_group.group_name
            for feature_name in feature_group.feature_names:
                feature = name_to_feature[feature_name]
                if feature.is_sparse:
                    emb_config = feature.emb_config
                    # pyre-ignore [16]
                    emb_name = emb_config.name
                    feat_to_group_to_emb_name[feature_name][group_name] = emb_name

        shared_feature_flag = dict()
        for feature_name, group_to_emb_name in feat_to_group_to_emb_name.items():
            if len(set(group_to_emb_name.values())) > 1:
                shared_feature_flag[feature_name] = True
            else:
                shared_feature_flag[feature_name] = False

        need_input_tile = is_input_tile()
        need_input_tile_emb = is_input_tile_emb()
        dim_to_emb_configs_user = defaultdict(OrderedDict)
        dim_to_emb_configs_item = defaultdict(OrderedDict)

        for feature_group in feature_groups:
            query_dim = 0
            sequence_dim = 0
            query_dims = []
            sequence_dims = []
            group_name = feature_group.group_name
            feature_names = list(feature_group.feature_names)
            shared_query = []
            shared_sequence = []

            for name in feature_names:
                shared_name = name
                feature = name_to_feature[name]
                if feature.is_sparse:
                    output_dim = feature.output_dim
                    emb_config = feature.emb_config
                    # we may/could modify ec name at feat_to_group_to_emb_name
                    emb_config.name = feat_to_group_to_emb_name[name][group_name]
                    assert emb_config is not None
                    emb_configs = dim_to_emb_configs[emb_config.embedding_dim]

                    if need_input_tile_emb:
                        if feature.is_user_feat:
                            add_embedding_config(
                                emb_configs=dim_to_emb_configs_user[
                                    emb_config.embedding_dim
                                ],
                                emb_config=emb_config,
                            )
                        else:
                            add_embedding_config(
                                emb_configs=dim_to_emb_configs_item[
                                    emb_config.embedding_dim
                                ],
                                emb_config=emb_config,
                            )
                    else:
                        add_embedding_config(
                            emb_configs=emb_configs,
                            emb_config=emb_config,
                        )
                    if need_input_tile_emb and feature.is_user_feat:
                        self.has_sparse_user = True
                    else:
                        self.has_sparse = True
                    if shared_feature_flag[name]:
                        shared_name = shared_name + "@" + emb_config.name
                else:
                    output_dim = feature.output_dim
                    if feature.is_sequence:
                        self.has_sequence_dense = True
                    else:
                        if need_input_tile and feature.is_user_feat:
                            self.has_dense_user = True
                        else:
                            self.has_dense = True

                is_user_feat = feature.is_user_feat if need_input_tile else False
                if feature.is_sequence:
                    shared_sequence.append(
                        (shared_name, feature.is_sparse, is_user_feat)
                    )
                    sequence_dim += output_dim
                    sequence_dims.append(output_dim)
                else:
                    shared_query.append((shared_name, feature.is_sparse, is_user_feat))
                    query_dim += output_dim
                    query_dims.append(output_dim)

            self._group_to_shared_query[group_name] = shared_query
            self._group_to_shared_sequence[group_name] = shared_sequence
            self._group_total_dim[f"{group_name}.query"] = query_dim
            self._group_total_dim[f"{group_name}.sequence"] = sequence_dim
            self._group_output_dims[f"{group_name}.query"] = query_dims
            self._group_output_dims[f"{group_name}.sequence"] = sequence_dims

        if need_input_tile_emb:
            self.ec_list_user = nn.ModuleList()
            for _, emb_configs in dim_to_emb_configs_user.items():
                self.ec_list_user.append(
                    EmbeddingCollection(list(emb_configs.values()), device=device)
                )

            self.ec_list_item = nn.ModuleList()
            for _, emb_configs in dim_to_emb_configs_item.items():
                self.ec_list_item.append(
                    EmbeddingCollection(list(emb_configs.values()), device=device)
                )
        else:
            self.ec_list = nn.ModuleList()
            for _, emb_configs in dim_to_emb_configs.items():
                self.ec_list.append(
                    EmbeddingCollection(list(emb_configs.values()), device=device)
                )

    def group_dims(self, group_name: str) -> List[int]:
        """Output dimension of each feature in a feature group."""
        return self._group_output_dims[group_name]

    def group_total_dim(self, group_name: str) -> int:
        """Total output dimension of a feature group."""
        return self._group_total_dim[group_name]

    def all_group_total_dim(self) -> Dict[str, int]:
        """Total output dimension of all feature group."""
        return self._group_total_dim

    def has_group(self, group_name: str) -> bool:
        """Check the feature group exist or not."""
        true_name = group_name.split(".")[0] if "." in group_name else group_name
        query_name = true_name + ".query"
        sequnce_name = true_name + ".sequence"
        return (
            query_name in self._group_output_dims.keys()
            and sequnce_name in self._group_output_dims.keys()
        )

    def forward(
        self,
        sparse_feature: KeyedJaggedTensor,
        dense_feature: KeyedTensor,
        sequence_dense_features: Dict[str, JaggedTensor],
        sparse_feature_user: KeyedJaggedTensor,
        dense_feature_user: KeyedTensor,
        batch_size: int = -1,
    ) -> Dict[str, torch.Tensor]:
        """Forward the module."""
        # TODO (hongsheng.jhs): deal with tag sequence feature.
        sparse_jt_dict_list = []
        dense_t_dict = {}
        dense_t_dict_user = {}

        need_input_tile = is_input_tile()
        need_input_tile_emb = is_input_tile_emb()
        if self.has_sparse:
            if need_input_tile_emb:
                for ec in self.ec_list_item:
                    sparse_jt_dict_list.append(ec(sparse_feature))
            else:
                for ec in self.ec_list:
                    sparse_jt_dict_list.append(ec(sparse_feature))

        if self.has_sparse_user:
            for ec in self.ec_list_user:
                sparse_jt_dict_list.append(ec(sparse_feature_user))

        sparse_jt_dict = _merge_list_of_jt_dict(sparse_jt_dict_list)

        if self.has_dense:
            dense_t_dict = dense_feature.to_dict()

        if self.has_dense_user:
            dense_t_dict_user = dense_feature_user.to_dict()

        results = {}
        for group_name, v in self._group_to_shared_query.items():
            query_t_list = []
            for name, is_sparse, is_user in v:
                if is_sparse:
                    # TODO(hongsheng.jhs): support multi-value id feature
                    query_t = sparse_jt_dict[name].to_padded_dense(1).squeeze(1)
                    if is_user and need_input_tile_emb:
                        query_t = query_t.tile(batch_size, 1)
                else:
                    if is_user and need_input_tile:
                        query_t = dense_t_dict_user[name]
                    else:
                        query_t = dense_t_dict[name]
                query_t_list.append(query_t)
            if len(query_t_list) > 0:
                results[f"{group_name}.query"] = torch.cat(query_t_list, dim=1)
        for group_name, v in self._group_to_shared_sequence.items():
            seq_t_list = []

            group_sequence_length = 1
            for i, (name, is_sparse, is_user) in enumerate(v):
                # when is_user is True
                #   sequence_sparse_features
                #       when input_tile_emb need to tile(batch_size,1):
                #   sequence_dense_features always need to tile
                need_tile = is_user and (
                    (need_input_tile and not is_sparse) or need_input_tile_emb
                )
                jt = (
                    sparse_jt_dict[name] if is_sparse else sequence_dense_features[name]
                )
                if i == 0:
                    sequence_length = jt.lengths()
                    group_sequence_length = _int_item(torch.max(sequence_length))
                    if need_tile:
                        results[f"{group_name}.sequence_length"] = sequence_length.tile(
                            batch_size
                        )
                    else:
                        results[f"{group_name}.sequence_length"] = sequence_length
                jt = jt.to_padded_dense(group_sequence_length)

                if need_tile:
                    jt = jt.tile(batch_size, 1, 1)
                seq_t_list.append(jt)

            if seq_t_list:
                results[f"{group_name}.sequence"] = torch.cat(seq_t_list, dim=2)

        return results
