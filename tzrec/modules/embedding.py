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
from torch import Tensor, nn
from torchrec.modules.embedding_configs import EmbeddingBagConfig, EmbeddingConfig
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
    EmbeddingCollection,
)
from torchrec.modules.mc_embedding_modules import (
    ManagedCollisionEmbeddingBagCollection,
    ManagedCollisionEmbeddingCollection,
)
from torchrec.modules.mc_modules import (
    ManagedCollisionCollection,
    ManagedCollisionModule,
    MCHManagedCollisionModule,
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
        self._grouped_features_keys = list()

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
                self._grouped_features_keys.append(group_name + ".query")
                self._grouped_features_keys.append(group_name + ".sequence")
                self._grouped_features_keys.append(group_name + ".sequence_length")
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
                self._grouped_features_keys.append(group_name)

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

        self._grouped_features_keys.sort()

    def grouped_features_keys(self) -> List[str]:
        """grouped_features_keys."""
        return self._grouped_features_keys

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

    def predict(
        self,
        batch: Batch,
    ) -> List[torch.Tensor]:
        """Predict embedding module and return a list of grouped embedding features."""
        grouped_features = self.forward(batch)
        values_list = []
        for key in self._grouped_features_keys:
            values_list.append(grouped_features[key])
        return values_list


def _add_embedding_bag_config(
    emb_bag_configs: Dict[str, EmbeddingBagConfig], emb_bag_config: EmbeddingBagConfig
) -> None:
    """Add embedding bag config to a dict of embedding bag config.

    Args:
        emb_bag_configs(Dict[str, EmbeddingBagConfig]): a dict contains emb_bag_configs
        emb_bag_config(EmbeddingBagConfig): an instance of EmbeddingBagConfig
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


def _add_embedding_config(
    emb_configs: Dict[str, EmbeddingConfig], emb_config: EmbeddingConfig
) -> None:
    """Add embedding config to a dict of embedding config.

    Args:
        emb_configs(Dict[str, EmbeddingConfig]): a dict contains emb_configs
        emb_config(EmbeddingConfig): an instance of EmbeddingConfig
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


def _add_mc_module(
    mc_modules: Dict[str, ManagedCollisionModule],
    emb_name: str,
    mc_module: ManagedCollisionModule,
) -> None:
    """Add ManagedCollisionModule to a dict of ManagedCollisionModule.

    Args:
        mc_modules(Dict[str, ManagedCollisionModule]): a dict of ManagedCollisionModule.
        emb_name(str): embedding_name.
        mc_module(ManagedCollisionModule): an instance of ManagedCollisionModule.
    """
    if emb_name in mc_modules:
        existed_mc_module = mc_modules[emb_name]
        if isinstance(mc_module, MCHManagedCollisionModule):
            assert isinstance(existed_mc_module, MCHManagedCollisionModule)
            assert mc_module._zch_size == existed_mc_module._zch_size
            assert mc_module._eviction_interval == existed_mc_module._eviction_interval
            assert repr(mc_module._eviction_policy) == repr(mc_module._eviction_policy)
    mc_modules[emb_name] = mc_module


class AutoDisModule(nn.Module):
    """An Embedding Learning Framework for Numerical Features in CTR Prediction.

    https://arxiv.org/pdf/2012.08986
    """

    def __init__(
        self,
        num_dense_feature: int,
        embedding_dim: int,
        num_channels: int,
        temperature: float = 0.1,
        keep_prob: float = 0.8,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.num_dense_feature = num_dense_feature
        self.embedding_dim = embedding_dim
        self.keep_prob = keep_prob
        self.temperature = temperature

        self.meta_emb = nn.Parameter(
            torch.randn(num_dense_feature, num_channels, embedding_dim, device=device)
        )
        self.proj_w = nn.Parameter(
            torch.randn(num_dense_feature, num_channels, device=device)
        )
        self.proj_m = nn.Parameter(
            torch.randn(num_dense_feature, num_channels, num_channels, device=device)
        )
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, dense_input: Tensor):
        """Forward the module.

        Args:
            dense_input (Tensor): dense input feature, shape = [b, n],
                where b is batch_size, n is the number of dense features

        Returns:
            atde (Tensor): Tensor of autodis embedding.
        """
        hidden = self.leaky_relu(
            torch.einsum("nc,bn->bnc", self.proj_w, dense_input)
        )  # shape [b, n, c]
        x_bar = (
            torch.einsum("nij,bnj->bni", self.proj_m, hidden) + self.keep_prob * hidden
        )  # shape [b, n, c]
        x_hat = nn.Softmax(dim=-1)(x_bar / self.temperature)  # shape = [b, n, c]
        emb = torch.einsum("ncd,bnc->bnd", self.meta_emb, x_hat)  # shape = [b, n, d]
        output = emb.reshape(
            (-1, self.num_dense_feature * self.embedding_dim)
        )  # shape = [b, n * d]
        return output


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
        mc_emb_bag_configs = OrderedDict()
        mc_modules = OrderedDict()
        autodis_config = OrderedDict()

        self.has_sparse = False
        self.has_sparse_user = False
        self.has_mc_sparse = False
        self.has_mc_sparse_user = False
        self.has_dense = False
        self.has_dense_user = False
        self.has_autodis_dense = False

        self._group_to_feature_names = OrderedDict()
        self._group_to_shared_feature_names = OrderedDict()
        self._group_total_dim = dict()
        self._group_feature_output_dims = dict()
        self._group_num_of_autodis_features = dict()
        self._group_autodis_feature_names = dict()
        self._group_dense_feature_names = dict()

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
        mc_emb_bag_configs_user = OrderedDict()
        mc_emb_bag_configs_item = OrderedDict()
        mc_modules_item = OrderedDict()
        mc_modules_user = OrderedDict()

        num_autodis_features = 0
        autodis_feature_names = []
        dense_feature_names = []
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
                    mc_module = feature.mc_module(device)
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
                            _add_embedding_bag_config(
                                emb_bag_configs=mc_emb_bag_configs_user
                                if mc_module
                                else emb_bag_configs_user,
                                emb_bag_config=emb_bag_config,
                            )
                            if mc_module:
                                _add_mc_module(
                                    mc_modules_user, emb_bag_config.name, mc_module
                                )
                                self.has_mc_sparse_user = True
                            else:
                                self.has_sparse_user = True
                        else:
                            _add_embedding_bag_config(
                                emb_bag_configs=mc_emb_bag_configs_item
                                if mc_module
                                else emb_bag_configs_item,
                                emb_bag_config=emb_bag_config,
                            )
                            if mc_module:
                                _add_mc_module(
                                    mc_modules_item, emb_bag_config.name, mc_module
                                )
                                self.has_mc_sparse = True
                            else:
                                self.has_sparse = True
                    else:
                        _add_embedding_bag_config(
                            emb_bag_configs=mc_emb_bag_configs
                            if mc_module
                            else emb_bag_configs,
                            emb_bag_config=emb_bag_config,
                        )
                        if mc_module:
                            _add_mc_module(mc_modules, emb_bag_config.name, mc_module)
                            self.has_mc_sparse = True
                        else:
                            self.has_sparse = True

                    if shared_feature_flag[name]:
                        shared_name = shared_name + "@" + emb_bag_config.name
                else:
                    output_dim = feature.output_dim
                    # TODO (hongsheng.jhs) make dense feature support embedding_dim

                    if is_wide:
                        raise ValueError(
                            f"dense feature [{name}] should not be configured in "
                            "wide group."
                        )
                    else:
                        if input_tile and feature.is_user_feat:
                            self.has_dense_user = True
                        else:
                            self.has_dense = True
                            dense_feature_names.append(name)
                            if hasattr(
                                feature.config, "autodis"
                            ) and feature.config.HasField("autodis"):
                                num_autodis_features += 1
                                self.has_autodis_dense = True
                                autodis_feature_names.append(name)
                                autodis_conf_tmp = feature.autodis_config
                                output_dim = autodis_conf_tmp["embedding_dim"]
                                if len(autodis_config) == 0:
                                    autodis_config.update(autodis_conf_tmp)
                                else:
                                    assert (
                                        autodis_conf_tmp == autodis_config
                                    ), "Autodis config should be \
                                    the same for all dense features"

                total_dim += output_dim
                feature_output_dims[name] = output_dim
                shared_feature_names.append(shared_name)
            self._group_to_feature_names[group_name] = feature_names
            if len(shared_feature_names) > 0:
                self._group_to_shared_feature_names[group_name] = shared_feature_names
            self._group_total_dim[group_name] = total_dim
            self._group_feature_output_dims[group_name] = feature_output_dims
            self._group_num_of_autodis_features[group_name] = num_autodis_features
            self._group_autodis_feature_names[group_name] = autodis_feature_names
            self._group_dense_feature_names[group_name] = dense_feature_names

        if input_tile_emb:
            self.ebc_user = EmbeddingBagCollection(
                list(emb_bag_configs_user.values()), device=device
            )
            self.ebc_item = EmbeddingBagCollection(
                list(emb_bag_configs_item.values()), device=device
            )
            if self.has_mc_sparse_user:
                self.mc_ebc_user = ManagedCollisionEmbeddingBagCollection(
                    EmbeddingBagCollection(
                        list(mc_emb_bag_configs_user.values()), device=device
                    ),
                    ManagedCollisionCollection(
                        mc_modules_user, list(mc_emb_bag_configs_user.values())
                    ),
                )
            if self.has_mc_sparse:
                self.mc_ebc_item = ManagedCollisionEmbeddingBagCollection(
                    EmbeddingBagCollection(
                        list(mc_emb_bag_configs_item.values()), device=device
                    ),
                    ManagedCollisionCollection(
                        mc_modules_item, list(mc_emb_bag_configs_item.values())
                    ),
                )
        else:
            self.ebc = EmbeddingBagCollection(
                list(emb_bag_configs.values()), device=device
            )
            if self.has_mc_sparse:
                self.mc_ebc = ManagedCollisionEmbeddingBagCollection(
                    EmbeddingBagCollection(
                        list(mc_emb_bag_configs.values()), device=device
                    ),
                    ManagedCollisionCollection(
                        mc_modules, list(mc_emb_bag_configs.values())
                    ),
                )
            if self.has_autodis_dense:
                self.autodis_module = AutoDisModule(
                    num_dense_feature=num_autodis_features,
                    embedding_dim=autodis_config["embedding_dim"],
                    num_channels=autodis_config["num_channels"],
                    keep_prob=autodis_config["keep_prob"],
                    temperature=autodis_config["temperature"],
                    device=device,
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

        if self.has_mc_sparse:
            if is_input_tile_emb():
                kts.append(self.mc_ebc_item(sparse_feature)[0])
            else:
                kts.append(self.mc_ebc(sparse_feature)[0])

        if self.has_sparse_user:
            keyed_tensor_user = self.ebc_user(sparse_feature_user)
            values_tile = keyed_tensor_user.values().tile(batch_size, 1)
            keyed_tensor_user_tile = KeyedTensor(
                keys=keyed_tensor_user.keys(),
                length_per_key=keyed_tensor_user.length_per_key(),
                values=values_tile,
            )
            kts.append(keyed_tensor_user_tile)

        if self.has_mc_sparse_user:
            keyed_tensor_user = self.mc_ebc_user(sparse_feature_user)[0]
            values_tile = keyed_tensor_user.values().tile(batch_size, 1)
            keyed_tensor_user_tile = KeyedTensor(
                keys=keyed_tensor_user.keys(),
                length_per_key=keyed_tensor_user.length_per_key(),
                values=values_tile,
            )
            kts.append(keyed_tensor_user_tile)

        if self.has_dense:
            if self.has_autodis_dense:
                autodis_names = [
                    f
                    for g, flist in self._group_autodis_feature_names.items()
                    for f in flist
                ]
                autodis_dense_tensor = torch.concat(
                    [dense_feature[key] for key in autodis_names], dim=1
                )
                autodis_embeddings = self.autodis_module(autodis_dense_tensor)
                kts_autodis = KeyedTensor(
                    keys=autodis_names,
                    length_per_key=[self.autodis_module.embedding_dim]
                    * len(autodis_names),
                    values=autodis_embeddings,
                    key_dim=1,
                )
                kts.append(kts_autodis)

                other_dense_feature_names = [
                    f
                    for g, flist in self._group_dense_feature_names.items()
                    for f in flist
                    if f not in autodis_names
                ]
                if len(other_dense_feature_names) > 0:
                    other_dense_features = KeyedTensor(
                        keys=other_dense_feature_names,
                        length_per_key=[1] * len(other_dense_feature_names),
                        values=torch.concat(
                            [dense_feature[key] for key in other_dense_feature_names],
                            dim=1,
                        ),
                        key_dim=1,
                    )
                    kts.append(other_dense_features)

            else:
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
        dim_to_mc_emb_configs = defaultdict(OrderedDict)
        dim_to_mc_modules = defaultdict(OrderedDict)

        self.has_sparse = False
        self.has_sparse_user = False
        self.has_mc_sparse = False
        self.has_mc_sparse_user = False
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
        dim_to_mc_emb_configs_user = defaultdict(OrderedDict)
        dim_to_mc_emb_configs_item = defaultdict(OrderedDict)
        dim_to_mc_modules_user = defaultdict(OrderedDict)
        dim_to_mc_modules_item = defaultdict(OrderedDict)

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
                    mc_module = feature.mc_module(device)
                    assert emb_config is not None
                    # we may/could modify ec name at feat_to_group_to_emb_name
                    emb_config.name = feat_to_group_to_emb_name[name][group_name]
                    embedding_dim = emb_config.embedding_dim

                    if need_input_tile_emb:
                        if feature.is_user_feat:
                            emb_configs = (
                                dim_to_mc_emb_configs_user[embedding_dim]
                                if mc_module
                                else dim_to_emb_configs_user[embedding_dim]
                            )
                            _add_embedding_config(
                                emb_configs=emb_configs,
                                emb_config=emb_config,
                            )
                            if mc_module:
                                _add_mc_module(
                                    dim_to_mc_modules_user[embedding_dim],
                                    emb_config.name,
                                    mc_module,
                                )
                                self.has_mc_sparse_user = True
                            else:
                                self.has_sparse_user = True
                        else:
                            emb_configs = (
                                dim_to_mc_emb_configs_item[embedding_dim]
                                if mc_module
                                else dim_to_emb_configs_item[embedding_dim]
                            )
                            _add_embedding_config(
                                emb_configs=emb_configs,
                                emb_config=emb_config,
                            )
                            if mc_module:
                                _add_mc_module(
                                    dim_to_mc_modules_item[embedding_dim],
                                    emb_config.name,
                                    mc_module,
                                )
                                self.has_mc_sparse = True
                            else:
                                self.has_sparse = True
                    else:
                        emb_configs = (
                            dim_to_mc_emb_configs[embedding_dim]
                            if mc_module
                            else dim_to_emb_configs[embedding_dim]
                        )
                        _add_embedding_config(
                            emb_configs=emb_configs,
                            emb_config=emb_config,
                        )
                        if mc_module:
                            _add_mc_module(
                                dim_to_mc_modules[embedding_dim],
                                emb_config.name,
                                mc_module,
                            )
                            self.has_mc_sparse = True
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
            self.mc_ec_list_user = nn.ModuleList()
            for k, emb_configs in dim_to_mc_emb_configs_user.items():
                self.mc_ec_list_user.append(
                    ManagedCollisionEmbeddingCollection(
                        EmbeddingCollection(list(emb_configs.values()), device=device),
                        ManagedCollisionCollection(
                            dim_to_mc_modules_user[k], list(emb_configs.values())
                        ),
                    )
                )
            self.mc_ec_list_item = nn.ModuleList()
            for k, emb_configs in dim_to_mc_emb_configs_item.items():
                self.mc_ec_list_item.append(
                    ManagedCollisionEmbeddingCollection(
                        EmbeddingCollection(list(emb_configs.values()), device=device),
                        ManagedCollisionCollection(
                            dim_to_mc_modules_item[k], list(emb_configs.values())
                        ),
                    )
                )
        else:
            self.ec_list = nn.ModuleList()
            for _, emb_configs in dim_to_emb_configs.items():
                self.ec_list.append(
                    EmbeddingCollection(list(emb_configs.values()), device=device)
                )
            self.mc_ec_list = nn.ModuleList()
            for k, emb_configs in dim_to_mc_emb_configs.items():
                self.mc_ec_list.append(
                    ManagedCollisionEmbeddingCollection(
                        EmbeddingCollection(list(emb_configs.values()), device=device),
                        ManagedCollisionCollection(
                            dim_to_mc_modules[k], list(emb_configs.values())
                        ),
                    )
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

        if self.has_mc_sparse:
            if need_input_tile_emb:
                for ec in self.mc_ec_list_item:
                    sparse_jt_dict_list.append(ec(sparse_feature)[0])
            else:
                for ec in self.mc_ec_list:
                    sparse_jt_dict_list.append(ec(sparse_feature)[0])

        if self.has_sparse_user:
            for ec in self.ec_list_user:
                sparse_jt_dict_list.append(ec(sparse_feature_user))

        if self.has_mc_sparse_user:
            for ec in self.mc_ec_list_user:
                sparse_jt_dict_list.append(ec(sparse_feature_user)[0])

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
                need_tile = False
                if is_user:
                    if is_sparse:
                        need_tile = need_input_tile_emb
                    else:
                        need_tile = need_input_tile
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
