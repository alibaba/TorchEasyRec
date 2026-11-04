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

import os
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple, Union, cast

import torch
from torch import nn
from torchrec.distributed.planner.types import ParameterConstraints
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
from tzrec.modules.dense_embedding_collection import (
    DenseEmbeddingCollection,
)
from tzrec.modules.sequence import create_seq_encoder
from tzrec.protos import model_pb2
from tzrec.protos.model_pb2 import FeatureGroupConfig, SeqGroupConfig
from tzrec.utils import init_util
from tzrec.utils.fx_util import (
    fx_int_item,
    fx_mark_keyed_tensor,
    fx_mark_seq_ec_jt,
    fx_mark_seq_len,
    fx_mark_seq_tensor,
    fx_mark_tensor,
)

EMPTY_KJT = KeyedJaggedTensor.empty()


torch.fx.wrap(fx_int_item)
torch.fx.wrap(fx_mark_keyed_tensor)
torch.fx.wrap(fx_mark_tensor)
torch.fx.wrap(fx_mark_seq_tensor)
torch.fx.wrap(fx_mark_seq_len)
torch.fx.wrap(fx_mark_seq_ec_jt)


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
def _merge_list_of_dict_of_jt_dict(
    list_of_dict_of_jt_odict: List[Dict[str, Dict[str, JaggedTensor]]],
) -> Dict[str, Dict[str, JaggedTensor]]:
    result: Dict[str, Dict[str, JaggedTensor]] = {}
    for tensor_d_d in list_of_dict_of_jt_odict:
        result.update(tensor_d_d)
    return result


@torch.fx.wrap
def _merge_list_of_jt_dict(
    list_of_jt_dict: List[Dict[str, JaggedTensor]],
) -> Dict[str, JaggedTensor]:
    result: Dict[str, JaggedTensor] = {}
    for jt_dict in list_of_jt_dict:
        result.update(jt_dict)
    return result


def _remap_mark_jt_dict(
    d_jt: Dict[str, JaggedTensor],
    feature_keys: List[Tuple[str, str]],
) -> Dict[str, JaggedTensor]:
    """Remap EmbeddingCollection output keys to shared keys and mark seq jts.

    EmbeddingCollection output keys are paired with the shared feature names the
    downstream feature groups consume; each shared-name JaggedTensor is also
    marked for torch.fx sequence tracing. Left unwrapped so the per-feature
    ``fx_mark_seq_ec_jt`` calls stay individual graph nodes.

    Args:
        d_jt: EmbeddingCollection output keyed by output key.
        feature_keys: list of (output_key, shared_key) pairs.

    Returns:
        Dict keyed by shared_key.
    """
    new_d_jt: Dict[str, JaggedTensor] = {}
    for output_key, shared_key in feature_keys:
        val = d_jt[output_key]
        fx_mark_seq_ec_jt(shared_key, val)
        new_d_jt[shared_key] = val
    return new_d_jt


@torch.fx.wrap
def _tile_and_combine_dense_kt(
    user_kt: Optional[KeyedTensor], item_kt: Optional[KeyedTensor], tile_size: int
) -> KeyedTensor:
    kt_keys: List[str] = []
    kt_length_per_key: List[int] = []
    kt_values: List[torch.Tensor] = []
    if user_kt is not None:
        kt_keys.extend(user_kt.keys())
        kt_length_per_key.extend(user_kt.length_per_key())
        kt_values.append(user_kt.values().tile(tile_size, 1))
    if item_kt is not None:
        kt_keys.extend(item_kt.keys())
        kt_length_per_key.extend(item_kt.length_per_key())
        kt_values.append(item_kt.values())
    return KeyedTensor(
        keys=kt_keys,
        length_per_key=kt_length_per_key,
        values=torch.cat(kt_values, dim=1),
    )


@torch.fx.wrap
def _dense_to_jt(t: torch.Tensor) -> JaggedTensor:
    return JaggedTensor(
        values=t,
        lengths=torch.ones(
            t.shape[0],
            dtype=torch.int64,
            device=t.device,
        ),
    )


class EmbeddingGroup(nn.Module):
    """Applies embedding lookup transformation for feature group.

    Args:
        features (list): list of features.
        feature_groups (list): list of feature group config.
        wide_embedding_dim (int, optional): wide group feature embedding dim.
        wide_init_fn (str, optional): wide embedding table init function.
        device (torch.device): embedding device, default is meta.
    """

    def __init__(
        self,
        features: List[BaseFeature],
        feature_groups: List[FeatureGroupConfig],
        wide_embedding_dim: Optional[int] = None,
        wide_init_fn: Optional[str] = None,
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

        (
            self._impl_key_to_feat_groups,
            self._impl_key_to_seq_groups,
            self._group_name_to_impl_key,
        ) = build_embedding_group_mappings(features, feature_groups)
        for feature_group in feature_groups:
            group_name = feature_group.group_name
            if feature_group.group_type in (
                model_pb2.SEQUENCE,
                model_pb2.JAGGED_SEQUENCE,
            ):
                self._grouped_features_keys.append(group_name + ".query")
                self._grouped_features_keys.append(group_name + ".sequence")
                self._grouped_features_keys.append(group_name + ".sequence_length")
            else:
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
                wide_init_fn=wide_init_fn,
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
            if feature_group.group_type not in (
                model_pb2.SEQUENCE,
                model_pb2.JAGGED_SEQUENCE,
            ):
                feature_dim = OrderedDict()
                impl_key = self._group_name_to_impl_key[group_name]
                feature_emb = self.emb_impls[impl_key]
                feature_dim.update(feature_emb.group_feature_dims(group_name))
                if group_name in self._group_name_to_seq_encoders:
                    seq_encoders = cast(
                        nn.ModuleList, self._group_name_to_seq_encoders[group_name]
                    )
                    for i, seq_encoder in enumerate(seq_encoders):
                        feature_dim[f"{group_name}_seq_encoder_{i}"] = (
                            seq_encoder.output_dim()
                        )
                self._group_feature_dims[group_name] = feature_dim

        self._grouped_features_keys.sort()

    def grouped_features_keys(self) -> List[str]:
        """grouped_features_keys."""
        return self._grouped_features_keys

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
        if feature_group.group_type in (model_pb2.SEQUENCE, model_pb2.JAGGED_SEQUENCE):
            return self.seq_emb_impls[impl_key].group_dims(group_name)
        else:
            dims = self.emb_impls[impl_key].group_dims(group_name)
            if group_name in self._group_name_to_seq_encoders:
                for seq_encoder in cast(
                    nn.ModuleList, self._group_name_to_seq_encoders[group_name]
                ):
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
        if feature_group.group_type in (model_pb2.SEQUENCE, model_pb2.JAGGED_SEQUENCE):
            return self.seq_emb_impls[impl_key].group_total_dim(group_name)
        else:
            return sum(self._group_feature_dims[group_name].values())

    def group_feature_dims(self, group_name: str) -> Dict[str, int]:
        """Every feature group each feature dim."""
        true_name = group_name.split(".")[0] if "." in group_name else group_name
        feature_group = self._name_to_feature_group[true_name]
        if feature_group.group_type in (model_pb2.SEQUENCE, model_pb2.JAGGED_SEQUENCE):
            raise ValueError("not support sequence group")
        return self._group_feature_dims[group_name]

    def group_type(self, group_name: str) -> model_pb2.FeatureGroupType:
        """Get feature group type."""
        true_name = group_name.split(".")[0] if "." in group_name else group_name
        feature_group = self._name_to_feature_group[true_name]
        return feature_group.group_type

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

        need_input_tile = is_input_tile()

        if need_input_tile:
            emb_keys = list(self.emb_impls.keys())
            seq_emb_keys = list(self.seq_emb_impls.keys())
            unique_keys = list(set(emb_keys + seq_emb_keys))
            # tile user dense feat & combine item dense feat, when has user dense feat
            for key in unique_keys:
                user_kt = batch.dense_features.get(key + "_user", None)
                if user_kt is not None:
                    item_kt = batch.dense_features.get(key, None)
                    batch.dense_features[key] = _tile_and_combine_dense_kt(
                        user_kt, item_kt, batch.tile_size
                    )

        for key, emb_impl in self.emb_impls.items():
            sparse_feat_kjt = None
            sparse_feat_kjt_user = None
            dense_feat_kt = None

            if emb_impl.has_dense:
                dense_feat_kt = batch.dense_features[key]
            if emb_impl.has_sparse or emb_impl.has_mc_sparse:
                sparse_feat_kjt = batch.sparse_features[key]
            if emb_impl.has_sparse_user or emb_impl.has_mc_sparse_user:
                sparse_feat_kjt_user = batch.sparse_features[key + "_user"]

            result_dicts.append(
                emb_impl(
                    sparse_feat_kjt,
                    dense_feat_kt,
                    sparse_feat_kjt_user,
                    batch.tile_size,
                )
            )

        for key, seq_emb_impl in self.seq_emb_impls.items():
            sparse_feat_kjt = None
            sparse_feat_kjt_user = None
            dense_feat_kt = None
            sequence_mulval_length_kjt = None
            sequence_mulval_length_kjt_user = None

            if seq_emb_impl.has_dense:
                dense_feat_kt = batch.dense_features[key]
            if seq_emb_impl.has_sparse or seq_emb_impl.has_mc_sparse:
                sparse_feat_kjt = batch.sparse_features[key]
            if seq_emb_impl.has_mulval_seq:
                sequence_mulval_length_kjt = batch.sequence_mulval_lengths[key]
            if seq_emb_impl.has_sparse_user or seq_emb_impl.has_mc_sparse_user:
                sparse_feat_kjt_user = batch.sparse_features[key + "_user"]
            if seq_emb_impl.has_mulval_seq_user:
                sequence_mulval_length_kjt_user = batch.sequence_mulval_lengths[
                    key + "_user"
                ]

            result_dicts.append(
                seq_emb_impl(
                    sparse_feat_kjt,
                    dense_feat_kt,
                    batch.sequence_dense_features,
                    sequence_mulval_length_kjt,
                    sparse_feat_kjt_user,
                    sequence_mulval_length_kjt_user,
                    batch.tile_size,
                )
            )

        result = _merge_list_of_tensor_dict(result_dicts)
        seq_feature_dict = {}
        for group_name, seq_encoders in self._group_name_to_seq_encoders.items():
            new_feature = []
            for seq_encoder in cast(nn.ModuleList, seq_encoders):
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


def _check_emb_name_suffix_collisions(
    feat_to_group_to_emb_name: Dict[str, Dict[str, str]],
    group_name_to_suffix: Dict[str, str],
) -> None:
    """Detect silent emb-name collisions across groups with different suffixes.

    A composed emb_name reachable from two groups whose embedding_name_suffix
    values differ (e.g. one suffixed group lands on the same name as another
    group's un-suffixed default) would silently merge tables that the user
    meant to keep independent. Raise rather than share.
    """
    emb_name_to_suffixes = defaultdict(set)
    for group_to_emb_name in feat_to_group_to_emb_name.values():
        for grp_name, emb_name in group_to_emb_name.items():
            emb_name_to_suffixes[emb_name].add(group_name_to_suffix.get(grp_name, ""))
    for emb_name, suffixes in emb_name_to_suffixes.items():
        if len(suffixes) > 1:
            raise ValueError(
                f"embedding table name {emb_name!r} is produced by groups with "
                f"different embedding_name_suffix values {sorted(suffixes)}; "
                "this would silently merge tables that were meant to be "
                "independent. Rename the feature's embedding_name or change "
                "the conflicting suffix."
            )


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
        # a shared table is weighted when any of its features needs weight
        # pyre-ignore [16]
        existed_emb_bag_config.is_weighted |= emb_bag_config.is_weighted
    else:
        emb_bag_configs[emb_bag_config.name] = emb_bag_config


def _is_weighted(emb_bag_configs: Dict[str, EmbeddingBagConfig]) -> bool:
    """Any table of the embedding bag collection is looked up with weight or not.

    Args:
        emb_bag_configs(Dict[str, EmbeddingBagConfig]): a dict contains emb_bag_configs

    Returns:
        whether the EmbeddingBagCollection should be built as weighted.
    """
    return any(x.is_weighted for x in emb_bag_configs.values())


def _merge_embedding_constraints(
    store: Dict[str, ParameterConstraints],
    name: str,
    feature_name: str,
    const: Optional[ParameterConstraints],
) -> None:
    """Record a table's constraints, one entry per input feature.

    pooling_factors keeps one raw id factor per feature of the table, in
    config order, unset features contributing one id per sample;
    unique_id_factor aggregates the members' configured values with max and
    stays unset when none is configured.

    Args:
        store(Dict[str, ParameterConstraints]): a dict of table constraints.
        name(str): embedding table name.
        feature_name(str): the feature the constraints come from.
        const(Optional[ParameterConstraints]): the feature's parameter
            constraints, or None when it has none.
    """
    const = (
        const
        if const is not None
        else ParameterConstraints(pooling_factors=[1.0], feature_names=[feature_name])
    )
    old = store.get(name)
    if old is None:
        store[name] = const
        return
    if const.feature_names[0] in (old.feature_names or []):
        return
    old.feature_names = list(old.feature_names or []) + list(const.feature_names or [])
    old.pooling_factors = list(old.pooling_factors) + list(const.pooling_factors)
    if const.sharding_types is not None:
        old.sharding_types = const.sharding_types
    if const.compute_kernels is not None:
        old.compute_kernels = const.compute_kernels
    value = getattr(const, "unique_id_factor", None)
    if value is not None:
        old_value = getattr(old, "unique_id_factor", None)
        old.unique_id_factor = value if old_value is None else max(old_value, value)


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


def _ec_output_key_to_shared_name(
    ec: EmbeddingCollection,
    emb_name_to_feature_to_shared_name: Dict[str, Dict[str, str]],
) -> List[Tuple[str, str]]:
    """Pair EmbeddingCollection output keys with shared feature names.

    EmbeddingCollection suffixes its output key with the table name when a feature
    is shared by more than one table of the collection, so the output key can not
    be assumed to be the feature name.

    Args:
        ec(EmbeddingCollection): an instance of EmbeddingCollection.
        emb_name_to_feature_to_shared_name(Dict[str, Dict[str, str]]): a dict of
            embedding_name to a dict of feature_name to shared feature name.

    Returns:
        a list of (embedding collection output key, shared feature name).
    """
    feat_list = []
    for emb_config, output_keys in zip(
        ec._embedding_configs, ec.embedding_names_by_table()
    ):
        for feature_name, output_key in zip(emb_config.feature_names, output_keys):
            shared_name = emb_name_to_feature_to_shared_name[emb_config.name].get(
                feature_name, feature_name
            )
            feat_list.append((output_key, shared_name))
    return feat_list


def _validate_and_complete_feature_group(
    feature_group: FeatureGroupConfig, seq_group_names: List[str]
) -> None:
    """Inspect feature group sequence_groups and sequence_encoders."""
    group_name = feature_group.group_name
    sequence_groups = list(feature_group.sequence_groups)
    sequence_encoders = list(feature_group.sequence_encoders)
    is_deep = feature_group.group_type == model_pb2.DEEP
    if is_deep:
        if len(sequence_groups) == 0 and len(sequence_encoders) == 0:
            return
        elif len(sequence_groups) > 0 and len(sequence_encoders) == 0:
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

        for sequence_group in sequence_groups:
            if sequence_group.group_name in seq_group_names:
                raise ValueError(
                    f"has repeat sequences groups_name: {sequence_group.group_name}"
                )
            else:
                seq_group_names.append(sequence_group.group_name)

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


def build_embedding_group_mappings(
    features: List[BaseFeature],
    feature_groups: List[FeatureGroupConfig],
) -> Tuple[
    Dict[str, List[FeatureGroupConfig]],
    Dict[str, List[FeatureGroupConfig]],
    Dict[str, str],
]:
    """Split feature groups into embedding implementations, from config.

    Features of one feature group must share a data group; that data group
    keys the embedding implementations, so groups of different data groups
    never merge their tables. SEQUENCE and JAGGED groups and the nested
    sequence_groups of ordinary groups - copied with the parent's
    embedding_name_suffix when the child sets none - go to the sequence
    implementation.

    Args:
        features (list): list of features.
        feature_groups (list): list of feature group config.

    Returns:
        (impl_key -> feature groups, impl_key -> sequence group configs,
        group name -> impl key).
    """
    name_to_feature = {x.name: x for x in features}
    seq_group_names: List[str] = []
    impl_key_to_feat_groups: Dict[str, List[FeatureGroupConfig]] = defaultdict(list)
    impl_key_to_seq_groups: Dict[str, List[FeatureGroupConfig]] = defaultdict(list)
    group_name_to_impl_key: Dict[str, str] = dict()
    for feature_group in feature_groups:
        group_name = feature_group.group_name
        _validate_and_complete_feature_group(feature_group, seq_group_names)
        features_data_group: Dict[str, List[str]] = defaultdict(list)
        for feature_name in feature_group.feature_names:
            feature = name_to_feature[feature_name]
            features_data_group[feature.data_group].append(feature_name)
        for sequence_group in feature_group.sequence_groups:
            for feature_name in sequence_group.feature_names:
                feature = name_to_feature[feature_name]
                features_data_group[feature.data_group].append(feature_name)
        if len(features_data_group) > 1:
            error_info = [",".join(v) for v in features_data_group.values()]
            raise ValueError(
                f"Feature {error_info} should not belong to same feature group."
            )
        impl_key = list(features_data_group.keys())[0]
        group_name_to_impl_key[group_name] = impl_key
        if feature_group.group_type in (
            model_pb2.SEQUENCE,
            model_pb2.JAGGED_SEQUENCE,
        ):
            impl_key_to_seq_groups[impl_key].append(feature_group)
        else:
            impl_key_to_feat_groups[impl_key].append(feature_group)
            if len(feature_group.sequence_groups) > 0:
                # Copy each nested sequence_group so the caller's proto is
                # not mutated, then propagate the parent suffix when the
                # child doesn't set its own. Keeps construction idempotent
                # across repeated EmbeddingGroup builds (train/eval/export).
                for sequence_group in feature_group.sequence_groups:
                    sg_copy = model_pb2.SeqGroupConfig()
                    sg_copy.CopyFrom(sequence_group)
                    if (
                        feature_group.embedding_name_suffix
                        and not sg_copy.embedding_name_suffix
                    ):
                        sg_copy.embedding_name_suffix = (
                            feature_group.embedding_name_suffix
                        )
                    impl_key_to_seq_groups[impl_key].append(sg_copy)
    return impl_key_to_feat_groups, impl_key_to_seq_groups, group_name_to_impl_key


def _is_zch_feature(feature: BaseFeature) -> bool:
    """Whether the feature's table is managed-collision (ZCH), from config."""
    return hasattr(feature.config, "zch") and feature.config.HasField("zch")


def _build_feature_table_mappings(
    name_to_feature: Dict[str, BaseFeature],
    feature_groups: List[FeatureGroupConfig],
    wide_name: bool,
    config_name: Callable[[BaseFeature], str],
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, bool]]:
    """Final table name per feature and group, and the shared-feature flag.

    A table name carries the WIDE mark and the group's embedding_name_suffix;
    a feature that reaches more than one final name is shared across those
    tables. Naming runs over every group before any config is touched, so a
    shared feature's later names never build on an already renamed table.
    """
    feat_to_group_to_emb_name: Dict[str, Dict[str, str]] = defaultdict(dict)
    group_name_to_suffix = {
        fg.group_name: fg.embedding_name_suffix for fg in feature_groups
    }
    for feature_group in feature_groups:
        group_name = feature_group.group_name
        for feature_name in feature_group.feature_names:
            feature = name_to_feature[feature_name]
            if not feature.is_sparse:
                continue
            emb_name = config_name(feature)
            if wide_name and feature_group.group_type == model_pb2.WIDE:
                emb_name = emb_name + "_wide"
            if feature_group.embedding_name_suffix:
                emb_name = emb_name + "_" + feature_group.embedding_name_suffix
            feat_to_group_to_emb_name[feature_name][group_name] = emb_name
    _check_emb_name_suffix_collisions(feat_to_group_to_emb_name, group_name_to_suffix)
    shared_feature_flag = {
        feature_name: len(set(group_to_emb_name.values())) > 1
        for feature_name, group_to_emb_name in feat_to_group_to_emb_name.items()
    }
    return feat_to_group_to_emb_name, shared_feature_flag


@dataclass
class EmbeddingBagTableConfigs:
    """Final EBC table configs of one embedding implementation.

    Attributes:
        ebc: ordered table name -> EmbeddingBagConfig whose feature_names
            lists every input feature of the table.
        mc_ebc: the managed-collision (ZCH) tables, same shape.
        ebc_user: the input-tile user copy of the user-feature tables.
        mc_ebc_user: the managed-collision user tables.
        feat_to_group_to_emb_name: feature -> group -> final table name.
        shared_feature_flag: whether a feature reaches more than one table.
    """

    ebc: "OrderedDict[str, EmbeddingBagConfig]"
    mc_ebc: "OrderedDict[str, EmbeddingBagConfig]"
    ebc_user: "OrderedDict[str, EmbeddingBagConfig]"
    mc_ebc_user: "OrderedDict[str, EmbeddingBagConfig]"
    feat_to_group_to_emb_name: Dict[str, Dict[str, str]]
    shared_feature_flag: Dict[str, bool]


def build_embedding_bag_configs(
    features: List[BaseFeature],
    feature_groups: List[FeatureGroupConfig],
    wide_embedding_dim: Optional[int] = None,
    wide_init_fn: Optional[str] = None,
) -> EmbeddingBagTableConfigs:
    """Resolve the EBC tables of feature groups from configuration.

    Walks the groups the way model construction does: WIDE groups rename and
    resize their tables, a table shared by several features keeps one input
    entry per feature, and an input-tile user feature's table lands in the
    user collection. Builds no module.

    Args:
        features (list): list of features.
        feature_groups (list): list of feature group config.
        wide_embedding_dim (int, optional): wide group feature embedding dim.
        wide_init_fn (str, optional): wide embedding table init function.
    """
    name_to_feature = {x.name: x for x in features}
    need_input_tile_emb = is_input_tile_emb()
    feat_to_group_to_emb_name, shared_feature_flag = _build_feature_table_mappings(
        name_to_feature,
        feature_groups,
        wide_name=True,
        config_name=lambda feature: feature.emb_bag_config.name,  # pyre-ignore[16]
    )
    tables = EmbeddingBagTableConfigs(
        OrderedDict(),
        OrderedDict(),
        OrderedDict(),
        OrderedDict(),
        feat_to_group_to_emb_name,
        shared_feature_flag,
    )
    for feature_group in feature_groups:
        group_name = feature_group.group_name
        is_wide = feature_group.group_type == model_pb2.WIDE
        for name in feature_group.feature_names:
            feature = name_to_feature[name]
            if not feature.is_sparse:
                continue
            emb_bag_config = feature.emb_bag_config
            assert emb_bag_config is not None
            if is_wide:
                emb_bag_config.embedding_dim = wide_embedding_dim or 4
                if wide_init_fn:
                    emb_bag_config.init_fn = init_util.create_init_fn(wide_init_fn)
            emb_bag_config.name = feat_to_group_to_emb_name[name][group_name]
            if need_input_tile_emb and feature.is_user_feat:
                emb_bag_configs = (
                    tables.mc_ebc_user if _is_zch_feature(feature) else tables.ebc_user
                )
            else:
                emb_bag_configs = (
                    tables.mc_ebc if _is_zch_feature(feature) else tables.ebc
                )
            _add_embedding_bag_config(
                emb_bag_configs=emb_bag_configs, emb_bag_config=emb_bag_config
            )
    return tables


@dataclass
class EmbeddingCollectionTableConfigs:
    """Final EC table configs of one sequence embedding implementation.

    Attributes:
        ec: embedding dim -> ordered table name -> EmbeddingConfig whose
            feature_names lists every input feature of the table.
        mc_ec: the managed-collision (ZCH) tables, same shape.
        ec_user: the input-tile user copy of the user-feature tables.
        mc_ec_user: the managed-collision user tables.
        feat_to_group_to_emb_name: feature -> group -> final table name.
        shared_feature_flag: whether a feature reaches more than one table.
        emb_name_to_feature_to_shared_name: table name -> feature name ->
            shared feature name (``<feature>@<table>`` when shared).
    """

    ec: Dict[int, "OrderedDict[str, EmbeddingConfig]"]
    mc_ec: Dict[int, "OrderedDict[str, EmbeddingConfig]"]
    ec_user: Dict[int, "OrderedDict[str, EmbeddingConfig]"]
    mc_ec_user: Dict[int, "OrderedDict[str, EmbeddingConfig]"]
    feat_to_group_to_emb_name: Dict[str, Dict[str, str]]
    shared_feature_flag: Dict[str, bool]
    emb_name_to_feature_to_shared_name: Dict[str, Dict[str, str]]


def build_embedding_collection_configs(
    features: List[BaseFeature],
    feature_groups: List[Union[FeatureGroupConfig, SeqGroupConfig]],
) -> EmbeddingCollectionTableConfigs:
    """Resolve the EC tables of sequence feature groups from configuration.

    Walks the groups the way model construction does: tables are collected
    per embedding dim, a table shared by several features keeps one input
    entry per feature, and an input-tile user feature's table lands in the
    user collection. Builds no module.

    Args:
        features (list): list of features.
        feature_groups (list): list of feature group config or seq group
            config.
    """
    name_to_feature = {x.name: x for x in features}
    need_input_tile_emb = is_input_tile_emb()
    feat_to_group_to_emb_name, shared_feature_flag = _build_feature_table_mappings(
        name_to_feature,
        feature_groups,
        wide_name=False,
        config_name=lambda feature: feature.emb_config.name,  # pyre-ignore[16]
    )
    dim_to_emb_configs: Dict[int, "OrderedDict[str, EmbeddingConfig]"] = defaultdict(
        OrderedDict
    )
    dim_to_mc_emb_configs: Dict[int, "OrderedDict[str, EmbeddingConfig]"] = defaultdict(
        OrderedDict
    )
    dim_to_emb_configs_user: Dict[int, "OrderedDict[str, EmbeddingConfig]"] = (
        defaultdict(OrderedDict)
    )
    dim_to_mc_emb_configs_user: Dict[int, "OrderedDict[str, EmbeddingConfig]"] = (
        defaultdict(OrderedDict)
    )
    emb_name_to_feature_to_shared_name: Dict[str, Dict[str, str]] = defaultdict(dict)
    for feature_group in feature_groups:
        group_name = feature_group.group_name
        for name in feature_group.feature_names:
            feature = name_to_feature[name]
            if not feature.is_sparse:
                continue
            emb_config = feature.emb_config
            assert emb_config is not None
            emb_config.name = feat_to_group_to_emb_name[name][group_name]
            shared_name = name
            if shared_feature_flag[name]:
                shared_name = shared_name + "@" + emb_config.name
            emb_name_to_feature_to_shared_name[emb_config.name][name] = shared_name
            if need_input_tile_emb and feature.is_user_feat:
                emb_configs = (
                    dim_to_mc_emb_configs_user[emb_config.embedding_dim]
                    if _is_zch_feature(feature)
                    else dim_to_emb_configs_user[emb_config.embedding_dim]
                )
            else:
                emb_configs = (
                    dim_to_mc_emb_configs[emb_config.embedding_dim]
                    if _is_zch_feature(feature)
                    else dim_to_emb_configs[emb_config.embedding_dim]
                )
            _add_embedding_config(emb_configs=emb_configs, emb_config=emb_config)
    return EmbeddingCollectionTableConfigs(
        dim_to_emb_configs,
        dim_to_mc_emb_configs,
        dim_to_emb_configs_user,
        dim_to_mc_emb_configs_user,
        feat_to_group_to_emb_name,
        shared_feature_flag,
        emb_name_to_feature_to_shared_name,
    )


class EmbeddingGroupImpl(nn.Module):
    """Applies embedding lookup transformation for feature group.

    Args:
        features (list): list of features.
        feature_groups (list): list of feature group config.
        wide_embedding_dim (int, optional): wide group feature embedding dim.
        wide_init_fn (str, optional): wide embedding table init function.
        device (torch.device): embedding device, default is meta.
    """

    def __init__(
        self,
        features: List[BaseFeature],
        feature_groups: List[FeatureGroupConfig],
        wide_embedding_dim: Optional[int] = None,
        wide_init_fn: Optional[str] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        if device is None:
            device = torch.device("meta")
        name_to_feature = {x.name: x for x in features}

        need_input_tile_emb = is_input_tile_emb()

        table_configs = build_embedding_bag_configs(
            features, feature_groups, wide_embedding_dim, wide_init_fn
        )
        emb_bag_configs = table_configs.ebc
        mc_emb_bag_configs = table_configs.mc_ebc
        emb_bag_configs_user = table_configs.ebc_user
        mc_emb_bag_configs_user = table_configs.mc_ebc_user
        feat_to_group_to_emb_name = table_configs.feat_to_group_to_emb_name
        shared_feature_flag = table_configs.shared_feature_flag

        mc_modules = OrderedDict()
        mc_modules_user = OrderedDict()
        dense_embedding_configs = []
        self._emb_bag_constraints = OrderedDict()
        self._emb_bag_constraints_user = OrderedDict()
        self.has_sparse = bool(emb_bag_configs)
        self.has_mc_sparse = bool(mc_emb_bag_configs)
        self.has_sparse_user = bool(emb_bag_configs_user)
        self.has_mc_sparse_user = bool(mc_emb_bag_configs_user)
        self.has_dense = False
        self.has_dense_embedding = False

        self._group_to_feature_names = OrderedDict()
        self._group_to_shared_feature_names = OrderedDict()
        self._group_total_dim = dict()
        self._group_feature_output_dims = dict()
        self._group_dense_feature_names = dict()
        self._group_dense_embedding_feature_names = dict()

        non_emb_dense_feature_to_dim = OrderedDict()
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
                    emb_bag_config = feature.emb_bag_config
                    assert emb_bag_config is not None
                    table_name = feat_to_group_to_emb_name[name][group_name]
                    output_dim = (
                        (wide_embedding_dim or 4) if is_wide else feature.output_dim
                    )
                    mc_module = (
                        feature.mc_module(device) if _is_zch_feature(feature) else None
                    )
                    const = feature.parameter_constraints(emb_bag_config)

                    if need_input_tile_emb and feature.is_user_feat:
                        if mc_module:
                            _add_mc_module(mc_modules_user, table_name, mc_module)
                            # mc module not support const now
                        else:
                            _merge_embedding_constraints(
                                self._emb_bag_constraints_user, table_name, name, const
                            )
                    else:
                        if mc_module:
                            _add_mc_module(mc_modules, table_name, mc_module)
                            # mc module not support const now
                        else:
                            _merge_embedding_constraints(
                                self._emb_bag_constraints, table_name, name, const
                            )

                    if shared_feature_flag[name]:
                        shared_name = shared_name + "@" + table_name
                else:
                    output_dim = feature.output_dim

                    if is_wide:
                        raise ValueError(
                            f"dense feature [{name}] should not be configured in "
                            "wide group."
                        )
                    else:
                        self.has_dense = True
                        if feature.dense_emb_config:
                            self.has_dense_embedding = True
                            conf_obj = feature.dense_emb_config
                            dense_embedding_configs.append(conf_obj)
                        else:
                            non_emb_dense_feature_to_dim[name] = output_dim

                total_dim += output_dim
                feature_output_dims[name] = output_dim
                shared_feature_names.append(shared_name)
            self._group_to_feature_names[group_name] = feature_names
            if len(shared_feature_names) > 0:
                self._group_to_shared_feature_names[group_name] = shared_feature_names
            self._group_total_dim[group_name] = total_dim
            self._group_feature_output_dims[group_name] = feature_output_dims
        self._all_group_str = "__".join(self._group_to_feature_names.keys())

        self.ebc = EmbeddingBagCollection(
            list(emb_bag_configs.values()),
            is_weighted=_is_weighted(emb_bag_configs),
            device=device,
        )
        if self.has_mc_sparse:
            self.mc_ebc = ManagedCollisionEmbeddingBagCollection(
                EmbeddingBagCollection(
                    list(mc_emb_bag_configs.values()),
                    is_weighted=_is_weighted(mc_emb_bag_configs),
                    device=device,
                ),
                ManagedCollisionCollection(
                    mc_modules, list(mc_emb_bag_configs.values())
                ),
            )
        if self.has_dense_embedding:
            self.dense_ec = DenseEmbeddingCollection(
                dense_embedding_configs,
                device=device,
                raw_dense_feature_to_dim=non_emb_dense_feature_to_dim,
            )

        if need_input_tile_emb:
            self.ebc_user = EmbeddingBagCollection(
                list(emb_bag_configs_user.values()),
                is_weighted=_is_weighted(emb_bag_configs_user),
                device=device,
            )
            if self.has_mc_sparse_user:
                self.mc_ebc_user = ManagedCollisionEmbeddingBagCollection(
                    EmbeddingBagCollection(
                        list(mc_emb_bag_configs_user.values()),
                        is_weighted=_is_weighted(mc_emb_bag_configs_user),
                        device=device,
                    ),
                    ManagedCollisionCollection(
                        mc_modules_user, list(mc_emb_bag_configs_user.values())
                    ),
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

    def parameter_constraints(
        self, prefix: str = ""
    ) -> Dict[str, ParameterConstraints]:
        """Embedding parameter constraint."""
        constraints = {}
        for k, v in self._emb_bag_constraints.items():
            constraints[f"{prefix}ebc.{k}"] = v
        for k, v in self._emb_bag_constraints_user.items():
            constraints[f"{prefix}ebc_user.{k}"] = v
        return constraints

    def forward(
        self,
        sparse_feature: KeyedJaggedTensor,
        dense_feature: KeyedTensor,
        sparse_feature_user: KeyedJaggedTensor,
        tile_size: int = -1,
    ) -> Dict[str, torch.Tensor]:
        """Forward the module.

        Args:
            sparse_feature (KeyedJaggedTensor): sparse id feature.
            dense_feature (dense_feature): dense feature.
            sparse_feature_user (KeyedJaggedTensor): user-side sparse feature
                with batch_size=1, when use INPUT_TILE=3.
            tile_size: size for user-side feature input tile.

        Returns:
            group_features (dict): dict of feature_group to embedded tensor.
        """
        kts: List[KeyedTensor] = []
        if self.has_sparse:
            kt = self.ebc(sparse_feature)
            fx_mark_keyed_tensor(self._all_group_str + "__ebc", kt)
            kts.append(kt)

        if self.has_mc_sparse:
            kt = self.mc_ebc(sparse_feature)[0]
            fx_mark_keyed_tensor(self._all_group_str + "__mc_ebc", kt)
            kts.append(kt)

        # do user-side embedding input-tile
        if self.has_sparse_user:
            keyed_tensor_user = self.ebc_user(sparse_feature_user)
            values_tile = keyed_tensor_user.values().tile(tile_size, 1)
            kt = KeyedTensor(
                keys=keyed_tensor_user.keys(),
                length_per_key=keyed_tensor_user.length_per_key(),
                values=values_tile,
            )
            fx_mark_keyed_tensor(self._all_group_str + "__ebc_user", kt)
            kts.append(kt)

        # do user-side mc embedding input-tile
        if self.has_mc_sparse_user:
            keyed_tensor_user = self.mc_ebc_user(sparse_feature_user)[0]
            values_tile = keyed_tensor_user.values().tile(tile_size, 1)
            kt = KeyedTensor(
                keys=keyed_tensor_user.keys(),
                length_per_key=keyed_tensor_user.length_per_key(),
                values=values_tile,
            )
            fx_mark_keyed_tensor(self._all_group_str + "__mc_ebc_user", kt)
            kts.append(kt)

        if self.has_dense:
            fx_mark_keyed_tensor(
                self._all_group_str + "__dense", dense_feature, is_dense=True
            )
            if self.has_dense_embedding:
                kts.append(self.dense_ec(dense_feature))
            else:
                kts.append(dense_feature)

        group_tensors = KeyedTensor.regroup_as_dict(
            kts,
            list(self._group_to_shared_feature_names.values()),
            list(self._group_to_shared_feature_names.keys()),
        )

        return group_tensors


class _SequenceEmbeddingInfo(NamedTuple):
    """One Embedding info in SequenceGroup."""

    name: str
    raw_name: str
    is_sparse: bool
    pooling: str
    value_dim: int
    is_user: bool
    is_sequence: bool


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

        need_input_tile = is_input_tile()
        need_input_tile_emb = is_input_tile_emb()

        dim_to_emb_configs = defaultdict(OrderedDict)
        dim_to_mc_emb_configs = defaultdict(OrderedDict)
        dim_to_mc_modules = defaultdict(OrderedDict)
        self._dim_to_emb_constraints = defaultdict(OrderedDict)
        self.has_sparse = False
        self.has_mc_sparse = False
        self.has_dense = False
        self.has_sequence_dense = False
        self.has_mulval_seq = False

        # for sparse input-tile-emb
        dim_to_mc_modules_user = defaultdict(OrderedDict)
        self._dim_to_emb_constraints_user = defaultdict(OrderedDict)
        self.has_mulval_seq_user = False

        self._group_to_shared_query = OrderedDict()
        self._group_to_shared_sequence = OrderedDict()
        self._group_total_dim = dict()
        self._group_output_dims = dict()
        self._group_to_is_jagged = dict()
        self._group_to_sequence_length = OrderedDict()

        ec_table_configs = build_embedding_collection_configs(features, feature_groups)
        dim_to_emb_configs = ec_table_configs.ec
        dim_to_mc_emb_configs = ec_table_configs.mc_ec
        dim_to_emb_configs_user = ec_table_configs.ec_user
        dim_to_mc_emb_configs_user = ec_table_configs.mc_ec_user
        feat_to_group_to_emb_name = ec_table_configs.feat_to_group_to_emb_name
        shared_feature_flag = ec_table_configs.shared_feature_flag
        emb_name_to_feature_to_shared_name = (
            ec_table_configs.emb_name_to_feature_to_shared_name
        )
        self.has_sparse = bool(dim_to_emb_configs)
        self.has_mc_sparse = bool(dim_to_mc_emb_configs)
        self.has_sparse_user = bool(dim_to_emb_configs_user)
        self.has_mc_sparse_user = bool(dim_to_mc_emb_configs_user)

        for feature_group in feature_groups:
            group_name = feature_group.group_name
            self._group_to_is_jagged[group_name] = (
                hasattr(feature_group, "group_type")
                and feature_group.group_type == model_pb2.JAGGED_SEQUENCE
            )

        for feature_group in feature_groups:
            query_dim = 0
            sequence_dim = 0
            query_dims = []
            sequence_dims = []
            output_dims = []
            group_name = feature_group.group_name
            feature_names = list(feature_group.feature_names)
            shared_query = []
            shared_sequence = []
            group_sequence_length = None

            for name in feature_names:
                shared_name = name
                feature = name_to_feature[name]

                if feature.is_sparse:
                    output_dim = feature.output_dim
                    emb_config = feature.emb_config
                    assert emb_config is not None
                    table_name = feat_to_group_to_emb_name[name][group_name]
                    mc_module = (
                        feature.mc_module(device) if _is_zch_feature(feature) else None
                    )
                    if shared_feature_flag[name]:
                        shared_name = shared_name + "@" + table_name
                    embedding_dim = emb_config.embedding_dim
                    const = feature.parameter_constraints(emb_config)

                    if need_input_tile_emb and feature.is_user_feat:
                        if mc_module:
                            _add_mc_module(
                                dim_to_mc_modules_user[embedding_dim],
                                table_name,
                                mc_module,
                            )
                            # mc module not support const now
                        else:
                            _merge_embedding_constraints(
                                self._dim_to_emb_constraints_user[embedding_dim],
                                table_name,
                                name,
                                const,
                            )
                        if feature.is_sequence and feature.value_dim != 1:
                            self.has_mulval_seq_user = True
                    else:
                        if mc_module:
                            _add_mc_module(
                                dim_to_mc_modules[embedding_dim],
                                table_name,
                                mc_module,
                            )
                            # mc module not support const now
                        else:
                            _merge_embedding_constraints(
                                self._dim_to_emb_constraints[embedding_dim],
                                table_name,
                                name,
                                const,
                            )
                        if feature.is_sequence and feature.value_dim != 1:
                            self.has_mulval_seq = True
                else:
                    output_dim = feature.output_dim
                    if feature.is_sequence:
                        self.has_sequence_dense = True
                    else:
                        self.has_dense = True

                is_user_feat = feature.is_user_feat if need_input_tile else False
                shared_info = _SequenceEmbeddingInfo(
                    name=shared_name,
                    raw_name=name,
                    is_sparse=feature.is_sparse,
                    pooling=feature.pooling_type.value.lower(),
                    value_dim=feature.value_dim,
                    is_user=is_user_feat,
                    is_sequence=feature.is_sequence,
                )
                if feature.is_sequence:
                    shared_sequence.append(shared_info)
                    sequence_dim += output_dim
                    sequence_dims.append(output_dim)
                    group_sequence_length = feature.sequence_length
                else:
                    shared_query.append(shared_info)
                    query_dim += output_dim
                    query_dims.append(output_dim)
                output_dims.append(output_dim)

            self._group_to_shared_query[group_name] = shared_query
            self._group_to_shared_sequence[group_name] = shared_sequence
            self._group_to_sequence_length[group_name] = group_sequence_length
            self._group_total_dim[f"{group_name}.query"] = query_dim
            self._group_total_dim[f"{group_name}.sequence"] = sequence_dim
            self._group_output_dims[f"{group_name}.query"] = query_dims
            self._group_output_dims[f"{group_name}.sequence"] = sequence_dims
            self._group_output_dims[group_name] = output_dims

        self.ec_dict = nn.ModuleDict()
        for k, emb_configs in dim_to_emb_configs.items():
            self.ec_dict[str(k)] = EmbeddingCollection(
                list(emb_configs.values()), device=device
            )

        self.ec_dict_features = OrderedDict()
        for k, ec in self.ec_dict.items():
            self.ec_dict_features[k] = _ec_output_key_to_shared_name(
                ec, emb_name_to_feature_to_shared_name
            )

        self.mc_ec_dict = nn.ModuleDict()
        for k, emb_configs in dim_to_mc_emb_configs.items():
            self.mc_ec_dict[str(k)] = ManagedCollisionEmbeddingCollection(
                EmbeddingCollection(list(emb_configs.values()), device=device),
                ManagedCollisionCollection(
                    dim_to_mc_modules[k], list(emb_configs.values())
                ),
            )

        self.mc_ec_dict_features = OrderedDict()
        for k, mc_ec in self.mc_ec_dict.items():
            self.mc_ec_dict_features[k] = _ec_output_key_to_shared_name(
                mc_ec._embedding_module, emb_name_to_feature_to_shared_name
            )

        if need_input_tile_emb:
            self.ec_dict_user = nn.ModuleDict()
            for k, emb_configs in dim_to_emb_configs_user.items():
                self.ec_dict_user[str(k)] = EmbeddingCollection(
                    list(emb_configs.values()), device=device
                )
            self.ec_dict_features_user = OrderedDict()
            for k, ec in self.ec_dict_user.items():
                self.ec_dict_features_user[k] = _ec_output_key_to_shared_name(
                    ec, emb_name_to_feature_to_shared_name
                )

            self.mc_ec_dict_user = nn.ModuleDict()
            for k, emb_configs in dim_to_mc_emb_configs_user.items():
                self.mc_ec_dict_user[str(k)] = ManagedCollisionEmbeddingCollection(
                    EmbeddingCollection(list(emb_configs.values()), device=device),
                    ManagedCollisionCollection(
                        dim_to_mc_modules_user[k], list(emb_configs.values())
                    ),
                )

            self.mc_ec_dict_features_user = OrderedDict()
            for k, mc_ec in self.mc_ec_dict_user.items():
                self.mc_ec_dict_features_user[k] = _ec_output_key_to_shared_name(
                    mc_ec._embedding_module, emb_name_to_feature_to_shared_name
                )

    def group_dims(self, group_name: str) -> List[int]:
        """Output dimension of each feature in a feature group."""
        return self._group_output_dims[group_name]

    def group_total_dim(self, group_name: str) -> int:
        """Total output dimension of a feature group."""
        if "." in group_name:
            return self._group_total_dim[group_name]
        else:
            return (
                self._group_total_dim[f"{group_name}.query"]
                + self._group_total_dim[f"{group_name}.sequence"]
            )

    def all_group_total_dim(self) -> Dict[str, int]:
        """Total output dimension of all feature group."""
        return self._group_total_dim

    def parameter_constraints(
        self, prefix: str = ""
    ) -> Dict[str, ParameterConstraints]:
        """Embedding parameter constraints."""
        constraints = {}
        for dim, emb_constraints in self._dim_to_emb_constraints.items():
            for k, v in emb_constraints.items():
                constraints[f"{prefix}ec_dict.{dim}.{k}"] = v
        for dim, emb_constraints in self._dim_to_emb_constraints_user.items():
            for k, v in emb_constraints.items():
                constraints[f"{prefix}ec_dict_user.{dim}.{k}"] = v
        return constraints

    def has_group(self, group_name: str) -> bool:
        """Check the feature group exist or not."""
        true_name = group_name.split(".")[0] if "." in group_name else group_name
        return true_name in self._group_output_dims.keys()

    def _forward_impl(
        self,
        sparse_feature: KeyedJaggedTensor,
        dense_feature: KeyedTensor,
        sequence_dense_features: Dict[str, JaggedTensor],
        sequence_mulval_lengths: KeyedJaggedTensor,
        sparse_feature_user: KeyedJaggedTensor,
        sequence_mulval_lengths_user: KeyedJaggedTensor,
    ) -> Tuple[Dict[str, JaggedTensor], Dict[str, torch.Tensor]]:
        sparse_jt_dict_list: List[Dict[str, JaggedTensor]] = []
        seq_mulval_length_jt_dict_list: List[Dict[str, JaggedTensor]] = []
        dense_t_dict: Dict[str, torch.Tensor] = {}

        if self.has_sparse:
            for feature_keys, ec in zip(
                self.ec_dict_features.values(), self.ec_dict.values()
            ):
                sparse_jt_dict_list.append(
                    _remap_mark_jt_dict(ec(sparse_feature), feature_keys)
                )

        if self.has_mc_sparse:
            for feature_keys, mc_ec in zip(
                self.mc_ec_dict_features.values(), self.mc_ec_dict.values()
            ):
                sparse_jt_dict_list.append(
                    _remap_mark_jt_dict(mc_ec(sparse_feature)[0], feature_keys)
                )

        if self.has_mulval_seq:
            seq_mulval_length_jt_dict_list.append(sequence_mulval_lengths.to_dict())

        if self.has_sparse_user:
            for feature_keys, ec in zip(
                self.ec_dict_features_user.values(), self.ec_dict_user.values()
            ):
                sparse_jt_dict_list.append(
                    _remap_mark_jt_dict(ec(sparse_feature_user), feature_keys)
                )

        if self.has_mc_sparse_user:
            for feature_keys, mc_ec in zip(
                self.mc_ec_dict_features_user.values(), self.mc_ec_dict_user.values()
            ):
                sparse_jt_dict_list.append(
                    _remap_mark_jt_dict(mc_ec(sparse_feature_user)[0], feature_keys)
                )

        if self.has_mulval_seq_user:
            seq_mulval_length_jt_dict_list.append(
                sequence_mulval_lengths_user.to_dict()
            )

        sparse_jt_dict = _merge_list_of_jt_dict(sparse_jt_dict_list)
        seq_mulval_length_jt_dict = _merge_list_of_jt_dict(
            seq_mulval_length_jt_dict_list
        )

        if self.has_dense:
            dense_t_dict = dense_feature.to_dict()

        seq_jt_dict: Dict[str, JaggedTensor] = {}
        for _, v in self._group_to_shared_sequence.items():
            for info in v:
                if info.name in seq_jt_dict:
                    continue
                jt = (
                    sparse_jt_dict[info.name]
                    if info.is_sparse
                    else sequence_dense_features[info.name]
                )
                if info.is_sparse and info.value_dim != 1:
                    length_jt = seq_mulval_length_jt_dict[info.raw_name]
                    # length_jt.values is sequence key_lengths
                    # length_jt.lengths is sequence seq_lengths
                    jt_values = torch.segment_reduce(
                        jt.values(), info.pooling, lengths=length_jt.values()
                    )
                    if info.pooling == "mean":
                        jt_values = torch.nan_to_num(jt_values, nan=0.0)
                    jt = JaggedTensor(
                        values=jt_values,
                        lengths=length_jt.lengths(),
                    )
                seq_jt_dict[info.name] = jt

        if len(seq_jt_dict) > 0:
            jt_dict = _merge_list_of_jt_dict([sparse_jt_dict, seq_jt_dict])
        else:
            jt_dict = sparse_jt_dict

        return jt_dict, dense_t_dict

    def forward(
        self,
        sparse_feature: KeyedJaggedTensor,
        dense_feature: KeyedTensor,
        sequence_dense_features: Dict[str, JaggedTensor],
        sequence_mulval_lengths: KeyedJaggedTensor,
        sparse_feature_user: KeyedJaggedTensor,
        sequence_mulval_lengths_user: KeyedJaggedTensor,
        tile_size: int = -1,
    ) -> Dict[str, torch.Tensor]:
        """Forward the module.

        Args:
            sparse_feature (KeyedJaggedTensor): sparse id feature.
            dense_feature (dense_feature): dense feature.
            sequence_dense_features (Dict[str, JaggedTensor]): dense sequence feature.
            sequence_mulval_lengths (KeyedJaggedTensor): key_lengths and seq_lengths for
                multi-value sparse sequence features, kjt.values is key_lengths,
                kjt.lengths is seq_lengths.
            sparse_feature_user (KeyedJaggedTensor): user-side sparse feature
                with batch_size=1, when use INPUT_TILE=3.
            sequence_mulval_lengths_user (KeyedJaggedTensor):key_lengths and seq_lengths
                of user-side multi-value sparse sequence features, with batch_size=1,
                when use INPUT_TILE=3.
            tile_size: size for user-side feature input tile.

        Returns:
            group_features (dict): dict of feature_group to embedded tensor.
        """
        need_input_tile = is_input_tile()
        need_input_tile_emb = is_input_tile_emb()

        jt_dict, dense_t_dict = self._forward_impl(
            sparse_feature,
            dense_feature,
            sequence_dense_features,
            sequence_mulval_lengths,
            sparse_feature_user,
            sequence_mulval_lengths_user,
        )

        results = {}
        sparse_query_cache: Dict[str, torch.Tensor] = {}
        for group_name, v in self._group_to_shared_query.items():
            query_t_list = []
            query_t_keys = []
            for info in v:
                if info.name in sparse_query_cache:
                    query_t = sparse_query_cache[info.name]
                elif info.is_sparse:
                    query_jt = jt_dict[info.name]
                    if info.value_dim == 1:
                        # for single-value id feature
                        query_t = jt_dict[info.name].to_padded_dense(1).squeeze(1)
                    else:
                        # for multi-value id feature
                        query_t = torch.segment_reduce(
                            query_jt.values(), info.pooling, lengths=query_jt.lengths()
                        )
                        if info.pooling == "mean":
                            query_t = torch.nan_to_num(query_t, nan=0.0)
                    if info.is_user and need_input_tile_emb:
                        query_t = query_t.tile(tile_size, 1)
                    sparse_query_cache[info.name] = query_t
                else:
                    query_t = dense_t_dict[info.name]
                query_t_list.append(query_t)
                query_t_keys.append(info.name)
            if len(query_t_list) > 0:
                query_cat_t = torch.cat(query_t_list, dim=1)
                fx_mark_tensor(f"{group_name}__query", query_cat_t, keys=query_t_keys)
                results[f"{group_name}.query"] = query_cat_t

        for group_name, v in self._group_to_shared_sequence.items():
            seq_t_list = []
            seq_t_keys = []

            group_sequence_length = 1
            for i, info in enumerate(v):
                # when is_user is True
                #   sequence_sparse_features
                #       when input_tile_emb need to tile(tile_size,1):
                #   sequence_dense_features always need to tile
                need_tile = False
                if info.is_user:
                    if info.is_sparse:
                        need_tile = need_input_tile_emb
                    else:
                        need_tile = need_input_tile
                jt = jt_dict[info.name]
                if i == 0:
                    sequence_length = jt.lengths()
                    group_sequence_length = fx_int_item(torch.max(sequence_length))

                    if need_tile:
                        sequence_length = sequence_length.tile(tile_size)
                    fx_mark_seq_len(f"{group_name}", sequence_length)
                    results[f"{group_name}.sequence_length"] = sequence_length

                if (
                    int(os.getenv("INPUT_TILE_3_ONLINE", "0")) == 1 and info.is_user
                ) or self._group_to_is_jagged[group_name]:
                    seq_t = jt.values()
                else:
                    seq_t = jt.to_padded_dense(group_sequence_length)

                if need_tile:
                    seq_t = seq_t.tile(tile_size, 1, 1)
                seq_t_list.append(seq_t)
                seq_t_keys.append(info.name)

            if seq_t_list:
                seq_cat_t = torch.cat(seq_t_list, dim=-1)
                fx_mark_seq_tensor(
                    f"{group_name}",
                    seq_cat_t,
                    keys=seq_t_keys,
                    max_seq_len=self._group_to_sequence_length[group_name],
                    is_jagged_seq=self._group_to_is_jagged[group_name],
                )
                results[f"{group_name}.sequence"] = seq_cat_t

        return results
