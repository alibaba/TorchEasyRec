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
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyfg
import torch
from torchrec import JaggedTensor, KeyedJaggedTensor, KeyedTensor

from tzrec.acc.utils import is_input_tile, is_input_tile_emb
from tzrec.datasets.utils import (
    Batch,
    DenseData,
    SequenceDenseData,
    SequenceSparseData,
    SparseData,
)
from tzrec.features.feature import BaseFeature, FgMode, create_fg_json
from tzrec.utils.logging_util import logger


def _to_tensor(x: npt.NDArray) -> torch.Tensor:
    if not x.flags.writeable:
        x = np.array(x)
    return torch.from_numpy(x)


class DataParser:
    """Input Data Parser.

    Args:
        features (list): a list of features.
        labels (list, optional): a list of label names.
        is_training (bool): is training or not.
        fg_threads (int): fg thread number.
        force_base_data_group (bool): force padding data into same
            data group with same batch_size.
    """

    def __init__(
        self,
        features: List[BaseFeature],
        labels: Optional[List[str]] = None,
        sample_weights: Optional[List[str]] = None,
        is_training: bool = False,
        fg_threads: int = 1,
        force_base_data_group: bool = False,
    ) -> None:
        self._features = features
        self._labels = labels or []
        self._sample_weights = sample_weights or []
        self._is_training = is_training
        self._force_base_data_group = force_base_data_group

        self._fg_mode = features[0].fg_mode
        self._fg_threads = fg_threads
        self._fg_handler = None

        self.dense_keys = defaultdict(list)
        self.dense_length_per_key = defaultdict(list)
        self.sparse_keys = defaultdict(list)
        self.sequence_dense_keys = []
        self.has_weight_keys = defaultdict(list)

        for feature in self._features:
            if feature.is_sequence:
                if feature.is_sparse:
                    self.sparse_keys[feature.data_group].append(feature.name)
                else:
                    self.sequence_dense_keys.append(feature.name)
            elif feature.is_sparse:
                self.sparse_keys[feature.data_group].append(feature.name)
            else:
                self.dense_keys[feature.data_group].append(feature.name)
                self.dense_length_per_key[feature.data_group].append(feature.output_dim)
            if feature.is_weighted:
                self.has_weight_keys[feature.data_group].append(feature.name)

        self.feature_input_names = set()
        if self._fg_mode == FgMode.DAG:
            self._init_fg_hander()
            self.feature_input_names = (
                self._fg_handler.user_inputs()
                | self._fg_handler.item_inputs()
                | self._fg_handler.context_inputs()
            ) - set(self._fg_handler.sequence_feature_pks().values())
        else:
            for feature in features:
                self.feature_input_names |= set(feature.inputs)

        self.user_inputs = []
        self.user_feats = []
        if is_input_tile():
            self._init_fg_hander()
            self.user_inputs = self._fg_handler.user_inputs() | set(
                self._fg_handler.sequence_input_to_name().keys()
            )
            self.user_feats = self._fg_handler.user_features() | set(
                self._fg_handler.sequence_feature_to_name().keys()
            )
            for feature in features:
                feature.is_user_feat = feature.name in self.user_feats
            is_rank_zero = os.environ.get("RANK", "0") == "0"
            if is_rank_zero:
                logger.info(f"self.user_feats: {self.user_feats}")
                logger.info(f"self.user_inputs: {self.user_inputs}")

    def _init_fg_hander(self) -> None:
        """Init pyfg dag handler."""
        if not self._fg_handler:
            fg_json = create_fg_json(self._features)
            # pyre-ignore [16]
            self._fg_handler = pyfg.FgArrowHandler(fg_json, self._fg_threads)

    def parse(self, input_data: Dict[str, pa.Array]) -> Dict[str, torch.Tensor]:
        """Parse input data dict and build batch.

        Args:
            input_data (dict): raw input data.

        Return:
            output_data (dict): parsed feature data.
        """
        output_data = {}
        if is_input_tile():
            flag = False
            for k, v in input_data.items():
                if self._fg_mode == FgMode.ENCODED:
                    if k in self.user_feats:
                        input_data[k] = v.take([0])
                else:
                    if k in self.user_inputs:
                        input_data[k] = v.take([0])
                if not flag:
                    output_data["batch_size"] = torch.tensor(v.__len__())
                    flag = True

        if self._fg_mode == FgMode.DAG:
            self._parse_feature_fg_dag(input_data, output_data)
        else:
            self._parse_feature_normal(input_data, output_data)

        for label_name in self._labels:
            output_data[label_name] = _to_tensor(input_data[label_name].to_numpy())
        
        for weight in self._sample_weights:
            output_data[weight] = _to_tensor(input_data[weight].to_numpy())

        return output_data

    def _parse_feature_normal(
        self, input_data: Dict[str, pa.Array], output_data: Dict[str, torch.Tensor]
    ) -> None:
        max_batch_size = (
            max([len(v) for v in input_data.values()])
            if self._force_base_data_group
            else 0
        )

        for feature in self._features:
            feat_data = feature.parse(input_data, is_training=self._is_training)

            if isinstance(feat_data, SequenceSparseData):
                output_data[f"{feature.name}.values"] = _to_tensor(feat_data.values)
                if self._force_base_data_group:
                    feat_data.seq_lengths = np.pad(
                        feat_data.seq_lengths,
                        (0, max_batch_size - len(feat_data.seq_lengths)),
                    )
                output_data[f"{feature.name}.lengths"] = _to_tensor(
                    feat_data.seq_lengths
                )
            elif isinstance(feat_data, SequenceDenseData):
                output_data[f"{feature.name}.values"] = _to_tensor(feat_data.values)
                if self._force_base_data_group:
                    feat_data.seq_lengths = np.pad(
                        feat_data.seq_lengths,
                        (0, max_batch_size - len(feat_data.seq_lengths)),
                    )
                output_data[f"{feature.name}.lengths"] = _to_tensor(
                    feat_data.seq_lengths
                )
            elif isinstance(feat_data, SparseData):
                output_data[f"{feature.name}.values"] = _to_tensor(feat_data.values)
                if self._force_base_data_group:
                    feat_data.lengths = np.pad(
                        feat_data.lengths, (0, max_batch_size - len(feat_data.lengths))
                    )
                output_data[f"{feature.name}.lengths"] = _to_tensor(feat_data.lengths)
                if feat_data.weights is not None:
                    output_data[f"{feature.name}.weights"] = _to_tensor(
                        feat_data.weights
                    )
            elif isinstance(feat_data, DenseData):
                if self._force_base_data_group:
                    feat_data.values = np.pad(
                        feat_data.values,
                        ((0, max_batch_size - len(feat_data.values)), (0, 0)),
                    )
                output_data[f"{feature.name}.values"] = _to_tensor(feat_data.values)

    def _parse_feature_fg_dag(
        self, input_data: Dict[str, pa.Array], output_data: Dict[str, torch.Tensor]
    ) -> None:
        max_batch_size = (
            max([len(v) for v in input_data.values()])
            if self._force_base_data_group
            else 0
        )

        input_data = {
            k: v for k, v in input_data.items() if k in self.feature_input_names
        }
        fg_output, status = self._fg_handler.process_arrow(input_data)
        assert status.ok(), status.message()

        for feature in self._features:
            feat_name = feature.name
            feat_data = fg_output[feat_name]
            if feature.is_sequence:
                if feature.is_sparse:
                    output_data[f"{feat_name}.values"] = torch.tensor(
                        feat_data.values, dtype=torch.int64
                    )
                    feat_lengths = np.asarray(feat_data.lengths, dtype=np.int32)
                    if self._force_base_data_group:
                        feat_lengths = np.pad(
                            feat_lengths, (0, max_batch_size - len(feat_lengths))
                        )
                    output_data[f"{feat_name}.lengths"] = _to_tensor(feat_lengths)
                else:
                    output_data[f"{feat_name}.values"] = _to_tensor(
                        feat_data.dense_values
                    )
                    feat_lengths = np.asarray(feat_data.lengths, dtype=np.int32)
                    if self._force_base_data_group:
                        feat_lengths = np.pad(
                            feat_lengths, (0, max_batch_size - len(feat_lengths))
                        )
                    output_data[f"{feat_name}.lengths"] = _to_tensor(feat_lengths)
            else:
                if feature.is_sparse:
                    output_data[f"{feat_name}.values"] = torch.tensor(
                        feat_data.values, dtype=torch.int64
                    )
                    feat_lengths = np.asarray(feat_data.lengths, dtype=np.int32)
                    if self._force_base_data_group:
                        feat_lengths = np.pad(
                            feat_lengths, (0, max_batch_size - len(feat_lengths))
                        )
                    output_data[f"{feat_name}.lengths"] = _to_tensor(feat_lengths)
                    if feature.is_weighted:
                        output_data[f"{feat_name}.weights"] = torch.tensor(
                            feat_data.weights, dtype=torch.float32
                        )
                else:
                    dense_values = feat_data.dense_values
                    if self._force_base_data_group:
                        dense_values = np.pad(
                            dense_values,
                            ((0, max_batch_size - len(dense_values)), (0, 0)),
                        )
                    output_data[f"{feat_name}.values"] = _to_tensor(dense_values)

    def to_batch(
        self, input_data: Dict[str, torch.Tensor], force_no_tile: bool = False
    ) -> Batch:
        """Convert input data dict to Batch.

        Args:
            input_data (dict): input tensor dict.
            force_no_tile (bool): force no tile sparse features when INPUT_TILE=2.

        Returns:
            an instance of Batch.
        """
        input_tile = is_input_tile()
        input_tile_emb = is_input_tile_emb()

        batch_size = -1
        if input_tile:
            batch_size = input_data["batch_size"].item()

        if input_tile_emb:
            # For INPUT_TILE = 3 mode, batch_size of user features for sparse and dense
            # are all equal to 1, we tile it after embedding lookup in EmbeddingGroup
            dense_features = self._to_dense_features_user1_itemb(input_data)
            sparse_features = self._to_sparse_features_user1_itemb(input_data)
        elif input_tile:
            # For INPUT_TILE = 2 mode,
            # batch_size of user features for sparse are all equal to 1, we tile it
            #   here to B and do not tile in EmbeddingGroup
            # batch_size of item features for dense are all equal to 1, we tile it
            #   in EmbeddingGroup
            dense_features = self._to_dense_features_user1_itemb(input_data)
            if force_no_tile:
                # we prevent tile twice in PREDICT mode.
                sparse_features = self._to_sparse_features_user1_itemb(input_data)
            else:
                sparse_features = self._to_sparse_features_user1tile_itemb(input_data)
        else:
            dense_features = self._to_dense_features(input_data)
            sparse_features = self._to_sparse_features(input_data)

        sequence_dense_features = {}
        for key in self.sequence_dense_keys:
            sequence_dense_feature = JaggedTensor(
                values=input_data[f"{key}.values"],
                lengths=input_data[f"{key}.lengths"],
            )
            sequence_dense_features[key] = sequence_dense_feature

        labels = {}
        for label_name in self._labels:
            labels[label_name] = input_data[label_name]
        
        sample_weights = {}
        for weight in self._sample_weights:
            sample_weights[weight] = input_data[weight]

        batch = Batch(
            dense_features=dense_features,
            sparse_features=sparse_features,
            sequence_dense_features=sequence_dense_features,
            labels=labels,
            sample_weights=sample_weights,
            # pyre-ignore [6]
            batch_size=batch_size,
        )
        return batch

    def _to_dense_features(
        self, input_data: Dict[str, torch.Tensor]
    ) -> Dict[str, KeyedTensor]:
        """Convert input data dict to batch dense features.

        Args:
            input_data (dict): input tensor dict.

        Returns:
            a dict of KeyedTensor.
        """
        dense_features = {}
        for dg, keys in self.dense_keys.items():
            values = []
            for key in keys:
                values.append(input_data[f"{key}.values"])
            dense_feature = KeyedTensor(
                keys=keys,
                length_per_key=self.dense_length_per_key[dg],
                values=torch.cat(values, dim=-1),
            )
            dense_features[dg] = dense_feature
        return dense_features

    def _to_sparse_features(
        self, input_data: Dict[str, torch.Tensor]
    ) -> Dict[str, KeyedJaggedTensor]:
        """Convert to batch sparse features.

        Args:
            input_data (dict): input tensor dict.

        Returns:
            a dict of KeyedTensor.
        """
        sparse_features = {}
        for dg, keys in self.sparse_keys.items():
            values = []
            lengths = []
            weights = []
            dg_has_weight_keys = self.has_weight_keys[dg]
            for key in keys:
                values.append(input_data[f"{key}.values"])
                lengths.append(input_data[f"{key}.lengths"])
                if len(dg_has_weight_keys) > 0:
                    if key in dg_has_weight_keys:
                        weights.append(input_data[f"{key}.weights"])
                    else:
                        weights.append(
                            torch.ones_like(
                                input_data[f"{key}.values"], dtype=torch.float32
                            )
                        )
            sparse_feature = KeyedJaggedTensor(
                keys=keys,
                values=torch.cat(values, dim=-1),
                lengths=torch.cat(lengths, dim=-1),
                weights=torch.cat(weights, dim=-1)
                if len(dg_has_weight_keys) > 0
                else None,
            )
            sparse_features[dg] = sparse_feature
        return sparse_features

    def _to_dense_features_user1_itemb(
        self, input_data: Dict[str, torch.Tensor]
    ) -> Dict[str, KeyedTensor]:
        """Convert to batch dense features user_bs = 1 and item_bs = B.

        User feature and item feature use separate KeyedTensor because
            user and item with different batch_size.
        User feature use batch_size = 1 with suffix _user.
        Item feature use batch_size = actual batch size with suffix _item.

        Args:
            input_data (dict): input tensor dict.

        Returns:
            a dict of KeyedTensor.
        """
        dense_features = {}
        for dg, keys in self.dense_keys.items():
            values_item = []
            keys_item = []
            length_per_key_item = []
            values_user = []
            keys_user = []
            length_per_key_user = []
            for index, key in enumerate(keys):
                if key in self.user_feats:
                    values_user.append(input_data[f"{key}.values"])
                    keys_user.append(key)
                    length_per_key_user.append(self.dense_length_per_key[dg][index])
                else:
                    values_item.append(input_data[f"{key}.values"])
                    keys_item.append(key)
                    length_per_key_item.append(self.dense_length_per_key[dg][index])
            if len(keys_user) > 0:
                dense_feature_user = KeyedTensor(
                    keys=keys_user,
                    length_per_key=length_per_key_user,
                    values=torch.cat(values_user, dim=-1),
                )
                dense_features[dg + "_user"] = dense_feature_user
            if len(keys_item) > 0:
                dense_feature_item = KeyedTensor(
                    keys=keys_item,
                    length_per_key=length_per_key_item,
                    values=torch.cat(values_item, dim=-1),
                )
                dense_features[dg + "_item"] = dense_feature_item
        return dense_features

    def _to_sparse_features_user1tile_itemb(
        self, input_data: Dict[str, torch.Tensor]
    ) -> Dict[str, KeyedJaggedTensor]:
        """Convert batch sparse features user_bs = 1 then tile and item_bs = B.

        User feature and item feature use one KeyedJaggedTensor.
        In input data, batch_size of **user** features = **1**, we **tile** it with
        batch_size and combine it with item features into one KeyedJaggedTensor.

        Args:
            input_data (dict): input tensor dict.

        Returns:
            a dict of KeyedJaggedTensor.
        """
        sparse_features = {}
        batch_size = input_data["batch_size"].item()

        for dg, keys in self.sparse_keys.items():
            values = []
            lengths = []
            weights = []
            dg_has_weight_keys = self.has_weight_keys[dg]

            for key in keys:
                value = input_data[f"{key}.values"]
                length = input_data[f"{key}.lengths"]
                if key in self.user_feats:
                    # pyre-ignore [6]
                    value = value.tile(batch_size)
                    # pyre-ignore [6]
                    length = length.tile(batch_size)
                values.append(value)
                lengths.append(length)

                if len(dg_has_weight_keys) > 0:
                    if key in dg_has_weight_keys:
                        weight = input_data[f"{key}.weights"]
                    else:
                        weight = torch.ones_like(
                            input_data[f"{key}.values"], dtype=torch.float32
                        )
                    if key in self.user_feats:
                        # pyre-ignore [6]
                        weight = weight.tile(batch_size)
                    weights.append(weights)

            sparse_feature = KeyedJaggedTensor(
                keys=keys,
                values=torch.cat(values, dim=-1),
                lengths=torch.cat(lengths, dim=-1),
                weights=torch.cat(weights, dim=-1)
                if len(dg_has_weight_keys) > 0
                else None,
            )
            sparse_features[dg] = sparse_feature
        return sparse_features

    def _to_sparse_features_user1_itemb(
        self, input_data: Dict[str, torch.Tensor]
    ) -> Dict[str, KeyedJaggedTensor]:
        """Convert to batch sparse features user_bs = 1 and item_bs = B.

        User feature and item feature use separate KeyedJaggedTensor.  because
            user and item with different batch_size.
        User feature use batch_size = 1 with suffix _user.
        Item feature use batch_size = actual batch size with suffix _item.

        Args:
            input_data (dict): input tensor dict.

        Returns:
            a dict of KeyedJaggedTensor.
        """
        sparse_features = {}

        for dg, keys in self.sparse_keys.items():
            values_item = []
            lengths_item = []
            weights_item = []
            keys_item = []

            values_user = []
            lengths_user = []
            weights_user = []
            keys_user = []

            dg_has_weight_keys = self.has_weight_keys[dg]
            for key in keys:
                value = input_data[f"{key}.values"]
                length = input_data[f"{key}.lengths"]
                if key in self.user_feats:
                    values_user.append(value)
                    lengths_user.append(length)
                    keys_user.append(key)
                else:
                    values_item.append(value)
                    lengths_item.append(length)
                    keys_item.append(key)

                if len(dg_has_weight_keys) > 0:
                    if key in dg_has_weight_keys:
                        weight = input_data[f"{key}.weights"]
                    else:
                        weight = torch.ones_like(
                            input_data[f"{key}.values"], dtype=torch.float32
                        )
                    if key in self.user_feats:
                        weights_user.append(weight)
                    else:
                        weights_item.append(weight)

            if len(keys_user) > 0:
                sparse_feature_user = KeyedJaggedTensor(
                    keys=keys_user,
                    values=torch.cat(values_user, dim=-1),
                    lengths=torch.cat(lengths_user, dim=-1),
                    weights=torch.cat(weights_user, dim=-1)
                    if len(dg_has_weight_keys) > 0
                    else None,
                )
                sparse_features[dg + "_user"] = sparse_feature_user
            if len(keys_item) > 0:
                sparse_feature_item = KeyedJaggedTensor(
                    keys=keys_item,
                    values=torch.cat(values_item, dim=-1),
                    lengths=torch.cat(lengths_item, dim=-1),
                    weights=torch.cat(weights_item, dim=-1)
                    if len(dg_has_weight_keys) > 0
                    else None,
                )
                sparse_features[dg + "_item"] = sparse_feature_item

        return sparse_features

    def dump_parsed_inputs(self, input_data: Dict[str, torch.Tensor]) -> pa.Array:
        """Dump parsed inputs for debug."""
        feature_rows = defaultdict(dict)
        for f in self._features:
            if f.is_sparse:
                lengths = input_data[f"{f.name}.lengths"]
                values = input_data[f"{f.name}.values"].cpu().numpy()
                cnt = 0
                # pyre-ignore [16]
                sep = f.sequence_delim if f.is_sequence else ","
                for i, ll in enumerate(lengths):
                    cur_v = values[cnt : cnt + ll]
                    cnt += ll
                    feature_rows[i][f.name] = sep.join(cur_v.astype(str))
            else:
                if f.is_sequence:
                    lengths = input_data[f"{f.name}.lengths"]
                    values = input_data[f"{f.name}.values"].cpu().numpy()
                    cnt = 0
                    for i, ll in enumerate(lengths):
                        cur_v = values[cnt : cnt + ll]
                        cnt += ll
                        feature_rows[i][f.name] = f.sequence_delim.join(
                            map(",".join, cur_v.astype(str))
                        )
                else:
                    values = input_data[f"{f.name}.values"].cpu().numpy()
                    for i, cur_v in enumerate(values):
                        feature_rows[i][f.name] = ",".join(cur_v.astype(str))

        result = []
        for i in range(len(feature_rows)):
            result.append(" | ".join([f"{k}:{v}" for k, v in feature_rows[i].items()]))
        return pa.array(result)

    def __del__(self) -> None:
        if self._fg_handler:
            self._fg_handler.reset_executor()
