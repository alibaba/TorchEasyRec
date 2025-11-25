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

import copy
from collections import OrderedDict
from enum import Enum
from math import sqrt
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor, nn
from torchrec.sparse.jagged_tensor import KeyedTensor


class DenseEmbeddingType(Enum):
    """Dense Embedding Type."""

    MLP = 0
    AUTO_DIS = 1


class DenseEmbeddingConfig:
    """DenseEmbeddingConfig base class."""

    def __init__(
        self,
        embedding_dim: int,
        feature_names: List[str],
        embedding_type: DenseEmbeddingType,
        value_dim: int = 1,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.feature_names = feature_names
        self.embedding_type = embedding_type
        self.value_dim = value_dim

    @property
    def group_key(self) -> str:
        """Config group key."""
        raise NotImplementedError(
            "Subclasses of DenseEmbeddingConfig should implement this."
        )


class MLPDenseEmbeddingConfig(DenseEmbeddingConfig):
    """MLPDenseEmbeddingConfig class."""

    def __init__(
        self, embedding_dim: int, feature_names: List[str], value_dim: int
    ) -> None:
        super().__init__(
            embedding_dim, feature_names, DenseEmbeddingType.MLP, value_dim
        )

    @property
    def group_key(self) -> str:
        """Config group key."""
        if self.value_dim != 1:
            return f"mlp#{self.embedding_dim}#vdim#{self.value_dim}#feature_name#\
            {self.feature_names[0]}"
        return f"mlp#{self.embedding_dim}"


class AutoDisEmbeddingConfig(DenseEmbeddingConfig):
    """AutoDisEmbeddingConfig class."""

    def __init__(
        self,
        embedding_dim: int,
        n_channels: int,
        temperature: float,
        keep_prob: float,
        feature_names: List[str],
    ) -> None:
        super().__init__(embedding_dim, feature_names, DenseEmbeddingType.AUTO_DIS)
        self.n_channels = n_channels
        self.temperature = temperature
        self.keep_prob = keep_prob

    @property
    def group_key(self) -> str:
        """Config group key."""
        return (
            f"autodis#{self.embedding_dim}#{self.n_channels}#{self.keep_prob:.6f}"
            f"#{self.temperature:.6f}".replace(".", "_")
        )


class AutoDisEmbedding(nn.Module):
    """An Embedding Learning Framework for Numerical Features in CTR Prediction.

    https://arxiv.org/pdf/2012.08986
    """

    def __init__(
        self,
        num_dense_feature: int,
        feature_names: List[str],
        embedding_dim: int,
        num_channels: int,
        temperature: float = 0.1,
        keep_prob: float = 0.8,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.num_dense_feature = num_dense_feature
        self.feature_names = feature_names
        self.embedding_dim = embedding_dim
        self.keep_prob = keep_prob
        self.temperature = temperature
        self.num_channels = num_channels

        self.meta_emb = nn.Parameter(
            torch.randn(num_dense_feature, num_channels, embedding_dim, device=device)
        )

        # glorot normal initialization, std = sqrt(2 /(1+c))
        self.proj_w = nn.Parameter(
            torch.randn(num_dense_feature, num_channels, device=device)
            * sqrt(2 / (1 + num_channels))
        )

        # glorot normal initialization, std = sqrt(2 /(c+c))
        self.proj_m = nn.Parameter(
            torch.randn(num_dense_feature, num_channels, num_channels, device=device)
            * sqrt(1 / num_channels)
        )
        self.leaky_relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.reset_parameters()

    def reset_parameters(
        self,
    ) -> None:
        """Reset the parameters."""
        nn.init.normal_(self.meta_emb, 0, 1.0)
        nn.init.normal_(self.proj_w, 0, sqrt(2 / (1 + self.num_channels)))
        nn.init.normal_(self.proj_m, 0, sqrt(1 / self.num_channels))

    def forward(self, dense_input: Tensor) -> Tensor:
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
        x_hat = self.softmax(x_bar / self.temperature)  # shape = [b, n, c]
        emb = torch.einsum("ncd,bnc->bnd", self.meta_emb, x_hat)  # shape = [b, n, d]
        output = emb.reshape(
            (-1, self.num_dense_feature * self.embedding_dim)
        )  # shape = [b, n * d]
        return output

    # pyre-ignore[14]
    def state_dict(
        self,
        destination: Optional[Dict[str, Any]] = None,
        prefix: str = "",
        keep_vars: bool = False,
        no_snapshot: bool = True,
    ) -> Dict[str, Any]:
        """Override.

        Split the parameters so that they can be exported when using shared
        embedding models like dssm_v2.
        """
        if destination is None:
            destination = OrderedDict()
            # pyre-ignore[16]
            destination._metadata = OrderedDict()
        for i in range(self.meta_emb.shape[0]):
            destination[f"{prefix}meta_emb_{self.feature_names[i]}.weight"] = (
                self.meta_emb[i]
            )
            destination[f"{prefix}proj_w_{self.feature_names[i]}.weight"] = self.proj_w[
                i
            ]
            destination[f"{prefix}proj_m_{self.feature_names[i]}.weight"] = self.proj_m[
                i
            ]
        return destination

    def _load_from_state_dict(
        self,
        state_dict: Dict[str, Any],
        prefix: str,
        local_metadata: Dict[str, Any],
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ) -> None:
        """Override."""
        for key, param in self.state_dict(prefix=prefix).items():
            param.detach().copy_(state_dict[key])


class MLPEmbedding(nn.Module):
    """MLP embedding for dense features."""

    def __init__(
        self,
        num_dense_feature: int,
        feature_names: List[str],
        embedding_dim: int,
        value_dim: int = 1,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.num_dense_feature = num_dense_feature
        self.feature_names = feature_names
        self.embedding_dim = embedding_dim
        self.value_dim = value_dim

        if value_dim == 1:
            self.proj_w = nn.Parameter(
                torch.randn(num_dense_feature, embedding_dim)
                * sqrt(2 / (1 + embedding_dim))  # glorot normal initialization
            )
        else:
            assert num_dense_feature == 1, (
                "MLP embedding is applied to features one by one when value_dim > 1."
            )
            self.proj_w = nn.Parameter(
                torch.randn(value_dim, embedding_dim)
                * sqrt(2 / (value_dim + embedding_dim))  # glorot normal initialization
            )

    def forward(self, input: Tensor) -> Tensor:
        """Forward the module.

        Args:
            input (Tensor): dense input feature, shape = [b, n],
                where b is batch_size, n is the number of dense features.

        Returns:
            output (Tensor): Tensor of mlp embedding, shape = [b, n * d],
                where d is the embedding_dim.
        """
        if self.value_dim > 1:
            return torch.einsum("vi,bv->bi", self.proj_w, input)
        else:
            return torch.einsum("ni,bn->bni", self.proj_w, input).reshape(
                (-1, self.num_dense_feature * self.embedding_dim)
            )

    # pyre-ignore[14]
    def state_dict(
        self,
        destination: Optional[Dict[str, Any]] = None,
        prefix: str = "",
        keep_vars: bool = False,
        no_snapshot: bool = True,
    ) -> Dict[str, Any]:
        """Override.

        Split the parameter proj_w so that they can be exported when using shared
        embedding models like dssm_v2.
        """
        if destination is None:
            destination = OrderedDict()
            # pyre-ignore[16]
            destination._metadata = OrderedDict()
        if self.value_dim > 1:
            destination[f"{prefix}proj_w_{self.feature_names[0]}.weight"] = self.proj_w
        else:
            for i in range(self.proj_w.shape[0]):
                destination[f"{prefix}proj_w_{self.feature_names[i]}.weight"] = (
                    self.proj_w[i]
                )
        return destination

    def _load_from_state_dict(
        self,
        state_dict: Dict[str, Any],
        prefix: str,
        local_metadata: Dict[str, Any],
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ) -> None:
        """Override."""
        for key, param in self.state_dict(prefix=prefix).items():
            param.detach().copy_(state_dict[key])


def merge_same_config_features(
    conf_list: List[DenseEmbeddingConfig],
) -> List[DenseEmbeddingConfig]:
    """Merge features with same group_key configs.

    For example:

        conf_list: List[DenseEmbeddingConfig]
    = [
        {'embedding_dim': 128, 'n_channels': 32, 'keep_prob': 0.5,
            'temperature': 1.0, 'feature_names': ['f1']},
        {'embedding_dim': 128, 'n_channels': 32, 'keep_prob': 0.5,
            'temperature': 1.0, 'feature_names': ['f2', 'f3']},
        {'embedding_dim': 256, 'n_channels': 64, 'keep_prob': 0.5,
            'temperature': 0.8, 'feature_names': ['f4']}
    ]

    will be merged as:
        [
            {'embedding_dim': 128, 'n_channels': 32, 'keep_prob': 0.5,
                'temperature': 1.0, 'feature_names': ['f1', 'f2', 'f3']},
            {'embedding_dim': 256, 'n_channels': 64, 'keep_prob': 0.5,
                'temperature': 0.8, 'feature_names': ['f4']}
        ]
    """
    unique_dict = {}

    for conf in conf_list:
        if conf.group_key in unique_dict:
            unique_dict[conf.group_key].feature_names.extend(conf.feature_names)
        else:
            unique_dict[conf.group_key] = copy.copy(conf)

    for key in unique_dict:
        unique_dict[key].feature_names = sorted(
            list(set(unique_dict[key].feature_names))
        )

    unique_list = list(unique_dict.values())
    return unique_list


class DenseEmbeddingCollection(nn.Module):
    """DenseEmbeddingCollection module.

    Args:
        emb_dense_configs (list): list of DenseEmbeddingConfig.
        device (torch.device): embedding device, default is meta.
        raw_dense_feature_to_dim (dict): a feature_name to feature dim dict for
            raw dense features do not need to do embedding. If specified,
            the returned keyed tensor will also include these features.
    """

    def __init__(
        self,
        emb_dense_configs: List[DenseEmbeddingConfig],
        device: Optional[torch.device] = None,
        raw_dense_feature_to_dim: Optional[Dict[str, int]] = None,
    ) -> None:
        super(DenseEmbeddingCollection, self).__init__()

        self.emb_dense_configs = emb_dense_configs
        self._raw_dense_feature_to_dim = raw_dense_feature_to_dim

        self.grouped_configs = merge_same_config_features(emb_dense_configs)

        self.all_dense_names = []
        self.all_dense_dims = []
        self._group_to_feature_names = OrderedDict()

        self.dense_embs = nn.ModuleDict()
        for conf in self.grouped_configs:
            feature_names = conf.feature_names
            embedding_dim = conf.embedding_dim
            value_dim = conf.value_dim
            if conf.embedding_type == DenseEmbeddingType.MLP:
                self.dense_embs[conf.group_key] = MLPEmbedding(
                    num_dense_feature=len(feature_names),
                    feature_names=feature_names,
                    embedding_dim=embedding_dim,
                    value_dim=value_dim,
                    device=device,
                )
            elif conf.embedding_type == DenseEmbeddingType.AUTO_DIS:
                self.dense_embs[conf.group_key] = AutoDisEmbedding(
                    num_dense_feature=len(feature_names),
                    feature_names=feature_names,
                    embedding_dim=embedding_dim,
                    num_channels=conf.n_channels,
                    temperature=conf.temperature,
                    keep_prob=conf.keep_prob,
                    device=device,
                )
            self.all_dense_names.extend(feature_names)
            self.all_dense_dims.extend([embedding_dim] * len(feature_names))
            self._group_to_feature_names[conf.group_key] = feature_names

        if raw_dense_feature_to_dim is not None and len(raw_dense_feature_to_dim) > 0:
            feature_names, feature_dims = (
                list(raw_dense_feature_to_dim.keys()),
                list(raw_dense_feature_to_dim.values()),
            )
            self._group_to_feature_names["__raw_dense_group__"] = feature_names
            self.all_dense_names.extend(feature_names)
            self.all_dense_dims.extend(feature_dims)

    def forward(self, dense_feature: KeyedTensor) -> KeyedTensor:
        """Forward the module."""
        grouped_features = KeyedTensor.regroup_as_dict(
            [dense_feature],
            list(self._group_to_feature_names.values()),
            list(self._group_to_feature_names.keys()),
        )
        emb_list = []
        for group_key, emb_module in self.dense_embs.items():
            emb_list.append(emb_module(grouped_features[group_key]))
        if self._raw_dense_feature_to_dim:
            emb_list.append(grouped_features["__raw_dense_group__"])

        kt_dense_emb = KeyedTensor(
            keys=self.all_dense_names,
            length_per_key=self.all_dense_dims,
            values=torch.cat(emb_list, dim=1),
            key_dim=1,
        )
        return kt_dense_emb
