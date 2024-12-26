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

from math import sqrt
from typing import Dict, List, Optional

import torch
from torch import Tensor, nn
from torchrec.sparse.jagged_tensor import KeyedTensor


class DenseEmbeddingConfig:
    """DenseEmbeddingConfig base class."""

    def __init__(self, embedding_dim: int, feature_names: List[str]) -> None:
        self.embedding_dim = embedding_dim
        self.feature_names = feature_names


class MLPDenseEmbeddingConfig(DenseEmbeddingConfig):
    """MLPDenseEmbeddingConfig class."""

    def __init__(self, embedding_dim: int, feature_names: List[str]) -> None:
        super().__init__(embedding_dim, feature_names)
        self.embedding_type = "MLP"

    def to_dict(self) -> Dict:
        """Convert the config to a dict."""
        return {
            "embedding_dim": self.embedding_dim,
            "embedding_type": self.embedding_type,
            "feature_names": self.feature_names,
        }


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
        super().__init__(embedding_dim, feature_names)
        self.n_channels = n_channels
        self.temperature = temperature
        self.keep_prob = keep_prob
        self.embedding_type = "AutoDis"

    def to_dict(self) -> Dict:
        """Convert to a dict."""
        return {
            "embedding_dim": self.embedding_dim,
            "n_channels": self.n_channels,
            "temperature": self.temperature,
            "keep_prob": self.keep_prob,
            "embedding_type": self.embedding_type,
            "feature_names": self.feature_names,
        }


class AutoDisEmbedding(nn.Module):
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
        x_hat = self.softmax(x_bar / self.temperature)  # shape = [b, n, c]
        emb = torch.einsum("ncd,bnc->bnd", self.meta_emb, x_hat)  # shape = [b, n, d]
        output = emb.reshape(
            (-1, self.num_dense_feature * self.embedding_dim)
        )  # shape = [b, n * d]
        return output


class MLPEmbedding(nn.Module):
    """MLP embedding for dense features."""

    def __init__(
        self,
        num_dense_feature: int,
        embedding_dim: int,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.num_dense_feature = num_dense_feature
        self.embedding_dim = embedding_dim
        self.proj_w = nn.Parameter(
            torch.randn(num_dense_feature, embedding_dim, 1, device=device)
            * sqrt(2 / (1 + embedding_dim))  # glorot normal initialization
        )

    def forward(self, input: Tensor):
        """Forward the module.

        Args:
            input (Tensor): dense input feature, shape = [b, n, 1],
                where b is batch_size, n is the number of dense features.

        Returns:
            output (Tensor): Tensor of mlp embedding, shape = [b, n * d],
                where d is the embedding_dim.
        """
        return torch.einsum("nij,bnj->bni", self.proj_w, input).reshape(
            (-1, self.num_dense_feature * self.embedding_dim)
        )


def merge_same_config_features(dict_list: List[Dict], keys: List[str]):
    """Merge features with same configs.

    For example:

    dict_list = [
        {'embedding_dim': 128, 'n_channels': 32, 'keep_prob': 0.5,
            'temperature': 1.0, 'feature_names': ['f1']},
        {'embedding_dim': 128, 'n_channels': 32, 'keep_prob': 0.5,
            'temperature': 1.0, 'feature_names': ['f2', 'f3']},
        {'embedding_dim': 256, 'n_channels': 64, 'keep_prob': 0.5,
            'temperature': 0.8, 'feature_names': ['f4']}
    ],
    keys = ['embedding_dim', 'n_channels', 'keep_prob', 'temperature']

    will be merged as:
        [
            {'embedding_dim': 128, 'n_channels': 32, 'keep_prob': 0.5,
                'temperature': 1.0, 'feature_names': ['f1', 'f2', 'f3']},
            {'embedding_dim': 256, 'n_channels': 64, 'keep_prob': 0.5,
                'temperature': 0.8, 'feature_names': ['f4']}
        ]
    """
    unique_dict = {}

    for item in dict_list:
        key = tuple(item[k] for k in keys)
        if key in unique_dict:
            unique_dict[key]["feature_names"].extend(item["feature_names"])
        else:
            unique_dict[key] = item.copy()

    for key in unique_dict:
        unique_dict[key]["feature_names"] = list(set(unique_dict[key]["feature_names"]))

    unique_list = list(unique_dict.values())
    return unique_list


class DenseEmbeddingCollection(nn.Module):
    """DenseEmbeddingCollection module."""

    def __init__(
        self,
        emb_dense_configs: List[DenseEmbeddingConfig],
        device: Optional[torch.device] = None,
    ) -> None:
        super(DenseEmbeddingCollection, self).__init__()

        self.emb_dense_configs = emb_dense_configs
        mlp_configs, autodis_configs = [], []
        for config in emb_dense_configs:
            if config.embedding_type == "MLP":
                mlp_configs.append(config.to_dict())
            elif config.embedding_type == "AutoDis":
                autodis_configs.append(config.to_dict())

        self.mlp_grouped_configs = merge_same_config_features(
            mlp_configs, keys=["embedding_dim"]
        )
        self.autodis_grouped_configs = merge_same_config_features(
            autodis_configs,
            keys=["embedding_dim", "n_channels", "keep_prob", "temperature"],
        )

        self.all_dense_names = []
        self.all_dense_dims = []

        if len(self.mlp_grouped_configs) > 0:
            self.mlp_emb_module_list = nn.ModuleList(
                [
                    MLPEmbedding(
                        num_dense_feature=len(
                            self.mlp_grouped_configs[i]["feature_names"]
                        ),
                        embedding_dim=self.mlp_grouped_configs[i]["embedding_dim"],
                        device=device,
                    )
                    for i in range(len(self.mlp_grouped_configs))
                ]
            )
            mlp_names = [
                name
                for conf in self.mlp_grouped_configs
                for name in conf["feature_names"]
            ]
            mlp_dims = [
                dim
                for conf in self.mlp_grouped_configs
                for dim in [conf["embedding_dim"]] * len(conf["feature_names"])
            ]

            self.all_dense_names.extend(mlp_names)
            self.all_dense_dims.extend(mlp_dims)

        if len(self.autodis_grouped_configs) > 0:
            self.autodis_module_list = nn.ModuleList(
                [
                    AutoDisEmbedding(
                        num_dense_feature=len(
                            self.autodis_grouped_configs[i]["feature_names"]
                        ),
                        embedding_dim=self.autodis_grouped_configs[i]["embedding_dim"],
                        num_channels=self.autodis_grouped_configs[i]["n_channels"],
                        temperature=self.autodis_grouped_configs[i]["temperature"],
                        keep_prob=self.autodis_grouped_configs[i]["keep_prob"],
                        device=device,
                    )
                    for i in range(len(self.autodis_grouped_configs))
                ]
            )
            autodis_names = [
                name
                for conf in self.autodis_grouped_configs
                for name in conf["feature_names"]
            ]
            autodis_dims = [
                dim
                for conf in self.autodis_grouped_configs
                for dim in [conf["embedding_dim"]] * len(conf["feature_names"])
            ]

            self.all_dense_names.extend(autodis_names)
            self.all_dense_dims.extend(autodis_dims)

    def forward(self, dense_feature: KeyedTensor) -> KeyedTensor:
        """Forward the module."""
        emb_list = []

        if hasattr(self, "mlp_emb_module_list") and len(self.mlp_emb_module_list) > 0:
            mlp_emb_list = []
            for i, config in enumerate(self.mlp_grouped_configs):
                feature_tensor = KeyedTensor.regroup_as_dict(
                    [dense_feature], [config["feature_names"]], ["grp"]
                )["grp"]
                mlp_emb_list.append(
                    self.mlp_emb_module_list[i](torch.unsqueeze(feature_tensor, dim=-1))
                )
            mlp_emb = torch.cat(mlp_emb_list, dim=1)
            emb_list.append(mlp_emb)

        if hasattr(self, "autodis_module_list") and len(self.autodis_module_list) > 0:
            autodis_emb_list = []
            for i, config in enumerate(self.autodis_grouped_configs):
                feature_tensor = KeyedTensor.regroup_as_dict(
                    [dense_feature], [config["feature_names"]], ["grp"]
                )["grp"]
                autodis_emb_list.append(self.autodis_module_list[i](feature_tensor))
            autodis_emb = torch.cat(autodis_emb_list, dim=1)
            emb_list.append(autodis_emb)

        kts_dense_emb = KeyedTensor(
            keys=self.all_dense_names,
            length_per_key=self.all_dense_dims,
            values=torch.cat(emb_list, dim=1),
            key_dim=1,
        )
        return kts_dense_emb
