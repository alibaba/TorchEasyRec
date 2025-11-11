# Copyright (c) 2025, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# We use the ContentEncoder from generative-recommenders a starting point.
# https://github.com/facebookresearch/generative-recommenders
# thanks to their public work.

import abc
from typing import Any, Dict, Union

import torch

from tzrec.modules.mlp import MLP
from tzrec.modules.utils import BaseModule
from tzrec.ops.jagged_tensors import concat_2D_jagged
from tzrec.protos import module_pb2
from tzrec.utils.config_util import config_to_kwargs


class ContentEncoder(BaseModule):
    """Abstract Content encoder for HSTU."""

    @property
    def output_dim(self) -> int:
        """Output dimension of the module."""
        raise NotImplementedError

    @abc.abstractmethod
    def forward(
        self,
        uih_embeddings: torch.Tensor,
        target_embeddings: torch.Tensor,
        max_uih_len: int,
        max_targets: int,
        uih_offsets: torch.Tensor,
        target_offsets: torch.Tensor,
        total_uih_len: int,
        total_targets: int,
    ) -> torch.Tensor:
        """Forward the module.

        Args:
            uih_embeddings (torch.Tensor): input uih sequence embeddings.
            target_embeddings (torch.Tensor): input target sequence embeddings.
            max_uih_len (int): maximum user history sequence length.
            max_targets (int): maximum targets sequence length.
            uih_offsets (torch.Tensor): input user history sequence offsets.
            target_offsets (torch.Tensor): target sequence lengths.
            total_uih_len (int): total user history sequence length.
            total_targets (int): total targets sequence length.

        Returns:
            content_embeddings: output content embedding tensor.
        """
        pass


class SliceContentEncoder(ContentEncoder):
    """Slice Content encoder for HSTU.

    Padding uid embedding to same dim with target embedding.

    Args:
        uih_embedding_dim (int): dimension of input uih embeddings.
        target_embedding_dim (int): dimension of input candidate embeddings.
            dim in payloads.
        is_inference (bool): whether to run in inference mode.
    """

    def __init__(
        self,
        uih_embedding_dim: int,
        target_embedding_dim: int,
        is_inference: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(is_inference)
        self._uih_embedding_dim = uih_embedding_dim
        self._target_embedding_dim = target_embedding_dim
        assert self._target_embedding_dim >= self._uih_embedding_dim

    @property
    def output_dim(self) -> int:
        """Output dimension of the module."""
        return self._uih_embedding_dim

    def forward(
        self,
        uih_embeddings: torch.Tensor,
        target_embeddings: torch.Tensor,
        max_uih_len: int,
        max_targets: int,
        uih_offsets: torch.Tensor,
        target_offsets: torch.Tensor,
        total_uih_len: int,
        total_targets: int,
    ) -> torch.Tensor:
        """Forward the module.

        Args:
            uih_embeddings (torch.Tensor): input uih sequence embeddings.
            target_embeddings (torch.Tensor): input target sequence embeddings.
            max_uih_len (int): maximum user history sequence length.
            max_targets (int): maximum targets sequence length.
            uih_offsets (torch.Tensor): input user history sequence offsets.
            target_offsets (torch.Tensor): target sequence lengths.
            total_uih_len (int): total user history sequence length.
            total_targets (int): total targets sequence length.

        Returns:
            content_embeddings: output content embedding tensor.
        """
        content_embeddings = concat_2D_jagged(
            values_left=uih_embeddings,
            values_right=target_embeddings[:, : self._uih_embedding_dim],
            max_len_left=max_uih_len,
            max_len_right=max_targets,
            offsets_left=uih_offsets,
            offsets_right=target_offsets,
            kernel=self.kernel(),
        )
        return content_embeddings


class PadContentEncoder(ContentEncoder):
    """Pad Content encoder for HSTU.

    Padding uid embedding to same dim with target embedding.

    Args:
        uih_embedding_dim (int): dimension of input uih embeddings.
        target_embedding_dim (int): dimension of input candidate embeddings.
            dim in payloads.
        is_inference (bool): whether to run in inference mode.
    """

    def __init__(
        self,
        uih_embedding_dim: int,
        target_embedding_dim: int,
        is_inference: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(is_inference)
        self._uih_embedding_dim = uih_embedding_dim
        self._target_embedding_dim = target_embedding_dim
        assert self._target_embedding_dim > self._uih_embedding_dim
        self._target_enrich_dummy_embeddings = torch.nn.Parameter(
            torch.empty(
                (1, self._target_embedding_dim - self._uih_embedding_dim)
            ).normal_(mean=0, std=0.1),
        )

    @property
    def output_dim(self) -> int:
        """Output dimension of the module."""
        return self._target_embedding_dim

    def forward(
        self,
        uih_embeddings: torch.Tensor,
        target_embeddings: torch.Tensor,
        max_uih_len: int,
        max_targets: int,
        uih_offsets: torch.Tensor,
        target_offsets: torch.Tensor,
        total_uih_len: int,
        total_targets: int,
    ) -> torch.Tensor:
        """Forward the module.

        Args:
            uih_embeddings (torch.Tensor): input uih sequence embeddings.
            target_embeddings (torch.Tensor): input target sequence embeddings.
            max_uih_len (int): maximum user history sequence length.
            max_targets (int): maximum targets sequence length.
            uih_offsets (torch.Tensor): input user history sequence offsets.
            target_offsets (torch.Tensor): target sequence lengths.
            total_uih_len (int): total user history sequence length.
            total_targets (int): total targets sequence length.

        Returns:
            content_embeddings: output content embedding tensor.
        """
        enrich_embeddings_uih = self._target_enrich_dummy_embeddings.tile(
            total_uih_len, 1
        ).to(uih_embeddings.dtype)
        uih_embeddings = torch.cat([uih_embeddings, enrich_embeddings_uih], dim=1)
        content_embeddings = concat_2D_jagged(
            values_left=uih_embeddings,
            values_right=target_embeddings,
            max_len_left=max_uih_len,
            max_len_right=max_targets,
            offsets_left=uih_offsets,
            offsets_right=target_offsets,
            kernel=self.kernel(),
        )
        return content_embeddings


class MLPContentEncoder(BaseModule):
    """MLP Content encoder for HSTU.

    Args:
        uih_embedding_dim (int): dimension of uih input embeddings.
        target_embedding_dim (int): dimension of target input embeddings.
        uih_mlp (Dict[str, int]): uih mlp config for uih sequence group.
        target_mlp (Dict[str, int]): target mlp config for candidate sequence group.
        is_inference (bool): whether to run in inference mode.
    """

    def __init__(
        self,
        uih_embedding_dim: int,
        target_embedding_dim: int,
        uih_mlp: Dict[str, Any],
        target_mlp: Dict[str, Any],
        is_inference: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(is_inference=is_inference)
        self._uih_embedding_dim = uih_embedding_dim
        self._target_embedding_dim = target_embedding_dim
        self.uih_mlp = MLP(in_features=self._uih_embedding_dim, **uih_mlp)
        self.target_mlp = MLP(in_features=self._target_embedding_dim, **target_mlp)
        assert self.uih_mlp.output_dim() == self.target_mlp.output_dim(), (
            "uih_mlp output_dim must be equal to target_mlp output_dim"
        )

    @property
    def output_dim(self) -> int:
        """Output dimension of the module."""
        return self.uih_mlp.output_dim()

    def forward(
        self,
        uih_embeddings: torch.Tensor,
        target_embeddings: torch.Tensor,
        max_uih_len: int,
        max_targets: int,
        uih_offsets: torch.Tensor,
        target_offsets: torch.Tensor,
        total_uih_len: int,
        total_targets: int,
    ) -> torch.Tensor:
        """Forward the module.

        Args:
            uih_embeddings (torch.Tensor): input uih sequence embeddings.
            target_embeddings (torch.Tensor): input target sequence embeddings.
            max_uih_len (int): maximum user history sequence length.
            max_targets (int): maximum targets sequence length.
            uih_offsets (torch.Tensor): input user history sequence offsets.
            target_offsets (torch.Tensor): target sequence lengths.
            total_uih_len (int): total user history sequence length.
            total_targets (int): total targets sequence length.

        Returns:
            content_embeddings: output content embedding tensor.
        """
        uih_embeddings = self.uih_mlp(uih_embeddings)
        target_embeddings = self.target_mlp(target_embeddings)
        content_embeddings = concat_2D_jagged(
            values_left=uih_embeddings,
            values_right=target_embeddings,
            max_len_left=max_uih_len,
            max_len_right=max_targets,
            offsets_left=uih_offsets,
            offsets_right=target_offsets,
            kernel=self.kernel(),
        )
        return content_embeddings


def create_content_encoder(
    content_encoder_cfg: Union[module_pb2.GRContentEncoder, Dict[str, Any]],
    **kwargs: Any,
) -> ContentEncoder:
    """Create ContentEncoder."""
    if isinstance(content_encoder_cfg, module_pb2.GRContentEncoder):
        content_encoder_type = content_encoder_cfg.WhichOneof("content_encoder")
        config_dict = config_to_kwargs(
            getattr(content_encoder_cfg, content_encoder_type)
        )
    else:
        assert len(content_encoder_cfg) == 1, (
            f"content_encoder_cfg should be {{content_encoder_type: content_encoder_kwargs}}, "  # NOQA
            f"but got {content_encoder_type}"
        )
        content_encoder_type, config_dict = content_encoder_cfg.popitem()

    config_dict = dict(config_dict, **kwargs)
    if content_encoder_type == "slice_content_encoder":
        return SliceContentEncoder(**config_dict)
    elif content_encoder_type == "pad_content_encoder":
        return PadContentEncoder(**config_dict)
    elif content_encoder_type == "mlp_content_encoder":
        return MLPContentEncoder(**config_dict)
    else:
        raise RuntimeError(f"Unknown content encoder type: {content_encoder_type}")
