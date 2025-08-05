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

from typing import Dict, List, Optional

import torch

from tzrec.modules.utils import BaseModule


class ContentEncoder(BaseModule):
    """Content encoder for HSTU.

    Args:
        input_embedding_dim (int): dimension of input embeddings.
        additional_content_feature_names (List[str]): content feature names in payloads.
        target_enrich_feature_names (List[str]): target feature names in payloads.
        is_inference (bool): whether to run in inference mode.
    """

    def __init__(
        self,
        input_embedding_dim: int,
        additional_content_feature_names: Optional[List[str]] = None,
        target_enrich_feature_names: Optional[List[str]] = None,
        is_inference: bool = False,
    ) -> None:
        super().__init__(is_inference=is_inference)
        self._input_embedding_dim: int = input_embedding_dim
        self._additional_content_feature_names: List[str] = (
            additional_content_feature_names
            if additional_content_feature_names is not None
            else []
        )
        self._target_enrich_feature_names: List[str] = (
            target_enrich_feature_names
            if target_enrich_feature_names is not None
            else []
        )
        self._target_enrich_dummy_embeddings: torch.nn.ParameterList = (
            torch.nn.ParameterList(
                [
                    torch.nn.Parameter(
                        torch.empty((self._input_embedding_dim,)).normal_(
                            mean=0, std=0.1
                        ),
                    )
                    for _ in self._target_enrich_feature_names
                ]
            )
        )

    @property
    def output_dim(self) -> int:
        """Output dimension of the module."""
        return self._input_embedding_dim * (
            1
            + len(self._additional_content_feature_names)
            + len(self._target_enrich_feature_names)
        )

    def forward(
        self,
        max_seq_len: int,
        seq_embeddings: torch.Tensor,
        seq_lengths: torch.Tensor,
        seq_offsets: torch.Tensor,
        seq_payloads: Dict[str, torch.Tensor],
        num_targets: torch.Tensor,
    ) -> torch.Tensor:
        """Forward the module.

        Args:
            max_seq_len (int): maximum sequence length.
            seq_embeddings (torch.Tensor): input sequence embeddings.
            seq_lengths (torch.Tensor): input sequence lengths.
            seq_offsets (torch.Tensor): input sequence offsets.
            seq_payloads (Dict[str, torch.Tensor]): sequence payload features.
            num_targets (int): number of targets.

        Returns:
            torch.Tensor: output content embedding tensor.
        """
        content_embeddings_list: List[torch.Tensor] = []

        if len(self._additional_content_feature_names) > 0:
            content_embeddings_list = [seq_embeddings] + [
                (seq_payloads[x].to(seq_embeddings.dtype))
                for x in self._additional_content_feature_names
            ]

        for i, f in enumerate(self._target_enrich_feature_names):
            padded_enrich_embeddings = (
                self._target_enrich_dummy_embeddings[i]
                .view(1, 1, -1)
                .repeat(seq_lengths.size(0), max_seq_len, 1)
            ).to(seq_embeddings.dtype)
            mask = torch.arange(max_seq_len, device=seq_offsets.device).view(
                1, max_seq_len
            )
            mask = torch.logical_and(
                mask >= (seq_lengths - num_targets).unsqueeze(1),
                mask < seq_lengths.unsqueeze(1),
            )
            padded_enrich_embeddings[mask] = seq_payloads[f].to(seq_embeddings.dtype)
            enrich_embeddings = torch.ops.fbgemm.dense_to_jagged(
                padded_enrich_embeddings,
                [seq_offsets],
            )[0]
            content_embeddings_list.append(enrich_embeddings)

        if (
            len(self._target_enrich_feature_names) == 0
            and len(self._additional_content_feature_names) == 0
        ):
            return seq_embeddings
        else:
            content_embeddings = torch.cat(
                content_embeddings_list,
                dim=1,
            )
        return content_embeddings
