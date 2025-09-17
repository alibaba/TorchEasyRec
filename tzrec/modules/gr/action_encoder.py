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

# We use the ActionEncoder from generative-recommenders a starting point.
# https://github.com/facebookresearch/generative-recommenders
# thanks to their public work.

from typing import Any, Dict, List, Optional, Tuple

import torch

from tzrec.modules.utils import BaseModule
from tzrec.ops.jagged_tensors import concat_2D_jagged


class ActionEncoder(BaseModule):
    """Action encoder for HSTU.

    Args:
        action_embedding_dim (int): dimension of action embedding.
        action_feature_name (str): name of action feature in payloads.
        action_weights (List[int]): bitmask of each action.
        watchtime_feature_name (str): name of watchtime feature in payloads.
        watchtime_to_action_thresholds (List[int]): threshold for watchtime of each
            action.
        watchtime_to_action_weights (List[int]): bitmask for watchtime of each action.
        is_inference (bool): whether to run in inference mode.
    """

    def __init__(
        self,
        action_embedding_dim: int,
        action_feature_name: str,
        action_weights: List[int],
        watchtime_feature_name: str = "",
        watchtime_to_action_thresholds: Optional[List[int]] = None,
        watchtime_to_action_weights: Optional[List[int]] = None,
        is_inference: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(is_inference=is_inference)
        self._watchtime_feature_name: str = watchtime_feature_name
        self._action_feature_name: str = action_feature_name
        self._watchtime_to_action_thresholds_and_weights: List[Tuple[int, int]] = []
        if watchtime_to_action_thresholds is not None:
            assert watchtime_to_action_weights is not None
            assert len(watchtime_to_action_thresholds) == len(
                watchtime_to_action_weights
            )
            self._watchtime_to_action_thresholds_and_weights = list(
                zip(watchtime_to_action_thresholds, watchtime_to_action_weights)
            )
        self.register_buffer(
            "_combined_action_weights",
            torch.tensor(
                action_weights
                + [x[1] for x in self._watchtime_to_action_thresholds_and_weights]
            ),
        )
        self._num_action_types: int = len(action_weights) + len(
            self._watchtime_to_action_thresholds_and_weights
        )
        self._action_embedding_dim = action_embedding_dim
        self._action_embedding_table: torch.nn.Parameter = torch.nn.Parameter(
            torch.empty((self._num_action_types, action_embedding_dim)).normal_(
                mean=0, std=0.1
            ),
        )
        self._target_action_embedding_table: torch.nn.Parameter = torch.nn.Parameter(
            torch.empty((1, self._num_action_types * action_embedding_dim)).normal_(
                mean=0, std=0.1
            ),
        )

    @property
    def output_dim(self) -> int:
        """Output dimension of the module."""
        return self._action_embedding_dim * self._num_action_types

    def forward(
        self,
        max_uih_len: int,
        max_targets: int,
        uih_offsets: torch.Tensor,
        target_offsets: torch.Tensor,
        seq_embeddings: torch.Tensor,
        seq_payloads: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Forward the module.

        Args:
            max_uih_len (int): maximum user history sequence length.
            max_targets (int): maximum targets sequence length.
            uih_offsets (torch.Tensor): input user history sequence offsets.
            target_offsets (torch.Tensor): target sequence lengths.
            seq_embeddings (torch.Tensor): input sequence embeddings.
            seq_payloads (Dict[str, torch.Tensor]): sequence payload features.

        Returns:
            torch.Tensor: output action embedding tensor.
        """
        seq_actions = seq_payloads[self._action_feature_name]
        if len(self._watchtime_to_action_thresholds_and_weights) > 0:
            watchtimes = seq_payloads[self._watchtime_feature_name]
            for threshold, weight in self._watchtime_to_action_thresholds_and_weights:
                seq_actions = torch.bitwise_or(
                    seq_actions, (watchtimes >= threshold).to(torch.int64) * weight
                )
        exploded_actions = (
            torch.bitwise_and(
                seq_actions.unsqueeze(-1), self._combined_action_weights.unsqueeze(0)
            )
            > 0
        )
        action_embeddings = (
            exploded_actions.unsqueeze(-1) * self._action_embedding_table.unsqueeze(0)
        ).view(-1, self._num_action_types * self._action_embedding_dim)

        total_targets: int = seq_embeddings.size(0) - action_embeddings.size(0)
        action_embeddings = concat_2D_jagged(
            values_left=action_embeddings,
            values_right=self._target_action_embedding_table.tile(
                total_targets,
                1,
            ),
            max_len_left=max_uih_len,
            max_len_right=max_targets,
            offsets_left=uih_offsets,
            offsets_right=target_offsets,
            kernel=self.kernel(),
        )
        return action_embeddings
