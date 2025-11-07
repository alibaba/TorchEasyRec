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

# We use the HSTU transducer from generative-recommenders a starting point.
# https://github.com/facebookresearch/generative-recommenders
# thanks to their public work.

from typing import Any, Dict, Optional, Tuple

import torch
from torch.profiler import record_function

from tzrec.modules.gr.positional_encoder import HSTUPositionalEncoder
from tzrec.modules.gr.postprocessors import (
    OutputPostprocessor,
    create_output_postprocessor,
)
from tzrec.modules.gr.preprocessors import InputPreprocessor, create_input_preprocessor
from tzrec.modules.gr.stu import STU, STULayer, STUStack
from tzrec.modules.utils import BaseModule
from tzrec.ops.jagged_tensors import split_2D_jagged
from tzrec.utils.fx_util import fx_unwrap_optional_tensor

torch.fx.wrap("len")


class HSTUTransducer(BaseModule):
    """HSTU module.

    Args:
        uih_embedding_dim (int): The dimension of the uih sequence embeddings.
        target_embedding_dim (int): The dimension of the candidate sequence embeddings.
        stu (dict): STULayer config.
        attn_num_layers (int): number of STULayer.
        input_preprocessor (dict): InputPreprocessor config.
        output_postprocessor (dict): OutputPostprocessor config.
        input_dropout_ratio (float): dropout ratio after input_preprocessor.
        positional_encoder (dict): HSTUPositionalEncoder config.
        contextual_feature_dim (int): contextual feature dimension.
        max_contextual_seq_len (int): contextual feature num.
        is_inference (bool): whether to run in inference mode.
        return_full_embeddings (bool): return all embeddings or not.
        listwise (bool): listwise training or not.
    """

    def __init__(
        self,
        uih_embedding_dim: int,
        target_embedding_dim: int,
        stu: Dict[str, Any],
        attn_num_layers: int,
        input_preprocessor: Dict[str, Any],
        output_postprocessor: Dict[str, Any],
        input_dropout_ratio: float = 0.0,
        positional_encoder: Optional[Dict[str, Any]] = None,
        contextual_feature_dim: int = 0,
        max_contextual_seq_len: int = 0,
        is_inference: bool = True,
        return_full_embeddings: bool = False,
        listwise: bool = False,
    ) -> None:
        super().__init__(is_inference=is_inference)
        self._stu_module: STU = STUStack(
            stu_list=[STULayer(**stu) for _ in range(attn_num_layers)],
        )
        self._input_preprocessor: InputPreprocessor = create_input_preprocessor(
            input_preprocessor,
            uih_embedding_dim=uih_embedding_dim,
            target_embedding_dim=target_embedding_dim,
            contextual_feature_dim=contextual_feature_dim,
            max_contextual_seq_len=max_contextual_seq_len,
            output_embedding_dim=stu["embedding_dim"],
        )
        self._output_postprocessor: OutputPostprocessor = create_output_postprocessor(
            output_postprocessor, embedding_dim=stu["embedding_dim"]
        )
        self._positional_encoder: Optional[HSTUPositionalEncoder] = None
        if positional_encoder is not None:
            self._positional_encoder = HSTUPositionalEncoder(
                embedding_dim=stu["embedding_dim"],
                contextual_seq_len=self._input_preprocessor.contextual_seq_len(),
                **positional_encoder,
            )
        self._input_dropout_ratio: float = input_dropout_ratio
        self._return_full_embeddings: bool = return_full_embeddings
        self._listwise_training: bool = listwise and self.is_train

    def _preprocess(
        self, grouped_features
    ) -> Tuple[
        int,
        int,
        int,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        with record_function("hstu_input_preprocessor"):
            (
                output_max_seq_len,
                output_total_uih_len,
                output_total_targets,
                output_seq_lengths,
                output_seq_offsets,
                output_seq_timestamps,
                output_seq_embeddings,
                output_num_targets,
            ) = self._input_preprocessor(grouped_features)

        with record_function("hstu_positional_encoder"):
            if self._positional_encoder is not None:
                output_seq_embeddings = self._positional_encoder(
                    max_seq_len=output_max_seq_len,
                    seq_lengths=output_seq_lengths,
                    seq_offsets=output_seq_offsets,
                    seq_timestamps=output_seq_timestamps,
                    seq_embeddings=output_seq_embeddings,
                    num_targets=(
                        None if self._listwise_training else output_num_targets
                    ),
                )

        output_seq_embeddings = torch.nn.functional.dropout(
            output_seq_embeddings,
            p=self._input_dropout_ratio,
            training=self.training,
        )

        return (
            output_max_seq_len,
            output_total_uih_len,
            output_total_targets,
            output_seq_lengths,
            output_seq_offsets,
            output_seq_timestamps,
            output_seq_embeddings,
            output_num_targets,
        )

    def _hstu_compute(
        self,
        max_seq_len: int,
        seq_lengths: torch.Tensor,
        seq_offsets: torch.Tensor,
        seq_timestamps: torch.Tensor,
        seq_embeddings: torch.Tensor,
        num_targets: torch.Tensor,
    ) -> torch.Tensor:
        with record_function("hstu"):
            seq_embeddings = self._stu_module(
                max_seq_len=max_seq_len,
                x=seq_embeddings,
                x_offsets=seq_offsets,
                num_targets=(None if self._listwise_training else num_targets),
            )
        return seq_embeddings

    def _postprocess(
        self,
        max_seq_len: int,
        total_uih_len: int,
        total_targets: int,
        seq_lengths: torch.Tensor,
        seq_timestamps: torch.Tensor,
        seq_embeddings: torch.Tensor,
        num_targets: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        with record_function("hstu_output_postprocessor"):
            if self._return_full_embeddings:
                seq_embeddings = self._output_postprocessor(
                    seq_embeddings=seq_embeddings,
                    seq_timestamps=seq_timestamps,
                )
            uih_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
                seq_lengths - num_targets
            )
            candidates_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
                num_targets
            )
            _, candidate_embeddings = split_2D_jagged(
                values=seq_embeddings,
                max_seq_len=max_seq_len,
                total_len_left=total_uih_len,
                total_len_right=total_targets,
                offsets_left=uih_offsets,
                offsets_right=candidates_offsets,
                kernel=self.kernel(),
            )
            interleave_targets: bool = self._input_preprocessor.interleave_targets()
            if interleave_targets:
                candidate_embeddings = candidate_embeddings.view(
                    -1, 2, candidate_embeddings.size(-1)
                )[:, 0, :]
            if not self._return_full_embeddings:
                _, candidate_timestamps = split_2D_jagged(
                    values=seq_timestamps.unsqueeze(-1),
                    max_seq_len=max_seq_len,
                    total_len_left=total_uih_len,
                    total_len_right=total_targets,
                    offsets_left=uih_offsets,
                    offsets_right=candidates_offsets,
                    kernel=self.kernel(),
                )
                candidate_timestamps = candidate_timestamps.squeeze(-1)
                if interleave_targets:
                    candidate_timestamps = candidate_timestamps.view(-1, 2)[:, 0]
                candidate_embeddings = self._output_postprocessor(
                    seq_embeddings=candidate_embeddings,
                    seq_timestamps=candidate_timestamps,
                )

            return (
                seq_embeddings if self._return_full_embeddings else None,
                candidate_embeddings,
            )

    def forward(
        self, grouped_features: Dict[str, torch.Tensor]
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
    ]:
        """Forward the module.

        Args:
            grouped_features (Dict[str, torch.Tensor]): embedding group features.

        Returns:
            encoded_candidate_embeddings (torch.Tensor): output embedding of candidates.
            encoded_embeddings (torch.Tensor): full output embeddings.
        """
        (
            max_seq_len,
            total_uih_len,
            total_targets,
            seq_lengths,
            seq_offsets,
            seq_timestamps,
            seq_embeddings,
            num_targets,
        ) = self._preprocess(grouped_features)

        encoded_embeddings = self._hstu_compute(
            max_seq_len=max_seq_len,
            seq_lengths=seq_lengths,
            seq_offsets=seq_offsets,
            seq_timestamps=seq_timestamps,
            seq_embeddings=seq_embeddings,
            num_targets=num_targets,
        )

        encoded_embeddings, encoded_candidate_embeddings = self._postprocess(
            max_seq_len=max_seq_len,
            total_uih_len=total_uih_len,
            total_targets=total_targets,
            seq_lengths=seq_lengths,
            seq_embeddings=encoded_embeddings,
            seq_timestamps=seq_timestamps,
            num_targets=num_targets,
        )

        if not self._is_inference:
            if self._return_full_embeddings:
                fx_unwrap_optional_tensor(encoded_embeddings)
        return (
            encoded_candidate_embeddings,
            encoded_embeddings,
        )
