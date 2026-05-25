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
from tzrec.modules.gr.stu import STULayer, STUStack
from tzrec.modules.utils import BaseModule
from tzrec.ops import Kernel
from tzrec.ops.hstu_attention_utils import (
    STUTruncationPlan,
    apply_stu_truncation_plan,
)
from tzrec.ops.jagged_tensors import split_2D_jagged
from tzrec.utils.fx_util import fx_unwrap_optional_tensor

torch.fx.wrap("len")


class _HSTUPipelineBase(BaseModule):
    """Shared HSTU encode pipeline.

    Owns the construction and forward chain of:
        input_preprocessor -> positional_encoder -> dropout -> STU stack
        -> mid-stack-truncation state replay.

    Subclasses (`HSTUTransducer`, `HSTUMatchEncoder`) set their own
    `_output_postprocessor` and implement `_compose_output` to produce the
    domain-specific final embedding(s) from the shared post-STU state.

    State_dict keys for subclasses are flat on `self` -- the base sets
    `self._input_preprocessor`, `self._positional_encoder`, `self._stu_module`,
    so a subclass that previously held these submodules directly keeps the
    same serialization keys.
    """

    def __init__(
        self,
        uih_embedding_dim: int,
        target_embedding_dim: int,
        stu: Dict[str, Any],
        attn_num_layers: int,
        input_preprocessor: Dict[str, Any],
        input_dropout_ratio: float = 0.0,
        positional_encoder: Optional[Dict[str, Any]] = None,
        contextual_feature_dim: int = 0,
        max_contextual_seq_len: int = 0,
        contextual_group_name: str = "contextual",
        scaling_seqlen: int = -1,
        is_inference: bool = True,
        attn_truncation_split_layer: int = 0,
        attn_truncation_tail_len: int = 0,
        name: str = "",
        query_time_key: str = "",
    ) -> None:
        super().__init__(is_inference=is_inference)
        # Grouped-feature key of the per-row request time used as the time-bias
        # anchor. Empty -> anchor on the last in-sequence timestamp (canonical
        # HSTU / DLRM-HSTU, which concatenates the candidate request time).
        self._query_time_key: str = query_time_key
        self._input_preprocessor: InputPreprocessor = create_input_preprocessor(
            input_preprocessor,
            uih_embedding_dim=uih_embedding_dim,
            target_embedding_dim=target_embedding_dim,
            contextual_feature_dim=contextual_feature_dim,
            max_contextual_seq_len=max_contextual_seq_len,
            contextual_group_name=contextual_group_name,
            output_embedding_dim=stu["embedding_dim"],
            name=name,
        )
        stu = dict(stu)
        # `< 0` sentinel = fall back; explicit 0 means "no contextual".
        # `not in stu` is unreachable -- config_to_kwargs's
        # including_default_value_fields=True always populates the key.
        if stu.get("contextual_seq_len", -1) < 0:
            stu["contextual_seq_len"] = self._input_preprocessor.contextual_seq_len()
        if stu.get("scaling_seqlen", -1) < 0:
            stu["scaling_seqlen"] = scaling_seqlen
        self._stu_module: STUStack = STUStack(
            stu_list=[STULayer(**stu) for _ in range(attn_num_layers)],
            truncate_split_layer=attn_truncation_split_layer,
            truncate_tail_len=attn_truncation_tail_len,
        )
        self._positional_encoder: Optional[HSTUPositionalEncoder] = None
        if positional_encoder is not None:
            self._positional_encoder = HSTUPositionalEncoder(
                embedding_dim=stu["embedding_dim"],
                contextual_seq_len=self._input_preprocessor.contextual_seq_len(),
                **positional_encoder,
            )
        self._input_dropout_ratio: float = input_dropout_ratio

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

        # Per-row request time anchor (HSTUMatch). Read from grouped_features
        # rather than the preprocessor tuple so the shared ranking path is
        # untouched. `[B, 1]` raw values -> the op reshapes to `[B]`.
        query_time: Optional[torch.Tensor] = None
        if self._query_time_key != "":
            query_time = grouped_features[self._query_time_key]

        with record_function("hstu_positional_encoder"):
            if self._positional_encoder is not None:
                output_seq_embeddings = self._positional_encoder(
                    max_seq_len=output_max_seq_len,
                    seq_lengths=output_seq_lengths,
                    seq_offsets=output_seq_offsets,
                    seq_timestamps=output_seq_timestamps,
                    seq_embeddings=output_seq_embeddings,
                    num_targets=output_num_targets,
                    query_time=query_time,
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
    ) -> Tuple[torch.Tensor, torch.Tensor, int, Optional[STUTruncationPlan]]:
        with record_function("hstu"):
            return self._stu_module(
                max_seq_len=max_seq_len,
                x=seq_embeddings,
                x_offsets=seq_offsets,
                num_targets=num_targets,
            )

    @staticmethod
    def _replay_truncation_state(
        seq_timestamps: torch.Tensor,
        seq_lengths: torch.Tensor,
        seq_offsets: torch.Tensor,
        max_seq_len: int,
        total_uih_len: int,
        total_targets: int,
        post_stu_seq_offsets: torch.Tensor,
        post_stu_max_seq_len: int,
        plan: Optional[STUTruncationPlan],
        kernel: Kernel,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        """Replay ``plan`` on ``seq_timestamps`` and refresh dependent metadata.

        ``plan is None`` -> inputs unchanged.  Otherwise the returned
        ``post_truncation_total_uih_len`` is ``plan.total_kept +
        plan.total_prefix - total_targets`` (static int -- skips
        ``split_2D_jagged``'s ``.item()`` fallback).  ``total_prefix``
        is the add-back for the contextual prefix that ``total_kept``
        excludes; see ``STUTruncationPlan.total_kept``.
        """
        if plan is None:
            return (
                seq_timestamps,
                seq_lengths,
                seq_offsets,
                max_seq_len,
                total_uih_len,
            )
        seq_timestamps = apply_stu_truncation_plan(
            seq_timestamps.unsqueeze(-1), plan, kernel=kernel
        ).squeeze(-1)
        return (
            seq_timestamps,
            plan.new_lengths,
            post_stu_seq_offsets,
            post_stu_max_seq_len,
            plan.total_kept + plan.total_prefix - total_targets,
        )

    def _compose_output(
        self,
        encoded_embeddings: torch.Tensor,
        seq_timestamps: torch.Tensor,
        seq_lengths: torch.Tensor,
        seq_offsets: torch.Tensor,
        max_seq_len: int,
        total_uih_len: int,
        total_targets: int,
        num_targets: torch.Tensor,
    ) -> Any:
        """Compose the final output from the shared post-STU state.

        Subclasses implement domain-specific finalization (e.g., UIH/candidate
        split + candidate postprocess for ranking; full-postprocess +
        last-position extraction for match).
        """
        raise NotImplementedError

    def forward(self, grouped_features: Dict[str, torch.Tensor]) -> Any:
        """Forward the module.

        Args:
            grouped_features (Dict[str, torch.Tensor]): embedding group features.

        Returns:
            subclass-determined output; see `_compose_output`.
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

        (
            encoded_embeddings,
            post_stu_seq_offsets,
            post_stu_max_seq_len,
            plan,
        ) = self._hstu_compute(
            max_seq_len=max_seq_len,
            seq_lengths=seq_lengths,
            seq_offsets=seq_offsets,
            seq_timestamps=seq_timestamps,
            seq_embeddings=seq_embeddings,
            num_targets=num_targets,
        )

        # When STUStack truncated mid-stack, replay the same split on the
        # parallel jagged seq_timestamps so downstream finalization lines up
        # with the truncated embeddings.
        (
            seq_timestamps,
            seq_lengths,
            seq_offsets,
            max_seq_len,
            post_truncation_total_uih_len,
        ) = self._replay_truncation_state(
            seq_timestamps=seq_timestamps,
            seq_lengths=seq_lengths,
            seq_offsets=seq_offsets,
            max_seq_len=max_seq_len,
            total_uih_len=total_uih_len,
            total_targets=total_targets,
            post_stu_seq_offsets=post_stu_seq_offsets,
            post_stu_max_seq_len=post_stu_max_seq_len,
            plan=plan,
            kernel=self.kernel(),
        )

        return self._compose_output(
            encoded_embeddings=encoded_embeddings,
            seq_timestamps=seq_timestamps,
            seq_lengths=seq_lengths,
            seq_offsets=seq_offsets,
            max_seq_len=max_seq_len,
            total_uih_len=post_truncation_total_uih_len,
            total_targets=total_targets,
            num_targets=num_targets,
        )


class HSTUTransducer(_HSTUPipelineBase):
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
        contextual_group_name (str): contextual group name in grouped features.
        scaling_seqlen (int): sequence length used as the divisor in the
            attention output scaling of every STULayer. ``-1`` (the default)
            preserves legacy behavior (divides by runtime ``max_seq_len``).
            Pass a fixed positive int (typically the model's ``max_seq_len``
            config) to make attention output invariant to batch-level
            seq-length.
        is_inference (bool): whether to run in inference mode.
        return_full_embeddings (bool): return all embeddings or not.
        attn_truncation_split_layer (int): layer index ``N1`` after which
            mid-stack attention truncation fires.  Must be in
            ``(0, attn_num_layers)`` when truncation is enabled, else 0.
        attn_truncation_tail_len (int): number of trailing UIH tokens kept
            on layers ``>= N1``.  Both ``attn_truncation_split_layer`` and
            ``attn_truncation_tail_len`` must be ``> 0`` to enable
            truncation; setting only one is rejected at construction.
        name (str): MoT channel name; forwarded to the input
            preprocessor (replaces the ``uih`` prefix on UIH-side keys).
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
        contextual_group_name: str = "contextual",
        scaling_seqlen: int = -1,
        is_inference: bool = True,
        return_full_embeddings: bool = False,
        attn_truncation_split_layer: int = 0,
        attn_truncation_tail_len: int = 0,
        name: str = "",
    ) -> None:
        super().__init__(
            uih_embedding_dim=uih_embedding_dim,
            target_embedding_dim=target_embedding_dim,
            stu=stu,
            attn_num_layers=attn_num_layers,
            input_preprocessor=input_preprocessor,
            input_dropout_ratio=input_dropout_ratio,
            positional_encoder=positional_encoder,
            contextual_feature_dim=contextual_feature_dim,
            max_contextual_seq_len=max_contextual_seq_len,
            contextual_group_name=contextual_group_name,
            scaling_seqlen=scaling_seqlen,
            is_inference=is_inference,
            attn_truncation_split_layer=attn_truncation_split_layer,
            attn_truncation_tail_len=attn_truncation_tail_len,
            name=name,
        )
        self._output_postprocessor: OutputPostprocessor = create_output_postprocessor(
            output_postprocessor, embedding_dim=stu["embedding_dim"]
        )
        self._return_full_embeddings: bool = return_full_embeddings

    def _compose_output(
        self,
        encoded_embeddings: torch.Tensor,
        seq_timestamps: torch.Tensor,
        seq_lengths: torch.Tensor,
        seq_offsets: torch.Tensor,
        max_seq_len: int,
        total_uih_len: int,
        total_targets: int,
        num_targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        with record_function("hstu_output_postprocessor"):
            if self._return_full_embeddings:
                encoded_embeddings = self._output_postprocessor(
                    seq_embeddings=encoded_embeddings,
                    seq_timestamps=seq_timestamps,
                )
            uih_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
                seq_lengths - num_targets
            )
            candidates_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
                num_targets
            )
            _, candidate_embeddings = split_2D_jagged(
                values=encoded_embeddings,
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

        full = encoded_embeddings if self._return_full_embeddings else None
        if not self._is_inference and self._return_full_embeddings:
            full = fx_unwrap_optional_tensor(full)
        return (candidate_embeddings, full)


class HSTUMatchEncoder(_HSTUPipelineBase):
    """HSTU encoder for two-tower match user side.

    UIH-only (no candidate concatenation). Postprocesses the full encoded
    sequence and extracts the last-position embedding per user -- the
    user embedding consumed by `MatchModel.sim`.

    Truncation (`attn_truncation_split_layer` / `attn_truncation_tail_len`)
    is supported and benefits long-UIH retrieval: early layers do broad
    UIH context aggregation; later layers attend to the recent tail only.
    The last-position embedding lives in the tail by construction, so
    truncation is safe.

    Args:
        uih_embedding_dim (int): The dimension of the uih sequence embeddings.
        stu (dict): STULayer config.
        attn_num_layers (int): number of STULayer.
        input_preprocessor (dict): InputPreprocessor config (must resolve to a
            UIH-only variant; e.g. `uih_preprocessor`).
        output_postprocessor (dict): OutputPostprocessor config.
        input_dropout_ratio (float): dropout ratio after input_preprocessor.
        positional_encoder (dict): HSTUPositionalEncoder config.
        contextual_feature_dim (int): contextual feature dimension.
        max_contextual_seq_len (int): contextual feature num.
        contextual_group_name (str): contextual group name in grouped features.
        scaling_seqlen (int): see `HSTUTransducer`.
        is_inference (bool): whether to run in inference mode.
        attn_truncation_split_layer (int): see `HSTUTransducer`.
        attn_truncation_tail_len (int): see `HSTUTransducer`.
        query_time_key (str): grouped-feature key of the per-row request time
            used as the time-bias anchor. Empty (default) anchors on the last
            UIH timestamp; pass a scalar request-time group to anchor on the
            actual request time (decoupled from UIH staleness).
    """

    def __init__(
        self,
        uih_embedding_dim: int,
        stu: Dict[str, Any],
        attn_num_layers: int,
        input_preprocessor: Dict[str, Any],
        output_postprocessor: Dict[str, Any],
        input_dropout_ratio: float = 0.0,
        positional_encoder: Optional[Dict[str, Any]] = None,
        contextual_feature_dim: int = 0,
        max_contextual_seq_len: int = 0,
        contextual_group_name: str = "contextual",
        scaling_seqlen: int = -1,
        is_inference: bool = True,
        attn_truncation_split_layer: int = 0,
        attn_truncation_tail_len: int = 0,
        name: str = "",
        query_time_key: str = "",
    ) -> None:
        super().__init__(
            uih_embedding_dim=uih_embedding_dim,
            target_embedding_dim=0,  # UIH-only; no candidate in sequence.
            stu=stu,
            attn_num_layers=attn_num_layers,
            input_preprocessor=input_preprocessor,
            input_dropout_ratio=input_dropout_ratio,
            positional_encoder=positional_encoder,
            contextual_feature_dim=contextual_feature_dim,
            max_contextual_seq_len=max_contextual_seq_len,
            contextual_group_name=contextual_group_name,
            scaling_seqlen=scaling_seqlen,
            is_inference=is_inference,
            attn_truncation_split_layer=attn_truncation_split_layer,
            attn_truncation_tail_len=attn_truncation_tail_len,
            name=name,
            query_time_key=query_time_key,
        )
        self._output_postprocessor: OutputPostprocessor = create_output_postprocessor(
            output_postprocessor, embedding_dim=stu["embedding_dim"]
        )

    def _compose_output(
        self,
        encoded_embeddings: torch.Tensor,
        seq_timestamps: torch.Tensor,
        seq_lengths: torch.Tensor,
        seq_offsets: torch.Tensor,
        max_seq_len: int,
        total_uih_len: int,
        total_targets: int,
        num_targets: torch.Tensor,
    ) -> torch.Tensor:
        with record_function("hstu_match_output_postprocessor"):
            encoded = self._output_postprocessor(
                seq_embeddings=encoded_embeddings,
                seq_timestamps=seq_timestamps,
            )
        # Last-position-per-user. For empty UIH (cold-start users) the clamped
        # index points to an unrelated row; mask those rows back to zero.
        last_idx = torch.clamp_min(seq_offsets[1:] - 1, 0)
        last = encoded[last_idx]
        last.masked_fill_((seq_lengths == 0).unsqueeze(-1), 0)
        return last
