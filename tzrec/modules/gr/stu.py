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

# We use the STU from generative-recommenders a starting point.
# https://github.com/facebookresearch/generative-recommenders
# thanks to their public work.

import abc
from typing import List, Optional, Tuple

import torch
from torch.autograd.profiler import record_function

from tzrec.modules.utils import BaseModule
from tzrec.ops import Kernel
from tzrec.ops.hstu_attention import delta_hstu_mha
from tzrec.ops.hstu_attention_utils import (
    apply_stu_truncation,
    build_sla_func_tensor,
)
from tzrec.ops.hstu_compute import (
    hstu_compute_output,
    hstu_compute_uqvk,
    hstu_preprocess_and_attention,
)
from tzrec.ops.jagged_tensors import concat_2D_jagged, split_2D_jagged
from tzrec.utils import env_util
from tzrec.utils.fx_util import fx_unwrap_optional_tensor


class STU(BaseModule, abc.ABC):
    """STU abstract module."""

    def cached_forward(
        self,
        delta_x: torch.Tensor,
        num_targets: torch.Tensor,
        max_kv_caching_len: int = 0,
        kv_caching_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward with cached key-value tensors.

        Args:
            delta_x (torch.Tensor): delta input sequence embedding tensor.
            num_targets (torch.Tensor): number of targets.
            max_kv_caching_len (int): maximum key-value caching length.
            kv_caching_lengths (Optional[torch.Tensor]): key-value caching lengths.

        Returns:
            torch.Tensor: output sequence embedding tensor.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        x_offsets: torch.Tensor,
        max_seq_len: int,
        num_targets: Optional[torch.Tensor],
        max_kv_caching_len: int = 0,
        kv_caching_lengths: Optional[torch.Tensor] = None,
        attn_func: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward the layer.

        Args:
            x (torch.Tensor): input sequence embedding tensor.
            x_offsets (torch.Tensor): input sequence offsets.
            max_seq_len (int): maximum sequence length.
            num_targets (Optional[torch.Tensor]): number of targets per batch
                element (None in listwise-training mode).
            max_kv_caching_len (int): maximum key-value caching length.
            kv_caching_lengths (Optional[torch.Tensor]): key-value caching lengths.
            attn_func (Optional[torch.Tensor]): pre-built NFUNC mask tensor.

        Returns:
            torch.Tensor: output sequence embedding tensor.
        """
        pass


def _update_kv_cache(
    max_seq_len: int,
    seq_offsets: torch.Tensor,
    k: Optional[torch.Tensor],
    v: Optional[torch.Tensor],
    max_kv_caching_len: int,
    kv_caching_lengths: Optional[torch.Tensor],
    orig_k_cache: Optional[torch.Tensor],
    orig_v_cache: Optional[torch.Tensor],
    orig_max_kv_caching_len: int,
    orig_kv_caching_offsets: Optional[torch.Tensor],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], int, Optional[torch.Tensor]]:
    if kv_caching_lengths is not None:
        kv_caching_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
            kv_caching_lengths
        )
        delta_offsets = seq_offsets - kv_caching_offsets
        k_cache, _ = split_2D_jagged(
            max_seq_len=max_seq_len,
            values=fx_unwrap_optional_tensor(k).flatten(1, 2),
            max_len_left=None,
            max_len_right=None,
            offsets_left=kv_caching_offsets,
            offsets_right=delta_offsets,
        )
        v_cache, _ = split_2D_jagged(
            max_seq_len=max_seq_len,
            values=fx_unwrap_optional_tensor(v).flatten(1, 2),
            max_len_left=None,
            max_len_right=None,
            offsets_left=kv_caching_offsets,
            offsets_right=delta_offsets,
        )
        if max_kv_caching_len == 0:
            max_kv_caching_len = int(kv_caching_lengths.max().item())
        return (
            k_cache,
            v_cache,
            max_kv_caching_len,
            kv_caching_offsets,
        )
    else:
        return (
            orig_k_cache,
            orig_v_cache,
            orig_max_kv_caching_len,
            orig_kv_caching_offsets,
        )


@torch.fx.wrap
def _construct_full_kv(
    delta_k: torch.Tensor,
    delta_v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    max_kv_caching_len: int,
    kv_caching_offsets: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
    L, _ = delta_k.shape
    B = kv_caching_offsets.shape[0] - 1
    delta_size = L // B
    full_k = concat_2D_jagged(
        values_left=k_cache,
        values_right=delta_k,
        max_len_left=max_kv_caching_len,
        max_len_right=delta_size,
        offsets_left=kv_caching_offsets,
        offsets_right=None,
    )
    full_v = concat_2D_jagged(
        values_left=v_cache,
        values_right=delta_v,
        max_len_left=max_kv_caching_len,
        max_len_right=delta_size,
        offsets_left=kv_caching_offsets,
        offsets_right=None,
    )
    full_kv_caching_offsets = kv_caching_offsets + delta_size * torch.arange(
        B + 1, device=delta_k.device
    )
    return (
        full_k,
        full_v,
        max_kv_caching_len + delta_size,
        full_kv_caching_offsets,
    )


class STULayer(STU):
    """A jagged sequential transduction unit for variable-length sequences.

    Args:
        embedding_dim (int): dimension of input embeddings
        num_heads (int): number of attention heads
        hidden_dim (int): dimension of hidden linear layers
        attention_dim (int): dimension of attention mechanism
        output_dropout_ratio (float): dropout probability for linear layers
        causal (bool): whether to use causal mask in attention
        target_aware (bool): whether to target mask in attention
        max_attn_len (int): maximum length of attention window
        attn_alpha (float): alpha for mha attention
        use_group_norm (bool): use group normalization or layer normalization.
        recompute_normed_x (bool): whether to recompute normed_x in backward
        recompute_uvqk (bool): whether to recompute uvqk in backward
        recompute_y (bool): whether to recompute y in backward
        sort_by_length (bool): whether to sort by length when forwarding
        contextual_seq_len (int): sequence length of contextual feature
        is_inference (bool): whether to run in inference mode.
    """

    max_kv_caching_len: int
    k_cache: Optional[torch.Tensor]
    v_cache: Optional[torch.Tensor]
    kv_caching_offsets: Optional[torch.Tensor]

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        hidden_dim: int,
        attention_dim: int,
        output_dropout_ratio: float = 0.3,
        causal: bool = True,
        target_aware: bool = True,
        max_attn_len: Optional[int] = None,
        attn_alpha: Optional[float] = None,
        use_group_norm: bool = False,
        recompute_normed_x: bool = True,
        recompute_uvqk: bool = True,
        recompute_y: bool = True,
        sort_by_length: bool = True,
        contextual_seq_len: int = 0,
        sla_k1: int = 0,
        sla_k2: int = 0,
        is_inference: bool = False,
    ) -> None:
        super().__init__(
            is_inference=is_inference,
        )
        self.reset_kv_cache()
        self._num_heads: int = num_heads
        self._embedding_dim: int = embedding_dim
        self._hidden_dim: int = hidden_dim
        self._attention_dim: int = attention_dim
        self._output_dropout_ratio: float = output_dropout_ratio
        self._target_aware: bool = target_aware
        self._causal: bool = causal
        self._max_attn_len: int = max_attn_len or 0
        self._attn_alpha: float = attn_alpha or 1.0 / (self._attention_dim**0.5)
        self._use_group_norm: bool = use_group_norm
        self._recompute_normed_x: bool = recompute_normed_x
        self._recompute_uvqk: bool = recompute_uvqk
        self._recompute_y: bool = recompute_y
        self._sort_by_length: bool = sort_by_length
        self._contextual_seq_len: int = contextual_seq_len
        self._sla_k1: int = sla_k1
        self._sla_k2: int = sla_k2

        self._uvqk_weight: torch.nn.Parameter = torch.nn.Parameter(
            torch.empty(
                (
                    self._embedding_dim,
                    (self._hidden_dim * 2 + self._attention_dim * 2) * self._num_heads,
                )
            ),
        )
        torch.nn.init.xavier_uniform_(self._uvqk_weight)
        self._uvqk_beta: torch.nn.Parameter = torch.nn.Parameter(
            torch.zeros(
                (self._hidden_dim * 2 + self._attention_dim * 2) * self._num_heads,
            ),
        )
        self._input_norm_weight: torch.nn.Parameter = torch.nn.Parameter(
            torch.ones((self._embedding_dim,)),
        )
        self._input_norm_bias: torch.nn.Parameter = torch.nn.Parameter(
            torch.zeros((self._embedding_dim,)),
        )
        self._output_weight = torch.nn.Parameter(
            torch.empty(
                (
                    self._hidden_dim * self._num_heads * 3,
                    self._embedding_dim,
                )
            ),
        )
        torch.nn.init.xavier_uniform_(self._output_weight)
        output_norm_shape: int = (
            self._hidden_dim * self._num_heads
            if not self._use_group_norm
            else self._num_heads
        )
        self._output_norm_weight: torch.nn.Parameter = torch.nn.Parameter(
            torch.ones((output_norm_shape,)),
        )
        self._output_norm_bias: torch.nn.Parameter = torch.nn.Parameter(
            torch.zeros((output_norm_shape,)),
        )
        self._enable_tma = env_util.enable_tma()

    def reset_kv_cache(self) -> None:
        """Reset the key-value cache."""
        self.k_cache = None
        self.v_cache = None
        self.kv_caching_offsets = None
        self.max_kv_caching_len = 0

    def update_kv_cache(
        self,
        max_seq_len: int,
        seq_offsets: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        max_kv_caching_len: int,
        kv_caching_lengths: Optional[torch.Tensor],
    ) -> None:
        """Update the key-value cache.

        Args:
            max_seq_len (int): maximum sequence length
            seq_offsets (torch.Tensor): sequence offsets
            k (Optional[torch.Tensor]): key tensor
            v (Optional[torch.Tensor]): value tensor
            max_kv_caching_len (int): maximum key-value caching length
            kv_caching_lengths (Optional[torch.Tensor]): key-value caching lengths
        """
        self.k_cache, self.v_cache, self.max_kv_caching_len, self.kv_caching_offsets = (
            _update_kv_cache(
                max_seq_len=max_seq_len,
                seq_offsets=seq_offsets,
                k=k,
                v=v,
                max_kv_caching_len=max_kv_caching_len,
                kv_caching_lengths=kv_caching_lengths,
                orig_k_cache=self.k_cache,
                orig_v_cache=self.v_cache,
                orig_max_kv_caching_len=self.max_kv_caching_len,
                orig_kv_caching_offsets=self.kv_caching_offsets,
            )
        )

    def construct_full_kv(
        self,
        delta_k: torch.Tensor,
        delta_v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
        """Construct full key-value tensor.

        Args:
            delta_k (torch.Tensor): delta key tensor.
            delta_v (torch.Tensor): delta value tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
                full key-value tensor,
                full value tensor,
                full key-value caching length,
                full key-value caching offsets.
        """
        return _construct_full_kv(
            delta_k=delta_k,
            delta_v=delta_v,
            k_cache=fx_unwrap_optional_tensor(self.k_cache),
            v_cache=fx_unwrap_optional_tensor(self.v_cache),
            max_kv_caching_len=self.max_kv_caching_len,
            kv_caching_offsets=self.kv_caching_offsets,
        )

    def forward(
        self,
        x: torch.Tensor,
        x_offsets: torch.Tensor,
        max_seq_len: int,
        num_targets: Optional[torch.Tensor],
        max_kv_caching_len: int = 0,
        kv_caching_lengths: Optional[torch.Tensor] = None,
        attn_func: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward the layer.

        Args:
            x (torch.Tensor): input sequence embedding tensor.
            x_offsets (torch.Tensor): input sequence offsets.
            max_seq_len (int): maximum sequence length.
            num_targets (Optional[torch.Tensor]): number of targets per batch
                element (None in listwise-training mode).
            max_kv_caching_len (int): maximum key-value caching length.
            kv_caching_lengths (Optional[torch.Tensor]): key-value caching lengths.
            attn_func (Optional[torch.Tensor]): pre-built arbitrary-mask func
                tensor for the CUTLASS NFUNC path. When ``None`` and SLA is
                enabled on this layer (``self._sla_k1 > 0 or self._sla_k2 >
                0``), it is built from ``self._sla_*`` / ``x_offsets`` /
                ``num_targets``. When supplied by the caller (typically
                ``STUStack.forward`` which hoists construction to batch
                scope) the build is skipped.

        Returns:
            torch.Tensor: output sequence embedding tensor.
        """
        if attn_func is None and (self._sla_k1 > 0 or self._sla_k2 > 0):
            attn_func = build_sla_func_tensor(
                nheads=self._num_heads,
                sla_k1=self._sla_k1,
                sla_k2=self._sla_k2,
                seq_offsets=x_offsets,
                total_q=x.size(0),
                num_targets=num_targets if self._target_aware else None,
                contextual_seq_len=self._contextual_seq_len,
            )

        with record_function("## stu_preprocess_and_attention ##"):
            u, attn_output, k, v = hstu_preprocess_and_attention(
                x=x,
                norm_weight=self._input_norm_weight.to(x.dtype),
                norm_bias=self._input_norm_bias.to(x.dtype),
                norm_eps=1e-6,
                num_heads=self._num_heads,
                attn_dim=self._attention_dim,
                hidden_dim=self._hidden_dim,
                uvqk_weight=self._uvqk_weight.to(x.dtype),
                uvqk_bias=self._uvqk_beta.to(x.dtype),
                max_seq_len=max_seq_len,
                seq_offsets=x_offsets,
                attn_alpha=self._attn_alpha,
                causal=self._causal,
                num_targets=num_targets if self._target_aware else None,
                max_attn_len=self._max_attn_len,
                contextual_seq_len=self._contextual_seq_len,
                recompute_uvqk_in_backward=self._recompute_uvqk,
                recompute_normed_x_in_backward=self._recompute_normed_x,
                sort_by_length=self._sort_by_length,
                prefill=kv_caching_lengths is not None,
                kernel=self.kernel(),
                enable_tma=self._enable_tma,
                attn_func=attn_func,
            )

        self.update_kv_cache(
            max_seq_len=max_seq_len,
            seq_offsets=x_offsets,
            k=k,
            v=v,
            max_kv_caching_len=max_kv_caching_len,
            kv_caching_lengths=kv_caching_lengths,
        )

        with record_function("## stu_compute_output ##"):
            return hstu_compute_output(
                attn=attn_output,
                u=u,
                x=x,
                norm_weight=self._output_norm_weight.to(x.dtype),
                norm_bias=self._output_norm_bias.to(x.dtype),
                norm_eps=1e-6,
                dropout_ratio=self._output_dropout_ratio,
                output_weight=self._output_weight.to(x.dtype),
                group_norm=self._use_group_norm,
                num_heads=self._num_heads,
                linear_dim=self._hidden_dim,
                concat_ux=True,
                training=self.training,
                kernel=self.kernel(),
                recompute_y_in_backward=self._recompute_y,
            )

    def cached_forward(
        self,
        delta_x: torch.Tensor,
        num_targets: torch.Tensor,
        max_kv_caching_len: int = 0,
        kv_caching_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward with cached key-value tensors.

        Args:
            delta_x (torch.Tensor): delta input sequence embedding tensor.
            num_targets (torch.Tensor): number of targets.
            max_kv_caching_len (int): maximum key-value caching length.
            kv_caching_lengths (Optional[torch.Tensor]): key-value caching lengths.

        Returns:
            torch.Tensor: output sequence embedding tensor.
        """
        with record_function("## stu_compute_uqvk ##"):
            delta_u, delta_q, delta_k, delta_v = hstu_compute_uqvk(
                x=delta_x,
                norm_weight=self._input_norm_weight.to(delta_x.dtype),
                norm_bias=self._input_norm_bias.to(delta_x.dtype),
                norm_eps=1e-6,
                num_heads=self._num_heads,
                attn_dim=self._attention_dim,
                hidden_dim=self._hidden_dim,
                uvqk_weight=self._uvqk_weight.to(delta_x.dtype),
                uvqk_bias=self._uvqk_beta.to(delta_x.dtype),
                kernel=self.kernel(),
            )
        k, v, max_seq_len, seq_offsets = self.construct_full_kv(
            delta_k=delta_k.flatten(1, 2),
            delta_v=delta_v.flatten(1, 2),
        )
        self.update_kv_cache(
            max_seq_len=max_seq_len,
            seq_offsets=seq_offsets,
            k=k,
            v=v,
            max_kv_caching_len=max_kv_caching_len,
            kv_caching_lengths=kv_caching_lengths,
        )
        k = k.view(-1, self._num_heads, self._attention_dim)
        v = v.view(-1, self._num_heads, self._hidden_dim)
        with record_function("## delta_hstu_mha ##"):
            delta_attn_output = delta_hstu_mha(
                max_seq_len=max_seq_len,
                alpha=self._attn_alpha,
                delta_q=delta_q,
                k=k,
                v=v,
                seq_offsets=seq_offsets,
                num_targets=num_targets if self._target_aware else None,
                max_attn_len=self._max_attn_len,
                contextual_seq_len=self._contextual_seq_len,
                kernel=self.kernel(),
                enable_tma=self._enable_tma,
            ).view(-1, self._hidden_dim * self._num_heads)
        with record_function("## stu_compute_output ##"):
            return hstu_compute_output(
                attn=delta_attn_output,
                u=delta_u,
                x=delta_x,
                norm_weight=self._output_norm_weight.to(delta_x.dtype),
                norm_bias=self._output_norm_bias.to(delta_x.dtype),
                norm_eps=1e-6,
                dropout_ratio=self._output_dropout_ratio,
                output_weight=self._output_weight.to(delta_x.dtype),
                group_norm=self._use_group_norm,
                num_heads=self._num_heads,
                linear_dim=self._hidden_dim,
                concat_ux=True,
                training=self.training,
                kernel=self.kernel(),
                recompute_y_in_backward=self._recompute_y,
            )


class STUStack(STU):
    """Stack of STU layers with optional attention truncation.

    Args:
        stu_list (List[STU]): list of STU layers.
        truncate_split_layer (int): after this many layers, truncate to tail.
            0 means disabled (all layers run on full sequence).
        truncate_tail_len (int): number of trailing tokens to keep after
            truncation. Only used when truncate_split_layer > 0.
        is_inference (bool): whether to run in inference mode.
    """

    def __init__(
        self,
        stu_list: List[STU],
        truncate_split_layer: int = 0,
        truncate_tail_len: int = 0,
        is_inference: bool = False,
    ) -> None:
        super().__init__(is_inference=is_inference)
        self._stu_layers: torch.nn.ModuleList = torch.nn.ModuleList(modules=stu_list)
        self._truncate_split_layer: int = truncate_split_layer
        self._truncate_tail_len: int = truncate_tail_len
        # Without this check, truncate_split_layer >= len(stu_list) silently
        # no-ops (the loop never hits the split index) and truncation is
        # lost without any signal.
        if self._truncate_tail_len > 0 and not (
            0 < self._truncate_split_layer < len(self._stu_layers)
        ):
            raise ValueError(
                f"truncate_split_layer must be in (0, {len(self._stu_layers)}) "
                f"when truncate_tail_len > 0; got truncate_split_layer="
                f"{self._truncate_split_layer}, truncate_tail_len="
                f"{self._truncate_tail_len}."
            )

    def forward(
        self,
        x: torch.Tensor,
        x_offsets: torch.Tensor,
        max_seq_len: int,
        num_targets: Optional[torch.Tensor],
        max_kv_caching_len: int = 0,
        kv_caching_lengths: Optional[torch.Tensor] = None,
        attn_func: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], int]:
        """Forward stack of stu layer.

        Args:
            x (torch.Tensor): input sequence embedding tensor.
            x_offsets (torch.Tensor): input sequence offsets.
            max_seq_len (int): maximum sequence length.
            num_targets (Optional[torch.Tensor]): number of targets per batch
                element (None in listwise-training mode).
            max_kv_caching_len (int): maximum key-value caching length.
            kv_caching_lengths (Optional[torch.Tensor]): key-value caching lengths.
            attn_func: accepted for signature parity with ``STULayer.forward``.
                STUStack builds its own per-layer ``attn_func`` (memoized on
                SLA config + offsets identity), so any caller-supplied value
                is overwritten on the first iteration that needs it.

        Returns:
            A 4-tuple ``(x, x_offsets, num_targets, max_seq_len)``.  When
            attention truncation is enabled (``truncate_tail_len > 0``) the
            three metadata fields reflect the post-truncation state so that
            downstream consumers (e.g. ``HSTUTransducer._postprocess``) see
            offsets/lengths matching the returned ``x``.  When truncation is
            disabled the metadata equals the inputs.
        """
        seq_lengths = x_offsets[1:] - x_offsets[:-1]
        # SLA (sla_k1 > 0 or sla_k2 > 0) runs on the CUTLASS production
        # kernel or the PyTorch reference kernel; Kernel.TRITON has no
        # NFUNC path. Surface a loud error instead of silently dropping
        # the mask when a layer's kernel and SLA config disagree.
        for layer in self._stu_layers:
            if (
                getattr(layer, "_sla_k1", 0) > 0 or getattr(layer, "_sla_k2", 0) > 0
            ) and layer.kernel() == Kernel.TRITON:
                raise ValueError(
                    f"STULayer has SLA enabled (sla_k1="
                    f"{getattr(layer, '_sla_k1', 0)}, sla_k2="
                    f"{getattr(layer, '_sla_k2', 0)}) but kernel is "
                    f"Kernel.TRITON. SLA requires Kernel.CUTLASS or "
                    f"Kernel.PYTORCH."
                )
        # Hoist SLA func construction out of the per-layer attention op so
        # a stack of N layers doesn't rebuild the same (nheads, 3, total_q)
        # tensor N times per forward.  We memoize on the SLA config +
        # current offsets identity and only rebuild when those change (in
        # particular, once across the truncation boundary).
        cur_attn_func: Optional[torch.Tensor] = attn_func
        attn_func_sig: Optional[tuple] = None
        for i, layer in enumerate(self._stu_layers):
            if (
                i == self._truncate_split_layer
                and self._truncate_split_layer > 0
                and self._truncate_tail_len > 0
            ):
                x, x_offsets, seq_lengths, num_targets, max_seq_len = (
                    apply_stu_truncation(
                        x=x,
                        x_offsets=x_offsets,
                        seq_lengths=seq_lengths,
                        num_targets=num_targets,
                        max_seq_len=max_seq_len,
                        truncate_tail_len=self._truncate_tail_len,
                        contextual_seq_len=getattr(
                            self._stu_layers[0], "_contextual_seq_len", 0
                        ),
                        kernel=self.kernel(),
                    )
                )

            sla_k1 = getattr(layer, "_sla_k1", 0)
            sla_k2 = getattr(layer, "_sla_k2", 0)
            if sla_k1 > 0 or sla_k2 > 0:
                ctx_len = getattr(layer, "_contextual_seq_len", 0)
                nheads = getattr(layer, "_num_heads", 0)
                tgt_aware = getattr(layer, "_target_aware", False)
                sig = (
                    sla_k1,
                    sla_k2,
                    ctx_len,
                    nheads,
                    tgt_aware,
                    id(x_offsets),
                )
                if sig != attn_func_sig:
                    cur_attn_func = build_sla_func_tensor(
                        nheads=nheads,
                        sla_k1=sla_k1,
                        sla_k2=sla_k2,
                        seq_offsets=x_offsets,
                        total_q=x.size(0),
                        num_targets=num_targets if tgt_aware else None,
                        contextual_seq_len=ctx_len,
                    )
                    attn_func_sig = sig
            else:
                cur_attn_func = None
                attn_func_sig = None

            x = layer(
                x=x,
                x_offsets=x_offsets,
                max_seq_len=max_seq_len,
                num_targets=num_targets,
                max_kv_caching_len=max_kv_caching_len,
                kv_caching_lengths=kv_caching_lengths,
                attn_func=cur_attn_func,
            )
        return x, x_offsets, num_targets, max_seq_len

    def cached_forward(
        self,
        delta_x: torch.Tensor,
        num_targets: torch.Tensor,
        max_kv_caching_len: int = 0,
        kv_caching_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward stack of stu layer with cached key-value tensors.

        Args:
            delta_x (torch.Tensor): delta input sequence embedding tensor.
            num_targets (torch.Tensor): number of targets.
            max_kv_caching_len (int): maximum key-value caching length.
            kv_caching_lengths (Optional[torch.Tensor]): key-value caching lengths.

        Returns:
            torch.Tensor: output sequence embedding tensor.
        """
        if self._truncate_tail_len > 0 and self._truncate_split_layer > 0:
            # The cached / delta path operates per-layer on a rolling KV
            # cache; mid-stack truncation would drop prefix tokens that the
            # post-truncation layers still reference via the cache, causing
            # train/serve skew. Refuse the config rather than silently
            # diverge -- proper support is a follow-up.
            raise NotImplementedError(
                "STUStack attention truncation is not supported in "
                "cached_forward (serving path). Either disable truncation "
                "or use the non-cached forward path."
            )
        for layer in self._stu_layers:
            delta_x = layer.cached_forward(
                delta_x=delta_x,
                num_targets=num_targets,
                max_kv_caching_len=max_kv_caching_len,
                kv_caching_lengths=kv_caching_lengths,
            )
        return delta_x
