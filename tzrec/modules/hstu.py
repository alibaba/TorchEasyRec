import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import abc

from typing import Callable, Optional, Tuple

TIMESTAMPS_KEY = "timestamps"


class RelativeAttentionBiasModule(torch.nn.Module):

    @abc.abstractmethod
    def forward(
        self,
        all_timestamps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            all_timestamps: [B, N] x int64
        Returns:
            torch.float tensor broadcastable to [B, N, N]
        """
        pass


class RelativePositionalBias(RelativeAttentionBiasModule):

    def __init__(self, max_seq_len: int) -> None:
        super().__init__()

        self._max_seq_len: int = max_seq_len
        self._w = torch.nn.Parameter(
            torch.empty(2 * max_seq_len - 1).normal_(mean=0, std=0.02),
        )

    def forward(
        self,
        all_timestamps: torch.Tensor,
    ) -> torch.Tensor:
        del all_timestamps
        n: int = self._max_seq_len
        t = F.pad(self._w[: 2 * n - 1], [0, n]).repeat(n)
        t = t[..., :-n].reshape(1, n, 3 * n - 2)
        r = (2 * n - 1) // 2
        return t[..., r:-r]


class RelativeBucketedTimeAndPositionBasedBias(RelativeAttentionBiasModule):
    """
    Bucketizes timespans based on ts(next-item) - ts(current-item).
    """

    def __init__(
        self,
        max_seq_len: int,
        num_buckets: int,
        bucketization_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        super().__init__()

        self._max_seq_len: int = max_seq_len
        self._ts_w = torch.nn.Parameter(
            torch.empty(num_buckets + 1).normal_(mean=0, std=0.02),
        )
        self._pos_w = torch.nn.Parameter(
            torch.empty(2 * max_seq_len - 1).normal_(mean=0, std=0.02),
        )
        self._num_buckets: int = num_buckets
        self._bucketization_fn: Callable[[torch.Tensor], torch.Tensor] = (
            bucketization_fn
        )

    def forward(
        self,
        all_timestamps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            all_timestamps: (B, N).
        Returns:
            (B, N, N).
        """
        B = all_timestamps.size(0)
        N = self._max_seq_len
        t = F.pad(self._pos_w[: 2 * N - 1], [0, N]).repeat(N)
        t = t[..., :-N].reshape(1, N, 3 * N - 2)
        r = (2 * N - 1) // 2

        # [B, N + 1] to simplify tensor manipulations.
        ext_timestamps = torch.cat(
            [all_timestamps, all_timestamps[:, N - 1 : N]], dim=1
        )
        # causal masking. Otherwise [:, :-1] - [:, 1:] works
        bucketed_timestamps = torch.clamp(
            self._bucketization_fn(
                ext_timestamps[:, 1:].unsqueeze(2) - ext_timestamps[:, :-1].unsqueeze(1)
            ),
            min=0,
            max=self._num_buckets,
        ).detach()
        rel_pos_bias = t[:, :, r:-r]
        rel_ts_bias = torch.index_select(
            self._ts_w, dim=0, index=bucketed_timestamps.view(-1)
        ).view(B, N, N)
        return rel_pos_bias + rel_ts_bias


HSTUCacheState = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def _hstu_attention_maybe_from_cache(
    num_heads: int,
    attention_dim: int,
    linear_dim: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cached_q: Optional[torch.Tensor],
    cached_k: Optional[torch.Tensor],
    delta_x_offsets: Optional[Tuple[torch.Tensor, torch.Tensor]],
    x_offsets: torch.Tensor,
    all_timestamps: Optional[torch.Tensor],
    invalid_attn_mask: torch.Tensor,
    rel_attn_bias: RelativeAttentionBiasModule,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B: int = x_offsets.size(0) - 1
    n: int = invalid_attn_mask.size(-1)
    if delta_x_offsets is not None:
        padded_q, padded_k = cached_q, cached_k
        flattened_offsets = delta_x_offsets[1] + torch.arange(
            start=0,
            end=B * n,
            step=n,
            device=delta_x_offsets[1].device,
            dtype=delta_x_offsets[1].dtype,
        )
        assert isinstance(padded_q, torch.Tensor)
        assert isinstance(padded_k, torch.Tensor)
        padded_q = (
            padded_q.view(B * n, -1)
            .index_copy_(
                dim=0,
                index=flattened_offsets,
                source=q,
            )
            .view(B, n, -1)
        )
        padded_k = (
            padded_k.view(B * n, -1)
            .index_copy_(
                dim=0,
                index=flattened_offsets,
                source=k,
            )
            .view(B, n, -1)
        )
    else:
        padded_q = torch.ops.fbgemm.jagged_to_padded_dense(
            values=q, offsets=[x_offsets], max_lengths=[n], padding_value=0.0
        )
        padded_k = torch.ops.fbgemm.jagged_to_padded_dense(
            values=k, offsets=[x_offsets], max_lengths=[n], padding_value=0.0
        )

    qk_attn = torch.einsum(
        "bnhd,bmhd->bhnm",
        padded_q.view(B, n, num_heads, attention_dim),
        padded_k.view(B, n, num_heads, attention_dim),
    )
    if all_timestamps is not None:
        qk_attn = qk_attn + rel_attn_bias(all_timestamps).unsqueeze(1)
    qk_attn = F.silu(qk_attn) / n
    qk_attn = qk_attn * invalid_attn_mask.unsqueeze(0).unsqueeze(0)
    attn_output = torch.ops.fbgemm.dense_to_jagged(
        torch.einsum(
            "bhnm,bmhd->bnhd",
            qk_attn,
            torch.ops.fbgemm.jagged_to_padded_dense(v, [x_offsets], [n]).reshape(
                B, n, num_heads, linear_dim
            ),
        ).reshape(B, n, num_heads * linear_dim),
        [x_offsets],
    )[0]
    return attn_output, padded_q, padded_k


class SequentialTransductionUnitJagged(torch.nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        linear_hidden_dim: int,
        attention_dim: int,
        dropout_ratio: float,
        attn_dropout_ratio: float,
        num_heads: int,
        linear_activation: str,
        relative_attention_bias_module: Optional[RelativeAttentionBiasModule] = None,
        normalization: str = "rel_bias",
        linear_config: str = "uvqk",
        concat_ua: bool = False,
        epsilon: float = 1e-6,
        max_length: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._embedding_dim: int = embedding_dim
        self._linear_dim: int = linear_hidden_dim
        self._attention_dim: int = attention_dim
        self._dropout_ratio: float = dropout_ratio
        self._attn_dropout_ratio: float = attn_dropout_ratio
        self._num_heads: int = num_heads
        self._rel_attn_bias: Optional[RelativeAttentionBiasModule] = (
            relative_attention_bias_module
        )
        self._normalization: str = normalization
        self._linear_config: str = linear_config
        if self._linear_config == "uvqk":
            self._uvqk: torch.nn.Parameter = torch.nn.Parameter(
                torch.empty(
                    (
                        embedding_dim,
                        linear_hidden_dim * 2 * num_heads
                        + attention_dim * num_heads * 2,
                    )
                ).normal_(mean=0, std=0.02),
            )
        else:
            raise ValueError(f"Unknown linear_config {self._linear_config}")
        self._linear_activation: str = linear_activation
        self._concat_ua: bool = concat_ua
        self._o = torch.nn.Linear(
            in_features=linear_hidden_dim * num_heads * (3 if concat_ua else 1),
            out_features=embedding_dim,
        )
        torch.nn.init.xavier_uniform_(self._o.weight)
        self._eps: float = epsilon

    def _norm_input(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, normalized_shape=[self._embedding_dim], eps=self._eps)

    def _norm_attn_output(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x, normalized_shape=[self._linear_dim * self._num_heads], eps=self._eps
        )

    def forward(  # pyre-ignore [3]
        self,
        x: torch.Tensor,
        x_offsets: torch.Tensor,
        all_timestamps: Optional[torch.Tensor],
        invalid_attn_mask: torch.Tensor,
        delta_x_offsets: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache: Optional[HSTUCacheState] = None,
        return_cache_states: bool = False,
    ):
        """
        Args:
            x: (\sum_i N_i, D) x float.
            x_offsets: (B + 1) x int32.
            all_timestamps: optional (B, N) x int64.
            invalid_attn_mask: (B, N, N) x float, each element in {0, 1}.
            delta_x_offsets: optional 2-tuple ((B,) x int32, (B,) x int32).
                For the 1st element in the tuple, each element is in [0, x_offsets[-1]). For the
                2nd element in the tuple, each element is in [0, N).
            cache: Optional 4-tuple of (v, padded_q, padded_k, output) from prior runs,
                where all except padded_q, padded_k are jagged.
        Returns:
            x' = f(x), (\sum_i N_i, D) x float.
        """
        n: int = invalid_attn_mask.size(-1)
        cached_q = None
        cached_k = None
        if delta_x_offsets is not None:
            # In this case, for all the following code, x, u, v, q, k become restricted to
            # [delta_x_offsets[0], :].
            assert cache is not None
            x = x[delta_x_offsets[0], :]
            cached_v, cached_q, cached_k, cached_outputs = cache

        normed_x = self._norm_input(x)

        if self._linear_config == "uvqk":
            batched_mm_output = torch.mm(normed_x, self._uvqk)
            if self._linear_activation == "silu":
                batched_mm_output = F.silu(batched_mm_output)
            elif self._linear_activation == "none":
                batched_mm_output = batched_mm_output
            u, v, q, k = torch.split(
                batched_mm_output,
                [
                    self._linear_dim * self._num_heads,
                    self._linear_dim * self._num_heads,
                    self._attention_dim * self._num_heads,
                    self._attention_dim * self._num_heads,
                ],
                dim=1,
            )
        else:
            raise ValueError(f"Unknown self._linear_config {self._linear_config}")

        if delta_x_offsets is not None:
            v = cached_v.index_copy_(dim=0, index=delta_x_offsets[0], source=v)

        B: int = x_offsets.size(0) - 1
        if self._normalization == "rel_bias" or self._normalization == "hstu_rel_bias":
            assert self._rel_attn_bias is not None
            attn_output, padded_q, padded_k = _hstu_attention_maybe_from_cache(
                num_heads=self._num_heads,
                attention_dim=self._attention_dim,
                linear_dim=self._linear_dim,
                q=q,
                k=k,
                v=v,
                cached_q=cached_q,
                cached_k=cached_k,
                delta_x_offsets=delta_x_offsets,
                x_offsets=x_offsets,
                all_timestamps=all_timestamps,
                invalid_attn_mask=invalid_attn_mask,
                rel_attn_bias=self._rel_attn_bias,
            )
        elif self._normalization == "softmax_rel_bias":
            if delta_x_offsets is not None:
                B = x_offsets.size(0) - 1
                padded_q, padded_k = cached_q, cached_k
                flattened_offsets = delta_x_offsets[1] + torch.arange(
                    start=0,
                    end=B * n,
                    step=n,
                    device=delta_x_offsets[1].device,
                    dtype=delta_x_offsets[1].dtype,
                )
                assert padded_q is not None
                assert padded_k is not None
                padded_q = (
                    padded_q.view(B * n, -1)
                    .index_copy_(
                        dim=0,
                        index=flattened_offsets,
                        source=q,
                    )
                    .view(B, n, -1)
                )
                padded_k = (
                    padded_k.view(B * n, -1)
                    .index_copy_(
                        dim=0,
                        index=flattened_offsets,
                        source=k,
                    )
                    .view(B, n, -1)
                )
            else:
                padded_q = torch.ops.fbgemm.jagged_to_padded_dense(
                    values=q, offsets=[x_offsets], max_lengths=[n], padding_value=0.0
                )
                padded_k = torch.ops.fbgemm.jagged_to_padded_dense(
                    values=k, offsets=[x_offsets], max_lengths=[n], padding_value=0.0
                )

            qk_attn = torch.einsum("bnd,bmd->bnm", padded_q, padded_k)
            if self._rel_attn_bias is not None:
                qk_attn = qk_attn + self._rel_attn_bias(all_timestamps)
            qk_attn = F.softmax(qk_attn / math.sqrt(self._attention_dim), dim=-1)
            qk_attn = qk_attn * invalid_attn_mask
            attn_output = torch.ops.fbgemm.dense_to_jagged(
                torch.bmm(
                    qk_attn,
                    torch.ops.fbgemm.jagged_to_padded_dense(v, [x_offsets], [n]),
                ),
                [x_offsets],
            )[0]
        else:
            raise ValueError(f"Unknown normalization method {self._normalization}")

        attn_output = (
            attn_output
            if delta_x_offsets is None
            else attn_output[delta_x_offsets[0], :]
        )
        if self._concat_ua:
            a = self._norm_attn_output(attn_output)
            o_input = torch.cat([u, a, u * a], dim=-1)
        else:
            o_input = u * self._norm_attn_output(attn_output)

        new_outputs = (
            self._o(
                F.dropout(
                    o_input,
                    p=self._dropout_ratio,
                    training=self.training,
                )
            )
            + x
        )

        if delta_x_offsets is not None:
            new_outputs = cached_outputs.index_copy_(
                dim=0, index=delta_x_offsets[0], source=new_outputs
            )

        if return_cache_states and delta_x_offsets is None:
            v = v.contiguous()

        return new_outputs, (v, padded_q, padded_k, new_outputs)