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

"""OneRank structured tokenization (paper 2.1 / 2.2).

Every candidate is expanded into a group of ``2K`` tokens that alternates a
candidate replica with a task token::

    X   = [ prefix (contextual + UIH, length H) | G_1 | G_2 | ... | G_N ]
    G_i = [ c_i^(1), t_1, c_i^(2), t_2, ..., c_i^(K), t_K ]      |G_i| = 2K

The task tokens ``t_1..t_K`` are ``K`` learned vectors shared by every
candidate and every request; what makes ``t_k`` candidate-specific is the
mask, not the content.

Why replicas instead of the paper's ``[e^C_i, t_1, ..., t_K]``: HSTU's
arbitrary-mask path encodes exactly **two** column intervals per query row
(``NFUNC=3``, see ``build_sla_func_tensor``).  Under the paper layout a task
token needs ``[0, H)`` + ``{e^C_i}`` + ``{itself}`` -- three intervals,
because the mutually invisible ``t_1..t_{k-1}`` sit in between.  Pairing
each task token with its own adjacent candidate replica collapses the last
two into one interval, so the layout fits the existing kernels with no
kernel change.  Semantically it is equivalent: the replicas carry identical
content, so ``t_k`` still reads the same candidate the paper gives it.

The cost is ``N * 2K`` extra tokens per request, so ``N`` must stay in
the few-candidates-per-request regime a rerank stage works in (a few to
a few dozen); it does not scale to full-corpus candidate sets.
"""

import math
from typing import Any, Dict, Optional, Tuple

import torch
from torch.profiler import record_function

from tzrec.modules.gr.hstu_transducer import HSTUTransducer
from tzrec.modules.gr.stu import STU, STULayer
from tzrec.modules.utils import BaseModule
from tzrec.ops.hstu_attention_utils import STUTruncationPlan
from tzrec.ops.jagged_tensors import concat_2D_jagged, split_2D_jagged
from tzrec.utils.fx_util import fx_int_item

torch.fx.wrap(fx_int_item)


def _fx_check_max_num_candidates(
    max_new_targets: int, group_size: int, max_num_candidates: int
) -> None:
    """Raise when a request carries more candidates than configured.

    An fx-wrapped leaf: under ``torch.fx`` symbolic tracing the bounds are
    proxies whose comparison cannot be evaluated as a Python bool, so the
    check must be recorded as a graph node and run for real at execution
    time (``fx_int_item`` returns a proxy while tracing, an ``int``
    otherwise).
    """
    if max_num_candidates > 0 and max_new_targets > max_num_candidates * group_size:
        raise ValueError(
            f"a request carries {max_new_targets // group_size} "
            f"candidates, more than onerank.max_num_candidates "
            f"({max_num_candidates}); the inflated sequence would exceed "
            f"the static max sequence length every jagged kernel was "
            f"autotuned for. Raise onerank.max_num_candidates or filter "
            f"the input."
        )


torch.fx.wrap(_fx_check_max_num_candidates)


def build_onerank_func_tensor(
    nheads: int,
    seq_offsets: torch.Tensor,
    total_q: int,
    num_targets: torch.Tensor,
    group_size: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Build the NFUNC=3 func tensor for the OneRank group layout.

    Same encoding as :func:`build_sla_func_tensor`: shape
    ``(nheads, 3, total_q)`` int32, jagged along ``total_q``, where query
    row ``p`` attends to ``[0, col_max0) u [col_min0, col_max1)``.

    With ``H_b = L_b - T_b`` the prefix boundary and ``j = (p - H_b) mod
    group_size`` the slot inside a candidate group (local positions):

    ===================== ========== ========== ========== ==================
    query row             col_max0   col_min0   col_max1   visible columns
    ===================== ========== ========== ========== ==================
    prefix ``p < H_b``    ``p + 1``  ``p + 1``  ``p + 1``  ``[0, p+1)``
    candidate replica     ``H_b``    ``p``      ``p + 1``  ``[0,H_b) u {p}``
    task token            ``H_b``    ``p - 1``  ``p + 1``  ``[0,H_b) u
                                                           {p-1, p}``
    ===================== ========== ========== ========== ==================

    Consequences, all of them intended:

    - the prefix is plain causal. Note the ``DlrmHSTU`` baseline resolves
      an unset ``stu.contextual_seq_len`` sentinel to the number of
      contextual features (> 0) and applies a *bidirectional* contextual
      block; OneRank pins the value to 0 so contextual tokens are ordinary
      causal history, which is a deliberate (minor) deviation from the
      baseline's attention pattern;
    - candidate groups are mutually invisible, matching the data (candidate
      timestamps are flat -- there is no causal order between candidates);
    - within a group, ``t_k`` sees only its own replica, never ``t_j`` for
      ``j != k``, so a task token cannot leak another task's state.

    Args:
        nheads: number of attention heads.
        seq_offsets: cumulative sequence offsets ``(B+1,)`` of the
            **expanded** sequence.
        total_q: total jagged tokens in the batch (= ``seq_offsets[-1]``);
            taken from the caller's tensor metadata to avoid a D->H sync.
        num_targets: **expanded** per-sample target counts ``(B,)``, i.e.
            ``candidates * group_size``.
        group_size: tokens per candidate group, ``2 * num_tasks``.
        device: target device (inferred from ``seq_offsets`` if None).

    Returns:
        func tensor of shape ``(nheads, 3, total_q)``, dtype int32, as a
        strided view (stride 0 on the head dim).
    """
    if group_size <= 0 or group_size % 2 != 0:
        raise ValueError(
            f"group_size must be a positive even int (2 * num_tasks); got {group_size}"
        )
    if device is None:
        device = seq_offsets.device
    # The tensor-plumbing below deliberately mirrors build_sla_func_tensor:
    # unconditional int32 cast (no `Proxy.dtype` compare under fx), diff +
    # repeat_interleave instead of searchsorted on a slice, and no
    # `.contiguous()` on the returned view.  Each of those shapes is a
    # workaround for an Inductor / AOT-compile failure documented there.
    seq_offsets_i32 = seq_offsets.to(torch.int32)
    seq_lengths = torch.diff(seq_offsets_i32)  # (B,)
    B = seq_lengths.size(0)
    pos_global = torch.arange(total_q, device=device, dtype=torch.int32)
    seq_offsets_starts = seq_offsets_i32.narrow(0, 0, B).contiguous()
    pos_local = pos_global - torch.repeat_interleave(
        seq_offsets_starts, seq_lengths, output_size=total_q
    )
    L = torch.repeat_interleave(seq_lengths, seq_lengths, output_size=total_q)
    T = torch.repeat_interleave(
        num_targets.to(torch.int32), seq_lengths, output_size=total_q
    )
    # Clamp so a pathological num_targets[b] > seq_lengths[b] cannot produce
    # a negative boundary that collapses every row to an empty interval.
    prefix_boundary = torch.clamp(L - T, min=0)

    is_prefix = pos_local < prefix_boundary
    # Slot inside the candidate group: even -> candidate replica, odd ->
    # task token.  torch.remainder takes the divisor's sign, so with a
    # positive group_size the slot is non-negative even on prefix rows,
    # where the value is meaningless but those rows are selected away
    # below.
    slot = torch.remainder(pos_local - prefix_boundary, group_size)
    is_task_token = torch.remainder(slot, 2) == 1

    causal = pos_local + 1
    # A task token reaches one column further left than a replica: back to
    # the replica it is paired with.
    group_col_min0 = torch.where(is_task_token, pos_local - 1, pos_local)
    col_max0 = torch.where(is_prefix, causal, prefix_boundary)
    col_min0 = torch.where(is_prefix, causal, group_col_min0)
    # Prefix rows get an empty second interval (col_min0 == col_max1);
    # group rows get exactly their own group columns.
    col_max1 = causal

    func_2d = torch.stack([col_max0, col_min0, col_max1], dim=0)  # (3, total_q)
    return func_2d.unsqueeze(0).expand(nheads, 3, total_q)


class OneRankTokenizer(BaseModule):
    """Expands each candidate into its ``[replica, task token] * K`` group.

    Args:
        num_tasks (int): number of task tokens ``K``.
        embedding_dim (int): STU embedding dimension.
        is_inference (bool): whether to run in inference mode.
    """

    def __init__(
        self,
        num_tasks: int,
        embedding_dim: int,
        is_inference: bool = False,
    ) -> None:
        super().__init__(is_inference=is_inference)
        if num_tasks <= 0:
            raise ValueError(f"num_tasks must be positive; got {num_tasks}")
        self._num_tasks: int = num_tasks
        self._embedding_dim: int = embedding_dim
        self._task_tokens: torch.nn.Parameter = torch.nn.Parameter(
            torch.empty((num_tasks, embedding_dim)).normal_(
                0.0, 1.0 / math.sqrt(embedding_dim)
            )
        )

    @property
    def num_tasks(self) -> int:
        """Number of task tokens ``K``."""
        return self._num_tasks

    @property
    def group_size(self) -> int:
        """Tokens per candidate group, ``2K``."""
        return 2 * self._num_tasks

    def forward(self, target_embeddings: torch.Tensor) -> torch.Tensor:
        """Expand the candidate segment of a jagged sequence.

        Args:
            target_embeddings (torch.Tensor): candidate embeddings
                ``(total_targets, D)``, jagged and grouped by request.

        Returns:
            torch.Tensor: ``(total_targets * 2K, D)``, each candidate
                replaced in place by ``[c^(1), t_1, ..., c^(K), t_K]``.
        """
        dim = target_embeddings.size(-1)
        # expand (not repeat): the K replicas are read-only views, so the
        # backward pass sums their grads straight back into the candidate.
        replicas = target_embeddings.unsqueeze(1).expand(-1, self._num_tasks, -1)
        tokens = self._task_tokens.to(target_embeddings.dtype)
        tokens = tokens.unsqueeze(0).expand(target_embeddings.size(0), -1, -1)
        # (total_targets, K, 2, D) -> flatten preserves the alternating
        # replica / task-token order the mask assumes.
        return torch.stack([replicas, tokens], dim=2).reshape(-1, dim)


class OneRankSTULayer(STULayer):
    """``STULayer`` masked with the OneRank group layout instead of SLA.

    Args:
        group_size (int): tokens per candidate group, ``2 * num_tasks``.
        **kwargs: forwarded verbatim to :class:`STULayer`.
    """

    def __init__(self, group_size: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if group_size <= 0 or group_size % 2 != 0:
            raise ValueError(
                f"group_size must be a positive even int (2 * num_tasks); "
                f"got {group_size}"
            )
        self._group_size: int = group_size
        if self._sla_k1 > 0 or self._sla_k2 > 0:
            raise ValueError(
                f"OneRank builds its own NFUNC mask and cannot also apply SLA; "
                f"keep stu.sla_k1 / stu.sla_k2 at 0, got sla_k1={self._sla_k1}, "
                f"sla_k2={self._sla_k2}."
            )
        if self._contextual_seq_len != 0:
            raise ValueError(
                f"OneRank requires stu.contextual_seq_len == 0, got "
                f"{self._contextual_seq_len}. NFUNC=3 encodes two column "
                f"intervals per query row and the group layout already spends "
                f"both on `prefix + own group`; a bidirectional contextual "
                f"block would need a third. With 0 the contextual tokens are "
                f"ordinary causal prefix; note this differs from the "
                f"DlrmHSTU baseline, whose unset (-1) sentinel resolves to "
                f"the number of contextual features and gives contextual "
                f"tokens a bidirectional block."
            )
        if self._max_attn_len > 0:
            raise ValueError(
                f"OneRank does not support stu.max_attn_len (got "
                f"{self._max_attn_len}); a local window would have to be "
                f"folded into the func tensor's two intervals, which the "
                f"group layout already uses up."
            )
        if not self._causal:
            raise ValueError("OneRank requires stu.causal = true.")

    @property
    def uses_arbitrary_mask(self) -> bool:
        """OneRank always drives attention through its own func tensor."""
        return True

    @property
    def attn_func_static_sig(self) -> str:
        """OneRank NFUNC cache key.

        Prefixed so it can never collide with ``STULayer``'s SLA signature
        in a mixed stack.
        """
        return f"onerank:{self._group_size}:{self._num_heads}"

    def _build_attn_func(
        self,
        x: torch.Tensor,
        x_offsets: torch.Tensor,
        num_targets: torch.Tensor,
    ) -> torch.Tensor:
        """Build the OneRank group mask.

        ``num_targets`` is used unconditionally: unlike SLA, the group
        layout is meaningless without the prefix boundary, so
        ``target_aware`` cannot switch it off.
        """
        return build_onerank_func_tensor(
            nheads=self._num_heads,
            seq_offsets=x_offsets,
            total_q=x.size(0),
            num_targets=num_targets,
            group_size=self._group_size,
        )

    def truncate_input(
        self,
        x: torch.Tensor,
        x_offsets: torch.Tensor,
        max_seq_len: int,
        num_targets: Optional[torch.Tensor],
        *,
        truncate_tail_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, int, STUTruncationPlan]:
        """Unsupported: truncation would desynchronize the group layout."""
        raise NotImplementedError(
            "OneRank does not support mid-stack attention truncation: the "
            "func tensor is cached across layers keyed on a static signature, "
            "and truncation changes total_q and the prefix boundary. Leave "
            "attn_truncation_split_layer / attn_truncation_tail_len at 0."
        )

    def cached_forward(
        self,
        delta_x: torch.Tensor,
        num_targets: torch.Tensor,
        max_kv_caching_len: int = 0,
        kv_caching_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Unsupported: the incremental path cannot carry a func tensor."""
        raise NotImplementedError(
            "OneRank has no incremental (KV-cache) serving path: "
            "delta_hstu_mha takes no attn_func and downgrades Kernel.CUTLASS "
            "to Kernel.TRITON, so the task-token mask would be silently "
            "dropped. Serve with the full forward pass."
        )


class OneRankHSTUTransducer(HSTUTransducer):
    """``HSTUTransducer`` that tokenizes candidates into OneRank groups.

    The expansion is spliced in between the positional encoder and the STU
    stack by overriding :meth:`_preprocess`: the base implementation runs
    the input preprocessor plus positional encoding, and only then are the
    candidates replaced by their groups.  Task tokens therefore carry no
    positional / time encoding (they are pure learned queries) while every
    candidate replica keeps the encoding of the candidate it copies.

    Output is ``(total_targets, K, D)`` -- the per-candidate per-task
    representation ``r^i_k`` of the paper -- instead of the baseline's
    ``(total_targets, D)``.

    Args:
        num_task_tokens (int): number of task tokens ``K``; must equal the
            number of task towers.
        max_num_candidates (int): upper bound on candidates per request
            (``onerank.max_num_candidates``); requests that exceed it fail
            with a clear error instead of silently overflowing the static
            max sequence length the jagged kernels were autotuned for.
            0 disables the runtime check (unit-test convenience).
        **kwargs: forwarded verbatim to :class:`HSTUTransducer`.
    """

    def __init__(
        self,
        num_task_tokens: int,
        max_num_candidates: int = 0,
        **kwargs: Any,
    ) -> None:
        if num_task_tokens <= 0:
            raise ValueError(f"num_task_tokens must be positive; got {num_task_tokens}")
        if max_num_candidates < 0:
            raise ValueError(
                f"max_num_candidates must be >= 0; got {max_num_candidates}"
            )
        self._max_num_candidates: int = max_num_candidates
        stu = dict(kwargs.pop("stu"))
        # Routed through the stu dict because `_build_stu_layer` runs inside
        # `super().__init__()`, before subclass attributes exist.
        stu["group_size"] = 2 * num_task_tokens
        # The NFUNC=3 group layout spends both func-tensor column intervals
        # on `prefix + own group`, so contextual tokens must be ordinary
        # causal prefix. Resolve the `-1` sentinel ("inherit from the
        # preprocessor", which the base class would resolve to the number
        # of contextual features, >= 1) to 0 here, so a `dlrm_hstu` config
        # block can be reused verbatim; an explicit non-zero value is left
        # for the `OneRankSTULayer` check to reject with a clear message.
        if stu.get("contextual_seq_len", -1) < 0:
            stu["contextual_seq_len"] = 0
        # Rejected at construction rather than at the first training step:
        # a reused `dlrm_hstu` block may carry truncation tuning, and the
        # stack would otherwise only surface `OneRankSTULayer`'s
        # NotImplementedError after torchrun, sharding and the data
        # pipeline are fully up.
        if (
            kwargs.get("attn_truncation_split_layer", 0) != 0
            or kwargs.get("attn_truncation_tail_len", 0) != 0
        ):
            raise ValueError(
                "OneRank does not support mid-stack attention truncation: "
                "the func tensor is cached across layers keyed on a static "
                "signature, and truncation changes total_q and the prefix "
                "boundary. Leave attn_truncation_split_layer / "
                "attn_truncation_tail_len at 0."
            )
        super().__init__(stu=stu, **kwargs)
        if self._return_full_embeddings:
            raise ValueError(
                "OneRank does not return full sequence embeddings; the "
                "expanded sequence is an internal layout."
            )
        # Interleaving doubles the candidate segment before this module
        # sees it, which would break the fixed group stride. Checked via
        # the public construction-time accessor: `interleave_targets()` is
        # train/eval dependent and would read False at construction.
        if self._input_preprocessor.has_interleaving():
            raise ValueError(
                "OneRank requires an input preprocessor without target "
                "interleaving (use `contextual_preprocessor`, not "
                "`contextual_interleave_preprocessor`)."
            )
        self._tokenizer: OneRankTokenizer = OneRankTokenizer(
            num_tasks=num_task_tokens,
            embedding_dim=stu["embedding_dim"],
            is_inference=self._is_inference,
        )

    def _build_stu_layer(self, stu: Dict[str, Any]) -> STU:
        return OneRankSTULayer(**stu)

    def _preprocess(
        self, grouped_features: Dict[str, torch.Tensor]
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
        (
            max_seq_len,
            total_uih_len,
            total_targets,
            seq_lengths,
            seq_offsets,
            seq_timestamps,
            seq_embeddings,
            num_targets,
        ) = super()._preprocess(grouped_features)

        group_size = self._tokenizer.group_size
        with record_function("onerank_tokenizer"):
            prefix_lengths = seq_lengths - num_targets
            prefix_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
                prefix_lengths
            )
            target_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(num_targets)
            prefix_embeddings, target_embeddings = split_2D_jagged(
                values=seq_embeddings,
                max_seq_len=max_seq_len,
                total_len_left=total_uih_len,
                total_len_right=total_targets,
                offsets_left=prefix_offsets,
                offsets_right=target_offsets,
                kernel=self.kernel(),
            )
            prefix_timestamps, target_timestamps = split_2D_jagged(
                values=seq_timestamps.unsqueeze(-1),
                max_seq_len=max_seq_len,
                total_len_left=total_uih_len,
                total_len_right=total_targets,
                offsets_left=prefix_offsets,
                offsets_right=target_offsets,
                kernel=self.kernel(),
            )

            expanded_embeddings = self._tokenizer(target_embeddings)
            # Every token of a group inherits the candidate's timestamp, so
            # the output postprocessor's time features are unchanged.
            expanded_timestamps = target_timestamps.repeat_interleave(group_size, dim=0)
            new_num_targets = num_targets * group_size
            new_target_offsets = target_offsets * group_size

            # Two D->H syncs for the padding bounds of the jagged concat.
            # `_preprocess` is already sync-bound (the input preprocessor
            # does four), and passing a loose bound would pad the dense
            # intermediate to the full expanded length for every sample.
            max_prefix_len = fx_int_item(prefix_lengths.max())
            max_new_targets = fx_int_item(new_num_targets.max())
            _fx_check_max_num_candidates(
                max_new_targets, group_size, self._max_num_candidates
            )

            new_seq_embeddings = concat_2D_jagged(
                values_left=prefix_embeddings,
                values_right=expanded_embeddings,
                max_len_left=max_prefix_len,
                max_len_right=max_new_targets,
                offsets_left=prefix_offsets,
                offsets_right=new_target_offsets,
                kernel=self.kernel(),
            )
            new_seq_timestamps = concat_2D_jagged(
                values_left=prefix_timestamps,
                values_right=expanded_timestamps,
                max_len_left=max_prefix_len,
                max_len_right=max_new_targets,
                offsets_left=prefix_offsets,
                offsets_right=new_target_offsets,
                kernel=self.kernel(),
            ).squeeze(-1)
            new_seq_lengths = prefix_lengths + new_num_targets
            new_seq_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
                new_seq_lengths
            )

        return (
            max_prefix_len + max_new_targets,
            total_uih_len,
            total_targets * group_size,
            new_seq_lengths,
            new_seq_offsets,
            new_seq_timestamps,
            new_seq_embeddings,
            new_num_targets,
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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Split off the expanded candidate segment and keep the task slots."""
        group_embeddings, full = super()._compose_output(
            encoded_embeddings=encoded_embeddings,
            seq_timestamps=seq_timestamps,
            seq_lengths=seq_lengths,
            seq_offsets=seq_offsets,
            max_seq_len=max_seq_len,
            total_uih_len=total_uih_len,
            total_targets=total_targets,
            num_targets=num_targets,
        )
        # (total_targets * 2K, D) -> (total_targets, K, D): slot 1 of each
        # (replica, task token) pair is the task token's output, r^i_k.
        dim = group_embeddings.size(-1)
        return (
            group_embeddings.view(-1, self._tokenizer.num_tasks, 2, dim)[:, :, 1, :],
            full,
        )
