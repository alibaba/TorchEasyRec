# Copyright (c) 2026, Alibaba Group;
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

Every candidate is expanded into a group of ``K + 1`` tokens -- the
candidate token followed by the task tokens::

    X   = [ prefix (contextual + UIH, length H) | G_1 | G_2 | ... | G_N ]
    G_i = [ e^C_i, t_1, t_2, ..., t_K ]                        |G_i| = K + 1

The task tokens ``t_1..t_K`` are ``K`` learned vectors shared by every
candidate and every request; what makes ``t_k`` candidate-specific is the
mask, not the content.

The mask gives ``t_k`` the visible set ``[0, H) u {e^C_i} u {t_k}`` --
three column intervals, because the mutually invisible ``t_1..t_{k-1}``
sit in between.  HSTU's arbitrary-mask path compiles with
``HSTU_ARBITRARY_NFUNC = 5``, exposing exactly ``(NFUNC + 1) // 2 = 3``
intervals per query row (see :func:`build_onerank_func_tensor`), so the
paper layout encodes directly with no candidate replicas.

The cost is ``N * (K + 1)`` tokens per request, so ``N`` must stay in
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
from tzrec.ops.hstu_attention_utils import (
    STUTruncationPlan,
    build_onerank_func_tensor,
)
from tzrec.ops.jagged_tensors import concat_2D_jagged, split_2D_jagged
from tzrec.utils.fx_util import fx_int_item

torch.fx.wrap(fx_int_item)


def _fx_check_max_seq_len(
    max_prefix_len: int, max_new_targets: int, max_seq_len: int
) -> None:
    """Raise when a request's inflated sequence exceeds ``max_seq_len``.

    An fx-wrapped leaf: under ``torch.fx`` symbolic tracing the bounds are
    proxies whose comparison cannot be evaluated as a Python bool, so the
    check must be recorded as a graph node and run for real at execution
    time (``fx_int_item`` returns a proxy while tracing, an ``int``
    otherwise).
    """
    if max_prefix_len + max_new_targets > max_seq_len:
        raise ValueError(
            f"a request's inflated sequence reaches "
            f"{max_prefix_len + max_new_targets} tokens "
            f"({max_prefix_len} history + {max_new_targets} expanded "
            f"candidate tokens), more than max_seq_len ({max_seq_len}); "
            f"every jagged kernel was autotuned for at most max_seq_len "
            f"tokens. Raise max_seq_len or filter the input."
        )


torch.fx.wrap(_fx_check_max_seq_len)


class OneRankTokenizer(BaseModule):
    """Expands each candidate into its ``[candidate, t_1..t_K]`` group.

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
        """Tokens per candidate group, ``K + 1``."""
        return self._num_tasks + 1

    def forward(self, target_embeddings: torch.Tensor) -> torch.Tensor:
        """Expand the candidate segment of a jagged sequence.

        Args:
            target_embeddings (torch.Tensor): candidate embeddings
                ``(total_targets, D)``, jagged and grouped by request.

        Returns:
            torch.Tensor: ``(total_targets * (K + 1), D)``, each candidate
                replaced in place by ``[e^C_i, t_1, ..., t_K]``.
        """
        dim = target_embeddings.size(-1)
        tokens = self._task_tokens.to(target_embeddings.dtype)
        tokens = tokens.unsqueeze(0).expand(target_embeddings.size(0), -1, -1)
        # (total_targets, K + 1, D) -> flatten preserves the
        # candidate-then-task-token order the mask assumes.
        return torch.cat([target_embeddings.unsqueeze(1), tokens], dim=1).reshape(
            -1, dim
        )


class OneRankSTULayer(STULayer):
    """``STULayer`` masked with the OneRank group layout instead of SLA.

    Args:
        group_size (int): tokens per candidate group, ``num_tasks + 1``.
        **kwargs: forwarded verbatim to :class:`STULayer`.
    """

    def __init__(self, group_size: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if group_size < 2:
            raise ValueError(
                f"group_size must be at least 2 (candidate token + one task "
                f"token, i.e. num_tasks + 1); got {group_size}"
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
                f"{self._contextual_seq_len}. The NFUNC mask encodes "
                f"(HSTU_ARBITRARY_NFUNC + 1) // 2 column intervals per query "
                f"row and the group layout already spends all of them on "
                f"`prefix + own group`; a bidirectional contextual block "
                f"would need another. With 0 the contextual tokens are "
                f"ordinary causal prefix; note this differs from the "
                f"DlrmHSTU baseline, whose unset (-1) sentinel resolves to "
                f"the number of contextual features and gives contextual "
                f"tokens a bidirectional block."
            )
        if self._max_attn_len > 0:
            raise ValueError(
                f"OneRank does not support stu.max_attn_len (got "
                f"{self._max_attn_len}); a local window would have to be "
                f"folded into the func tensor's intervals, which the group "
                f"layout already uses up."
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
    positional / time encoding (they are pure learned queries) while the
    candidate token keeps the encoding of the candidate it heads.

    Output is ``(total_targets, K, D)`` -- the per-candidate per-task
    representation ``r^i_k`` of the paper -- instead of the baseline's
    ``(total_targets, D)``.

    Args:
        num_task_tokens (int): number of task tokens ``K``; must equal the
            number of task towers.
        max_seq_len (int): upper bound on the inflated sequence length
            per request (history tokens + expanded candidate groups);
            requests that exceed it fail with a clear error instead of
            silently overflowing the static max sequence length the
            jagged kernels were autotuned for.
        **kwargs: forwarded verbatim to :class:`HSTUTransducer`.
    """

    def __init__(
        self,
        num_task_tokens: int,
        max_seq_len: int,
        **kwargs: Any,
    ) -> None:
        if num_task_tokens <= 0:
            raise ValueError(f"num_task_tokens must be positive; got {num_task_tokens}")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive; got {max_seq_len}")
        self._max_seq_len: int = max_seq_len
        stu = dict(kwargs.pop("stu"))
        # Routed through the stu dict because `_build_stu_layer` runs inside
        # `super().__init__()`, before subclass attributes exist.
        stu["group_size"] = num_task_tokens + 1
        # The paper layout spends all three func-tensor column intervals
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
        # Interleaving reorders the candidate segment before this module
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
            _fx_check_max_seq_len(max_prefix_len, max_new_targets, self._max_seq_len)

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
        # (total_targets * (K + 1), D) -> (total_targets, K, D): slots
        # 1..K of each group carry the task tokens' outputs, r^i_k.
        dim = group_embeddings.size(-1)
        return (
            group_embeddings.view(-1, self._tokenizer.group_size, dim)[:, 1:, :],
            full,
        )
