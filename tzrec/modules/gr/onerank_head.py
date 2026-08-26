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

"""OneRank prediction head (paper 2.3 - 2.4).

Consumes the per-candidate per-task representations ``r^i_k`` produced by
:class:`~tzrec.modules.gr.onerank_tokenizer.OneRankHSTUTransducer` and emits
one logit per ``(candidate, task)`` pair::

    z_k = CrossTask({ SD_k(s, {r^i_k}) })          request level, per task
    s^i_k = z_k . r^i_k / sqrt(D) + b_k            candidate level

Both stages are optional (paper ablations V5 and V3).  With neither, ``z_k``
degenerates to the mean of ``{r^i_k}`` over the request's candidates, which
still discriminates candidates through ``r^i_k``.

The inner-product scorer replaces ``FusionMTLTower``: the baseline funnels
every task through one shared MLP and splits only in the final linear
layer, so the tasks share nearly all capacity.  Here each task owns its
whole channel ``(z_k, r^i_k)`` and the only shared parameters are the STU
trunk.
"""

import math
from typing import Any, Dict, Optional, Sequence

import torch

from tzrec.modules.gr.onerank_cross_task import OneRankCrossTaskAttention
from tzrec.modules.gr.onerank_jagged import jagged_mean
from tzrec.modules.gr.onerank_sd import OneRankSituationDiscernment
from tzrec.modules.utils import BaseModule


class OneRankPredictionHead(BaseModule):
    """Turns ``r^i_k`` into one logit per ``(candidate, task)`` pair.

    Args:
        embedding_dim (int): STU embedding dim ``D``.
        task_names (Sequence[str]): task names in ``task_configs`` order;
            its length is the number of tasks ``K`` and must match the
            number of task tokens the transducer was built with, and its
            order is the cross-task cascade order.
        contextual_feature_dim (int): width of the flattened contextual
            token; only needed when ``situation_discernment`` is given.
        situation_discernment (Dict, optional): kwargs of
            :class:`~tzrec.modules.gr.onerank_sd.OneRankSituationDiscernment`;
            ``None`` falls back to mean pooling (ablation V5).
        cross_task_head (Dict, optional): kwargs of
            :class:`~tzrec.modules.gr.onerank_cross_task.OneRankCrossTaskAttention`;
            ``None`` scores the per-task vectors directly (ablation V3).
        is_inference (bool): whether to run in inference mode.
    """

    def __init__(
        self,
        embedding_dim: int,
        task_names: Sequence[str],
        contextual_feature_dim: int = 0,
        situation_discernment: Optional[Dict[str, Any]] = None,
        cross_task_head: Optional[Dict[str, Any]] = None,
        is_inference: bool = False,
    ) -> None:
        super().__init__(is_inference=is_inference)
        num_tasks = len(task_names)
        if num_tasks <= 0:
            raise ValueError("task_names must not be empty")
        self._embedding_dim: int = embedding_dim
        self._num_tasks: int = num_tasks
        # The STU output is layer-normed, so a raw dot product of two
        # D-dim vectors lands around +-D and saturates BCE from step 0.
        self._logit_scale: float = 1.0 / math.sqrt(embedding_dim)
        self._task_bias: torch.nn.Parameter = torch.nn.Parameter(torch.zeros(num_tasks))
        self._situation_discernment: Optional[OneRankSituationDiscernment] = None
        if situation_discernment is not None:
            self._situation_discernment = OneRankSituationDiscernment(
                embedding_dim=embedding_dim,
                num_tasks=num_tasks,
                contextual_feature_dim=contextual_feature_dim,
                is_inference=is_inference,
                **situation_discernment,
            )
        self._cross_task: Optional[OneRankCrossTaskAttention] = None
        if cross_task_head is not None:
            self._cross_task = OneRankCrossTaskAttention(
                embedding_dim=embedding_dim,
                task_names=task_names,
                is_inference=is_inference,
                **cross_task_head,
            )

    @property
    def num_tasks(self) -> int:
        """Number of tasks ``K``."""
        return self._num_tasks

    def _request_vectors(
        self,
        task_embeddings: torch.Tensor,
        num_candidates: torch.Tensor,
        contextual_embeddings: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Aggregate ``{r^i_k}`` into one vector per ``(request, task)``.

        Mean pooling is the no-Situation-Discernment fallback (ablation
        V5); it is order-free, which matches the data -- candidate
        timestamps are flat, so the candidates of a request are a set.
        Cross-task attention, when enabled, then mixes the ``K`` vectors
        under the task-order mask.
        """
        sd = self._situation_discernment
        if sd is None:
            pooled = jagged_mean(task_embeddings, num_candidates)
        elif contextual_embeddings is None:
            raise ValueError(
                "situation_discernment is configured but no contextual "
                "embeddings were passed to the head."
            )
        else:
            pooled = sd(
                contextual_embeddings=contextual_embeddings,
                task_embeddings=task_embeddings,
                num_candidates=num_candidates,
            )
        cross_task = self._cross_task
        if cross_task is not None:
            pooled = cross_task(pooled)
        return pooled

    def forward(
        self,
        task_embeddings: torch.Tensor,
        num_candidates: torch.Tensor,
        contextual_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Score every ``(candidate, task)`` pair.

        Args:
            task_embeddings (torch.Tensor): ``(total_candidates, K, D)``
                per-candidate per-task representations ``r^i_k``, jagged
                over requests.
            num_candidates (torch.Tensor): ``(B,)`` candidates per request,
                summing to ``total_candidates``.
            contextual_embeddings (torch.Tensor, optional): ``(B, *)``
                contextual token of each request; required only when
                Situation Discernment is enabled.

        Returns:
            torch.Tensor: ``(total_candidates, K)`` logits.
        """
        request_vectors = self._request_vectors(
            task_embeddings, num_candidates, contextual_embeddings
        )
        # Back to candidate granularity; repeat_interleave with a tensor of
        # repeats keeps this a single fused op instead of gather-by-index.
        broadcast = torch.repeat_interleave(request_vectors, num_candidates, dim=0)
        scores = (broadcast * task_embeddings).sum(dim=-1) * self._logit_scale
        return scores + self._task_bias.to(scores.dtype)
