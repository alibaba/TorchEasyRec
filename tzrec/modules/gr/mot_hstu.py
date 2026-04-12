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

"""Mixture of Transducers from ULTRA-HSTU.

Multiple HSTUTransducer instances process disjoint feature channels
(e.g. consumption, engagement) and their outputs are fused into a
single candidate embedding.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch

from tzrec.modules.gr.hstu_transducer import HSTUTransducer
from tzrec.modules.utils import BaseModule


class MoTHSTUTransducer(BaseModule):
    """Mixture of Transducers: parallel HSTU stacks with output fusion.

    Each sub-transducer processes a disjoint feature channel and produces
    candidate embeddings. The embeddings are fused via element-wise sum
    or a learned concat-MLP.

    Args:
        transducer_configs: list of kwargs dicts for HSTUTransducer.
        fusion: fusion strategy, one of ``"sum"`` or ``"concat_mlp"``.
        embedding_dim: output embedding dimension (must match all sub-
            transducers' STU embedding_dim for ``"sum"``; for
            ``"concat_mlp"`` the MLP projects the concatenation down
            to this dimension).
        is_inference: whether to run in inference mode.
    """

    def __init__(
        self,
        transducer_configs: List[Dict[str, Any]],
        fusion: str = "sum",
        embedding_dim: int = 0,
        is_inference: bool = False,
    ) -> None:
        super().__init__(is_inference=is_inference)
        assert len(transducer_configs) >= 1, (
            "Mixture of Transducers requires at least 1 transducer"
        )
        self._transducers: torch.nn.ModuleList = torch.nn.ModuleList(
            [HSTUTransducer(**cfg) for cfg in transducer_configs]
        )
        self._fusion = fusion
        self._embedding_dim = embedding_dim

        if fusion == "concat_mlp":
            n = len(transducer_configs)
            assert embedding_dim > 0, "embedding_dim required for concat_mlp"
            self._fusion_mlp = torch.nn.Sequential(
                torch.nn.Linear(embedding_dim * n, embedding_dim),
                torch.nn.GELU(),
                torch.nn.Linear(embedding_dim, embedding_dim),
            )
        elif fusion != "sum":
            raise ValueError(f"Unknown fusion strategy: {fusion}")

    def forward(
        self,
        grouped_features_list: List[Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward the module.

        Args:
            grouped_features_list: one feature dict per sub-transducer,
                in the same order as ``transducer_configs``.

        Returns:
            fused_candidate_embeddings: output embedding of candidates.
            fused_full_embeddings: full output embeddings (None when any
                sub-transducer does not return them).
        """
        candidate_list: List[torch.Tensor] = []
        full_list: List[Optional[torch.Tensor]] = []
        for transducer, gf in zip(self._transducers, grouped_features_list):
            cand, full = transducer(gf)
            candidate_list.append(cand)
            full_list.append(full)

        if self._fusion == "sum":
            fused_cand = candidate_list[0]
            for c in candidate_list[1:]:
                fused_cand = fused_cand + c
        else:
            fused_cand = self._fusion_mlp(torch.cat(candidate_list, dim=-1))

        # Full embeddings: sum if all present, else None.
        if all(f is not None for f in full_list):
            if self._fusion == "sum":
                fused_full: Optional[torch.Tensor] = full_list[0]
                for f in full_list[1:]:
                    fused_full = fused_full + f  # type: ignore[operator]
            else:
                fused_full = self._fusion_mlp(
                    torch.cat([f for f in full_list if f is not None], dim=-1)
                )
        else:
            fused_full = None

        return fused_cand, fused_full
