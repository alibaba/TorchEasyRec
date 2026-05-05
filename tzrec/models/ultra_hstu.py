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


from typing import Any, Dict, List, Optional, Tuple

import torch

from tzrec.features.feature import BaseFeature
from tzrec.models.dlrm_hstu import DlrmHSTU
from tzrec.models.rank_model import RankModel
from tzrec.modules.gr.hstu_transducer import HSTUTransducer
from tzrec.protos.model_pb2 import ModelConfig
from tzrec.protos.models import multi_task_rank_pb2
from tzrec.utils.config_util import config_to_kwargs


class _HSTUTransducerStack(torch.nn.Module):
    """N parallel HSTUTransducers; concat per-candidate outputs on dim=-1.

    Forward signature mirrors HSTUTransducer so DlrmHSTU.predict()
    treats this stack identically to a single transducer.
    """

    def __init__(self, transducers: List[HSTUTransducer]) -> None:
        super().__init__()
        self._transducers: torch.nn.ModuleList = torch.nn.ModuleList(transducers)

    def forward(
        self, grouped_features: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        cand_list: List[torch.Tensor] = []
        for transducer in self._transducers:
            cand, _ = transducer(grouped_features)
            cand_list.append(cand)
        return torch.cat(cand_list, dim=-1), None


class UltraHSTU(DlrmHSTU):
    """ULTRA-HSTU model with Mixture of Transducers.

    Builds N parallel HSTUTransducer stacks (one per channel in
    ``model_config.ultra_hstu.hstu``) and concatenates their
    per-candidate outputs along the embedding dim.  The ``candidate``
    and contextual groups are shared; each channel's UIH-side groups
    are named after the channel (``<name>``, ``<name>_action``,
    ``<name>_watchtime``, ``<name>_timestamp``).  Channels with the
    same ``embedding_name`` on a feature share the underlying table
    via ``EmbeddingGroup`` dedupe.

    Args:
        model_config (ModelConfig): an instance of ModelConfig.
        features (list): list of features.
        labels (list): list of label names.
        sample_weights (list): sample weight names.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        features: List[BaseFeature],
        labels: List[str],
        sample_weights: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        # Call grandparent RankModel.__init__ directly to skip DlrmHSTU's
        # dlrm_hstu-specific model-type assertion.
        RankModel.__init__(
            self, model_config, features, labels, sample_weights, **kwargs
        )
        assert model_config.WhichOneof("model") == "ultra_hstu", (
            "invalid model config: %s" % self._model_config.WhichOneof("model")
        )
        assert isinstance(self._model_config, multi_task_rank_pb2.UltraHSTU)
        channels = self._model_config.hstu
        assert len(channels) >= 1, "UltraHSTU requires at least 1 hstu channel"
        if len(channels) >= 2:
            names = [c.name for c in channels]
            assert all(names) and len(set(names)) == len(names), (
                "When UltraHSTU has >= 2 channels every channel must set a "
                f"unique non-empty `name`, got {names!r}"
            )
        self._init()

    def _build_transducer(
        self, contextual_feature_dim: int, max_contextual_seq_len: int
    ) -> torch.nn.Module:
        transducers: List[HSTUTransducer] = []
        for ch in self._model_config.hstu:
            uih_group = ch.name if ch.name else "uih"
            transducers.append(
                HSTUTransducer(
                    uih_embedding_dim=self.embedding_group.group_total_dim(uih_group),
                    target_embedding_dim=self.embedding_group.group_total_dim(
                        "candidate"
                    ),
                    contextual_feature_dim=contextual_feature_dim,
                    max_contextual_seq_len=max_contextual_seq_len,
                    contextual_group_name=self._contextual_group_name,
                    scaling_seqlen=self._model_config.max_seq_len,
                    **config_to_kwargs(ch),
                    return_full_embeddings=False,
                )
            )
        if len(transducers) == 1:
            return transducers[0]
        return _HSTUTransducerStack(transducers)

    def _stu_embedding_dim(self) -> int:
        return sum(c.stu.embedding_dim for c in self._model_config.hstu)
