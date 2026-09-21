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


from typing import Any, Dict, List, Optional

import torch
from torch.autograd.profiler import record_function

from tzrec.features.feature import BaseFeature
from tzrec.models.dlrm_hstu import DlrmHSTU
from tzrec.models.rank_model import RankModel
from tzrec.modules.gr.onerank_tokenizer import OneRankHSTUTransducer
from tzrec.modules.task_tower import OneRankPredictionHead
from tzrec.ops.utils import set_static_max_seq_lens
from tzrec.protos.model_pb2 import ModelConfig
from tzrec.protos.models import multi_task_rank_pb2
from tzrec.utils.config_util import config_to_kwargs


class DlrmHSTUOneRank(DlrmHSTU):
    """OneRank-style multi-task DLRM HSTU model.

    Differs from :class:`DlrmHSTU` in three places:

    1. every candidate is expanded into a group of ``K + 1`` tokens (the
       candidate token followed by the task tokens, paper 2.1 / 2.2) and
       attention runs under a task-private mask, so the trunk emits one
       representation ``r^i_k`` per ``(candidate, task)`` pair rather than
       a single per-candidate vector (see
       :mod:`tzrec.modules.gr.onerank_tokenizer`);
    2. scoring is a per-task inner product instead of ``FusionMTLTower``'s
       shared MLP (paper 2.4, see
       :class:`tzrec.modules.task_tower.OneRankPredictionHead`);
    3. the candidate-side ``_item_embedding_mlp`` is gone -- the candidate
       features already enter through the HSTU input preprocessor, and the
       task token's own channel replaces the fused ``user * item`` product.

    Everything else -- labels, bitmask decoding, losses (including the
    optional ``LossConfig.listwise_rank_loss``), metrics -- is inherited
    unchanged, so the two models are directly comparable on the same
    ``fusion_mtl_tower.task_configs``.

    Requires ``kernel: CUTLASS`` (with bf16/fp16 mixed precision) or the
    ``PYTORCH`` reference kernel (fp32-capable): the task-private mask is
    expressed as an NFUNC func tensor, which the Triton attention kernel
    does not implement, and the CUTLASS kernel only accepts fp16/bf16.

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
        assert model_config.WhichOneof("model") == "dlrm_hstu_onerank", (
            "invalid model config: %s" % self._model_config.WhichOneof("model")
        )
        assert isinstance(self._model_config, multi_task_rank_pb2.DlrmHSTUOneRank)
        self._init()

    def _init(self) -> None:
        super()._init()
        # Same contract as DlrmHSTU: `max_seq_len` both buckets the
        # jagged-kernel autotune and scales the attention output.  For
        # OneRank it bounds the *inflated* sequence (each candidate
        # becomes K + 1 tokens in the tokenizer), which the tokenizer
        # guards at runtime.
        set_static_max_seq_lens([self._model_config.max_seq_len])

    def _num_tasks(self) -> int:
        """Number of task tokens ``K``, one per task tower."""
        return len(self._task_configs)

    def _build_transducer(
        self, contextual_feature_dim: int, max_contextual_seq_len: int
    ) -> torch.nn.Module:
        num_tasks = self._num_tasks()
        if num_tasks == 0:
            raise ValueError(
                "dlrm_hstu_onerank requires at least one "
                "fusion_mtl_tower.task_configs entry: the number of task "
                "tokens K is taken from it."
            )
        for task_cfg in self._task_configs:
            if task_cfg.num_class > 1:
                raise ValueError(
                    f"dlrm_hstu_onerank scores each task with a single inner "
                    f"product, so num_class > 1 is not supported; task "
                    f"'{task_cfg.task_name}' has num_class={task_cfg.num_class}."
                )
        return OneRankHSTUTransducer(
            num_task_tokens=num_tasks,
            max_seq_len=self._model_config.max_seq_len,
            uih_embedding_dim=self.embedding_group.group_total_dim("uih"),
            target_embedding_dim=self.embedding_group.group_total_dim("candidate"),
            contextual_feature_dim=contextual_feature_dim,
            max_contextual_seq_len=max_contextual_seq_len,
            contextual_group_name=self._contextual_group_name,
            scaling_seqlen=self._model_config.max_seq_len,
            **config_to_kwargs(self._model_config.hstu),
            return_full_embeddings=False,
        )

    def _contextual_token_dim(self) -> int:
        """Flattened width of the contextual token ``s`` fed to the head.

        Equals ``max_contextual_seq_len * contextual_feature_dim`` for both
        branches of ``DlrmHSTU._init``, which is the layout of
        ``grouped_features[contextual_group_name]``.
        """
        return sum(self.embedding_group.group_dims(self._contextual_group_name))

    def _build_output_modules(self, stu_embedding_dim: int) -> None:
        onerank = self._model_config.onerank
        situation_discernment = None
        if onerank.HasField("situation_discernment"):
            situation_discernment = config_to_kwargs(onerank.situation_discernment)
        cross_task_head = None
        if onerank.HasField("cross_task_head"):
            cross_task_head = config_to_kwargs(onerank.cross_task_head)
        # config_to_kwargs emits enums as name strings and repeated fields
        # as lists, so cross_task_head.mask_type and
        # hybrid_chain_task_names already arrive in the exact form
        # build_cross_task_mask expects.  task_bias_init is wired by hand
        # for the empty-list -> None mapping; scorer_type is a plain
        # string in the proto, so it passes through verbatim and the head
        # validates the value at construction.
        task_bias_init = list(onerank.task_bias_init) or None
        self._onerank_head: torch.nn.Module = OneRankPredictionHead(
            embedding_dim=stu_embedding_dim,
            task_names=[task_cfg.task_name for task_cfg in self._task_configs],
            contextual_feature_dim=self._contextual_token_dim(),
            situation_discernment=situation_discernment,
            cross_task_head=cross_task_head,
            scorer_type=onerank.scorer_type,
            scorer_hidden_dim=onerank.scorer_hidden_dim,
            task_bias_init=task_bias_init,
        )

    def _predict_impl(
        self, grouped_features: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Score every candidate of the (possibly time-flipped) request.

        The head consumes the features in the same request order the
        transducer does -- reversed when
        ``sequence_timestamp_is_ascending`` is false; the flip /
        un-flip bookkeeping stays in :meth:`DlrmHSTU.predict`.
        """
        with record_function("## user_forward ##"):
            task_embeddings, _ = self._hstu_transducer(grouped_features)

        with record_function("## onerank_head ##"):
            # Post-flip lengths and contextual rows: `task_embeddings` is
            # jagged in the same (possibly reversed) request order the
            # transducer just consumed.
            scores = self._onerank_head(
                task_embeddings=task_embeddings,
                num_candidates=grouped_features["candidate.sequence_length"],
                contextual_embeddings=grouped_features[self._contextual_group_name],
            )

        # Keep the trailing class dim so `_output_to_prediction_impl`
        # sees the same `(N, 1)` shape FusionMTLTower produces.
        return {
            task_cfg.task_name: scores[:, i : i + 1]
            for i, task_cfg in enumerate(self._task_configs)
        }
